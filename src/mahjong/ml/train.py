"""Train the deal-in and win-probability models. Needs `.[ml]` extras.

    PYTHONPATH=src python3 -m mahjong.ml.train --data data/

Fits standardised logistic regressions and exports them as JSON for the
pure-Python evaluator in model.py. Prints, for every model:

  - ROC AUC / PR AUC against the hand-tuned heuristic baseline on a
    held-out 20% of GAMES (split by game_id — rows from one game never
    straddle the split)
  - a coefficient table: what the data says each signal is worth
  - a HistGradientBoosting ceiling check — how much accuracy a fancier
    model would buy over the interpretable linear one

The danger model trains on `waited_legal` (perfect-information: would
this discard deal in right now), then is sanity-checked on the tiles
agents actually threw against observed deal-ins.
"""

import argparse
import csv
import gzip
import os
import subprocess
from datetime import date

import numpy as np
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss,
    mean_absolute_error, r2_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from mahjong.ml.features import DANGER_FEATURES, OUTCOME_FEATURES
from mahjong.ml.model import (
    LinearModel, CompositeValueModel,
    DANGER_MODEL_PATH, WIN_MODEL_PATH, VALUE_MODEL_PATH,
)

TEST_FRACTION = 0.2
SPLIT_SEED = 0

# NOTE: the split is a seeded random choice of GAME ids, not a modulus.
# datagen assigns lineups by `game_id % len(LINEUPS)`, so any modulus
# that shares a factor with the lineup count silently sorts whole
# lineups into one side — the original `game_id % 5 == 4` put exactly
# one lineup in test and zero of it in train.


def load_table(path: str, chunk: int = 200_000):
    """CSV.gz → (header, float32 matrix), loaded in chunks.

    A single list-of-lists parse of the 10k-game danger table would
    peak at several GB of Python objects; chunking keeps the transient
    footprint at ~chunk rows.
    """
    parts = []
    with gzip.open(path, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        buf = []
        for row in reader:
            buf.append(row)
            if len(buf) >= chunk:
                parts.append(np.asarray(buf, dtype=np.float32))
                buf = []
        if buf:
            parts.append(np.asarray(buf, dtype=np.float32))
    data = np.vstack(parts) if parts else np.empty((0, len(header)), np.float32)
    return header, data


def split_by_game(data: np.ndarray, game_col: int = 0):
    """Hold out a seeded random 20% of GAMES (never rows within a game)."""
    gid = data[:, game_col].astype(np.int64)
    games = np.unique(gid)
    rng = np.random.default_rng(SPLIT_SEED)
    n_test = max(1, int(round(len(games) * TEST_FRACTION)))
    test_games = rng.permutation(games)[:n_test]
    test_mask = np.isin(gid, test_games)
    return data[~test_mask], data[test_mask]


def report_split_balance(data: np.ndarray, header: list) -> None:
    """Lineups must appear on both sides — this is what the old modulus
    split got wrong, so it is now checked out loud every run."""
    if "lineup" not in header:
        return
    lineup_col = header.index("lineup")
    train, test = split_by_game(data)
    tr = sorted(set(train[:, lineup_col].astype(int).tolist()))
    te = sorted(set(test[:, lineup_col].astype(int).tolist()))
    status = "OK" if tr == te else "*** IMBALANCED ***"
    print(f"  lineups — train {tr} / test {te}   {status}")


def fit_logreg(train_x, train_y):
    scaler = StandardScaler().fit(train_x)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.transform(train_x), train_y)
    return scaler, clf


def _git_commit() -> str:
    """Short hash of the code that produced these weights (provenance)."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def export(scaler, clf, feature_names, path, metadata, link="logistic"):
    metadata = {**metadata, "code_commit": _git_commit(),
                "n_features": len(feature_names)}
    coefs = np.atleast_2d(clf.coef_)[0]           # (n,) for regressors
    intercept = np.atleast_1d(clf.intercept_)[0]  # scalar for regressors
    model = LinearModel(
        features=list(feature_names),
        mean=[float(v) for v in scaler.mean_],
        scale=[float(v) for v in scaler.scale_],
        coef=[float(v) for v in coefs],
        intercept=float(intercept),
        metadata=metadata,
        link=link,
    )
    model.save(path)
    return model


def report(name, y_true, y_prob, baseline_score=None):
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    print(f"  {name:<28} ROC AUC {auc:.4f}   PR AUC {ap:.4f}   "
          f"log-loss {log_loss(y_true, y_prob):.4f}   "
          f"Brier {brier_score_loss(y_true, y_prob):.5f}")
    if baseline_score is not None:
        b_auc = roc_auc_score(y_true, baseline_score)
        b_ap = average_precision_score(y_true, baseline_score)
        print(f"  {'heuristic baseline':<28} ROC AUC {b_auc:.4f}   "
              f"PR AUC {b_ap:.4f}   (rank-only — not a probability)")
    return auc, ap


def coefficient_table(clf, feature_names, unit=10):
    coefs = np.atleast_2d(clf.coef_)[0]
    order = np.argsort(-np.abs(coefs))
    print("\n  Coefficients (standardised — comparable magnitudes):")
    for i in order:
        c = coefs[i]
        bar = "█" * min(30, int(abs(c) * unit))
        sign = "+" if c >= 0 else "−"
        print(f"    {feature_names[i]:<22} {sign}{abs(c):.3f}  {bar}")


def train_danger(data_dir: str):
    print("═" * 70)
    print("DANGER MODEL — P(discard deals in) — label: waited_legal")
    print("═" * 70)
    header, data = load_table(os.path.join(data_dir, "danger.csv.gz"))
    col = {name: i for i, name in enumerate(header)}
    feat_idx = [col[f] for f in DANGER_FEATURES]

    train, test = split_by_game(data)
    n_games = len(np.unique(data[:, 0]))
    print(f"  {len(data):,} rows from {n_games:,} games "
          f"({len(train):,} train / {len(test):,} test)")
    report_split_balance(data, header)
    y_name = "waited_legal"
    train_x, train_y = train[:, feat_idx], train[:, col[y_name]]
    test_x, test_y = test[:, feat_idx], test[:, col[y_name]]
    print(f"  positives: {int(train_y.sum()):,} train "
          f"/ {int(test_y.sum()):,} test "
          f"({100 * data[:, col[y_name]].mean():.2f}%)\n")

    scaler, clf = fit_logreg(train_x, train_y)
    probs = clf.predict_proba(scaler.transform(test_x))[:, 1]

    # Heuristic baseline: defense.py's current hand-tuned blend
    heuristic = (0.25 * test[:, col["sig_visibility"]]
                 + 0.25 * test[:, col["sig_discard_absence"]]
                 + 0.35 * test[:, col["sig_opponent_threat"]]
                 + 0.15 * test[:, col["sig_suit_safety"]])
    auc, ap = report("logistic regression", test_y, probs, heuristic)

    # Ceiling check: does a GBM see structure the linear model misses?
    gbm = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    gbm.fit(train_x, train_y)
    gbm_probs = gbm.predict_proba(test_x)[:, 1]
    report("hist gradient boosting", test_y, gbm_probs)

    # Sanity check on OBSERVED outcomes: only tiles agents actually
    # threw, scored against whether the game ended in a ron on them.
    chosen = test[test[:, col["chosen"]] == 1]
    if chosen[:, col["dealt_in"]].sum() > 0:
        c_probs = clf.predict_proba(
            scaler.transform(chosen[:, feat_idx]))[:, 1]
        c_auc = roc_auc_score(chosen[:, col["dealt_in"]], c_probs)
        print(f"\n  observed-outcome check: AUC {c_auc:.4f} on "
              f"{len(chosen):,} thrown tiles, "
              f"{int(chosen[:, col['dealt_in']].sum())} real deal-ins")

    coefficient_table(clf, DANGER_FEATURES)
    model = export(scaler, clf, DANGER_FEATURES, DANGER_MODEL_PATH, {
        "label": y_name, "games": int(n_games), "rows": int(len(data)),
        "test_roc_auc": round(float(auc), 4),
        "test_pr_auc": round(float(ap), 4),
        "trained": date.today().isoformat(),
    })
    print(f"\n  exported → {DANGER_MODEL_PATH}")
    return model


def train_win(data_dir: str):
    print("\n" + "═" * 70)
    print("WIN MODEL — P(this seat wins the hand) — label: won")
    print("═" * 70)
    header, data = load_table(os.path.join(data_dir, "outcome.csv.gz"))
    col = {name: i for i, name in enumerate(header)}
    feat_idx = [col[f] for f in OUTCOME_FEATURES]

    train, test = split_by_game(data)
    print(f"  {len(data):,} decision rows "
          f"({len(train):,} train / {len(test):,} test), "
          f"win rate {100 * data[:, col['won']].mean():.1f}%\n")
    train_x, train_y = train[:, feat_idx], train[:, col["won"]]
    test_x, test_y = test[:, feat_idx], test[:, col["won"]]

    scaler, clf = fit_logreg(train_x, train_y)
    probs = clf.predict_proba(scaler.transform(test_x))[:, 1]
    # Baseline: shanten alone (lower = better, negate for ranking)
    auc, ap = report("logistic regression", test_y, probs,
                     -test[:, col["shanten"]])
    coefficient_table(clf, OUTCOME_FEATURES)
    export(scaler, clf, OUTCOME_FEATURES, WIN_MODEL_PATH, {
        "label": "won", "rows": int(len(data)),
        "test_roc_auc": round(float(auc), 4),
        "trained": date.today().isoformat(),
    })
    print(f"\n  exported → {WIN_MODEL_PATH}")


def train_value(data_dir: str):
    """Phase C value model: a decomposed expectation, not one ridge.

        V = P(win) · E[points | win]  −  P(pay) · E[points | pay]

    net_points is 42% exact zeros with ±96-point tails; a single ridge
    on it hedges every prediction into about ±2 points (std 2.2 vs the
    label's 8.9), and the agent's EV comparison against the full-scale
    deal-in constant then systematically over-folds. Decomposed, each
    part is an easier problem — and the magnitude models never see a
    zero, so nothing drags them toward the mean.

    The monolithic ridge and a GBM are still fitted every run, as the
    printed baselines this has to beat.
    """
    print("\n" + "═" * 70)
    print("VALUE MODEL (composite) — P(win)·E[pts|win] − P(pay)·E[pts|pay]")
    print("═" * 70)
    header, data = load_table(os.path.join(data_dir, "outcome.csv.gz"))
    col = {name: i for i, name in enumerate(header)}
    feat_idx = [col[f] for f in OUTCOME_FEATURES]

    train, test = split_by_game(data)
    n_games = len(np.unique(data[:, 0]))
    print(f"  {len(data):,} rows from {n_games:,} games "
          f"({len(train):,} train / {len(test):,} test)")
    train_x, train_y = train[:, feat_idx], train[:, col["net_points"]]
    test_x, test_y = test[:, feat_idx], test[:, col["net_points"]]
    print(f"  label: mean {train_y.mean():+.2f}, std {train_y.std():.2f}, "
          f"zeros {100 * (train_y == 0).mean():.1f}%, "
          f"range [{train_y.min():.0f}, {train_y.max():.0f}]\n")

    # ── Components ───────────────────────────────────────────────────
    # Events are disjoint: won == net_points > 0, paid == net_points < 0
    # (tai-only accounting — the winner collects, payers pay, everyone
    # else is untouched). The two probabilities are modelled separately
    # rather than as one 3-class softmax so each stays a plain
    # LinearModel the pure-Python evaluator already knows how to read.
    train_won = (train_y > 0).astype(float)
    train_paid = (train_y < 0).astype(float)
    test_won = (test_y > 0).astype(float)
    test_paid = (test_y < 0).astype(float)

    sc_w, clf_w = fit_logreg(train_x, train_won)
    sc_p, clf_p = fit_logreg(train_x, train_paid)
    print("  P(win) component:")
    report("    logistic", test_won, clf_w.predict_proba(
        sc_w.transform(test_x))[:, 1])
    print("  P(pay) component:")
    report("    logistic", test_paid, clf_p.predict_proba(
        sc_p.transform(test_x))[:, 1])

    win_rows = train_y > 0
    pay_rows = train_y < 0
    sc_ws = StandardScaler().fit(train_x[win_rows])
    reg_ws = Ridge(alpha=1.0).fit(sc_ws.transform(train_x[win_rows]),
                                  train_y[win_rows])
    sc_ps = StandardScaler().fit(train_x[pay_rows])
    reg_ps = Ridge(alpha=1.0).fit(sc_ps.transform(train_x[pay_rows]),
                                  -train_y[pay_rows])
    t_win, t_pay = test_y > 0, test_y < 0
    print(f"  E[pts|win]  on {win_rows.sum():,} winner rows: "
          f"test MAE {mean_absolute_error(test_y[t_win], reg_ws.predict(sc_ws.transform(test_x[t_win]))):.3f}  "
          f"R² {r2_score(test_y[t_win], reg_ws.predict(sc_ws.transform(test_x[t_win]))):.4f}")
    print(f"  E[pts|pay]  on {pay_rows.sum():,} payer rows:  "
          f"test MAE {mean_absolute_error(-test_y[t_pay], reg_ps.predict(sc_ps.transform(test_x[t_pay]))):.3f}  "
          f"R² {r2_score(-test_y[t_pay], reg_ps.predict(sc_ps.transform(test_x[t_pay]))):.4f}\n")

    # ── Recomposed V on the held-out set ─────────────────────────────
    def _lin(scaler, est, link):
        return LinearModel(
            features=list(OUTCOME_FEATURES),
            mean=[float(v) for v in scaler.mean_],
            scale=[float(v) for v in scaler.scale_],
            coef=[float(v) for v in np.atleast_2d(est.coef_)[0]],
            intercept=float(np.atleast_1d(est.intercept_)[0]),
            metadata={}, link=link)

    composite = CompositeValueModel(
        win=_lin(sc_w, clf_w, "logistic"),
        win_size=_lin(sc_ws, reg_ws, "identity"),
        pay=_lin(sc_p, clf_p, "logistic"),
        pay_size=_lin(sc_ps, reg_ps, "identity"))

    p_w = clf_w.predict_proba(sc_w.transform(test_x))[:, 1]
    p_p = clf_p.predict_proba(sc_p.transform(test_x))[:, 1]
    e_w = np.maximum(0.0, reg_ws.predict(sc_ws.transform(test_x)))
    e_p = np.maximum(0.0, reg_ps.predict(sc_ps.transform(test_x)))
    pred = p_w * e_w - p_p * e_p

    def value_report(name, p):
        print(f"  {name:<28} MAE {mean_absolute_error(test_y, p):.3f}   "
              f"R² {r2_score(test_y, p):.4f}   spread σ {np.std(p):.2f}")

    value_report("composite", pred)
    # Baselines this has to beat, all on identical rows
    value_report("predict the mean", np.full_like(test_y, train_y.mean()))
    sc_m = StandardScaler().fit(train_x)
    mono = Ridge(alpha=1.0).fit(sc_m.transform(train_x), train_y)
    value_report("monolithic ridge", mono.predict(sc_m.transform(test_x)))
    gbm = HistGradientBoostingRegressor(max_iter=300, random_state=0)
    gbm.fit(train_x, train_y)
    value_report("hist gradient boosting", gbm.predict(test_x))
    print(f"  {'(label std — the target spread)':<28} σ {np.std(test_y):.2f}")

    # Decile lift — the number the agent actually lives on: sort the
    # held-out hands by predicted V, does actual value climb with it?
    order = np.argsort(pred)
    deciles = np.array_split(order, 10)
    actual = [float(test_y[idx].mean()) for idx in deciles]
    monotone = sum(1 for a, b in zip(actual, actual[1:]) if b >= a)
    print("\n  Decile lift (predicted V decile → actual mean points):")
    print("    " + "  ".join(f"{a:+.2f}" for a in actual))
    print(f"    monotone steps: {monotone}/9, "
          f"D10−D1 spread {actual[-1] - actual[0]:+.2f} pts")

    composite.metadata = {
        "label": "net_points (decomposed)", "games": int(n_games),
        "rows": int(len(data)),
        "test_mae": round(float(mean_absolute_error(test_y, pred)), 3),
        "test_r2": round(float(r2_score(test_y, pred)), 4),
        "pred_spread_std": round(float(np.std(pred)), 3),
        "decile_actual": [round(a, 3) for a in actual],
        "trained": date.today().isoformat(),
        "code_commit": _git_commit(), "n_features": len(OUTCOME_FEATURES),
    }
    composite.save(VALUE_MODEL_PATH)
    print(f"\n  exported → {VALUE_MODEL_PATH}")
    return composite


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    train_danger(args.data)
    train_win(args.data)
    train_value(args.data)


if __name__ == "__main__":
    main()
