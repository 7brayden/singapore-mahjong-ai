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
    LinearModel, DANGER_MODEL_PATH, WIN_MODEL_PATH, VALUE_MODEL_PATH,
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
    """The Phase A model: expected NET POINTS from a post-discard state.

    Same table and features as the win model, but regression on the
    realized net_points — so a live full-flush track is worth more than
    a chicken hand at the same shanten. Labels are per-HAND (every
    decision in a seat's hand shares one outcome), so the effective
    sample count is seats×games, not rows — the reason this trains on a
    10k-game run.
    """
    print("\n" + "═" * 70)
    print("VALUE MODEL — E[net points | post-discard state] — label: net_points")
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
          f"range [{train_y.min():.0f}, {train_y.max():.0f}]\n")

    scaler = StandardScaler().fit(train_x)
    reg = Ridge(alpha=1.0)
    reg.fit(scaler.transform(train_x), train_y)
    pred = reg.predict(scaler.transform(test_x))

    def value_report(name, p):
        print(f"  {name:<28} MAE {mean_absolute_error(test_y, p):.3f}   "
              f"R² {r2_score(test_y, p):.4f}")

    value_report("ridge regression", pred)
    # Baselines: the mean (no model), and shanten alone (what the win
    # model effectively saw before the tai features existed)
    value_report("predict the mean", np.full_like(test_y, train_y.mean()))
    sh = train[:, [col["shanten"]]]
    sh_reg = Ridge(alpha=1.0).fit(sh, train_y)
    value_report("shanten only", sh_reg.predict(test[:, [col["shanten"]]]))
    gbm = HistGradientBoostingRegressor(max_iter=300, random_state=0)
    gbm.fit(train_x, train_y)
    value_report("hist gradient boosting", gbm.predict(test_x))

    coefficient_table(reg, OUTCOME_FEATURES, unit=3)
    model = export(scaler, reg, OUTCOME_FEATURES, VALUE_MODEL_PATH, {
        "label": "net_points", "games": int(n_games), "rows": int(len(data)),
        "test_mae": round(float(mean_absolute_error(test_y, pred)), 3),
        "test_r2": round(float(r2_score(test_y, pred)), 4),
        "trained": date.today().isoformat(),
    }, link="identity")
    print(f"\n  exported → {VALUE_MODEL_PATH}")
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    train_danger(args.data)
    train_win(args.data)
    train_value(args.data)


if __name__ == "__main__":
    main()
