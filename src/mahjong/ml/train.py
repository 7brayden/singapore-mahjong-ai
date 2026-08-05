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
from datetime import date

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from mahjong.ml.features import DANGER_FEATURES, OUTCOME_FEATURES
from mahjong.ml.model import (
    LinearModel, DANGER_MODEL_PATH, WIN_MODEL_PATH,
)

TEST_FOLD = 4  # game_id % 5 == 4 → held out


def load_table(path: str):
    """CSV.gz → (header, float32 matrix)."""
    with gzip.open(path, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) for v in row] for row in reader]
    return header, np.asarray(rows, dtype=np.float32)


def split_by_game(data: np.ndarray, game_col: int = 0):
    test_mask = (data[:, game_col].astype(np.int64) % 5) == TEST_FOLD
    return data[~test_mask], data[test_mask]


def fit_logreg(train_x, train_y):
    scaler = StandardScaler().fit(train_x)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(scaler.transform(train_x), train_y)
    return scaler, clf


def export(scaler, clf, feature_names, path, metadata):
    model = LinearModel(
        features=list(feature_names),
        mean=[float(v) for v in scaler.mean_],
        scale=[float(v) for v in scaler.scale_],
        coef=[float(v) for v in clf.coef_[0]],
        intercept=float(clf.intercept_[0]),
        metadata=metadata,
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


def coefficient_table(clf, feature_names):
    order = np.argsort(-np.abs(clf.coef_[0]))
    print("\n  Coefficients (standardised — comparable magnitudes):")
    for i in order:
        c = clf.coef_[0][i]
        bar = "█" * min(30, int(abs(c) * 10))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    train_danger(args.data)
    train_win(args.data)


if __name__ == "__main__":
    main()
