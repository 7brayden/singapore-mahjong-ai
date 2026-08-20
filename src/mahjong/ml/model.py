"""Pure-Python inference for models trained by train.py.

train.py fits a standardised logistic regression with scikit-learn and
exports it as JSON: feature names, per-feature mean/scale, coefficients,
intercept. This module evaluates that JSON with nothing but math.exp,
so agents and the server need no numpy/sklearn at play time — and the
weights are human-readable, which matters for a project whose whole
point is showing players WHY.

    p = sigmoid( sum_i coef_i * (x_i - mean_i) / scale_i + intercept )
"""

import json
import math
import os
from typing import Dict, List, Optional

_MODEL_DIR = os.path.dirname(__file__)
DANGER_MODEL_PATH = os.path.join(_MODEL_DIR, "danger_model.json")
WIN_MODEL_PATH = os.path.join(_MODEL_DIR, "win_model.json")
VALUE_MODEL_PATH = os.path.join(_MODEL_DIR, "value_model.json")


class LinearModel:
    """A standardised linear model evaluated in pure Python.

    link="logistic" (default) squashes the linear score through a
    sigmoid — probabilities. link="identity" returns the raw score —
    regression in POINTS. link="exp2" returns 2**score, capped at
    output_cap — used by the payout-size heads, because this table's
    payouts are literally base·2^(tai−1): a linear model in log2 space
    prices each extra tai as a DOUBLING, which a points-space ridge
    structurally cannot (it can only subtract constants — the exact
    failure that underpriced the ping hu forfeit 2-5x).
    """

    def __init__(self, features: List[str], mean: List[float],
                 scale: List[float], coef: List[float], intercept: float,
                 metadata: Optional[Dict] = None, link: str = "logistic",
                 output_cap: Optional[float] = None):
        n = len(features)
        if not (len(mean) == len(scale) == len(coef) == n):
            raise ValueError("model arrays disagree on feature count")
        if link not in ("logistic", "identity", "exp2"):
            raise ValueError(f"unknown link {link!r}")
        self.output_cap = output_cap
        self.features = features
        self.mean = mean
        self.scale = scale
        self.coef = coef
        self.intercept = intercept
        self.metadata = metadata or {}
        self.link = link

    @classmethod
    def load(cls, path: str) -> "LinearModel":
        with open(path) as f:
            data = json.load(f)
        return cls(data["features"], data["mean"], data["scale"],
                   data["coef"], data["intercept"], data.get("metadata"),
                   data.get("link", "logistic"), data.get("output_cap"))

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({
                "features": self.features,
                "mean": self.mean,
                "scale": self.scale,
                "coef": self.coef,
                "intercept": self.intercept,
                "metadata": self.metadata,
                "link": self.link,
                "output_cap": self.output_cap,
            }, f, indent=2)
            f.write("\n")

    def predict(self, x: List[float]) -> float:
        """One feature vector (ordered as self.features — the shared
        extractors in features.py guarantee this) → probability for the
        logistic link, points for the identity link."""
        z = self.intercept
        for i in range(len(x)):
            z += self.coef[i] * (x[i] - self.mean[i]) / self.scale[i]
        if self.link == "identity":
            return z
        if self.link == "exp2":
            cap = self.output_cap if self.output_cap is not None else 192.0
            return min(2.0 ** min(z, 16.0), cap)
        # Numerically safe sigmoid
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    def explain(self, x: List[float]) -> List[Dict]:
        """Per-feature contribution to the logit, largest first.

        This is what makes the model coach-friendly: 'copies_visible
        pushed danger up by 0.8, suit_safety pulled it down by 0.3'.
        """
        rows = []
        for i, name in enumerate(self.features):
            contribution = self.coef[i] * (x[i] - self.mean[i]) / self.scale[i]
            rows.append({"feature": name, "value": x[i],
                         "contribution": contribution})
        rows.sort(key=lambda r: -abs(r["contribution"]))
        return rows


class CompositeValueModel:
    """Phase C hand value: decomposed expectation instead of one ridge.

        V(x) = P(win|x) · E[points | win, x]  −  P(pay|x) · E[points | pay, x]

    net_points is a terrible direct regression target — 42% exact
    zeros with ±96 tails — so a single ridge model hedges everything
    into ±2 points, and the agent's EV comparison against the fixed
    deal-in cost (8.55) is rigged toward folding. Each component here
    is an easier problem: two probabilities (logistic), and two
    magnitude regressions fitted only on the hands where something
    actually happened (no zeros to hedge toward). The recomposed V
    keeps a realistic spread.

    Duck-types LinearModel where it matters: .features, .predict(x),
    .metadata, .link — so the agent, the analysis endpoint, and the
    _fresh() staleness check need no special cases.
    """

    link = "identity"  # output is points, like the model it replaces

    def __init__(self, win: LinearModel, win_size: LinearModel,
                 pay: LinearModel, pay_size: LinearModel,
                 metadata: Optional[Dict] = None):
        self.win, self.win_size = win, win_size
        self.pay, self.pay_size = pay, pay_size
        self.features = win.features
        for part in (win_size, pay, pay_size):
            if part.features != self.features:
                raise ValueError("composite components disagree on features")
        self.metadata = metadata or {}

    @classmethod
    def load(cls, path: str) -> "CompositeValueModel":
        with open(path) as f:
            data = json.load(f)
        parts = {name: LinearModel(c["features"], c["mean"], c["scale"],
                                   c["coef"], c["intercept"], c.get("metadata"),
                                   c.get("link", "logistic"),
                                   c.get("output_cap"))
                 for name, c in data["components"].items()}
        return cls(parts["win"], parts["win_size"], parts["pay"],
                   parts["pay_size"], data.get("metadata"))

    def save(self, path: str) -> None:
        def _dump(m: LinearModel) -> Dict:
            return {"features": m.features, "mean": m.mean, "scale": m.scale,
                    "coef": m.coef, "intercept": m.intercept,
                    "metadata": m.metadata, "link": m.link,
                    "output_cap": m.output_cap}
        with open(path, "w") as f:
            json.dump({
                "kind": "composite_value",
                "components": {"win": _dump(self.win),
                               "win_size": _dump(self.win_size),
                               "pay": _dump(self.pay),
                               "pay_size": _dump(self.pay_size)},
                "metadata": self.metadata,
            }, f, indent=2)
            f.write("\n")

    def predict_components(self, x: List[float]) -> tuple:
        """(p_win, e_win, p_pay, e_pay) — the four numbers predict()
        multiplies. Exposed so the decision layer can clamp the win
        term with engine-truth tai arithmetic (see agents/learned)."""
        return (self.win.predict(x),
                max(0.0, self.win_size.predict(x)),
                self.pay.predict(x),
                max(0.0, self.pay_size.predict(x)))

    def predict(self, x: List[float]) -> float:
        """Expected net points. Magnitudes are clamped at zero — a
        ridge extrapolating a negative 'size of the win' is noise."""
        p_win, e_win, p_pay, e_pay = self.predict_components(x)
        return p_win * e_win - p_pay * e_pay


def load_danger_model() -> Optional[LinearModel]:
    """The packaged deal-in model, or None if not trained yet."""
    if not os.path.exists(DANGER_MODEL_PATH):
        return None
    return LinearModel.load(DANGER_MODEL_PATH)


def load_win_model() -> Optional[LinearModel]:
    if not os.path.exists(WIN_MODEL_PATH):
        return None
    return LinearModel.load(WIN_MODEL_PATH)


def load_value_model():
    """The packaged hand-value model: expected NET POINTS from a
    post-discard state. Since Phase C this is a CompositeValueModel;
    a pre-composite monolithic artifact still loads as a LinearModel,
    so an old checkout keeps working."""
    if not os.path.exists(VALUE_MODEL_PATH):
        return None
    with open(VALUE_MODEL_PATH) as f:
        kind = json.load(f).get("kind")
    if kind == "composite_value":
        return CompositeValueModel.load(VALUE_MODEL_PATH)
    return LinearModel.load(VALUE_MODEL_PATH)
