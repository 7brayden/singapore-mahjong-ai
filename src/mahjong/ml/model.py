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
    regression, used by the hand-value model, whose output is POINTS.
    """

    def __init__(self, features: List[str], mean: List[float],
                 scale: List[float], coef: List[float], intercept: float,
                 metadata: Optional[Dict] = None, link: str = "logistic"):
        n = len(features)
        if not (len(mean) == len(scale) == len(coef) == n):
            raise ValueError("model arrays disagree on feature count")
        if link not in ("logistic", "identity"):
            raise ValueError(f"unknown link {link!r}")
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
                   data.get("link", "logistic"))

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


def load_danger_model() -> Optional[LinearModel]:
    """The packaged deal-in model, or None if not trained yet."""
    if not os.path.exists(DANGER_MODEL_PATH):
        return None
    return LinearModel.load(DANGER_MODEL_PATH)


def load_win_model() -> Optional[LinearModel]:
    if not os.path.exists(WIN_MODEL_PATH):
        return None
    return LinearModel.load(WIN_MODEL_PATH)


def load_value_model() -> Optional[LinearModel]:
    """The packaged hand-value model: expected NET POINTS from a
    post-discard state. Identity link — its output is points."""
    if not os.path.exists(VALUE_MODEL_PATH):
        return None
    return LinearModel.load(VALUE_MODEL_PATH)
