"""Ventricular-beat classification and its evaluation protocol.

Two things in here matter more than the model itself.

**The split.** Beat classification results are only meaningful when training
and test sets contain *different subjects*. A random split over beats puts
the same person's heartbeats on both sides; since a subject's beats all look
alike, the model can recognise the subject rather than the pathology and
report accuracy that collapses on a new patient. The standard remedy is the
de Chazal DS1/DS2 division of MIT-BIH, which this module uses. It also
computes the intra-patient number on request, purely so the size of the
inflation can be seen rather than argued about.

**The class balance.** Ventricular beats are a small minority. A classifier
that answers "normal" to everything scores over 90% accuracy, so accuracy is
not reported at all -- only sensitivity and PPV for the ventricular class.

The model is logistic regression: with eight interpretable features, its
coefficients can be read directly and checked against clinical expectation.
A beat that looks unlike the patient's normal, is wide, and arrives early
should push the score up; if a fitted coefficient disagrees with that, the
feature pipeline is wrong and the coefficient is the thing that reveals it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# de Chazal et al. (2004) inter-patient division of MIT-BIH.
# The four paced records (102, 104, 107, 217) are excluded, per AAMI EC57:
# paced rhythms are a different problem and are conventionally not scored.
DS1_TRAIN = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]

DS2_TEST = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]

PACED_EXCLUDED = ["102", "104", "107", "217"]


@dataclass
class ClassificationScore:
    """Ventricular-class performance."""

    name: str
    n_beats: int
    n_ventricular: int
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def sensitivity(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else float("nan")

    @property
    def ppv(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else float("nan")

    @property
    def f1(self) -> float:
        se, pp = self.sensitivity, self.ppv
        if not np.isfinite(se) or not np.isfinite(pp) or (se + pp) == 0:
            return float("nan")
        return 2 * se * pp / (se + pp)

    def __str__(self) -> str:
        return (
            f"{self.name:<22} beats={self.n_beats:>7}  V={self.n_ventricular:>6}  "
            f"TP={self.true_positives:>5}  FP={self.false_positives:>5}  "
            f"FN={self.false_negatives:>5}  "
            f"Se={self.sensitivity * 100:6.2f}%  PPV={self.ppv * 100:6.2f}%  "
            f"F1={self.f1 * 100:6.2f}%"
        )


def score_classification(
    y_true: np.ndarray, y_pred: np.ndarray, name: str = ""
) -> ClassificationScore:
    """Score ventricular-class predictions (1 = ventricular, 0 = other)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.count_nonzero((y_pred == 1) & (y_true == 1)))
    fp = int(np.count_nonzero((y_pred == 1) & (y_true == 0)))
    fn = int(np.count_nonzero((y_pred == 0) & (y_true == 1)))

    return ClassificationScore(
        name=name,
        n_beats=int(y_true.size),
        n_ventricular=int(np.count_nonzero(y_true == 1)),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def _clean(X: np.ndarray) -> np.ndarray:
    """Replace non-finite feature values with column medians.

    NaNs arise legitimately at record edges, where a beat has no preceding or
    following interval. Dropping those beats would quietly change the test set
    between runs, so they are imputed instead.
    """
    X = np.array(X, dtype=np.float64, copy=True)
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.all():
            col[:] = 0.0
        elif bad.any():
            col[bad] = np.median(col[~bad])
    return X


class VentricularClassifier:
    """Logistic regression over morphology and rhythm-context features."""

    def __init__(self, class_weight: str | dict | None = "balanced", C: float = 1.0):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        # class_weight="balanced" counteracts the minority class; without it
        # the fit is dominated by normal beats and sensitivity collapses.
        self.model = LogisticRegression(
            class_weight=class_weight, C=C, max_iter=2000, solver="lbfgs"
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VentricularClassifier":
        X = self.scaler.fit_transform(_clean(X))
        self.model.fit(X, np.asarray(y).astype(int))
        self._fitted = True
        return self

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("classifier is not fitted")
        return self.model.predict_proba(self.scaler.transform(_clean(X)))[:, 1]

    def coefficients(self, feature_names: list[str]) -> list[tuple[str, float]]:
        """Fitted coefficients, largest magnitude first.

        On standardised features these are comparable to one another, so the
        ordering says which features the model actually relies on.
        """
        if not self._fitted:
            raise RuntimeError("classifier is not fitted")
        coefs = self.model.coef_[0]
        pairs = list(zip(feature_names, coefs))
        return sorted(pairs, key=lambda p: abs(p[1]), reverse=True)
