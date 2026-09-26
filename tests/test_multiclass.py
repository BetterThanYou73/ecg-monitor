"""Tests for AAMI multi-class beat classification."""

from __future__ import annotations

import numpy as np
import pytest

from ecgmon.analysis.multiclass import (
    AAMI_CLASSES,
    MULTICLASS_FEATURES,
    MultiClassBeatClassifier,
    confusion,
    drop_unclassifiable,
    format_confusion,
    multiclass_columns,
    score_multiclass,
    symbols_to_aami,
)
from ecgmon.analysis.morphology import FEATURE_NAMES


# ------------------------------------------------------------ symbol mapping


def test_symbols_map_to_expected_classes():
    syms = np.array(["N", "A", "V", "F", "Q"])
    assert symbols_to_aami(syms).tolist() == [0, 1, 2, 3, 4]


def test_bundle_branch_blocks_are_normal_class():
    """L and R are bundle-branch block beats; AAMI groups them with N."""
    assert symbols_to_aami(np.array(["L", "R"])).tolist() == [0, 0]


def test_supraventricular_variants_group_together():
    for s in ["A", "a", "J", "S"]:
        assert symbols_to_aami(np.array([s]))[0] == AAMI_CLASSES.index("S")


def test_ventricular_variants_group_together():
    for s in ["V", "E"]:
        assert symbols_to_aami(np.array([s]))[0] == AAMI_CLASSES.index("V")


def test_unknown_symbol_maps_to_q_rather_than_raising():
    assert symbols_to_aami(np.array(["~"]))[0] == AAMI_CLASSES.index("Q")


def test_empty_symbols():
    assert symbols_to_aami(np.array([])).size == 0


# -------------------------------------------------------------- Q handling


def test_drop_unclassifiable_removes_q_beats():
    X = np.arange(10, dtype=float).reshape(5, 2)
    y = np.array([0, 4, 2, 4, 1])
    X2, y2, n = drop_unclassifiable(X, y)
    assert n == 2
    assert y2.tolist() == [0, 2, 1]
    assert X2.shape[0] == 3


def test_drop_unclassifiable_with_no_q():
    X = np.zeros((3, 2))
    y = np.array([0, 1, 2])
    _, y2, n = drop_unclassifiable(X, y)
    assert n == 0
    assert y2.tolist() == [0, 1, 2]


# ------------------------------------------------------------------ scoring


def test_score_multiclass_one_vs_rest():
    y_true = np.array([0, 0, 2, 2, 1])
    y_pred = np.array([0, 2, 2, 2, 0])
    scores = {s.cls: s for s in score_multiclass(y_true, y_pred)}

    assert scores["N"].tp == 1
    assert scores["N"].fn == 1     # one N called V
    assert scores["N"].fp == 1     # one S called N
    assert scores["V"].tp == 2
    assert scores["V"].fp == 1


def test_absent_classes_are_skipped_not_reported_as_zero():
    """A class with no reference beats says nothing; omit it."""
    y = np.array([0, 0, 2])
    scores = {s.cls for s in score_multiclass(y, y.copy())}
    assert scores == {"N", "V"}
    assert "S" not in scores


def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 2, 3])
    for s in score_multiclass(y, y.copy()):
        assert s.sensitivity == 1.0
        assert s.ppv == 1.0


# --------------------------------------------------------------- confusion


def test_confusion_matrix_rows_are_truth():
    y_true = np.array([0, 0, 2])
    y_pred = np.array([0, 2, 2])
    m = confusion(y_true, y_pred)
    assert m[0, 0] == 1
    assert m[0, 2] == 1    # one N predicted as V
    assert m[2, 2] == 1


def test_confusion_totals_match_input():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, 200)
    y_pred = rng.integers(0, 5, 200)
    assert confusion(y_true, y_pred).sum() == 200


def test_format_confusion_labels_rows():
    m = confusion(np.array([0, 2]), np.array([0, 2]))
    text = format_confusion(m)
    assert "N" in text and "V" in text
    assert "predicted" in text


# ----------------------------------------------------------------- columns


def test_multiclass_columns_resolve_against_feature_names():
    cols = multiclass_columns(FEATURE_NAMES)
    assert len(cols) == len(MULTICLASS_FEATURES)
    for name, idx in zip(MULTICLASS_FEATURES, cols):
        assert FEATURE_NAMES[idx] == name


def test_multiclass_features_include_rhythm_and_p_wave():
    """S is separable by timing and atrial activity, not QRS shape."""
    assert "pre_rr_ratio" in MULTICLASS_FEATURES
    assert "p_corr" in MULTICLASS_FEATURES


# -------------------------------------------------------------- classifier


def _toy_dataset(n=400, seed=0):
    """Four separable clusters standing in for the AAMI classes."""
    rng = np.random.default_rng(seed)
    centres = {0: [0, 0], 1: [3, 0], 2: [0, 3], 3: [3, 3]}
    X, y = [], []
    for cls, c in centres.items():
        k = n if cls == 0 else n // 4
        X.append(rng.normal(c, 0.4, size=(k, 2)))
        y.append(np.full(k, cls))
    X = np.vstack(X)
    # pad to the expected feature width
    X = np.hstack([X, np.zeros((X.shape[0], len(MULTICLASS_FEATURES) - 2))])
    return X, np.concatenate(y)


def test_classifier_learns_separable_classes():
    X, y = _toy_dataset()
    clf = MultiClassBeatClassifier().fit(X, y)
    pred = clf.predict(X)
    assert (pred == y).mean() > 0.9


def test_classifier_predicts_minority_classes():
    """Unweighted, a model would answer N to everything; weighting prevents that."""
    X, y = _toy_dataset()
    clf = MultiClassBeatClassifier().fit(X, y)
    pred = clf.predict(X)
    for cls in (1, 2, 3):
        assert np.count_nonzero(pred == cls) > 0, f"class {cls} never predicted"


def test_default_weights_are_not_balanced():
    """'balanced' over-predicts tiny classes; the default is deliberately milder."""
    clf = MultiClassBeatClassifier()
    assert isinstance(clf.model.class_weight, dict)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        MultiClassBeatClassifier().predict(np.zeros((2, len(MULTICLASS_FEATURES))))


def test_proba_rows_sum_to_one():
    X, y = _toy_dataset()
    clf = MultiClassBeatClassifier().fit(X, y)
    p = clf.predict_proba(X)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_coefficients_cover_every_fitted_class():
    X, y = _toy_dataset()
    clf = MultiClassBeatClassifier().fit(X, y)
    coefs = clf.coefficients(MULTICLASS_FEATURES)
    assert set(coefs) == {"N", "S", "V", "F"}
    for pairs in coefs.values():
        assert len(pairs) == len(MULTICLASS_FEATURES)


def test_nan_features_do_not_break_fitting():
    X, y = _toy_dataset()
    X[::10, 0] = np.nan
    clf = MultiClassBeatClassifier().fit(X, y)
    assert np.all(np.isfinite(clf.predict_proba(X)))
