"""Tests for beat morphology extraction, templates and classification scoring."""

from __future__ import annotations

import numpy as np
import pytest

from ecgmon.analysis.beat_classifier import (
    DS1_TRAIN,
    DS2_TEST,
    PACED_EXCLUDED,
    _clean,
    score_classification,
)
from ecgmon.analysis.morphology import (
    BEFORE_S,
    FEATURE_NAMES,
    _corr_to_template,
    beat_features,
    build_template,
    extract_beats,
    features_to_array,
    local_median_rr,
    qrs_widths,
)

FS = 360.0


def gaussian_beat(n, centre, width_samples, amplitude=1.0):
    t = np.arange(n)
    return amplitude * np.exp(-0.5 * ((t - centre) / width_samples) ** 2)


def synth_record(n_beats=60, rr_s=1.0, fs=FS, width_s=0.012, amplitude=1.0,
                 odd_every=0, odd_width_s=0.040, odd_amplitude=1.0):
    """Build a signal with regular beats, optionally with wide odd ones."""
    n = int(n_beats * rr_s * fs) + int(fs)
    x = np.zeros(n)
    peaks = []
    for i in range(n_beats):
        c = int((i + 0.5) * rr_s * fs)
        if c - int(0.3 * fs) < 0 or c + int(0.5 * fs) >= n:
            continue
        is_odd = odd_every and (i % odd_every == 0) and i > 0
        w = odd_width_s if is_odd else width_s
        a = odd_amplitude if is_odd else amplitude
        x += gaussian_beat(n, c, w * fs, a)
        peaks.append(c)
    return x, np.array(peaks, dtype=int)


# ----------------------------------------------------------- beat extraction


def test_extract_beats_shape():
    x, peaks = synth_record(n_beats=30)
    beats, kept = extract_beats(x, FS, peaks)
    expected_len = int(round(0.20 * FS)) + int(round(0.35 * FS))
    assert beats.shape[1] == expected_len
    assert beats.shape[0] == kept.size


def test_extract_beats_drops_edge_beats_rather_than_padding():
    x = np.zeros(int(2 * FS))
    peaks = np.array([2, int(1.0 * FS), x.size - 2])  # first and last too close
    beats, kept = extract_beats(x, FS, peaks)
    assert kept.tolist() == [1]


def test_extract_beats_empty_input():
    beats, kept = extract_beats(np.zeros(100), FS, np.array([], dtype=int))
    assert beats.shape[0] == 0
    assert kept.size == 0


# ------------------------------------------------------------------ template


def test_template_of_identical_beats_is_that_beat():
    one = gaussian_beat(200, 100, 10)
    beats = np.tile(one, (20, 1))
    assert np.allclose(build_template(beats), one)


def test_template_resists_a_minority_of_odd_beats():
    normal = gaussian_beat(200, 100, 8)
    weird = gaussian_beat(200, 100, 40, amplitude=3.0)
    beats = np.vstack([np.tile(normal, (18, 1)), np.tile(weird, (2, 1))])

    template = build_template(beats)
    assert _corr_to_template(normal[None, :], template)[0] > 0.99
    assert _corr_to_template(weird[None, :], template)[0] < 0.95


def test_template_from_single_beat():
    one = gaussian_beat(100, 50, 5)
    assert np.allclose(build_template(one[None, :]), one)


def test_template_requires_beats():
    with pytest.raises(ValueError):
        build_template(np.empty((0, 10)))


# --------------------------------------------------------------- correlation


def test_correlation_identical_is_one():
    t = gaussian_beat(200, 100, 10)
    assert _corr_to_template(t[None, :], t)[0] == pytest.approx(1.0)


def test_correlation_inverted_is_minus_one():
    t = gaussian_beat(200, 100, 10)
    assert _corr_to_template(-t[None, :], t)[0] == pytest.approx(-1.0)


def test_correlation_of_flat_beat_is_zero_not_nan():
    t = gaussian_beat(200, 100, 10)
    flat = np.zeros(200)
    out = _corr_to_template(flat[None, :], t)[0]
    assert np.isfinite(out)
    assert out == pytest.approx(0.0)


# -------------------------------------------------------------- QRS width


def test_wider_beat_measures_wider():
    narrow = gaussian_beat(400, 200, 5)[None, :]
    wide = gaussian_beat(400, 200, 20)[None, :]
    w_narrow = qrs_widths(narrow, FS, r_offset=200)[0]
    w_wide = qrs_widths(wide, FS, r_offset=200)[0]
    assert w_wide > w_narrow


def test_qrs_width_of_flat_beat_is_nan():
    flat = np.zeros((1, 200))
    assert np.isnan(qrs_widths(flat, FS, r_offset=100)[0])


# -------------------------------------------------------------------- RR


def test_local_median_rr_of_constant_rhythm():
    peaks = np.arange(0, int(60 * FS), int(FS))
    med = local_median_rr(peaks, FS)
    assert np.allclose(med, 1.0)


def test_local_median_rr_too_few_peaks():
    assert local_median_rr(np.array([5]), FS).size == 0


# --------------------------------------------------------------- features


def test_feature_count_matches_names():
    x, peaks = synth_record(n_beats=40)
    feats, beats, template = beat_features(x, FS, peaks)
    assert features_to_array(feats).shape[1] == len(FEATURE_NAMES)


def test_uniform_beats_give_ratios_near_one():
    """With identical, regularly spaced beats every relative feature is ~1."""
    x, peaks = synth_record(n_beats=60, rr_s=1.0)
    feats, _, _ = beat_features(x, FS, peaks)
    A = features_to_array(feats)
    cols = {n: A[:, i] for i, n in enumerate(FEATURE_NAMES)}

    assert np.nanmedian(cols["template_corr"]) > 0.99
    assert np.nanmedian(cols["r_amp_ratio"]) == pytest.approx(1.0, abs=0.05)
    assert np.nanmedian(cols["width_ratio"]) == pytest.approx(1.0, abs=0.05)
    assert np.nanmedian(cols["pre_rr_ratio"]) == pytest.approx(1.0, abs=0.05)


def test_wide_odd_beats_are_flagged_by_the_features():
    """A wide, different-shaped beat must show lower correlation and more width."""
    x, peaks = synth_record(n_beats=60, odd_every=6, odd_width_s=0.045)
    feats, _, _ = beat_features(x, FS, peaks)
    A = features_to_array(feats)
    corr = A[:, FEATURE_NAMES.index("template_corr")]
    wr = A[:, FEATURE_NAMES.index("width_ratio")]

    odd = wr > 1.5
    assert odd.any(), "expected some beats measured as wide"
    # The wide beats should correlate less well with the normal template.
    assert np.nanmedian(corr[odd]) < np.nanmedian(corr[~odd])


def test_beat_features_on_empty_peaks():
    feats, beats, template = beat_features(np.zeros(1000), FS, np.array([], dtype=int))
    assert feats == []


# -------------------------------------------------- classification scoring


def test_score_classification_counts():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    s = score_classification(y_true, y_pred)
    assert s.true_positives == 2
    assert s.false_positives == 1
    assert s.false_negatives == 1
    assert s.sensitivity == pytest.approx(2 / 3)
    assert s.ppv == pytest.approx(2 / 3)


def test_score_perfect_prediction():
    y = np.array([0, 1, 0, 1])
    s = score_classification(y, y.copy())
    assert s.sensitivity == 1.0
    assert s.ppv == 1.0


def test_predicting_all_normal_scores_zero_sensitivity():
    """The failure mode accuracy would hide: never predicting the minority class."""
    y_true = np.array([0] * 95 + [1] * 5)
    y_pred = np.zeros(100, dtype=int)
    s = score_classification(y_true, y_pred)
    assert s.sensitivity == 0.0
    # Accuracy would be 95% here, which is why it is not reported.


# ----------------------------------------------------------- split hygiene


def test_train_and_test_subjects_do_not_overlap():
    """The whole point of the inter-patient protocol."""
    assert set(DS1_TRAIN).isdisjoint(set(DS2_TEST))


def test_paced_records_excluded_from_both_sets():
    for rec in PACED_EXCLUDED:
        assert rec not in DS1_TRAIN
        assert rec not in DS2_TEST


def test_clean_imputes_non_finite_values():
    X = np.array([[1.0, np.nan], [3.0, 2.0], [np.inf, 4.0]])
    out = _clean(X)
    assert np.all(np.isfinite(out))
    assert out[0, 1] == pytest.approx(3.0)  # median of [2, 4]


def test_clean_handles_all_nan_column():
    X = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    out = _clean(X)
    assert np.all(np.isfinite(out))
