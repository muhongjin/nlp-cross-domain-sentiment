"""Tests for the metric helper. Run with: python -m pytest tests -q"""
import numpy as np

from common import compute_metrics, fmt


def test_perfect_predictions():
    y = [0, 1, 1, 0]
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_confusion_matrix_matches_error_counts():
    y_true = [0, 0, 0, 1, 1, 1]
    y_pred = [0, 1, 1, 1, 0, 1]
    m = compute_metrics(y_true, y_pred)
    tn, fp, fn, tp = np.array(m["confusion_matrix"]).ravel()
    assert (tn, fp, fn, tp) == (1, 2, 1, 2)
    assert m["per_class"]["negative"]["support"] == 3
    assert m["per_class"]["positive"]["support"] == 3


def test_metrics_recompute_from_confusion_matrix():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 500)
    y_pred = rng.integers(0, 2, 500)
    m = compute_metrics(y_true.tolist(), y_pred.tolist())

    tn, fp, fn, tp = np.array(m["confusion_matrix"]).ravel()
    accuracy = (tn + tp) / (tn + fp + fn + tp)
    precision_pos = tp / (tp + fp)
    recall_pos = tp / (tp + fn)
    precision_neg = tn / (tn + fn)
    recall_neg = tn / (tn + fp)
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos)
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)

    assert abs(m["accuracy"] - accuracy) < 1e-12
    assert abs(m["macro_f1"] - (f1_pos + f1_neg) / 2) < 1e-12
    assert abs(m["per_class"]["positive"]["precision"] - precision_pos) < 1e-12
    assert abs(m["per_class"]["negative"]["recall"] - recall_neg) < 1e-12


def test_fmt_rounds_half_down_for_the_tie_value():
    # 0.9015 as a double is just below the decimal tie, so formatting keeps it at 0.901.
    # Tables and figures both use this helper, which is why they agree.
    assert fmt(0.9015) == "0.901"
    assert fmt(0.8085) == "0.808"
    assert fmt(1.0) == "1.000"
