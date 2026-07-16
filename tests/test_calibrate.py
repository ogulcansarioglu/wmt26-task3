import numpy as np
from sklearn.metrics import matthews_corrcoef

from src.calibrate import best_threshold, evaluate, mcc_sweep


def test_best_threshold_separable():
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    labels = np.array([0, 0, 0, 1, 1, 1])
    thr, mcc = best_threshold(scores, labels)
    assert mcc == 1.0
    assert 0.3 < thr < 0.7


def test_mcc_sweep_matches_sklearn():
    rng = np.random.default_rng(7)
    scores = rng.random(200)
    labels = (scores + rng.normal(0, 0.3, 200) > 0.5).astype(int)
    cands, mcc = mcc_sweep(scores, labels)
    for i in range(0, len(cands), 17):
        preds = (scores > cands[i]).astype(int)
        expected = 0.0 if len(np.unique(preds)) == 1 else matthews_corrcoef(labels, preds)
        assert abs(mcc[i] - expected) < 1e-9


def test_single_class_gold_does_not_crash():
    scores = np.array([0.2, 0.4, 0.6])
    labels = np.array([1, 1, 1])
    thr, mcc = best_threshold(scores, labels)
    assert mcc == 0.0
    assert np.isfinite(thr)


def test_evaluate_flags_degenerate_predictions():
    scores = np.array([0.9, 0.8, 0.85, 0.95])
    labels = np.array([1, 0, 1, 0])
    result = evaluate(scores, labels, threshold=0.0)  # everything predicted 1
    assert result["single_class_pred"]
    assert result["mcc"] == 0.0
