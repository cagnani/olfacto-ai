import numpy as np
import pytest

from olfacto.ensemble import combine_scores


def test_combine_scores_weighted_average():
    transformer = np.array([[1.0, 0.0]])
    fingerprint = np.array([[0.0, 1.0]])

    result = combine_scores(transformer, fingerprint, weight_fingerprint=0.25)

    np.testing.assert_allclose(result, [[0.75, 0.25]])


def test_combine_scores_rejects_invalid_weight():
    scores = np.zeros((1, 2))

    with pytest.raises(ValueError, match="between 0 and 1"):
        combine_scores(scores, scores, weight_fingerprint=1.5)


def test_combine_scores_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        combine_scores(np.zeros((1, 2)), np.zeros((2, 1)))
