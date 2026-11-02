import numpy as np
import pytest

pytest.importorskip("rdkit")

from olfacto.fingerprints import (  # noqa: E402
    fit_final_model,
    fit_predict_fold,
    morgan_matrix,
    predict_scores,
)


def test_morgan_matrix_shape_and_dtype():
    matrix = morgan_matrix(["CCO", "CCN", "c1ccccc1"], n_bits=128)

    assert matrix.shape == (3, 128)
    assert matrix.dtype == np.uint8
    assert set(np.unique(matrix)).issubset({0, 1})


def test_morgan_matrix_is_deterministic():
    first = morgan_matrix(["CCO", "c1ccccc1"], n_bits=64)
    second = morgan_matrix(["CCO", "c1ccccc1"], n_bits=64)

    assert np.array_equal(first, second)


def test_morgan_matrix_rejects_invalid_smiles():
    with pytest.raises(ValueError, match="invalid SMILES"):
        morgan_matrix(["isto não é química"])


def test_fit_predict_and_final_model():
    x = morgan_matrix(["CCO", "CCN", "c1ccccc1", "CCCl"], n_bits=64)
    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])

    fold_scores = fit_predict_fold(x, y, x)
    assert fold_scores.shape == (4, 2)

    models = fit_final_model(x, y)
    scores = predict_scores(x, models)
    assert scores.shape == (4, 2)
    assert scores[0, 0] > scores[2, 0]
