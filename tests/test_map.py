import numpy as np
import pytest

from olfacto.map import extract_embeddings, nearest_neighbors, project_umap


def test_nearest_neighbors_returns_closest():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    result = nearest_neighbors(np.array([1.0, 0.0, 0.0]), embeddings, k=2)

    assert result == [0, 1]


def test_nearest_neighbors_excludes_query_index():
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    result = nearest_neighbors(
        embeddings[0], embeddings, k=2, exclude_index=0
    )

    assert 0 not in result
    assert result == [1, 2]


def test_nearest_neighbors_rejects_bad_dimensions():
    with pytest.raises(ValueError, match="same dimension"):
        nearest_neighbors(np.array([1.0, 0.0]), np.zeros((3, 3)), k=1)


def test_nearest_neighbors_rejects_invalid_k():
    with pytest.raises(ValueError, match="k must"):
        nearest_neighbors(np.array([1.0]), np.array([[1.0]]), k=0)


def test_extract_embeddings_shape():
    torch = pytest.importorskip("torch")

    class FakeModel:
        def eval(self):
            return self

        def parameters(self):
            yield torch.zeros(1)

        def __call__(self, input_ids, attention_mask, output_hidden_states=False):
            batch, length = input_ids.shape
            hidden = torch.ones((batch, length, 8))
            return type("Out", (), {"hidden_states": [hidden]})()

    class FakeTokenizer:
        def __call__(self, batch, **kwargs):
            return {
                "input_ids": torch.ones((len(batch), 5), dtype=torch.long),
                "attention_mask": torch.ones((len(batch), 5), dtype=torch.long),
            }

    embeddings = extract_embeddings(FakeModel(), FakeTokenizer(), ["CCO", "CCN"])

    assert embeddings.shape == (2, 8)


@pytest.mark.slow
def test_umap_projection_is_deterministic():
    pytest.importorskip("umap")

    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(60, 16))

    first = project_umap(embeddings, n_neighbors=5, seed=7)
    second = project_umap(embeddings, n_neighbors=5, seed=7)

    assert first.shape == (60, 2)
    assert np.allclose(first, second)
