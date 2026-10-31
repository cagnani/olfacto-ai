"""Embeddings and 2D projection of the olfactory space."""

from __future__ import annotations

import numpy as np


def extract_embeddings(
    model,
    tokenizer,
    smiles: list[str],
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    """Mean-pooled last hidden state for each SMILES."""
    import torch

    model.eval()
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []

    for start in range(0, len(smiles), batch_size):
        batch = smiles[start : start + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        chunks.append(pooled.cpu().numpy())

    return np.concatenate(chunks, axis=0)


def project_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Project embeddings to 2D with UMAP."""
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        metric="cosine",
    )
    return reducer.fit_transform(np.asarray(embeddings))


def nearest_neighbors(
    query: np.ndarray,
    embeddings: np.ndarray,
    k: int = 10,
    exclude_index: int | None = None,
) -> list[int]:
    """Indices of the k most cosine-similar embeddings."""
    if k < 1:
        raise ValueError("k must be at least 1")

    query = np.asarray(query, dtype=float)
    embeddings = np.asarray(embeddings, dtype=float)
    if query.ndim != 1 or embeddings.ndim != 2:
        raise ValueError("query must be 1-D and embeddings must be 2-D")
    if query.shape[0] != embeddings.shape[1]:
        raise ValueError("query and embeddings must share the same dimension")

    query_norm = query / (np.linalg.norm(query) + 1e-12)
    embedding_norms = embeddings / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    )
    similarities = embedding_norms @ query_norm

    if exclude_index is not None:
        similarities[exclude_index] = -np.inf

    return np.argsort(-similarities)[:k].tolist()


def dominant_family(weights, families: list[str]) -> str:
    """Return the family with the largest summed weight."""
    totals: dict[str, float] = {}
    for value, family in zip(weights, families, strict=True):
        totals[family] = totals.get(family, 0.0) + float(value)
    return max(totals, key=totals.get)
