"""Sensory context retrieval over the curated dish corpus."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_corpus(path: str | Path) -> list[dict]:
    """Load a JSONL corpus of dish descriptions."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _document(entry: dict) -> str:
    parts = [
        entry["dish"],
        entry["category"],
        " ".join(entry["ingredients"]),
        entry["description"],
        " ".join(entry["aromas"]),
    ]
    return " ".join(parts)


class SensoryRetriever:
    """TF-IDF retriever for sensory descriptions of Brazilian dishes."""

    def __init__(self, corpus: list[dict]) -> None:
        if not corpus:
            raise ValueError("corpus must not be empty")

        self.corpus = corpus
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(
            [_document(entry) for entry in corpus]
        )

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        if k < 1:
            raise ValueError("k must be at least 1")
        if not query or not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix)[0]
        order = scores.argsort()[::-1][:k]

        return [
            {**self.corpus[index], "score": float(scores[index])}
            for index in order
            if scores[index] > 0
        ]
