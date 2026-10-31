"""Portuguese odor vocabulary: translation and olfactory family."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def load_lexicon(path: str | Path) -> dict[str, dict[str, str]]:
    """Load a JSON lexicon mapping English odor labels to PT-BR and family."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("lexicon must be a JSON object")
    return data


def translate_labels(
    labels: Iterable[str], lexicon: dict[str, dict[str, str]]
) -> list[str]:
    """Translate label names to PT-BR, preserving input order."""
    return [lexicon[label]["pt_br"] for label in labels]


def get_family(label: str, lexicon: dict[str, dict[str, str]]) -> str:
    """Return the olfactory family of a label."""
    return lexicon[label]["family"]
