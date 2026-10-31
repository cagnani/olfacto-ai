import os
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from olfacto.data import load_odor_dataset  # noqa: E402
from olfacto.train import (  # noqa: E402
    TrainConfig,
    compute_pos_weight,
    fit,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "curated_GS_LF_merged_4983.csv"


def test_config_from_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump({"epochs": 3, "batch_size": 8, "fold": 2}), encoding="utf-8"
    )

    config = TrainConfig.from_yaml(path)

    assert config.epochs == 3
    assert config.batch_size == 8
    assert config.fold == 2


def test_pos_weight_ratio_and_cap():
    y = np.array([[1, 0], [0, 0], [0, 0], [0, 0]], dtype=np.int8)

    weights = compute_pos_weight(y, cap=20.0)

    assert weights[0] == pytest.approx(3.0)
    assert weights[1] == pytest.approx(1.0)


def test_pos_weight_respects_cap():
    y = np.zeros((100, 1), dtype=np.int8)
    y[0, 0] = 1

    weights = compute_pos_weight(y, cap=20.0)

    assert weights[0] == pytest.approx(20.0)


@pytest.mark.slow
def test_fit_smoke(tmp_path):
    if not os.environ.get("RUN_SLOW"):
        pytest.skip("set RUN_SLOW=1 to run the training smoke test")

    df = load_odor_dataset(DATA_PATH).head(24).reset_index(drop=True)
    config = TrainConfig(
        max_length=32,
        batch_size=8,
        epochs=1,
        learning_rate=5e-5,
        early_stopping_patience=1,
        val_min_positives=1,
        output_dir=str(tmp_path / "model"),
    )

    metrics = fit(config, df.iloc[:20], df.iloc[20:])

    assert len(metrics["history"]) == 1
    assert np.isfinite(metrics["history"][0]["train_loss"])
    assert (tmp_path / "model" / "metrics.json").is_file()
    assert (tmp_path / "model" / "config.json").is_file()
