.PHONY: install dev test lint format train upload clean

install:
	uv sync --extra dev

dev: install

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .

train:
	uv run python -m olfacto.train --config configs/train_chemberta.yaml

upload:
	uv run python scripts/upload_hf.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
