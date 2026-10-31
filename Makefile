.PHONY: install dev test lint format train cv map publish upload clean

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
	uv run --extra train python -m olfacto.train --config configs/train_chemberta.yaml

cv:
	uv run --extra train python scripts/run_cv.py --output-dir outputs/cv --final

map:
	uv run --extra app python scripts/build_map.py

publish:
	uv run --extra hf python scripts/upload_hf.py

upload: publish

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
