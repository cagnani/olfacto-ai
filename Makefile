.PHONY: install dev test lint format train cv map publish upload space-static clean

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

space-static:
	uv run --extra train --extra app --extra chem --extra hf python scripts/build_static_space.py
	uv run --extra hf python scripts/upload_static_space.py

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
