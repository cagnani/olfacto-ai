"""Publish the static Hugging Face Space."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-id", default="SamuelCagnani/olfacto")
    parser.add_argument("--space-dir", type=Path, default=Path("space_static"))
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.space_id, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        folder_path=str(args.space_dir),
        repo_id=args.space_id,
        repo_type="space",
    )
    print(f"space published: https://huggingface.co/spaces/{args.space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
