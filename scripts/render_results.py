"""Render docs/results.md from cross-validation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from olfacto.report import render_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary", type=Path, default=Path("outputs/cv/cv_summary.json")
    )
    parser.add_argument(
        "--per-label", type=Path, default=Path("outputs/cv/per_label.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("docs/results.md"))
    parser.add_argument(
        "--fingerprint-summary",
        type=Path,
        default=Path("outputs/fingerprint/cv_summary.json"),
    )
    parser.add_argument(
        "--ensemble-summary",
        type=Path,
        default=Path("outputs/ensemble/summary.json"),
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    per_label = pd.read_csv(args.per_label)

    baselines = []
    if args.fingerprint_summary.is_file():
        fingerprint = json.loads(args.fingerprint_summary.read_text(encoding="utf-8"))
        baselines.append(
            {
                "name": "Morgan + LogReg",
                "description": "5-fold OOF, baseline clássico",
                "macro_roc_auc": fingerprint["oof"]["macro_roc_auc"],
            }
        )
    if args.ensemble_summary.is_file():
        ensemble = json.loads(args.ensemble_summary.read_text(encoding="utf-8"))
        weight = ensemble["weight_fingerprint"]
        baselines.append(
            {
                "name": f"Ensemble ({weight:.0%} fingerprint)",
                "description": "5-fold OOF, blend",
                "macro_roc_auc": ensemble["oof"]["macro_roc_auc"],
            }
        )

    args.output.write_text(
        render_results(summary, per_label, baselines=baselines), encoding="utf-8"
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
