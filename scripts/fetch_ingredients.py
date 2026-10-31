"""Fill missing SMILES in the Brazilian ingredient catalog from PubChem."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from olfacto.ingredients import load_ingredients, parse_pubchem_response

PUG_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}"
    "/property/IsomericSMILES/JSON"
)


def fetch_smiles(name: str, timeout: float = 15.0) -> str | None:
    """Fetch the isomeric SMILES for a compound name from PubChem."""
    url = PUG_URL.format(name=urllib.parse.quote(name))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

    return parse_pubchem_response(payload)


def is_valid_smiles(smiles: str) -> bool:
    """Validate a SMILES string with RDKit when available."""
    try:
        from rdkit import Chem
    except ImportError:
        return bool(smiles.strip())

    return Chem.MolFromSmiles(smiles) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/brazilian_ingredients.csv"),
    )
    parser.add_argument("--delay", type=float, default=0.25)
    args = parser.parse_args()

    df = load_ingredients(args.catalog)
    filled = 0
    failed: list[str] = []

    for index, row in df.iterrows():
        if row["smiles"].strip():
            continue

        smiles = fetch_smiles(row["pubchem_name"])
        if smiles and is_valid_smiles(smiles):
            df.at[index, "smiles"] = smiles
            filled += 1
            print(f"ok   {row['name_pt']:<22} {smiles}")
        else:
            failed.append(row["name_pt"])
            print(f"fail {row['name_pt']:<22} ({row['pubchem_name']})")
        time.sleep(args.delay)

    df.to_csv(args.catalog, index=False)
    print(f"\n{filled} SMILES preenchidos, {len(failed)} falhas")
    if failed:
        print("falhas:", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
