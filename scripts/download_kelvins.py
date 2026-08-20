#!/usr/bin/env python3
"""Download the ESA Kelvins Collision Avoidance Challenge dataset (CC-BY-4.0)
from its canonical Zenodo record and unpack it into data/kelvins_cdm/.

See docs/05-datasets.md for the license/provenance details this script
records into data/kelvins_cdm/SOURCE.md.
"""

import hashlib
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ZENODO_RECORD = "4463683"
FILE_NAME = "Collision Avoidance Challenge - Dataset.zip"
DOWNLOAD_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/"
    f"{FILE_NAME.replace(' ', '%20')}/content"
)
EXPECTED_MD5 = "d19dc8875229f2f6893253c38adddc87"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "kelvins_cdm"


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "kelvins_dataset.zip"

    if not zip_path.exists():
        print(f"Downloading {FILE_NAME} ({221_128_642 / 1e6:.0f} MB) from Zenodo...")
        urlretrieve(DOWNLOAD_URL, zip_path)
    else:
        print(f"{zip_path} already exists, skipping download.")

    actual_md5 = md5sum(zip_path)
    if actual_md5 != EXPECTED_MD5:
        print(
            f"ERROR: MD5 mismatch. Expected {EXPECTED_MD5}, got {actual_md5}. "
            "Deleting the downloaded file -- re-run this script to retry.",
            file=sys.stderr,
        )
        zip_path.unlink()
        sys.exit(1)
    print(f"MD5 verified: {actual_md5}")

    print("Unpacking...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)

    # The outer archive contains a *nested* zip
    # (kelvins_competition_data/train_data.zip) that must also be
    # extracted to get train_data.csv -- discovered by actually running
    # this end-to-end, not assumed from the Zenodo listing.
    nested_zips = list(DATA_DIR.rglob("*.zip"))
    for nested in nested_zips:
        if nested == zip_path:
            continue
        print(f"Unpacking nested archive {nested.relative_to(DATA_DIR)}...")
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(nested.parent)
        nested.unlink()  # redundant with the extracted CSV, drop it

    source_doc = DATA_DIR / "SOURCE.md"
    source_doc.write_text(
        f"""# Source and license

- Dataset: ESA Kelvins Collision Avoidance Challenge Dataset
- Zenodo DOI: 10.5281/zenodo.{ZENODO_RECORD}
- File: {FILE_NAME}
- MD5: {actual_md5}
- **License: CC-BY-4.0**

Attribution required per CC-BY-4.0: T. Uriot, D. Izzo, L. F. Simoes,
R. Abay, N. Einecke, S. Rebhan, J. Martinez-Heras, F. Letizia,
J. Siminski, K. Merz -- and ESA's Advanced Concepts Team / Space Debris
Office, with the US Space Surveillance Network as the underlying data
source. See docs/05-datasets.md for the full research notes on this
dataset (schema, redistribution terms, use in this project).
""",
        encoding="utf-8",
    )
    print(f"Wrote {source_doc}")

    # The zip is large and already MD5-verified once; keep the extracted
    # CSVs (small, useful) but drop the zip to save space -- comment this
    # out if you want to keep it for re-verification later.
    zip_path.unlink()
    print("Done.")


if __name__ == "__main__":
    main()
