"""Download the published v1 checkpoint; verify the release's SHA-256."""
import argparse
from pathlib import Path
from urllib.request import urlretrieve

from web.backend.catalog import V1_SHA256, sha256

URL = "https://github.com/ArmaanSayyad/satellite-rl/releases/download/checkpoint-v1/ppo_stage2_riskaware_run1.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/ppo_stage2_riskaware_run1.zip"))
    args = parser.parse_args()
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != V1_SHA256:
            raise RuntimeError(f"Existing {path} does not match the release; preserved without overwriting")
        print(f"Verified existing checkpoint: {V1_SHA256}")
        return
    temporary = path.with_suffix(".download")
    urlretrieve(URL, temporary)
    if sha256(temporary) != V1_SHA256:
        raise RuntimeError(f"SHA-256 mismatch; untrusted download retained at {temporary}, not installed")
    temporary.replace(path)
    print(f"Verified {path}: {V1_SHA256}")


if __name__ == "__main__":
    main()
