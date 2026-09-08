"""Package measured run artifacts and one experimental checkpoint for the application."""
import json
import shutil

from web.backend.catalog import ROOT, sha256


def main():
    source = ROOT / "runs/v2-imitation-hf-proof"
    target = ROOT / "assets/models/v2-imitation-hf"
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text())
    if sha256(source / "model.zip") != manifest["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint does not match its recorded training manifest")
    for name in ("model.zip", "manifest.json"):
        shutil.copyfile(source / name, target / name)
    destination = ROOT / "web/frontend/public/replays"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "replays/evaluation.json", destination / "evaluation.json")
    response = ROOT / "runs/teacher-response-validation.json"
    if response.exists():
        shutil.copyfile(response, destination / "planner-response-validation.json")
    documents = destination.parent / "docs"
    documents.mkdir(exist_ok=True)
    for name in ("EXPERIMENT_PROTOCOL.md", "MODEL_CARD.md", "DATA_CARD.md", "DEPLOYMENT.md", "ARCHITECTURE.md"):
        shutil.copyfile(ROOT / "docs" / name, documents / name)
    attribution = destination.parent / "data/kelvins_cdm"
    attribution.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "data/kelvins_cdm/SOURCE.md", attribution / "SOURCE.md")
    print(f"Packaged experimental model {manifest['checkpoint_sha256']}")


if __name__ == "__main__":
    main()
