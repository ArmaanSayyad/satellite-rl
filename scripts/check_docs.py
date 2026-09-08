"""Check repository Markdown links to local files and headings."""
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = [ROOT / "README.md", ROOT / "TECHNICAL.md", ROOT / "web/README.md"]
    files += list((ROOT / "docs").glob("*.md"))
    files += list((ROOT / "web/frontend/public/docs").glob("*.md"))
    failures = []
    for file in files:
        for link in re.findall(r"\]\(([^)]+)\)", file.read_text()):
            if "://" in link or link.startswith("mailto:"):
                continue
            path, _, anchor = unquote(link).partition("#")
            target = file.parent / path if path else file
            if not target.exists():
                failures.append(f"{file.relative_to(ROOT)}: missing {link}")
            elif anchor and target.suffix == ".md":
                headings = re.findall(r"^#+\s+(.*)$", target.read_text(), re.MULTILINE)
                slugs = {re.sub(r"[^\w\s-]", "", heading.lower()).replace(" ", "-") for heading in headings}
                if anchor not in slugs:
                    failures.append(f"{file.relative_to(ROOT)}: missing heading {link}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Verified local links in {len(files)} Markdown documents")


if __name__ == "__main__":
    main()
