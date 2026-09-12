"""Fail if a local README image is missing from the checkout or Git index."""

from pathlib import Path
import re
import subprocess


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    content = (root / "README.md").read_text(encoding="utf-8")
    sources = re.findall(r'<img\b[^>]*\bsrc="([^"]+)"', content)
    sources += re.findall(r'!\[[^\]]*\]\(([^\s)]+)\)', content)
    tracked = set(subprocess.check_output(
        ["git", "ls-files"], cwd=root, text=True
    ).splitlines())
    local = [s for s in sources if not s.startswith(("https://", "http://", "data:"))]
    failures = [s for s in local if not (root / s).is_file() or s not in tracked]
    if failures:
        raise SystemExit("Missing or untracked README images: " + ", ".join(failures))
    print(f"PASS: {len(local)} local README images exist and are tracked")


if __name__ == "__main__":
    main()
