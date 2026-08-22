"""Capture the immutable pre-fix state for the final math/anatomy/visual audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "submission_release"
    / "final_visual_anatomical_audit"
    / "PRE_FIX_HASHES.json"
)
HASH_ROOTS = (
    "submission_release",
    "outputs/lca_ssm/lca_population_model",
    "outputs/lca_ssm/lca_population_cohort",
    "outputs/lca_ssm/lca_population_motion",
    "outputs/lca_ssm/lca_population_export",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    ).stdout.rstrip()


def main() -> None:
    files: dict[str, dict[str, int | str]] = {}
    roots: dict[str, dict[str, int | str | bool]] = {}
    for relative_root in HASH_ROOTS:
        path = ROOT / relative_root
        members = sorted(p for p in path.rglob("*") if p.is_file()) if path.exists() else []
        aggregate = hashlib.sha256()
        size = 0
        for member in members:
            # The new manifest cannot be part of its own pre-write snapshot.
            if member.resolve() == OUTPUT.resolve():
                continue
            relative = member.relative_to(ROOT).as_posix()
            digest = sha256(member)
            member_size = member.stat().st_size
            files[relative] = {"sha256": digest, "size_bytes": member_size}
            size += member_size
            aggregate.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
        roots[relative_root] = {
            "exists": path.exists(),
            "file_count": sum(1 for key in files if key == relative_root or key.startswith(relative_root + "/")),
            "size_bytes": size,
            "aggregate_sha256": aggregate.hexdigest(),
        }

    payload = {
        "schema_version": 1,
        "purpose": "Pre-correction hashes for the final mathematical, anatomical, physiological, and visual audit.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT),
        "branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain=v1", "--untracked-files=all").splitlines(),
        "git_diff_stat": git("diff", "--stat").splitlines(),
        "hash_roots": roots,
        "files": files,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Hashed {len(files)} unique files")
    print(f"HEAD {payload['git_head']}")


if __name__ == "__main__":
    main()
