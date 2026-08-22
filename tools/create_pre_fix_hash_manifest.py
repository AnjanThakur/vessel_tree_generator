"""Create the immutable pre-fix evidence manifest for the final system audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "submission_release" / "final_audit" / "pre_fix_hash_manifest.json"

PROTECTED_ROOTS = (
    "outputs/lca_ssm/ppt_priority_completion",
    "outputs/lca_ssm/stage1_heart_scaffold_vtk",
    "outputs/lca_ssm/stage1_parametric_heart_surface",
    "outputs/lca_ssm/lca_population_model",
    "outputs/lca_ssm/lca_population_cohort",
    "outputs/lca_ssm/lca_population_motion",
    "outputs/lca_ssm/lca_population_export",
    "submission_release/demo_cases",
    "submission_release/reports",
)

PROTECTED_REPORTS = (
    "submission_release/FINAL_PROJECT_REPORT.docx",
    "submission_release/VALIDATION_SUMMARY.json",
    "README.md",
    "SUBMISSION_GUIDE.md",
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
    root_summary: dict[str, dict[str, int | str]] = {}

    for relative_root in PROTECTED_ROOTS:
        path = ROOT / relative_root
        members = sorted(p for p in path.rglob("*") if p.is_file()) if path.exists() else []
        total_bytes = 0
        root_digest = hashlib.sha256()
        for member in members:
            relative = member.relative_to(ROOT).as_posix()
            digest = sha256(member)
            size = member.stat().st_size
            files[relative] = {"sha256": digest, "size_bytes": size}
            total_bytes += size
            root_digest.update(relative.encode("utf-8"))
            root_digest.update(b"\0")
            root_digest.update(digest.encode("ascii"))
            root_digest.update(b"\n")
        root_summary[relative_root] = {
            "exists": path.exists(),
            "file_count": len(members),
            "size_bytes": total_bytes,
            "aggregate_sha256": root_digest.hexdigest(),
        }

    for relative in PROTECTED_REPORTS:
        path = ROOT / relative
        if path.is_file() and relative not in files:
            files[relative] = {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }

    manifest = {
        "schema_version": 1,
        "purpose": "Pre-fix hashes for the final system truth audit; this is evidence, not a claim of clinical validity.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(ROOT),
        "branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain=v1", "--untracked-files=all").splitlines(),
        "protected_roots": root_summary,
        "files": files,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Hashed {len(files)} files")


if __name__ == "__main__":
    main()
