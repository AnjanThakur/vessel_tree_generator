"""Compare the final repository artifacts with the pre-fix evidence manifest.

The comparison separates immutable evidence/statistics from artifacts that this
audit intentionally regenerated.  It is an engineering-integrity record, not a
claim of clinical validation.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "submission_release" / "final_audit"
PRE_PATH = AUDIT_DIR / "pre_fix_hash_manifest.json"
POST_PATH = AUDIT_DIR / "post_fix_hash_manifest.json"
COMPARISON_PATH = AUDIT_DIR / "protected_integrity_comparison.json"

IMMUTABLE_ROOTS = (
    "outputs/lca_ssm/ppt_priority_completion",
    "outputs/lca_ssm/stage1_heart_scaffold_vtk",
    "outputs/lca_ssm/stage1_parametric_heart_surface",
    "outputs/lca_ssm/lca_population_model",
)

INTENTIONALLY_REGENERATED_ROOTS = (
    "outputs/lca_ssm/lca_population_cohort",
    "outputs/lca_ssm/lca_population_motion",
    "outputs/lca_ssm/lca_population_export",
    "submission_release/demo_cases",
    "submission_release/reports",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def root_for(relative: str, roots: tuple[str, ...]) -> str | None:
    for root in roots:
        if relative == root or relative.startswith(root + "/"):
            return root
    return None


def empty_counts() -> dict[str, int]:
    return {"unchanged": 0, "changed": 0, "missing": 0, "added": 0}


def main() -> None:
    pre = json.loads(PRE_PATH.read_text(encoding="utf-8"))
    pre_files: dict[str, dict[str, int | str]] = pre["files"]
    roots = IMMUTABLE_ROOTS + INTENTIONALLY_REGENERATED_ROOTS
    summaries = {root: empty_counts() for root in roots}
    current: dict[str, dict[str, int | str | bool]] = {}
    differences: list[dict[str, str | int]] = []

    for relative, before in sorted(pre_files.items()):
        path = ROOT / relative
        root = root_for(relative, roots)
        if not path.is_file():
            state = "missing"
            current[relative] = {
                "exists": False,
                "pre_sha256": before["sha256"],
                "state": state,
            }
        else:
            digest = sha256(path)
            state = "unchanged" if digest == before["sha256"] else "changed"
            current[relative] = {
                "exists": True,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "pre_sha256": before["sha256"],
                "state": state,
            }
        if root is not None:
            summaries[root][state] += 1
        if state != "unchanged":
            differences.append({"path": relative, "state": state, "root": root or "other"})

    for root in roots:
        path = ROOT / root
        if not path.exists():
            continue
        for member in sorted(p for p in path.rglob("*") if p.is_file()):
            relative = member.relative_to(ROOT).as_posix()
            if relative in pre_files:
                continue
            digest = sha256(member)
            current[relative] = {
                "exists": True,
                "sha256": digest,
                "size_bytes": member.stat().st_size,
                "pre_sha256": None,
                "state": "added",
            }
            summaries[root]["added"] += 1
            differences.append({"path": relative, "state": "added", "root": root})

    immutable_pass = all(
        summaries[root]["changed"] == 0 and summaries[root]["missing"] == 0
        for root in IMMUTABLE_ROOTS
    )
    post = {
        "schema_version": 1,
        "purpose": "Post-fix hashes and states relative to the immutable pre-fix evidence manifest.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pre_fix_manifest": PRE_PATH.relative_to(ROOT).as_posix(),
        "pre_fix_git_head": pre["git_head"],
        "files": current,
    }
    comparison = {
        "schema_version": 1,
        "created_utc": post["created_utc"],
        "status": "PASS" if immutable_pass else "FAIL",
        "immutable_evidence_preserved": immutable_pass,
        "immutable_roots": list(IMMUTABLE_ROOTS),
        "intentionally_regenerated_roots": list(INTENTIONALLY_REGENERATED_ROOTS),
        "root_comparison": summaries,
        "difference_count": len(differences),
        "differences": differences,
        "interpretation": (
            "PASS means pre-existing PPT evidence, Stage-1 geometry, and frozen "
            "population model/statistics were neither changed nor removed. Changes "
            "under regenerated roots are expected final-pipeline outputs."
        ),
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    POST_PATH.write_text(json.dumps(post, indent=2) + "\n", encoding="utf-8")
    COMPARISON_PATH.write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {POST_PATH}")
    print(f"Wrote {COMPARISON_PATH}")
    print(f"Immutable evidence preserved: {immutable_pass}")


if __name__ == "__main__":
    main()
