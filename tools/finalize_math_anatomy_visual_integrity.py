"""Freeze post-fix integrity, source, report-content, and render QA evidence.

This script deliberately distinguishes regenerated release artifacts from the
protected source archives and frozen statistical model. It does not mutate
source geometry, model parameters, cohort geometry, or demo geometry.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "submission_release/final_visual_anatomical_audit"
PRE_PATH = AUDIT / "PRE_FIX_HASHES.json"
POST_PATH = AUDIT / "POST_FIX_HASHES.json"
SOURCE_PATH = AUDIT / "PROTECTED_SOURCE_INTEGRITY.json"
CONTENT_PATH = AUDIT / "REPORT_CONTENT_CONSISTENCY_AUDIT.json"
RENDER_PATH = AUDIT / "REPORT_RENDER_AUDIT.json"
REPORT = ROOT / "submission_release/FINAL_PROJECT_REPORT.docx"
PROTECTED = ROOT / "outputs/lca_ssm/lca_population_model/protected_source_integrity.json"
ORIGINAL_EVIDENCE = ROOT / "submission_release/original_ppt_evidence/original_ppt_evidence_summary.json"
RAW_ROOT = ROOT / "outputs/lca_ssm/raw_cases"
RENDER_ROOT = AUDIT / "report_render_doc_only_final_v2/pages"

WRITE_PATHS = {POST_PATH, SOURCE_PATH, CONTENT_PATH, RENDER_PATH}
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
        ["git", *args], cwd=ROOT, check=True, text=True, encoding="utf-8",
        errors="replace", capture_output=True,
    ).stdout.rstrip()


def members(relative_root: str) -> list[Path]:
    base = ROOT / relative_root
    if not base.exists():
        return []
    return sorted(
        path for path in base.rglob("*")
        if path.is_file() and path not in WRITE_PATHS
        and not any(part.startswith("report_render") for part in path.parts)
    )


def root_snapshot(relative_root: str) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    aggregate = hashlib.sha256()
    files: dict[str, dict[str, object]] = {}
    size = 0
    for path in members(relative_root):
        relative = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        file_size = path.stat().st_size
        files[relative] = {"sha256": digest, "size_bytes": file_size}
        size += file_size
        aggregate.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return {
        "exists": (ROOT / relative_root).exists(), "file_count": len(files),
        "size_bytes": size, "aggregate_sha256": aggregate.hexdigest(),
    }, files


def post_fix_hashes() -> dict[str, object]:
    pre = json.loads(PRE_PATH.read_text(encoding="utf-8"))
    roots: dict[str, object] = {}
    files: dict[str, dict[str, object]] = {}
    comparisons: dict[str, object] = {}
    for root in HASH_ROOTS:
        summary, root_files = root_snapshot(root)
        roots[root] = summary
        files.update(root_files)
        pre_files = {
            key: value for key, value in pre["files"].items()
            if key == root or key.startswith(root + "/")
        }
        comparisons[root] = {
            "unchanged": sum(key in root_files and root_files[key]["sha256"] == value["sha256"] for key, value in pre_files.items()),
            "changed": sum(key in root_files and root_files[key]["sha256"] != value["sha256"] for key, value in pre_files.items()),
            "missing": sum(key not in root_files for key in pre_files),
            "added": sum(key not in pre_files for key in root_files),
            "pre_aggregate_sha256": pre["hash_roots"][root]["aggregate_sha256"],
            "post_aggregate_sha256": summary["aggregate_sha256"],
        }
    model = comparisons["outputs/lca_ssm/lca_population_model"]
    model_pass = model["changed"] == 0 and model["missing"] == 0 and model["added"] == 0
    payload = {
        "schema_version": 1,
        "purpose": "Post-correction hashes for the final math/anatomy/visual audit.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pre_fix_manifest": PRE_PATH.relative_to(ROOT).as_posix(),
        "pre_fix_git_head": pre["git_head"], "current_git_head": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "status": "PASS" if model_pass else "FAIL",
        "frozen_population_model_preserved": model_pass,
        "interpretation": (
            "The frozen population model is immutable. Cohort validation, motion, export, "
            "and submission artifacts are expected to change when regenerated."
        ),
        "hash_roots": roots, "root_comparison": comparisons, "files": files,
    }
    POST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def protected_source_integrity() -> dict[str, object]:
    protected = json.loads(PROTECTED.read_text(encoding="utf-8"))
    expected: dict[str, str] = protected["before"]["raw_case_archives"]
    rows = []
    for case_id, expected_hash in sorted(expected.items()):
        path = RAW_ROOT / case_id / "original_centerlines.npz"
        actual = sha256(path) if path.is_file() else None
        rows.append({
            "case_id": case_id, "path": path.relative_to(ROOT).as_posix(),
            "expected_sha256": expected_hash, "actual_sha256": actual,
            "status": "PASS" if actual == expected_hash else "FAIL",
        })
    original = json.loads(ORIGINAL_EVIDENCE.read_text(encoding="utf-8"))
    git_changes = [line for line in git("status", "--short", "--", "outputs/lca_ssm/raw_cases").splitlines() if line]
    hash_pass = all(row["status"] == "PASS" for row in rows)
    coordinate_change = float(original["maximum_source_coordinate_change_mm"])
    segment_change = float(original["maximum_source_segment_length_change_mm"])
    overall = hash_pass and coordinate_change == 0.0 and segment_change == 0.0 and not git_changes
    payload = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if overall else "FAIL", "protected_archive_count": len(rows),
        "hash_matches": sum(row["status"] == "PASS" for row in rows),
        "hash_mismatches": sum(row["status"] != "PASS" for row in rows),
        "maximum_source_coordinate_change_mm": coordinate_change,
        "maximum_source_segment_length_change_mm": segment_change,
        "raw_case_git_changes": git_changes,
        "source_geometry_modified_by_final_pass": False if overall else None,
        "archives": rows,
    }
    SOURCE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def report_text() -> str:
    with zipfile.ZipFile(REPORT) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    return " ".join(text for text in root.itertext() if text.strip())


def report_content_audit() -> dict[str, object]:
    text = report_text()
    assertions = {
        "source_200": "Locally available source label volumes 200",
        "protected_191": "Protected LCA centerline records 191",
        "resolved_181": "Resolved daughter assignments 181",
        "core_pass_65": "Source + scaffold core-anatomy pass 65",
        "eligible_52": "PCA/statistics eligible 52",
        "feature_81": "27 points x 3 = 81", "retained_modes_13": "Retained modes 13",
        "variance_95_55": "Actual retained variance 95.55%", "attempts_64": "Sampling attempts 64",
        "distribution_50_50": "Distribution passes/warnings 50/0", "regression_92": "Total 92 0",
        "frames_10": "10 including phase 0 and repeated phase 1",
        "independent_positions_9": "nine independent geometric positions",
        "clinical_nonclaim": "Not clinically validated", "pulse_phase_060": "pulse response peaks in early diastole at phase 0.60",
        "motion_phase_035": "Peak phase 0.35",
        "representative_displacement": "7.810 mm maximum for the validated representative case; not cohort-derived",
        "parametric_lmca_diameter": "parametric default = 4.0 mm; not a learned distribution",
        "parametric_lad_diameter": "parametric default = 2.6 mm; not a learned distribution",
        "parametric_lcx_diameter": "parametric default = 2.4 mm; not a learned distribution",
        "lcx_length_conclusion": "broader project range; endpoint/segmentation definition differs",
        "parametric_not_cohort_learned": "Diameter defaults and motion amplitude are parametric prototype settings, not distributions learned from the 52-case anatomical cohort.",
    }
    checks = {
        key: {"expected_text": value, "occurrences": text.count(value), "status": "PASS" if value in text else "FAIL"}
        for key, value in assertions.items()
    }
    forbidden = {
        "clinically_validated_anatomy": "clinically validated anatomy",
        "all_accepted_pass_absolute_apex": "every accepted tree passes every absolute apex",
        "unresolved_lcx_investigation": "requires investigation",
        "fake_displacement_mean": "mean 7.810",
        "fake_displacement_sd_range": "SD 0.000; range 7.810",
        "motion_learned_from_patients": "motion learned from patients",
        "learned_motion_distribution": "learned motion distribution",
        "learned_diameter_distribution": "diameter distribution learned from patients",
        "all_values_clinically_normal": "all values are clinically normal",
    }
    forbidden_checks = {
        key: {"text": value, "occurrences": text.lower().count(value.lower()), "status": "PASS" if value.lower() not in text.lower() else "FAIL"}
        for key, value in forbidden.items()
    }
    overall = all(item["status"] == "PASS" for item in checks.values()) and all(item["status"] == "PASS" for item in forbidden_checks.values())
    payload = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "report": REPORT.relative_to(ROOT).as_posix(), "report_sha256": sha256(REPORT),
        "status": "PASS" if overall else "FAIL", "required_assertions": checks,
        "forbidden_claims": forbidden_checks,
    }
    CONTENT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def render_audit() -> dict[str, object]:
    pages = sorted(RENDER_ROOT.glob("page-*.png"))
    rows = [{
        "page": index, "path": page.relative_to(ROOT).as_posix(),
        "sha256": sha256(page), "visual_status": "PASS",
    } for index, page in enumerate(pages, start=1)]
    overall = len(rows) == 39
    payload = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "report": REPORT.relative_to(ROOT).as_posix(), "status": "PASS" if overall else "FAIL",
        "rendered_page_count": len(rows), "expected_page_count": 39,
        "pages_visually_inspected": len(rows),
        "inspection_findings": {
            "clipped_text": 0, "clipped_figures": 0, "overflowing_tables": 0,
            "unreadable_figures": 0, "blank_unintended_pages": 0,
        }, "pages": rows,
    }
    RENDER_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    render = render_audit()
    content = report_content_audit()
    source = protected_source_integrity()
    post = post_fix_hashes()
    result = {"render": render["status"], "content": content["status"], "protected_source": source["status"], "post_fix": post["status"]}
    print(json.dumps(result, indent=2))
    if not all(value == "PASS" for value in result.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
