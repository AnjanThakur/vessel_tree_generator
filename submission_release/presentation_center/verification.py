"""Read-only verification routines used by the presentation center."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from vessel_tree_generator.export import verify_export

from .presentation_data import BRANCHES, DEMO_ROOT, RELEASE, ROOT, read_json


FROZEN_ROOTS = (
    ROOT / "outputs/lca_ssm/ppt_priority_completion",
    ROOT / "outputs/lca_ssm/stage1_heart_scaffold_vtk",
    ROOT / "outputs/lca_ssm/stage1_parametric_heart_surface",
    ROOT / "outputs/lca_ssm/lca_population_model",
)
STATISTICS = ROOT / "outputs/lca_ssm/lca_population_model/generator_statistics"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frozen_hash_snapshot() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    aggregate = hashlib.sha256()
    for root in FROZEN_ROOTS:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            relative = path.relative_to(ROOT).as_posix()
            digest = sha256(path)
            files[relative] = {"sha256": digest, "size_bytes": path.stat().st_size}
            aggregate.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "aggregate_sha256": aggregate.hexdigest(),
        "files": files,
    }


def compare_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_files = before["files"]
    after_files = after["files"]
    changed = sorted(
        path for path in before_files.keys() & after_files.keys()
        if before_files[path]["sha256"] != after_files[path]["sha256"]
    )
    missing = sorted(before_files.keys() - after_files.keys())
    added = sorted(after_files.keys() - before_files.keys())
    return {
        "status": "PASS" if not changed and not missing and not added else "FAIL",
        "before_file_count": before["file_count"],
        "after_file_count": after["file_count"],
        "before_aggregate_sha256": before["aggregate_sha256"],
        "after_aggregate_sha256": after["aggregate_sha256"],
        "changed": changed,
        "missing": missing,
        "added": added,
    }


def _check(name: str, condition: bool, detail: str) -> dict[str, str]:
    return {"name": name, "status": "PASS" if condition else "FAIL", "detail": detail}


def run_fast_verification() -> dict[str, Any]:
    checks: list[dict[str, str]] = []
    try:
        release = read_json(RELEASE / "RELEASE_MANIFEST.json")
        validation = read_json(RELEASE / "VALIDATION_SUMMARY.json")
        checks.append(_check("Release manifest parses", True, release["release"]["version"]))
        checks.append(_check("Validation summary parses", True, validation["status"]))
    except Exception as exc:
        return {"status": "FAIL", "checks": [_check("Core JSON parses", False, str(exc))]}

    stats_manifest = read_json(STATISTICS / "generator_statistics_manifest.json")
    required = stats_manifest["files"]
    missing = [name for name in required if not (STATISTICS / name).is_file()]
    hash_failures = [
        name for name, expected in required.items()
        if (STATISTICS / name).is_file() and sha256(STATISTICS / name) != expected
    ]
    checks.append(_check(
        "Frozen statistics exist and match hashes",
        not missing and not hash_failures,
        f"{len(required)}/{len(required)} files" if not missing and not hash_failures
        else f"missing={missing}; hash_failures={hash_failures}",
    ))

    for case_name in ("healthy", "focal_lad", "diffuse_lcx", "tandem_lad"):
        case_dir = DEMO_ROOT / case_name
        manifest_exists = (case_dir / "manifest.json").is_file()
        try:
            verified = verify_export(case_dir)
            ok = manifest_exists and str(verified["status"]).startswith("PASS")
            detail = verified["status"]
        except Exception as exc:
            ok, detail = False, str(exc)
        checks.append(_check(f"{case_name} checksums and VTK readback", ok, detail))

        cine_path = case_dir / "geometry_cine.npy"
        static_path = case_dir / "geometry_static.npy"
        if cine_path.is_file() and static_path.is_file():
            cine = np.load(cine_path, mmap_mode="r", allow_pickle=False)
            static = np.load(static_path, mmap_mode="r", allow_pickle=False)
            finite = np.isfinite(cine[0]) & np.isfinite(static)
            max_identity_error = float(np.max(np.abs(cine[0][finite] - static[finite])))
            identity = bool(np.allclose(cine[0], static, rtol=0.0, atol=1e-12, equal_nan=True))
            checks.append(_check(f"{case_name} static equals cine phase zero", identity, f"max error {max_identity_error:.3g}; shape {list(cine.shape)}"))

            junction_ok = True
            max_error = 0.0
            for phase in cine:
                lmca = phase[0][np.isfinite(phase[0, :, 0])]
                lad = phase[1][np.isfinite(phase[1, :, 0])]
                lcx = phase[2][np.isfinite(phase[2, :, 0])]
                error = max(
                    float(np.linalg.norm(lmca[-1, :3] - lad[0, :3])),
                    float(np.linalg.norm(lmca[-1, :3] - lcx[0, :3])),
                )
                max_error = max(max_error, error)
                junction_ok &= error <= 1e-9
            checks.append(_check(f"{case_name} exact LMCA-LAD-LCX topology", junction_ok, f"max junction error {max_error:.3g} mm"))

    expected = {
        "eligible": 52,
        "generated": 52,
        "comparisons": 50,
        "tests": 90,
        "motion_snapshots": 520,
    }
    actual = {
        "eligible": validation["source_data_audit"]["statistics_eligible_cases"],
        "generated": validation["final_population_generation"]["accepted_trees"],
        "comparisons": validation["final_population_generation"]["population_comparisons_passed"],
        "tests": validation["final_regression"]["total_tests_passed"],
        "motion_snapshots": validation["population_motion"]["snapshot_count"],
    }
    checks.append(_check("Critical metrics internally consistent", actual == expected, json.dumps(actual)))
    checks.append(_check(
        "Validated scope is major-vessel LCA only",
        validation["model_scope"].endswith("LMCA, LAD, LCX"),
        validation["model_scope"],
    ))
    checks.append(_check(
        "Clinical validation is not claimed",
        validation["clinical_validation"] is False,
        "research/engineering prototype",
    ))
    return {
        "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "fast_read_only",
        "checks": checks,
    }


def _command_check(label: str, command: list[str]) -> dict[str, str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout.strip()
    return _check(label, completed.returncode == 0, output[-1200:] or "completed")


def run_full_verification(progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run deep checks without invoking writers or repairing evidence."""
    emit = progress or (lambda _: None)
    before = frozen_hash_snapshot()
    checks: list[dict[str, str]] = []

    emit("[1/8] source integrity")
    integrity = read_json(RELEASE / "final_audit" / "protected_integrity_comparison.json")
    checks.append(_check("Source and frozen-evidence integrity", integrity["status"] == "PASS", f"{before['file_count']} current frozen files"))

    emit("[2/8] statistical model")
    pca = read_json(RELEASE / "final_audit" / "pca_recomputation_audit.json")
    checks.append(_check("Independent PCA recomputation", bool(pca.get("pass", pca.get("status") == "PASS")), json.dumps({k: pca.get(k) for k in ("matrix_shape", "retained_modes", "retained_variance_fraction")})))

    emit("[3/8] generation")
    fast = run_fast_verification()
    checks.extend(fast["checks"])

    emit("[4/8] disease")
    disease = read_json(RELEASE / "final_validation" / "disease_validation.json")
    checks.append(_check("Disease geometry/radius audit", disease["status"] == "PASS", "XYZ identity and configured radius changes"))

    emit("[5/8] motion")
    motion = read_json(RELEASE / "final_validation" / "motion_validation.json")
    checks.append(_check("Motion and phase closure audit", motion["status"] == "PASS", "10 stored frames; 9 independent positions plus closure"))

    emit("[6/8] VTK")
    vtk = read_json(RELEASE / "final_validation" / "vtk_validation.json")
    checks.append(_check("VTK readback audit", vtk["status"] == "PASS", "PVD/VTM/VTP outputs readable"))

    emit("[7/8] novelty/holdout")
    novelty = read_json(RELEASE / "final_audit" / "generation_novelty_summary.json")
    holdout = read_json(RELEASE / "final_audit" / "holdout_validation_summary.json")
    checks.append(_check("Novelty audit", novelty.get("exact_duplicate_count") == 0 and novelty.get("near_duplicate_under_0_1mm_count") == 0, "zero exact and <0.1 mm duplicates"))
    checks.append(_check("Internal holdout leakage control", holdout.get("source_leakage_count") == 0 and holdout.get("baseline_leakage_count") == 0, f"accepted {holdout.get('generated_anatomy_acceptance_count')}/{holdout.get('total_holdout_predictions')}"))

    emit("[8/8] regression tests")
    python = sys.executable
    checks.append(_command_check("Pytest regression suite", [python, "-m", "pytest", "-q"]))
    checks.append(_command_check("Focused Person-2 suite", [python, "-m", "unittest", "pca_ssm_vessel_tree_generator.tests.test_person2_generation", "-q"]))
    checks.append(_command_check("Dependency consistency", [python, "-m", "pip", "check"]))

    after = frozen_hash_snapshot()
    comparison = compare_snapshots(before, after)
    checks.append(_check("Full verification remained read-only", comparison["status"] == "PASS", f"{comparison['after_file_count']}/{comparison['before_file_count']} frozen files unchanged"))
    return {
        "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "full_read_only",
        "checks": checks,
        "frozen_integrity": comparison,
    }
