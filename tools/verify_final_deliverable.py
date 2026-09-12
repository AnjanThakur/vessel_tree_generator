#!/usr/bin/env python
"""Read-only verification of the distributable Coronary4D repository."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from vessel_tree_generator.export import verify_export


ROOT = Path(__file__).resolve().parents[1]
STATISTICS = ROOT / "outputs/lca_ssm/lca_population_model/generator_statistics"
RELEASE = ROOT / "submission_release"
PRESENTATION = RELEASE / "final_presentation_52"
BRANCHES = ("LMCA", "LAD", "LCX")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_statistics() -> dict[str, Any]:
    manifest = load_json(STATISTICS / "generator_statistics_manifest.json")
    failures = []
    for relative, expected in manifest["files"].items():
        path = STATISTICS / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
        elif sha256(path) != expected:
            failures.append(f"hash mismatch {relative}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "files_verified": len(manifest["files"]) - len(failures),
        "files_expected": len(manifest["files"]),
        "failures": failures,
    }


def verify_multiblock(path: Path, expected_names: list[str]) -> list[str]:
    errors = []
    try:
        dataset = pv.read(path)
    except Exception as exc:
        return [f"{path}: read failed: {exc}"]
    if list(dataset.keys()) != expected_names:
        errors.append(f"{path}: block names {list(dataset.keys())}")
    for name in expected_names:
        block = dataset[name]
        if block is None or block.n_points <= 1:
            errors.append(f"{path}/{name}: empty block")
        elif not np.all(np.isfinite(block.points)):
            errors.append(f"{path}/{name}: non-finite coordinates")
    return errors


def verify_all_52_presentations() -> dict[str, Any]:
    manifest = load_json(PRESENTATION / "ALL_52_PRESENTATION_MANIFEST.json")
    errors = []
    frames = 0
    mesh_names = [f"{name}_MESH" for name in BRANCHES]
    overlay_names = [*mesh_names, *[f"{name}_CENTERLINE" for name in BRANCHES]]
    vtm_contracts = {
        "final_static_tree.vtm": list(BRANCHES),
        "final_tree_with_scaffold.vtm": ["ELLIPSOID_SCAFFOLD", *BRANCHES],
        "healthy_tapered_mesh.vtm": mesh_names,
        "healthy_mesh_with_centerlines.vtm": overlay_names,
        "diseased_tapered_mesh.vtm": mesh_names,
        "diseased_mesh_with_centerlines.vtm": overlay_names,
    }
    for index in range(1, 53):
        tree = PRESENTATION / f"tree_{index:04d}"
        for filename, names in vtm_contracts.items():
            errors.extend(verify_multiblock(tree / filename, names))
        pvd = tree / "cine.pvd"
        try:
            entries = [item.attrib["file"] for item in ET.parse(pvd).iter("DataSet")]
        except Exception as exc:
            errors.append(f"{pvd}: XML read failed: {exc}")
            continue
        if len(entries) != 10:
            errors.append(f"{pvd}: expected 10 phases, found {len(entries)}")
        for relative in entries:
            errors.extend(verify_multiblock(tree / relative, list(BRANCHES)))
            frames += 1
    contract_ok = (
        manifest.get("status") == "PASS"
        and manifest.get("tree_count_verified") == 52
        and manifest.get("total_cine_frames") == 520
        and manifest["maximum_errors"]["mesh_source_xyz_change_mm"] == 0.0
    )
    if not contract_ok:
        errors.append("all-52 manifest contract failed")
    return {
        "status": "PASS" if not errors else "FAIL",
        "trees_verified": 52 if not errors else None,
        "cine_frames_read": frames,
        "errors": errors,
    }


def verify_demo_cases() -> dict[str, Any]:
    results = {}
    for name in ("healthy", "focal_lad", "diffuse_lcx", "tandem_lad"):
        result = verify_export(RELEASE / "demo_cases" / name)
        results[name] = result["status"]
    passed = all(str(status).startswith("PASS") for status in results.values())
    return {"status": "PASS" if passed else "FAIL", "cases": results}


def verify_release_artifact_hashes() -> dict[str, Any]:
    manifest = load_json(RELEASE / "RELEASE_MANIFEST.json")
    failures = []
    for relative, expected in manifest["artifact_hashes"].items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
        elif path.stat().st_size != expected["size_bytes"] or sha256(path) != expected["sha256"]:
            failures.append(f"integrity mismatch {relative}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "files_verified": len(manifest["artifact_hashes"]) - len(failures),
        "files_expected": len(manifest["artifact_hashes"]),
        "failures": failures,
    }


def main() -> int:
    checks = {
        "frozen_statistics": verify_frozen_statistics(),
        "all_52_presentations": verify_all_52_presentations(),
        "demo_cases": verify_demo_cases(),
        "release_artifacts": verify_release_artifact_hashes(),
    }
    status = "PASS" if all(item["status"] == "PASS" for item in checks.values()) else "FAIL"
    print(json.dumps({"status": status, "checks": checks}, indent=2, allow_nan=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
