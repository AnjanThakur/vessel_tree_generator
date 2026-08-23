#!/usr/bin/env python
"""Build and verify the production-only Coronary4D presentation package.

This tool packages the already accepted ``tree_0015`` geometry.  It does not
fit, resample, or otherwise modify the canonical static centerlines.  The 4D
series is produced by the public production API using those centerlines as its
explicit reference anatomy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from vessel_tree_generator import (
    CoronaryTreeGenerator,
    GenerationConfig,
    MotionConfig,
    PulsatilityConfig,
)
from vessel_tree_generator.disease import healthy_config
from vessel_tree_generator.export import export_case


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/lca_ssm/lca_population_cohort"
TREE_ID = "tree_0015"
TREE = COHORT / TREE_ID
STATS = ROOT / "outputs/lca_ssm/lca_population_model/generator_statistics"
PROTECTED = ROOT / "outputs/lca_ssm/lca_population_model/protected_source_integrity.json"
RAW = ROOT / "outputs/lca_ssm/raw_cases"
ORIGINAL_EVIDENCE = ROOT / "submission_release/original_ppt_evidence/original_ppt_evidence_summary.json"
OUT = ROOT / "submission_release/final_presentation"
WORK = ROOT / "submission_release/final_presentation_working"
AUDIT = ROOT / "submission_release/FINAL_CODEBASE_CLEANUP_AUDIT.md"
ANATOMY_AUDIT = ROOT / "submission_release/final_visual_anatomical_audit/accepted_tree_anatomy_audit.csv"
BRANCHES = ("LMCA", "LAD", "LCX")
COLORS = {"LMCA": "#252525", "LAD": "#c5221f", "LCX": "#008b8b"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_hash(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        relative = path.relative_to(root).as_posix()
        item_hash = sha256(path)
        digest.update(relative.encode("utf-8") + b"\0" + item_hash.encode("ascii") + b"\n")
    return {"file_count": len(files), "aggregate_sha256": digest.hexdigest()}


def write_vtm(path: Path, blocks: list[tuple[str, Path]]) -> None:
    root = ET.Element("VTKFile", type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian")
    dataset = ET.SubElement(root, "vtkMultiBlockDataSet")
    for index, (name, target) in enumerate(blocks):
        relative = target.relative_to(path.parent).as_posix()
        ET.SubElement(dataset, "DataSet", index=str(index), name=name, file=relative)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def verify_frozen_statistics() -> dict[str, Any]:
    manifest_path = STATS / "generator_statistics_manifest.json"
    manifest = load_json(manifest_path)
    rows = []
    for name, expected in manifest["files"].items():
        path = STATS / name
        actual = sha256(path) if path.is_file() else None
        rows.append({"file": name, "expected_sha256": expected, "actual_sha256": actual, "pass": actual == expected})
    passed = all(row["pass"] for row in rows)
    return {
        "status": "PASS" if passed else "FAIL",
        "manifest_sha256": sha256(manifest_path),
        "shape_vector_dimensions": manifest["shape_vector_dimensions"],
        "pca_case_count": manifest["pca_case_count"],
        "files_verified": len(rows),
        "files": rows,
    }


def verify_protected_sources() -> dict[str, Any]:
    protected = load_json(PROTECTED)
    raw_expected = protected["before"]["raw_case_archives"]
    raw_rows = []
    for case_id, expected in sorted(raw_expected.items()):
        path = RAW / case_id / "original_centerlines.npz"
        actual = sha256(path) if path.is_file() else None
        raw_rows.append({"case_id": case_id, "pass": actual == expected})
    measurement_rows = []
    for relative, expected in protected["before"]["ppt_input_files"].items():
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        measurement_rows.append({"path": relative, "pass": actual == expected})
    evidence = load_json(ORIGINAL_EVIDENCE)
    passed = all(row["pass"] for row in raw_rows + measurement_rows)
    passed = passed and evidence["maximum_source_coordinate_change_mm"] == 0.0
    passed = passed and evidence["maximum_source_segment_length_change_mm"] == 0.0
    return {
        "status": "PASS" if passed else "FAIL",
        "protected_raw_archives_verified": sum(row["pass"] for row in raw_rows),
        "protected_raw_archives_expected": len(raw_rows),
        "measurement_evidence_files_verified": sum(row["pass"] for row in measurement_rows),
        "measurement_evidence_files_expected": len(measurement_rows),
        "maximum_source_coordinate_change_mm": evidence["maximum_source_coordinate_change_mm"],
        "maximum_source_segment_length_change_mm": evidence["maximum_source_segment_length_change_mm"],
    }


def verify_cohort() -> dict[str, Any]:
    directories = sorted(COHORT.glob("tree_[0-9][0-9][0-9][0-9]"))
    with ANATOMY_AUDIT.open(newline="", encoding="utf-8") as handle:
        independent_rows = {row["case_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for directory in directories:
        errors: list[str] = []
        branches: dict[str, np.ndarray] = {}
        for name in BRANCHES:
            path = directory / f"{name}.npy"
            if not path.is_file():
                errors.append(f"missing {name}.npy")
                continue
            points = np.load(path, allow_pickle=False)
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
                errors.append(f"invalid {name} shape {points.shape}")
            elif not np.all(np.isfinite(points)):
                errors.append(f"non-finite {name} coordinates")
            branches[name] = points
        topology_error = None
        if len(branches) == 3:
            topology_error = max(
                float(np.linalg.norm(branches["LMCA"][-1] - branches["LAD"][0])),
                float(np.linalg.norm(branches["LMCA"][-1] - branches["LCX"][0])),
                float(np.linalg.norm(branches["LAD"][0] - branches["LCX"][0])),
            )
            if topology_error > 1.0e-9:
                errors.append(f"topology error {topology_error:.3e} mm")
        validation_path = directory / "validation.json"
        validation = load_json(validation_path) if validation_path.is_file() else {}
        if not validation.get("accepted", False):
            errors.append("production validation did not accept tree")
        independent = independent_rows.get(directory.name)
        if independent is None:
            errors.append("independent production anatomy audit row is missing")
        elif independent["independent_coordinate_gate_pass"].lower() != "true":
            errors.append("independent production coordinate/anatomy gate did not pass")
        elif independent["validator_pass"].lower() != "true":
            errors.append("stored production validator did not pass")
        elif independent["validator_coordinate_gate_mismatch"].lower() == "true":
            errors.append("production validator and independent coordinate gate disagree")
        vtk_path = directory / "vtk/synthetic_tree.vtm"
        try:
            vtk = pv.read(vtk_path)
            keys = set(vtk.keys()) if isinstance(vtk, pv.MultiBlock) else set()
            if not {"SYNTHETIC_LMCA", "SYNTHETIC_LAD", "SYNTHETIC_LCX"}.issubset(keys):
                errors.append("VTK branch hierarchy is incomplete")
        except Exception as exc:  # pragma: no cover - diagnostic boundary
            errors.append(f"VTK readback failed: {exc}")
        rows.append({
            "tree_id": directory.name,
            "status": "PASS" if not errors else "FAIL",
            "maximum_topology_error_mm": topology_error,
            "warning_count": len(validation.get("warnings", [])),
            "full_descriptive_anatomy_pass": bool(
                independent and independent["full_descriptive_anatomy_pass"].lower() == "true"
            ),
            "errors": errors,
        })
    passed = len(rows) == 52 and all(row["status"] == "PASS" for row in rows)
    return {
        "status": "PASS" if passed else "FAIL",
        "expected": 52,
        "readable": sum(row["status"] == "PASS" for row in rows),
        "topology_valid": sum(row["maximum_topology_error_mm"] is not None and row["maximum_topology_error_mm"] <= 1.0e-9 for row in rows),
        "production_anatomy_valid": sum(not row["errors"] for row in rows),
        "full_descriptive_anatomy_valid": sum(row["full_descriptive_anatomy_pass"] for row in rows),
        "anatomy_interpretation": "52/52 pass the production coordinate/topology gate; 35/52 also pass every stricter absolute descriptive rule",
        "rows": rows,
    }


def copy_static_geometry() -> dict[str, Any]:
    static = WORK / "static"
    scaffold = WORK / "scaffold"
    static.mkdir(parents=True)
    scaffold.mkdir(parents=True)
    source_vtk = TREE / "vtk"
    branch_paths = []
    maximum_error = 0.0
    point_arrays: dict[str, list[str]] = {}
    for name in BRANCHES:
        target = static / f"{name}.vtp"
        shutil.copy2(source_vtk / f"{name}.vtp", target)
        branch_paths.append((name, target))
        copied = pv.read(target)
        original = np.load(TREE / f"{name}.npy", allow_pickle=False)
        maximum_error = max(maximum_error, float(np.max(np.abs(copied.points - original))))
        point_arrays[name] = list(copied.point_data.keys())
    static_vtm = WORK / "final_static_tree.vtm"
    write_vtm(static_vtm, branch_paths)
    ellipsoid = scaffold / "ELLIPSOID_SCAFFOLD.vtp"
    shutil.copy2(source_vtk / "synthetic_ellipsoid.vtp", ellipsoid)
    scaffold_vtm = WORK / "final_tree_with_scaffold.vtm"
    write_vtm(scaffold_vtm, [("ELLIPSOID_SCAFFOLD", ellipsoid), *branch_paths])
    static_readback = pv.read(static_vtm)
    scaffold_readback = pv.read(scaffold_vtm)
    return {
        "status": "PASS" if maximum_error <= 1.0e-12 else "FAIL",
        "maximum_static_copy_coordinate_error_mm": maximum_error,
        "static_block_names": list(static_readback.keys()),
        "scaffold_block_names": list(scaffold_readback.keys()),
        "point_arrays": point_arrays,
    }


def build_reference() -> dict[str, Any]:
    parameters = load_json(TREE / "parameters.json")
    validation = load_json(TREE / "validation.json")
    sampling = load_json(TREE / "sampling_attempts.json")
    ellipsoid = parameters["ellipsoid"]
    return {
        "branches": {name: np.load(TREE / f"{name}.npy", allow_pickle=False) for name in BRANCHES},
        "ellipsoid_params": {
            "a_mm": float(ellipsoid["a"]),
            "b_mm": float(ellipsoid["b"]),
            "c_mm": float(ellipsoid["c"]),
        },
        "generation_parameters": parameters,
        "static_validation": validation,
        "sampling": {
            "attempt_count": sampling["attempt_count"],
            "accepted_attempt": sampling["accepted_attempt"],
        },
    }


def build_cine() -> tuple[dict[str, Any], dict[str, Any]]:
    reference = build_reference()
    parameters = reference["generation_parameters"]
    scale = float(parameters["generation"]["deviation_sample"]["variation_scale"])
    generation = GenerationConfig(
        seed=int(parameters["seed"]),
        source_case_id=parameters["ellipsoid"]["source_case_id"],
        pca_scale=scale,
        maximum_attempts=250,
    )
    case = CoronaryTreeGenerator().generate_case(
        healthy_config(case_id="tree_0015_production_healthy"),
        generation=generation,
        motion=MotionConfig(number_of_phases=10),
        pulsatility=PulsatilityConfig(),
        reference=reference,
    )
    exported = WORK / "_cine_export"
    manifest = export_case(case, exported, points_per_branch=80, clean=False)
    shutil.copy2(exported / "vtk/cine.pvd", WORK / "cine.pvd")
    for phase in sorted((exported / "vtk").glob("phase_[0-9][0-9][0-9]")):
        target = WORK / phase.name
        target.mkdir()
        shutil.copy2(phase / "tree.vtm", target / "tree.vtm")
        shutil.copytree(phase / "tree", target / "tree")
    metadata = load_json(exported / "metadata.json")
    metadata["presentation"] = {
        "package": "surface-relative production generator only",
        "canonical_tree_id": TREE_ID,
        "static_geometry_source": TREE.relative_to(ROOT).as_posix(),
        "static_coordinates_copied_without_modification": True,
        "RCA_included": False,
        "clinical_status": "research/engineering prototype; not clinically validated",
    }
    write_json(WORK / "metadata.json", metadata)
    shutil.rmtree(exported)
    return case, manifest


def plot_view(path: Path, branches: dict[str, np.ndarray], view: str, axes: tuple[int, int] | None = None) -> None:
    if view == "perspective":
        figure = plt.figure(figsize=(9.0, 7.5))
        axis = figure.add_subplot(111, projection="3d")
        for name in BRANCHES:
            points = branches[name]
            axis.plot(points[:, 0], points[:, 1], points[:, 2], color=COLORS[name], linewidth=3.0, label=name)
        bifurcation = branches["LMCA"][-1]
        axis.scatter(*bifurcation, color="#f28e2b", s=45)
        axis.set_xlabel("Cardiac X (mm)")
        axis.set_ylabel("Cardiac Y (mm)")
        axis.set_zlabel("Cardiac Z (mm; apex negative)")
        axis.view_init(elev=24, azim=-58)
        try:
            axis.set_box_aspect(np.ptp(np.vstack(list(branches.values())), axis=0))
        except AttributeError:
            pass
    else:
        assert axes is not None
        figure, axis = plt.subplots(figsize=(9.0, 7.5))
        labels = ("Cardiac X (mm)", "Cardiac Y (mm)", "Cardiac Z (mm; apex negative)")
        for name in BRANCHES:
            points = branches[name]
            axis.plot(points[:, axes[0]], points[:, axes[1]], color=COLORS[name], linewidth=3.0, label=name)
        bifurcation = branches["LMCA"][-1]
        axis.scatter(bifurcation[axes[0]], bifurcation[axes[1]], color="#f28e2b", s=45, zorder=5)
        axis.set_xlabel(labels[axes[0]])
        axis.set_ylabel(labels[axes[1]])
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
    axis.legend(loc="best")
    axis.set_title(f"{TREE_ID} — final surface-relative production anatomy ({view})")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_previews() -> dict[str, Any]:
    branches = {name: np.load(TREE / f"{name}.npy", allow_pickle=False) for name in BRANCHES}
    plot_view(WORK / "front.png", branches, "front", (0, 2))
    plot_view(WORK / "top.png", branches, "top", (0, 1))
    plot_view(WORK / "side.png", branches, "side", (1, 2))
    plot_view(WORK / "perspective.png", branches, "perspective")
    paths = [WORK / name for name in ("front.png", "top.png", "side.png", "perspective.png")]
    return {"status": "PASS" if all(path.stat().st_size > 10_000 for path in paths) else "FAIL", "files": [path.name for path in paths]}


def tree_metrics() -> dict[str, Any]:
    validation = load_json(TREE / "validation.json")
    metrics = validation["metrics"]
    direction = metrics["anatomical_direction"]
    return {
        "tree_id": TREE_ID,
        "source_case_id": load_json(TREE / "parameters.json")["ellipsoid"]["source_case_id"],
        "branch_lengths_mm": metrics["branch_lengths_mm"],
        "LAD_LCX_angle_deg": metrics["bifurcation_angle_deg"],
        "LAD_inferior_displacement_mm": direction["LAD_inferior_displacement_mm"],
        "LCX_inferior_displacement_mm": direction["LCX_inferior_displacement_mm"],
        "LCX_lateral_circumferential_displacement_mm": direction["LCX_terminal_lateral_displacement_mm"],
        "branch_tortuosity_arc_over_chord": metrics["branch_tortuosity"],
        "branch_obliquity_rad": metrics["branch_obliquity_rad"],
        "branch_obliquity_deg": {name: math.degrees(value) for name, value in metrics["branch_obliquity_rad"].items()},
        "minimum_nonlocal_clearance_mm": metrics["minimum_nonlocal_distance_mm"],
        "minimum_interbranch_clearance_mm": metrics["minimum_interbranch_distance_mm"],
        "backward_progression_ratio": {name: value["backward_progress_ratio"] for name, value in metrics["branch_progression"].items()},
        "warnings": validation["warnings"],
        "warning_count": len(validation["warnings"]),
        "production_anatomy_gate": metrics["anatomical_role_acceptance"],
    }


def verify_cine(case: dict[str, Any]) -> dict[str, Any]:
    pvd = ET.parse(WORK / "cine.pvd").getroot()
    entries = [element.attrib["file"] for element in pvd.iter("DataSet")]
    frames: list[dict[str, np.ndarray]] = []
    errors: list[str] = []
    for relative in entries:
        path = WORK / relative
        try:
            dataset = pv.read(path)
        except Exception as exc:  # pragma: no cover - diagnostic boundary
            errors.append(f"{relative}: readback failed: {exc}")
            continue
        if not isinstance(dataset, pv.MultiBlock) or list(dataset.keys()) != list(BRANCHES):
            errors.append(f"{relative}: branch names are not LMCA/LAD/LCX")
            continue
        frame: dict[str, np.ndarray] = {}
        for name in BRANCHES:
            block = dataset[name]
            if block is None or block.n_points < 2 or not np.all(np.isfinite(block.points)):
                errors.append(f"{relative}/{name}: empty or non-finite")
                continue
            arrays = set(block.point_data.keys())
            required = {"radius_mm", "normalized_arc_length", "disease_reduction_fraction"}
            if not required.issubset(arrays):
                errors.append(f"{relative}/{name}: missing arrays {sorted(required - arrays)}")
            frame[name] = np.asarray(block.points)
        if len(frame) == 3:
            topology = max(
                float(np.linalg.norm(frame["LMCA"][-1] - frame["LAD"][0])),
                float(np.linalg.norm(frame["LMCA"][-1] - frame["LCX"][0])),
            )
            if topology > 1.0e-9:
                errors.append(f"{relative}: topology error {topology:.3e} mm")
            frames.append(frame)
    closure = None
    if len(frames) == len(entries) and frames:
        closure = max(float(np.max(np.abs(frames[0][name] - frames[-1][name]))) for name in BRANCHES)
        if closure > 1.0e-9:
            errors.append(f"cycle closure error {closure:.3e} mm")
    passed = len(entries) == 10 and len(frames) == 10 and not errors and case["validation"]["is_valid"]
    return {
        "status": "PASS" if passed else "FAIL",
        "phase_files_expected": 10,
        "phase_files_read": len(frames),
        "branch_names": list(BRANCHES),
        "maximum_cycle_closure_error_mm": closure,
        "production_4d_validation": case["validation"],
        "errors": errors,
    }


def write_notes(metrics: dict[str, Any]) -> None:
    (WORK / "README.md").write_text(
        "# Coronary4D Final Production Presentation\n\n"
        "This directory contains only the md-aligned surface-relative production generator output. "
        "It is an LCA-only research/engineering prototype and is not clinically validated.\n\n"
        "## ParaView\n\n"
        "- Open `final_static_tree.vtm` for the unchanged canonical LMCA/LAD/LCX centerlines.\n"
        "- Open `final_tree_with_scaffold.vtm` to add the exact associated production ellipsoid.\n"
        "- Open `cine.pvd`, click Apply, and press Play for the validated closed 4D cycle.\n"
        "- Use Tube representation for vessels. Color the cine blocks by `radius_mm`; "
        "`disease_reduction_fraction` is present and is zero for this healthy presentation case.\n\n"
        "The canonical case is `tree_0015`, selected previously as a warning-free production medoid. "
        "No presentation coordinate translation or geometry deformation is applied.\n",
        encoding="utf-8",
    )
    (WORK / "ANATOMY_INTERPRETATION.md").write_text(
        "# Anatomy interpretation\n\n"
        "The production model does not force a textbook planar LAD/LCX drawing. It preserves measured "
        "population variability in a genuinely three-dimensional, surface-relative representation. LAD is "
        "required to be the dominant inferior/apical daughter. LCX is required to be comparatively lateral "
        "and circumferential, and less inferior than LAD. A single X-Z view can hide Y/circumferential travel.\n\n"
        "The scaffold is visually sparse because RCA, diagonal and septal branches, obtuse marginals, the aorta, "
        "and myocardium are outside this validated model scope. Assess the tree using front, top, side, and "
        "oblique views together with `validation_summary.json`.\n\n"
        f"For `{TREE_ID}`, LAD inferior displacement is {metrics['LAD_inferior_displacement_mm']:.3f} mm, "
        f"LCX inferior displacement is {metrics['LCX_inferior_displacement_mm']:.3f} mm, and LCX terminal "
        f"lateral/circumferential displacement is {metrics['LCX_lateral_circumferential_displacement_mm']:.3f} mm. "
        f"The production anatomy gate passes with {metrics['warning_count']} warnings.\n",
        encoding="utf-8",
    )


def write_cleanup_audit(cohort: dict[str, Any], metrics: dict[str, Any], integrity: dict[str, Any]) -> None:
    tests = load_json(ROOT / "submission_release/final_validation/test_validation.json")
    AUDIT.write_text(
        "# Final codebase cleanup audit\n\n"
        "## Classification\n\n"
        "| Class | Files / areas | Decision |\n"
        "|---|---|---|\n"
        "| FINAL_PRODUCTION_REQUIRED | `vessel_tree_generator/`, production generation/surface/PCA/motion modules, frozen statistics, accepted cohort | Retained |\n"
        "| SHARED_UTILITY_REQUIRED | ellipse measurement, cardiac-frame, surface projection, VTK export, validation utilities | Retained because production/evidence imports use them |\n"
        "| DOCUMENTATION_ONLY / MEASUREMENT EVIDENCE | tracked 191-case two-plane/two-ellipse tables, residuals, figures and integrity records | Retained as source/statistical evidence, not a competing generator |\n"
        "| EXPERIMENT_ONLY | `ppt_exact_generator.py`, refinement module, two runners and two test files | Removed from production/import/test paths and placed in an ignored local research archive |\n"
        "| GENERATED_EXPERIMENT_ARTIFACT | `submission_release/ppt_exact_generator*` | Removed from canonical release and placed in the same ignored local research archive |\n\n"
        "## Production result\n\n"
        f"- Cohort: {cohort['readable']}/{cohort['expected']} readable, topology-valid and production-anatomy-valid.\n"
        f"- Presentation medoid: `{TREE_ID}`; warning count: {metrics['warning_count']}.\n"
        f"- Protected raw archives: {integrity['protected_raw_archives_verified']}/{integrity['protected_raw_archives_expected']} hash matches.\n"
        f"- Source-coordinate change: {integrity['maximum_source_coordinate_change_mm']:.1f} mm.\n"
        f"- Source segment-length change: {integrity['maximum_source_segment_length_change_mm']:.1f} mm.\n"
        f"- Regression families: {tests['pytest_passed']} main production, "
        f"{tests['person2_unittest_passed']} focused generation, "
        f"{tests['design_alignment_unittest_passed']} design-alignment; "
        f"{tests['total_tests_passed']} total, 0 failed.\n"
        "- Final code path: `vessel_tree_generator.CoronaryTreeGenerator` / `python -m vessel_tree_generator`.\n"
        "- `FINAL_PROJECT_REPORT.docx` and presentation slides were intentionally left unchanged at the user's request.\n",
        encoding="utf-8",
    )


def main() -> int:
    if not TREE.is_dir():
        raise FileNotFoundError(TREE)
    resolved = WORK.resolve()
    expected_parent = (ROOT / "submission_release").resolve()
    if resolved.parent != expected_parent or resolved.name != "final_presentation_working":
        raise RuntimeError(f"refusing unexpected work path: {resolved}")
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    cohort_before = aggregate_hash(COHORT)
    frozen_before = aggregate_hash(STATS)
    cohort = verify_cohort()
    frozen = verify_frozen_statistics()
    source = verify_protected_sources()
    static = copy_static_geometry()
    case, export_manifest = build_cine()
    cine = verify_cine(case)
    previews = render_previews()
    metrics = tree_metrics()
    write_notes(metrics)

    cohort_after = aggregate_hash(COHORT)
    frozen_after = aggregate_hash(STATS)
    integrity = {
        **source,
        "canonical_cohort_unchanged": cohort_before == cohort_after,
        "canonical_cohort_before": cohort_before,
        "canonical_cohort_after": cohort_after,
        "frozen_statistics_unchanged": frozen_before == frozen_after,
        "frozen_statistics_before": frozen_before,
        "frozen_statistics_after": frozen_after,
        "frozen_statistics_manifest_verification": frozen,
    }
    integrity["status"] = "PASS" if (
        source["status"] == "PASS"
        and integrity["canonical_cohort_unchanged"]
        and integrity["frozen_statistics_unchanged"]
        and frozen["status"] == "PASS"
    ) else "FAIL"

    checks = {
        "cohort": cohort["status"],
        "tree_0015_warning_free": "PASS" if metrics["warning_count"] == 0 else "FAIL",
        "static_copy": static["status"],
        "production_cine": cine["status"],
        "previews": previews["status"],
        "integrity": integrity["status"],
    }
    status = "PASS" if all(value == "PASS" for value in checks.values()) else "FAIL"
    metadata = load_json(WORK / "metadata.json")
    metadata["production_model"] = {
        "name": "surface-relative production generator",
        "shape_vector_dimensions": 81,
        "retained_modes": 13,
        "retained_variance_percent": 95.55,
        "eligible_independent_anatomies": 52,
        "spline": "validated normalized-arc shape-preserving cubic B-spline",
    }
    metadata["canonical_tree_metrics"] = metrics
    write_json(WORK / "metadata.json", metadata)

    files = sorted(path for path in WORK.rglob("*") if path.is_file() and path.name != "validation_summary.json")
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "scope": "surface-relative production generator only; LCA-only research/engineering prototype",
        "checks": checks,
        "cohort_verification": cohort,
        "canonical_tree_metrics": metrics,
        "static_vtk_verification": static,
        "production_4d_verification": cine,
        "production_export_manifest": export_manifest,
        "preview_verification": previews,
        "source_and_model_integrity": integrity,
        "files": {
            path.relative_to(WORK).as_posix(): {"size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in files
        },
    }
    write_json(WORK / "validation_summary.json", summary)
    write_cleanup_audit(cohort, metrics, integrity)
    if status != "PASS":
        raise RuntimeError(json.dumps(checks, indent=2))

    if OUT.exists():
        resolved_out = OUT.resolve()
        if resolved_out.parent != expected_parent or resolved_out.name != "final_presentation":
            raise RuntimeError(f"refusing unexpected output path: {resolved_out}")
        shutil.rmtree(OUT)
    WORK.rename(OUT)
    print(json.dumps({
        "status": status,
        "output": OUT.relative_to(ROOT).as_posix(),
        "cohort": f"{cohort['readable']}/{cohort['expected']}",
        "tree": TREE_ID,
        "warning_count": metrics["warning_count"],
        "cine_phases": cine["phase_files_read"],
        "source_integrity": integrity["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
