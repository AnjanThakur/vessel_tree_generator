#!/usr/bin/env python
"""Package the three ParaView entry files for all 52 production trees.

The builder copies accepted static/scaffold geometry byte-for-byte and converts
the already validated 52 x 10 production motion export into portable VTK.  It
does not sample PCA, refit anatomy, or modify the canonical cohort.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from lca_vessel_tree_generator.LCA_topology_generator.tube_surface import (
    centerline_radius_to_surface,
)
from vessel_tree_generator.disease import apply_disease_config, stenosis_config
from vessel_tree_generator.pulsatility import phase_radius


ROOT = Path(__file__).resolve().parents[1]
COHORT = ROOT / "outputs/lca_ssm/lca_population_cohort"
EXPORT = ROOT / "outputs/lca_ssm/lca_population_export"
MOTION_SUMMARY = ROOT / "outputs/lca_ssm/lca_population_motion/4d_trees_summary.json"
OUT = ROOT / "submission_release/final_presentation_52"
WORK = ROOT / "submission_release/final_presentation_52_working"
BRANCHES = ("LMCA", "LAD", "LCX")
BRANCH_IDS = {"LMCA": 1, "LAD": 2, "LCX": 3}
PULSATILITY_AMPLITUDE = 0.03
STENOSIS_COMPLIANCE_FACTOR = 0.35
PULSATILITY_PEAK_PHASE = 0.60
MESH_CIRCLE_POINTS = 24
DISEASE_BRANCH = "LAD"
DISEASE_POSITION = 0.45
DISEASE_LENGTH = 0.12
DISEASE_SEVERITY = 0.65


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


def relative_to(path: Path, parent: Path) -> str:
    return Path(*path.relative_to(parent).parts).as_posix()


def write_vtm(path: Path, blocks: list[tuple[str, Path]]) -> None:
    root = ET.Element("VTKFile", type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian")
    dataset = ET.SubElement(root, "vtkMultiBlockDataSet")
    for index, (name, target) in enumerate(blocks):
        ET.SubElement(
            dataset,
            "DataSet",
            index=str(index),
            name=name,
            file=relative_to(target, path.parent),
        )
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def polyline(
    xyz: np.ndarray,
    radius: np.ndarray,
    *,
    tree_id: str,
    branch: str,
    phase: float,
) -> pv.PolyData:
    points = np.asarray(xyz, dtype=float)
    mesh = pv.PolyData(points)
    mesh.lines = np.concatenate(([len(points)], np.arange(len(points), dtype=np.int64)))
    mesh.point_data["radius_mm"] = np.asarray(radius, dtype=float)
    mesh.point_data["normalized_arc_length"] = np.linspace(0.0, 1.0, len(points))
    mesh.point_data["disease_reduction_fraction"] = np.zeros(len(points), dtype=float)
    mesh.field_data["branch_id"] = np.asarray([BRANCH_IDS[branch]], dtype=np.int32)
    mesh.field_data["case_id"] = np.asarray([tree_id])
    mesh.field_data["phase"] = np.asarray([phase], dtype=float)
    return mesh


def surface_mesh(
    centerline: np.ndarray,
    reduction: np.ndarray,
    *,
    tree_id: str,
    branch: str,
    disease_case_id: str,
) -> pv.PolyData:
    """Create a capped, variable-radius display mesh from an Nx4 centerline."""
    values = np.asarray(centerline, dtype=float)
    reduction = np.asarray(reduction, dtype=float)
    surface = centerline_radius_to_surface(values, num_circle_points=MESH_CIRCLE_POINTS)
    ring_count, circle_count, _ = surface.shape
    points = surface.reshape(-1, 3)

    faces: list[int] = []
    for ring in range(ring_count - 1):
        for side in range(circle_count):
            next_side = (side + 1) % circle_count
            a = ring * circle_count + side
            b = ring * circle_count + next_side
            c = (ring + 1) * circle_count + next_side
            d = (ring + 1) * circle_count + side
            faces.extend((4, a, b, c, d))

    first_center = len(points)
    last_center = first_center + 1
    points = np.vstack((points, values[0, :3], values[-1, :3]))
    for side in range(circle_count):
        next_side = (side + 1) % circle_count
        faces.extend((3, first_center, next_side, side))
        a = (ring_count - 1) * circle_count + side
        b = (ring_count - 1) * circle_count + next_side
        faces.extend((3, last_center, a, b))

    mesh = pv.PolyData(points, np.asarray(faces, dtype=np.int64))
    radius = np.repeat(values[:, 3], circle_count)
    normalized = np.repeat(np.linspace(0.0, 1.0, ring_count), circle_count)
    disease = np.repeat(reduction, circle_count)
    mesh.point_data["radius_mm"] = np.concatenate((radius, values[[0, -1], 3]))
    mesh.point_data["diameter_mm"] = 2.0 * mesh.point_data["radius_mm"]
    mesh.point_data["normalized_arc_length"] = np.concatenate((normalized, [0.0, 1.0]))
    mesh.point_data["disease_reduction_fraction"] = np.concatenate(
        (disease, reduction[[0, -1]])
    )
    mesh.point_data["branch_id"] = np.full(mesh.n_points, BRANCH_IDS[branch], dtype=np.int32)
    mesh.field_data["case_id"] = np.asarray([tree_id])
    mesh.field_data["disease_case_id"] = np.asarray([disease_case_id])
    mesh.field_data["mesh_circle_points"] = np.asarray([circle_count], dtype=np.int32)
    return mesh


def mesh_centerline(
    centerline: np.ndarray,
    reduction: np.ndarray,
    *,
    tree_id: str,
    branch: str,
    disease_case_id: str,
) -> pv.PolyData:
    values = np.asarray(centerline, dtype=float)
    mesh = polyline(
        values[:, :3],
        values[:, 3],
        tree_id=tree_id,
        branch=branch,
        phase=0.0,
    )
    mesh.point_data["diameter_mm"] = 2.0 * values[:, 3]
    mesh.point_data["disease_reduction_fraction"] = np.asarray(reduction, dtype=float)
    mesh.field_data["disease_case_id"] = np.asarray([disease_case_id])
    return mesh


def write_mesh_presentations(tree_id: str, target: Path) -> dict[str, Any]:
    """Write separate surface-only and surface+centerline views.

    Disease is an explicit presentation scenario applied to radii only.  The
    source production XYZ anatomy is copied without alteration.
    """
    static = np.load(EXPORT / tree_id / "geometry_static.npy", allow_pickle=False)
    healthy = {branch: static[index].copy() for index, branch in enumerate(BRANCHES)}
    config = stenosis_config(
        DISEASE_BRANCH,
        DISEASE_POSITION,
        DISEASE_LENGTH,
        DISEASE_SEVERITY,
        "focal",
        case_id=f"{tree_id}_presentation_lad_focal_65",
    )
    diseased, disease_metadata = apply_disease_config(healthy, config)

    mesh_root = target / "mesh"
    centerline_root = target / "mesh_centerlines"
    maximum_surface_radius_error = 0.0
    maximum_xyz_change = 0.0
    written: dict[str, dict[str, Path]] = {}
    reductions: dict[str, dict[str, np.ndarray]] = {"healthy": {}, "diseased": {}}
    for state, tree in (("healthy", healthy), ("diseased", diseased)):
        (mesh_root / state).mkdir(parents=True)
        (centerline_root / state).mkdir(parents=True)
        written[state] = {}
        for branch in BRANCHES:
            reference_radius = healthy[branch][:, 3]
            reduction = 1.0 - tree[branch][:, 3] / reference_radius
            reduction[np.abs(reduction) < 1.0e-14] = 0.0
            reductions[state][branch] = reduction
            maximum_xyz_change = max(
                maximum_xyz_change,
                float(np.max(np.abs(tree[branch][:, :3] - healthy[branch][:, :3]))),
            )
            surface = surface_mesh(
                tree[branch],
                reduction,
                tree_id=tree_id,
                branch=branch,
                disease_case_id=config["case_id"] if state == "diseased" else "healthy",
            )
            surface_path = mesh_root / state / f"{branch}_MESH.vtp"
            surface.save(surface_path, binary=True)
            written[state][f"{branch}_MESH"] = surface_path

            centerline = mesh_centerline(
                tree[branch],
                reduction,
                tree_id=tree_id,
                branch=branch,
                disease_case_id=config["case_id"] if state == "diseased" else "healthy",
            )
            centerline_path = centerline_root / state / f"{branch}_CENTERLINE.vtp"
            centerline.save(centerline_path, binary=True)
            written[state][f"{branch}_CENTERLINE"] = centerline_path

            readback = pv.read(surface_path)
            ring_points = np.asarray(readback.points[: len(tree[branch]) * MESH_CIRCLE_POINTS])
            ring_points = ring_points.reshape(len(tree[branch]), MESH_CIRCLE_POINTS, 3)
            measured_radius = np.linalg.norm(ring_points - tree[branch][:, None, :3], axis=2)
            maximum_surface_radius_error = max(
                maximum_surface_radius_error,
                float(np.max(np.abs(measured_radius - tree[branch][:, None, 3]))),
            )

    healthy_mesh = [(f"{branch}_MESH", written["healthy"][f"{branch}_MESH"]) for branch in BRANCHES]
    diseased_mesh = [(f"{branch}_MESH", written["diseased"][f"{branch}_MESH"]) for branch in BRANCHES]
    healthy_lines = [
        (f"{branch}_CENTERLINE", written["healthy"][f"{branch}_CENTERLINE"]) for branch in BRANCHES
    ]
    diseased_lines = [
        (f"{branch}_CENTERLINE", written["diseased"][f"{branch}_CENTERLINE"]) for branch in BRANCHES
    ]
    write_vtm(target / "healthy_tapered_mesh.vtm", healthy_mesh)
    write_vtm(target / "healthy_mesh_with_centerlines.vtm", [*healthy_mesh, *healthy_lines])
    write_vtm(target / "diseased_tapered_mesh.vtm", diseased_mesh)
    write_vtm(target / "diseased_mesh_with_centerlines.vtm", [*diseased_mesh, *diseased_lines])

    taper = {
        branch: {
            "proximal_radius_mm": float(healthy[branch][0, 3]),
            "terminal_radius_mm": float(healthy[branch][-1, 3]),
            "proximal_diameter_mm": float(2.0 * healthy[branch][0, 3]),
            "terminal_diameter_mm": float(2.0 * healthy[branch][-1, 3]),
            "terminal_to_proximal_radius_ratio": float(healthy[branch][-1, 3] / healthy[branch][0, 3]),
        }
        for branch in BRANCHES
    }
    result = {
        "disease_config": config,
        "disease_metadata": disease_metadata,
        "taper": taper,
        "maximum_disease_reduction_fraction": {
            branch: float(np.max(reductions["diseased"][branch])) for branch in BRANCHES
        },
        "maximum_source_xyz_change_mm": maximum_xyz_change,
        "maximum_surface_radius_reconstruction_error_mm": maximum_surface_radius_error,
        "mesh_circle_points": MESH_CIRCLE_POINTS,
    }
    write_json(target / "mesh_visualization_metadata.json", result)
    return result


def copy_static(tree_id: str, target: Path) -> dict[str, Any]:
    source = COHORT / tree_id
    source_vtk = source / "vtk"
    static = target / "static"
    scaffold = target / "scaffold"
    static.mkdir()
    scaffold.mkdir()
    blocks: list[tuple[str, Path]] = []
    maximum_copy_error = 0.0
    for branch in BRANCHES:
        output = static / f"{branch}.vtp"
        shutil.copy2(source_vtk / f"{branch}.vtp", output)
        original = np.load(source / f"{branch}.npy", allow_pickle=False)
        copied = pv.read(output)
        maximum_copy_error = max(maximum_copy_error, float(np.max(np.abs(copied.points - original))))
        blocks.append((branch, output))
    write_vtm(target / "final_static_tree.vtm", blocks)

    ellipsoid = scaffold / "ELLIPSOID_SCAFFOLD.vtp"
    shutil.copy2(source_vtk / "synthetic_ellipsoid.vtp", ellipsoid)
    write_vtm(target / "final_tree_with_scaffold.vtm", [("ELLIPSOID_SCAFFOLD", ellipsoid), *blocks])
    return {"maximum_static_copy_coordinate_error_mm": maximum_copy_error}


def write_cine(tree_id: str, target: Path, phases: np.ndarray) -> dict[str, Any]:
    source = EXPORT / tree_id
    geometry = np.load(source / "geometry_cine.npy", allow_pickle=False)
    static = np.load(source / "geometry_static.npy", allow_pickle=False)
    expected_shape = (len(phases), len(BRANCHES), 50, 4)
    if geometry.shape != expected_shape or static.shape != expected_shape[1:]:
        raise ValueError(f"{tree_id}: unexpected export shapes {geometry.shape}/{static.shape}")
    if not np.all(np.isfinite(geometry)) or np.any(static[..., 3] <= 0.0):
        raise ValueError(f"{tree_id}: non-finite geometry or non-positive radius")

    phase_zero_error = float(np.max(np.abs(geometry[0, ..., :3] - static[..., :3])))
    source_closure_error = float(np.max(np.abs(geometry[0, ..., :3] - geometry[-1, ..., :3])))
    entries = []
    maximum_topology_error = 0.0
    maximum_radius_variation = 0.0
    first_radii: dict[str, np.ndarray] = {}
    last_radii: dict[str, np.ndarray] = {}
    for phase_index, phase in enumerate(phases):
        phase_dir = target / f"phase_{phase_index:03d}"
        phase_dir.mkdir()
        blocks = []
        for branch_index, branch in enumerate(BRANCHES):
            reference_radius = static[branch_index, :, 3]
            radius, _ = phase_radius(
                reference_radius,
                reference_radius,
                float(phase),
                amplitude=PULSATILITY_AMPLITUDE,
                stenosis_compliance_factor=STENOSIS_COMPLIANCE_FACTOR,
                peak_phase=PULSATILITY_PEAK_PHASE,
            )
            mesh = polyline(
                geometry[phase_index, branch_index, :, :3],
                radius,
                tree_id=tree_id,
                branch=branch,
                phase=float(phase),
            )
            path = phase_dir / f"{branch}.vtp"
            mesh.save(path, binary=True)
            blocks.append((branch, path))
            maximum_radius_variation = max(
                maximum_radius_variation,
                float(np.max(np.abs(radius / reference_radius - 1.0))),
            )
            if phase_index == 0:
                first_radii[branch] = radius
            if phase_index == len(phases) - 1:
                last_radii[branch] = radius
        write_vtm(phase_dir / "tree.vtm", blocks)
        maximum_topology_error = max(
            maximum_topology_error,
            float(np.linalg.norm(geometry[phase_index, 0, -1, :3] - geometry[phase_index, 1, 0, :3])),
            float(np.linalg.norm(geometry[phase_index, 0, -1, :3] - geometry[phase_index, 2, 0, :3])),
        )
        entries.append(
            f'    <DataSet timestep="{float(phase):.12g}" group="" part="0" '
            f'file="phase_{phase_index:03d}/tree.vtm"/>'
        )
    (target / "cine.pvd").write_text(
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <Collection>\n" + "\n".join(entries) + "\n  </Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )
    radius_closure_error = max(
        float(np.max(np.abs(first_radii[branch] - last_radii[branch]))) for branch in BRANCHES
    )
    return {
        "phase_count": len(phases),
        "point_count_per_branch": 50,
        "phase_zero_source_identity_error_mm": phase_zero_error,
        "source_motion_cycle_closure_error_mm": source_closure_error,
        "maximum_topology_error_mm": maximum_topology_error,
        "maximum_radius_variation_fraction": maximum_radius_variation,
        "radius_cycle_closure_error_mm": radius_closure_error,
    }


def verify_tree(tree_id: str, target: Path, source_checks: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    static = pv.read(target / "final_static_tree.vtm")
    scaffold = pv.read(target / "final_tree_with_scaffold.vtm")
    if list(static.keys()) != list(BRANCHES):
        errors.append("static block names differ from LMCA/LAD/LCX")
    if list(scaffold.keys()) != ["ELLIPSOID_SCAFFOLD", *BRANCHES]:
        errors.append("scaffold block names are invalid")
    mesh_expectations = {
        "healthy_tapered_mesh.vtm": [f"{branch}_MESH" for branch in BRANCHES],
        "diseased_tapered_mesh.vtm": [f"{branch}_MESH" for branch in BRANCHES],
        "healthy_mesh_with_centerlines.vtm": [
            *[f"{branch}_MESH" for branch in BRANCHES],
            *[f"{branch}_CENTERLINE" for branch in BRANCHES],
        ],
        "diseased_mesh_with_centerlines.vtm": [
            *[f"{branch}_MESH" for branch in BRANCHES],
            *[f"{branch}_CENTERLINE" for branch in BRANCHES],
        ],
    }
    for filename, names in mesh_expectations.items():
        dataset = pv.read(target / filename)
        if list(dataset.keys()) != names:
            errors.append(f"{filename}: invalid block names")
        for name in names:
            block = dataset[name]
            required = {"radius_mm", "diameter_mm", "normalized_arc_length", "disease_reduction_fraction"}
            if block is None or block.n_points <= 1 or not np.all(np.isfinite(block.points)):
                errors.append(f"{filename}/{name}: invalid or empty geometry")
                continue
            if not required.issubset(block.point_data.keys()):
                errors.append(f"{filename}/{name}: missing thickness/disease arrays")
            if np.any(np.asarray(block.point_data["radius_mm"]) <= 0.0):
                errors.append(f"{filename}/{name}: non-positive radius")
    pvd = ET.parse(target / "cine.pvd").getroot()
    entries = [element.attrib["file"] for element in pvd.iter("DataSet")]
    if len(entries) != 10:
        errors.append(f"cine phase count is {len(entries)}, expected 10")
    first: dict[str, np.ndarray] | None = None
    last: dict[str, np.ndarray] | None = None
    for relative in entries:
        path = target / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
            continue
        dataset = pv.read(path)
        if list(dataset.keys()) != list(BRANCHES):
            errors.append(f"{relative}: invalid branch names")
            continue
        frame = {}
        for branch in BRANCHES:
            block = dataset[branch]
            required = {"radius_mm", "normalized_arc_length", "disease_reduction_fraction"}
            if block is None or block.n_points != 50 or not np.all(np.isfinite(block.points)):
                errors.append(f"{relative}/{branch}: invalid points")
                continue
            if not required.issubset(block.point_data.keys()):
                errors.append(f"{relative}/{branch}: missing point arrays")
            if np.any(block.point_data["radius_mm"] <= 0.0):
                errors.append(f"{relative}/{branch}: non-positive radius")
            if np.any(block.point_data["disease_reduction_fraction"] != 0.0):
                errors.append(f"{relative}/{branch}: healthy disease reduction is not zero")
            frame[branch] = np.asarray(block.points)
        if len(frame) == 3:
            if first is None:
                first = frame
            last = frame
    readback_closure = None
    if first is not None and last is not None:
        readback_closure = max(float(np.max(np.abs(first[name] - last[name]))) for name in BRANCHES)
        if readback_closure > 1.0e-9:
            errors.append(f"readback closure error {readback_closure:.3e} mm")
    thresholds = {
        "maximum_static_copy_coordinate_error_mm": source_checks["maximum_static_copy_coordinate_error_mm"],
        "phase_zero_source_identity_error_mm": source_checks["phase_zero_source_identity_error_mm"],
        "source_motion_cycle_closure_error_mm": source_checks["source_motion_cycle_closure_error_mm"],
        "maximum_topology_error_mm": source_checks["maximum_topology_error_mm"],
        "radius_cycle_closure_error_mm": source_checks["radius_cycle_closure_error_mm"],
    }
    for name, value in thresholds.items():
        if value > 1.0e-9:
            errors.append(f"{name}={value:.3e}")
    if source_checks["maximum_source_xyz_change_mm"] > 1.0e-12:
        errors.append("disease visualization changed XYZ anatomy")
    if source_checks["maximum_surface_radius_reconstruction_error_mm"] > 1.0e-5:
        errors.append("surface thickness does not reconstruct the requested radius")
    disease_reduction = source_checks["maximum_disease_reduction_fraction"]
    if disease_reduction[DISEASE_BRANCH] < 0.60:
        errors.append("display stenosis is not visibly represented")
    if any(disease_reduction[branch] > 1.0e-12 for branch in BRANCHES if branch != DISEASE_BRANCH):
        errors.append("display stenosis affected an unintended branch")
    if source_checks["maximum_radius_variation_fraction"] > PULSATILITY_AMPLITUDE + 1.0e-8:
        errors.append("pulsatility exceeds configured amplitude")
    validation = load_json(COHORT / tree_id / "validation.json")
    if not validation.get("accepted", False):
        errors.append("canonical production validation is not accepted")
    return {
        "tree_id": tree_id,
        "status": "PASS" if not errors else "FAIL",
        "entry_files": [
            "final_static_tree.vtm",
            "final_tree_with_scaffold.vtm",
            "cine.pvd",
            *mesh_expectations,
        ],
        "canonical_warning_count": len(validation.get("warnings", [])),
        "readback_cycle_closure_error_mm": readback_closure,
        **source_checks,
        "errors": errors,
    }


def main() -> int:
    resolved = WORK.resolve()
    expected_parent = (ROOT / "submission_release").resolve()
    if resolved.parent != expected_parent or resolved.name != "final_presentation_52_working":
        raise RuntimeError(f"refusing unexpected work path: {resolved}")
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    source_before = aggregate_hash(COHORT)
    export_before = aggregate_hash(EXPORT)
    motion = load_json(MOTION_SUMMARY)["motion_summary"]
    phases = np.asarray(motion["phase_values"], dtype=float)
    tree_directories = sorted(COHORT.glob("tree_[0-9][0-9][0-9][0-9]"))
    export_directories = sorted(EXPORT.glob("tree_[0-9][0-9][0-9][0-9]"))
    if len(tree_directories) != 52 or len(export_directories) != 52:
        raise RuntimeError(f"expected 52 cohort/export trees, found {len(tree_directories)}/{len(export_directories)}")

    rows = []
    for index, source in enumerate(tree_directories, start=1):
        tree_id = source.name
        target = WORK / tree_id
        target.mkdir()
        static_checks = copy_static(tree_id, target)
        cine_checks = write_cine(tree_id, target, phases)
        mesh_checks = write_mesh_presentations(tree_id, target)
        row = verify_tree(tree_id, target, {**static_checks, **cine_checks, **mesh_checks})
        rows.append(row)
        if index % 10 == 0 or index == len(tree_directories):
            print(f"verified {index}/52", flush=True)

    source_after = aggregate_hash(COHORT)
    export_after = aggregate_hash(EXPORT)
    source_unchanged = source_before == source_after
    export_unchanged = export_before == export_after
    passed = all(row["status"] == "PASS" for row in rows) and source_unchanged and export_unchanged
    files = sorted(path for path in WORK.rglob("*") if path.is_file())
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "scope": (
            "52 accepted surface-relative production LCA trees; healthy production motion/pulsatility; "
            "static healthy taper and explicit parametric LAD disease visualization meshes"
        ),
        "tree_count_expected": 52,
        "tree_count_built": len(rows),
        "tree_count_verified": sum(row["status"] == "PASS" for row in rows),
        "entry_file_count_per_tree": 7,
        "entry_file_names": [
            "final_static_tree.vtm",
            "final_tree_with_scaffold.vtm",
            "cine.pvd",
            "healthy_tapered_mesh.vtm",
            "healthy_mesh_with_centerlines.vtm",
            "diseased_tapered_mesh.vtm",
            "diseased_mesh_with_centerlines.vtm",
        ],
        "cine_phases_per_tree": len(phases),
        "total_cine_frames": len(rows) * len(phases),
        "branch_order": list(BRANCHES),
        "pulsatility": {
            "amplitude": PULSATILITY_AMPLITUDE,
            "peak_phase": PULSATILITY_PEAK_PHASE,
            "healthy_case_disease_reduction_fraction": 0.0,
        },
        "mesh_visualization": {
            "circle_points": MESH_CIRCLE_POINTS,
            "surface_strategy": "separate capped branch meshes; no enclosing union or scaffold surface",
            "thickness_arrays": ["radius_mm", "diameter_mm"],
            "disease_array": "disease_reduction_fraction",
            "disease_is_parametric_not_learned": True,
            "disease_changes_radius_only": True,
            "disease_scenario": {
                "branch": DISEASE_BRANCH,
                "type": "focal",
                "normalized_position": DISEASE_POSITION,
                "normalized_length": DISEASE_LENGTH,
                "fractional_radius_reduction": DISEASE_SEVERITY,
            },
        },
        "source_integrity": {
            "canonical_cohort_unchanged": source_unchanged,
            "production_export_unchanged": export_unchanged,
            "canonical_cohort_before": source_before,
            "canonical_cohort_after": source_after,
            "production_export_before": export_before,
            "production_export_after": export_after,
        },
        "maximum_errors": {
            "static_copy_coordinate_mm": max(row["maximum_static_copy_coordinate_error_mm"] for row in rows),
            "phase_zero_source_identity_mm": max(row["phase_zero_source_identity_error_mm"] for row in rows),
            "source_cycle_closure_mm": max(row["source_motion_cycle_closure_error_mm"] for row in rows),
            "topology_mm": max(row["maximum_topology_error_mm"] for row in rows),
            "readback_cycle_closure_mm": max(row["readback_cycle_closure_error_mm"] for row in rows),
            "radius_cycle_closure_mm": max(row["radius_cycle_closure_error_mm"] for row in rows),
            "radius_variation_fraction": max(row["maximum_radius_variation_fraction"] for row in rows),
            "mesh_source_xyz_change_mm": max(row["maximum_source_xyz_change_mm"] for row in rows),
            "surface_radius_reconstruction_mm": max(
                row["maximum_surface_radius_reconstruction_error_mm"] for row in rows
            ),
        },
        "trees": rows,
        "file_count_before_manifest": len(files),
        "size_bytes_before_manifest": sum(path.stat().st_size for path in files),
    }
    write_json(WORK / "ALL_52_PRESENTATION_MANIFEST.json", manifest)
    if not passed:
        raise RuntimeError(json.dumps({"failed": [row for row in rows if row["status"] != "PASS"]}, indent=2))

    if OUT.exists():
        resolved_out = OUT.resolve()
        if resolved_out.parent != expected_parent or resolved_out.name != "final_presentation_52":
            raise RuntimeError(f"refusing unexpected output path: {resolved_out}")
        shutil.rmtree(OUT)
    WORK.rename(OUT)
    print(json.dumps({
        "status": manifest["status"],
        "output": OUT.relative_to(ROOT).as_posix(),
        "trees": f"{manifest['tree_count_verified']}/{manifest['tree_count_expected']}",
        "entry_files": 52 * 7,
        "cine_frames": manifest["total_cine_frames"],
        "size_bytes": manifest["size_bytes_before_manifest"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
