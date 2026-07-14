import argparse
import copy
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .anatomical_lca_reorientation import (
    BRANCH_COLORS,
    BRANCH_IDS,
    BRANCH_SLICES,
    _branch_lengths,
    save_branch_polyline_vtk,
    validate_anatomical_lca_tree,
)
from .apply_disease_to_lca import process_patient as process_disease_patient
from .bspline import interpolate_lca_tree
from .disease_model import DEFAULT_DISEASE_SETTINGS
from .lca_validation import validate_lca_tree
from .paths import lca_generated_control_points_dir, output_path
from .radius_model import (
    DEFAULT_STATIC_RADIUS_MODEL,
    build_lca_radius_tree,
    cube_law_parent_radius,
    validate_lca_radius_tree,
)
from .tight_mesh import (
    build_lca_tight_mesh,
    save_tight_mesh_ply,
    save_tight_mesh_preview,
    save_tight_mesh_stl,
    validate_tight_mesh,
)
from .tortuosity import calculate_lca_tortuosity
from .tube_surface import build_lca_tube_surfaces, validate_lca_tube_surfaces
from .visualize import set_axes_equal


BRANCH_ORDER = ["LMCA", "LAD", "LCX"]


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(_json_safe(data), json_file, indent=2)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


def _patient_index_from_id(patient_id: str):
    match = re.search(r"patient_(\d+)", patient_id)
    if match is None:
        return None
    return int(match.group(1))


def _radius_axis(points_with_radius: np.ndarray) -> np.ndarray:
    if len(points_with_radius) == 0:
        return np.zeros(0, dtype=float)
    if len(points_with_radius) == 1:
        return np.zeros(1, dtype=float)
    distances = np.zeros(len(points_with_radius), dtype=float)
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points_with_radius[:, :3], axis=0), axis=1))
    total = distances[-1]
    if total <= 1e-9:
        return np.zeros(len(points_with_radius), dtype=float)
    return distances / total


def _save_radius_profile_plot(path: Path, centerlines_with_radius: dict, title: str):
    fig, ax = plt.subplots(figsize=(5.4, 3.3))
    for branch_name in BRANCH_ORDER:
        points = centerlines_with_radius[branch_name]
        ax.plot(
            _radius_axis(points),
            points[:, 3],
            color=BRANCH_COLORS[branch_name],
            linewidth=2.0,
            label=branch_name,
        )

    ax.set_xlabel("Normalized branch arc length")
    ax.set_ylabel("Radius (mm)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_tree_with_radius_plot(path: Path, centerlines_with_radius: dict, title: str):
    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_subplot(projection="3d")

    for branch_name in BRANCH_ORDER:
        points = centerlines_with_radius[branch_name]
        radius = points[:, 3]
        linewidth = 0.9 + float(np.mean(radius)) * 1.25
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=BRANCH_COLORS[branch_name],
            linewidth=linewidth,
            alpha=0.88,
            label=f"{branch_name} {radius[0]:.2f}->{radius[-1]:.2f} mm",
        )
        ax.scatter(
            [points[0, 0], points[-1, 0]],
            [points[0, 1], points[-1, 1]],
            [points[0, 2], points[-1, 2]],
            color=BRANCH_COLORS[branch_name],
            edgecolor="white",
            linewidth=0.6,
            s=[42, 22],
            depthshade=False,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    set_axes_equal(ax)
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_tube_surface_plot(path: Path, tube_surfaces: dict, centerlines_with_radius: dict, title: str):
    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_subplot(projection="3d")
    surface_colors = {
        "LMCA": "#4a4a4a",
        "LAD": "#d62728",
        "LCX": "#1f77b4",
    }

    for branch_name in BRANCH_ORDER:
        surface = tube_surfaces[branch_name]
        ax.plot_surface(
            surface[:, :, 0],
            surface[:, :, 1],
            surface[:, :, 2],
            color=surface_colors[branch_name],
            alpha=0.72,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        centerline = centerlines_with_radius[branch_name]
        ax.plot(
            centerline[:, 0],
            centerline[:, 1],
            centerline[:, 2],
            color="white",
            linewidth=0.8,
            alpha=0.85,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    set_axes_equal(ax)
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_centerline_vtk(path: Path, centerlines: dict, title: str):
    save_branch_polyline_vtk(path, centerlines, title)


def _control_points_with_radius(control_tree: np.ndarray, radius_model: dict) -> np.ndarray:
    control_centerlines = {
        branch_name: control_tree[BRANCH_SLICES[branch_name]]
        for branch_name in BRANCH_ORDER
    }
    control_radius_tree, _ = build_lca_radius_tree(control_centerlines, radius_model)
    return np.vstack([control_radius_tree[branch_name] for branch_name in BRANCH_ORDER])


def _metadata_path_for_patient(metadata_dir: Path | None, patient_id: str):
    if metadata_dir is None:
        return None
    candidate = metadata_dir / patient_id / "patient_info.json"
    if candidate.exists():
        return candidate
    return None


def _radius_model_for_patient(metadata_dir: Path | None, metadata_patient_id: str | None) -> dict:
    model = copy.deepcopy(DEFAULT_STATIC_RADIUS_MODEL)
    metadata_path = _metadata_path_for_patient(metadata_dir, metadata_patient_id or "")
    metadata = _load_json(metadata_path) if metadata_path else {}
    model["metadata_source"] = str(metadata_path) if metadata_path else None
    model["branch_radius_source"] = {}

    for branch_name in BRANCH_ORDER:
        lower = branch_name.lower()
        radius_value = metadata.get(f"{lower}_radius_mm")
        diameter_value = metadata.get(f"{lower}_diameter_mm")
        if radius_value is not None:
            model["branches"][branch_name]["proximal_radius_mm"] = float(radius_value)
            model["branch_radius_source"][branch_name] = "patient_metadata_radius"
        elif diameter_value is not None:
            model["branches"][branch_name]["proximal_radius_mm"] = float(diameter_value) / 2.0
            model["branch_radius_source"][branch_name] = "patient_metadata_diameter"
        else:
            model["branch_radius_source"][branch_name] = "mvp_default"

    lad_radius = model["branches"]["LAD"]["proximal_radius_mm"]
    lcx_radius = model["branches"]["LCX"]["proximal_radius_mm"]
    lmca_required = cube_law_parent_radius(lad_radius, lcx_radius)
    lmca_radius = model["branches"]["LMCA"]["proximal_radius_mm"]
    if lmca_radius < lmca_required:
        model["branches"]["LMCA"]["proximal_radius_mm"] = lmca_required
        model["branch_radius_source"]["LMCA"] += "_cube_law_floor"

    return model


def load_anatomical_records(input_dir: Path, control_filename: str = "control_points_27x3_anatomical.npy") -> list:
    tree_dir = input_dir / "trees"
    if not tree_dir.exists():
        raise FileNotFoundError(f"Missing anatomical tree directory: {tree_dir}")

    records = []
    for patient_dir in sorted(path for path in tree_dir.glob("patient_*") if path.is_dir()):
        control_path = patient_dir / control_filename
        if not control_path.exists():
            alternate = patient_dir / "control_points_27x3_synthetic.npy"
            if alternate.exists():
                control_path = alternate
            else:
                continue
        control_tree = np.load(control_path)
        records.append({
            "patient_id": patient_dir.name,
            "metadata_patient_id": patient_dir.name,
            "tree_index": _patient_index_from_id(patient_dir.name),
            "control_tree": control_tree,
            "source_dir": str(patient_dir),
            "reference_lengths": None,
        })
    return records


def _save_patient_outputs(
    patient_dir: Path,
    record: dict,
    centerlines: dict,
    centerlines_with_radius: dict,
    control_points_with_radius: np.ndarray,
    tube_surfaces: dict,
    tight_mesh_vertices: np.ndarray,
    tight_mesh_faces: np.ndarray,
    reports: dict,
):
    patient_id = record["patient_id"]
    control_tree = record["control_tree"]
    patient_dir.mkdir(parents=True, exist_ok=True)

    np.save(patient_dir / "control_points_27x3.npy", control_tree)
    np.save(patient_dir / "control_points_27x3_anatomical.npy", control_tree)
    np.save(patient_dir / "control_points_27x4.npy", control_points_with_radius)
    for branch_name in BRANCH_ORDER:
        np.save(patient_dir / f"{branch_name.lower()}_centerline.npy", centerlines[branch_name])
        np.save(patient_dir / f"{branch_name.lower()}_centerline_radius.npy", centerlines_with_radius[branch_name])
    np.savez(
        patient_dir / "tree_centerlines.npz",
        LMCA=centerlines["LMCA"],
        LAD=centerlines["LAD"],
        LCX=centerlines["LCX"],
    )
    np.savez(
        patient_dir / "tree_centerline_radius.npz",
        LMCA=centerlines_with_radius["LMCA"],
        LAD=centerlines_with_radius["LAD"],
        LCX=centerlines_with_radius["LCX"],
    )
    np.savez(
        patient_dir / "tree_tube_surface.npz",
        LMCA=tube_surfaces["LMCA"],
        LAD=tube_surfaces["LAD"],
        LCX=tube_surfaces["LCX"],
    )
    np.savez(patient_dir / "tree_tight_mesh.npz", vertices=tight_mesh_vertices, faces=tight_mesh_faces)

    _write_json(patient_dir / "anatomical_validation.json", reports["anatomical_validation"])
    _write_json(patient_dir / "centerline_validation.json", reports["centerline_validation"])
    _write_json(patient_dir / "tortuosity_metrics.json", reports["tortuosity_metrics"])
    _write_json(patient_dir / "radius_validation.json", reports["radius_validation"])
    _write_json(patient_dir / "tube_surface_validation.json", reports["tube_validation"])
    _write_json(patient_dir / "tight_mesh_validation.json", reports["tight_mesh_validation"])
    _write_json(patient_dir / "radius_summary.json", reports["radius_summary"])
    _write_json(patient_dir / "patient_pipeline_summary.json", reports["patient_summary"])

    _save_centerline_vtk(patient_dir / "anatomical_pipeline_centerlines.vtk", centerlines, f"{patient_id} anatomical pipeline centerlines")
    _save_radius_profile_plot(patient_dir / "radius_profile.png", centerlines_with_radius, f"{patient_id} static radius taper")
    _save_tree_with_radius_plot(patient_dir / "tree_with_radius.png", centerlines_with_radius, f"{patient_id} anatomical tree with radius")
    _save_tube_surface_plot(patient_dir / "tree_tube_surface.png", tube_surfaces, centerlines_with_radius, f"{patient_id} tube surface")
    save_tight_mesh_preview(patient_dir / "tree_tight_mesh.png", tight_mesh_vertices, tight_mesh_faces)
    save_tight_mesh_ply(patient_dir / "tree_tight_mesh.ply", tight_mesh_vertices, tight_mesh_faces)
    save_tight_mesh_stl(patient_dir / "tree_tight_mesh.stl", tight_mesh_vertices, tight_mesh_faces)


def _process_anatomical_record(record: dict, patient_dir: Path, args) -> dict:
    patient_id = record["patient_id"]
    control_tree = np.asarray(record["control_tree"], dtype=float)
    reference_lengths = record.get("reference_lengths")
    anatomical_validation = validate_anatomical_lca_tree(control_tree, reference_lengths=reference_lengths)
    if not anatomical_validation["is_valid"]:
        return {
            "patient_id": patient_id,
            "status": "failed",
            "stage": "anatomical_validation",
            "errors": anatomical_validation["errors"],
            "anatomical_validation": anatomical_validation,
        }

    centerlines = interpolate_lca_tree(
        control_tree,
        lmca_points=args.lmca_points,
        lad_points=args.lad_points,
        lcx_points=args.lcx_points,
    )
    tortuosity_metrics = calculate_lca_tortuosity(centerlines)
    centerline_validation = validate_lca_tree(control_tree, centerlines, tortuosity_metrics)
    if not centerline_validation["is_valid"]:
        return {
            "patient_id": patient_id,
            "status": "failed",
            "stage": "centerline_validation",
            "errors": centerline_validation["errors"],
            "warnings": centerline_validation["warnings"],
            "anatomical_validation": anatomical_validation,
            "centerline_validation": centerline_validation,
        }

    radius_model = _radius_model_for_patient(args.metadata_dir, record.get("metadata_patient_id", patient_id))
    centerlines_with_radius, radius_metadata = build_lca_radius_tree(centerlines, radius_model)
    radius_validation = validate_lca_radius_tree(
        centerlines_with_radius,
        adjacent_jump_threshold_mm=args.radius_adjacent_jump_threshold,
    )
    if not radius_validation["is_valid"]:
        return {
            "patient_id": patient_id,
            "status": "failed",
            "stage": "radius_validation",
            "errors": radius_validation["errors"],
            "warnings": radius_validation["warnings"],
            "anatomical_validation": anatomical_validation,
            "centerline_validation": centerline_validation,
            "radius_validation": radius_validation,
        }

    control_points_with_radius = _control_points_with_radius(control_tree, radius_model)
    tube_surfaces, tube_metadata = build_lca_tube_surfaces(
        centerlines_with_radius,
        num_circle_points=args.tube_circle_points,
    )
    tube_validation = validate_lca_tube_surfaces(tube_surfaces, centerlines_with_radius)
    if not tube_validation["is_valid"]:
        return {
            "patient_id": patient_id,
            "status": "failed",
            "stage": "tube_surface_validation",
            "errors": tube_validation["errors"],
            "warnings": tube_validation["warnings"],
            "anatomical_validation": anatomical_validation,
            "centerline_validation": centerline_validation,
            "radius_validation": radius_validation,
            "tube_validation": tube_validation,
        }

    tight_mesh_vertices, tight_mesh_faces, tight_mesh_metadata = build_lca_tight_mesh(tube_surfaces)
    tight_mesh_validation = validate_tight_mesh(tight_mesh_vertices, tight_mesh_faces, tight_mesh_metadata)
    if not tight_mesh_validation["is_valid"]:
        return {
            "patient_id": patient_id,
            "status": "failed",
            "stage": "tight_mesh_validation",
            "errors": tight_mesh_validation["errors"],
            "warnings": tight_mesh_validation["warnings"],
            "anatomical_validation": anatomical_validation,
            "centerline_validation": centerline_validation,
            "radius_validation": radius_validation,
            "tube_validation": tube_validation,
            "tight_mesh_validation": tight_mesh_validation,
        }

    output_shapes = {
        "control_points_27x3": list(control_tree.shape),
        "control_points_27x4": list(control_points_with_radius.shape),
        "centerline_LMCA": list(centerlines["LMCA"].shape),
        "centerline_LAD": list(centerlines["LAD"].shape),
        "centerline_LCX": list(centerlines["LCX"].shape),
        "radius_LMCA": list(centerlines_with_radius["LMCA"].shape),
        "radius_LAD": list(centerlines_with_radius["LAD"].shape),
        "radius_LCX": list(centerlines_with_radius["LCX"].shape),
        "tube_surface_LMCA": list(tube_surfaces["LMCA"].shape),
        "tube_surface_LAD": list(tube_surfaces["LAD"].shape),
        "tube_surface_LCX": list(tube_surfaces["LCX"].shape),
        "tight_mesh_vertices": list(tight_mesh_vertices.shape),
        "tight_mesh_faces": list(tight_mesh_faces.shape),
    }
    radius_summary = {
        "units": "mm",
        "columns": ["x", "y", "z", "radius_mm"],
        "radius_model": radius_metadata,
        "branch_radius_source": radius_metadata.get("branch_radius_source", {}),
        "output_shapes": output_shapes,
        "validation": radius_validation,
        "tube_surface": tube_metadata,
        "tube_validation": tube_validation,
        "tight_mesh": tight_mesh_metadata,
        "tight_mesh_validation": tight_mesh_validation,
    }
    patient_summary = {
        "patient_id": patient_id,
        "source_dir": record.get("source_dir"),
        "metadata_patient_id": record.get("metadata_patient_id", patient_id),
        "status": "completed",
        "anatomical_score": anatomical_validation["anatomical_metrics"]["anatomical_score"],
        "lad_downward_score": anatomical_validation["anatomical_metrics"]["lad_downward_score"],
        "lcx_lateral_score": anatomical_validation["anatomical_metrics"]["lcx_lateral_score"],
        "branch_lengths_mm": _branch_lengths(control_tree),
        "validations": {
            "anatomical": anatomical_validation["is_valid"],
            "centerline": centerline_validation["is_valid"],
            "radius": radius_validation["is_valid"],
            "tube_surface": tube_validation["is_valid"],
            "tight_mesh": tight_mesh_validation["is_valid"],
        },
        "output_shapes": output_shapes,
        "tight_mesh_boundary_edge_count": tight_mesh_validation["boundary_edge_count"],
        "tight_mesh_nonmanifold_edge_count": tight_mesh_validation["nonmanifold_edge_count"],
        "tight_mesh_connected_component_count": tight_mesh_validation["connected_component_count"],
    }
    reports = {
        "anatomical_validation": anatomical_validation,
        "centerline_validation": centerline_validation,
        "tortuosity_metrics": tortuosity_metrics,
        "radius_validation": radius_validation,
        "tube_validation": tube_validation,
        "tight_mesh_validation": tight_mesh_validation,
        "radius_summary": radius_summary,
        "patient_summary": patient_summary,
    }
    _save_patient_outputs(
        patient_dir,
        record,
        centerlines,
        centerlines_with_radius,
        control_points_with_radius,
        tube_surfaces,
        tight_mesh_vertices,
        tight_mesh_faces,
        reports,
    )
    return {
        **patient_summary,
        "stage": "completed",
        "errors": [],
        "warnings": {
            "anatomical": anatomical_validation["warnings"],
            "centerline": centerline_validation["warnings"],
            "radius": radius_validation["warnings"],
            "tube_surface": tube_validation["warnings"],
            "tight_mesh": tight_mesh_validation["warnings"],
        },
    }


def _disease_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        config=None,
        minimum_radius_mm=args.minimum_radius_mm,
        adjacent_jump_threshold_mm=args.disease_adjacent_jump_threshold,
        tube_circle_points=args.tube_circle_points,
        skip_mesh=False,
    )


def _run_disease_pipeline(output_dir: Path, completed_patient_ids: list, args) -> tuple:
    disease_output_dir = output_dir / "disease"
    disease_tree_dir = disease_output_dir / "trees"
    disease_tree_dir.mkdir(parents=True, exist_ok=True)
    all_cases = []
    failures = []
    disease_args = _disease_args(args)

    for patient_id in completed_patient_ids:
        patient_dir = output_dir / "trees" / patient_id
        output_patient_dir = disease_tree_dir / patient_id
        output_patient_dir.mkdir(parents=True, exist_ok=True)
        try:
            all_cases.extend(process_disease_patient(patient_dir, output_patient_dir, disease_args))
        except Exception as exc:
            failures.append({
                "patient_id": patient_id,
                "stage": "disease_pipeline",
                "error": str(exc),
            })

    disease_summary = {
        "mode": "anatomical_lca_disease_reuse",
        "description": "Existing MVP disease cases applied to anatomical radius-enabled LCA centerlines.",
        "output_dir": str(disease_output_dir),
        "patients_processed": int(len(completed_patient_ids)),
        "cases_generated": int(len(all_cases)),
        "valid_cases": int(sum(1 for case in all_cases if case["validation_status"] == "valid")),
        "invalid_cases": int(sum(1 for case in all_cases if case["validation_status"] != "valid")),
        "valid_tight_mesh_cases": int(sum(1 for case in all_cases if case["tight_mesh_validation_status"] == "valid")),
        "invalid_tight_mesh_cases": int(sum(1 for case in all_cases if case["tight_mesh_validation_status"] == "invalid")),
        "failed_patients": failures,
        "cases": all_cases,
    }
    _write_json(disease_output_dir / "disease_summary.json", disease_summary)
    _write_json(
        disease_output_dir / "disease_validation.json",
        {
            "total_cases": len(all_cases),
            "valid_cases": disease_summary["valid_cases"],
            "invalid_cases": disease_summary["invalid_cases"],
            "cases": all_cases,
            "failed_patients": failures,
        },
    )
    return disease_summary, failures


def _average_metric(records: list, key: str):
    values = [
        float(record[key])
        for record in records
        if record.get("status") == "completed" and key in record
    ]
    if not values:
        return None
    return float(np.mean(values))


def _write_summary_markdown(path: Path, summary: dict):
    failed = summary["failed_cases"]
    lines = [
        "# Anatomical LCA Pipeline Summary",
        "",
        "## Counts",
        "",
        f"- Total anatomical trees discovered: {summary['total_anatomical_trees_discovered']}",
        f"- Total anatomical trees processed: {summary['total_anatomical_trees_processed']}",
        f"- Valid centerline outputs: {summary['valid_centerline_outputs']}",
        f"- Valid radius outputs: {summary['valid_radius_outputs']}",
        f"- Valid normal tight mesh outputs: {summary['valid_tight_mesh_outputs']}",
        f"- Disease cases generated: {summary['disease_cases_generated']}",
        f"- Valid diseased cases: {summary['valid_diseased_cases']}",
        f"- Valid diseased tight mesh cases: {summary['valid_diseased_tight_mesh_cases']}",
        "",
        "## Average Anatomical Scores",
        "",
        f"- Average LAD downward score: {summary['average_lad_downward_score']}",
        f"- Average LCX lateral score: {summary['average_lcx_lateral_score']}",
        f"- Average anatomical score: {summary['average_anatomical_score']}",
        "",
        "## Failed Cases",
        "",
    ]
    if failed:
        for case in failed:
            lines.append(f"- {case.get('patient_id')}: {case.get('stage')} - {case.get('errors') or case.get('error')}")
    else:
        lines.append("- None")
    lines.extend([
        "",
        "## Limitations",
        "",
        "- This is MVP anatomical realism, not clinical reconstruction.",
        "- Vessels are not fitted to a real heart surface.",
        "- Patient-specific apex, base, and groove landmarks are not available.",
        "- The corrected geometry is rule-based and designed to remain compatible with the current radius, disease, tube, and mesh pipeline.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline_records(records: list, output_dir: Path, args) -> dict:
    output_dir = Path(output_dir)
    tree_dir = output_dir / "trees"
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    tree_dir.mkdir(parents=True, exist_ok=True)

    patient_records = []
    failed_cases = []
    completed_patient_ids = []
    control_trees = []

    for record in records:
        patient_id = record["patient_id"]
        patient_dir = tree_dir / patient_id
        try:
            result = _process_anatomical_record(record, patient_dir, args)
        except Exception as exc:
            result = {
                "patient_id": patient_id,
                "status": "failed",
                "stage": "exception",
                "error": str(exc),
            }
        patient_records.append(result)
        if result.get("status") == "completed":
            completed_patient_ids.append(patient_id)
            control_trees.append(np.asarray(record["control_tree"], dtype=float))
        else:
            failed_cases.append(result)

    if control_trees:
        np.save(output_dir / "LCA_tree_ctrl_points_anatomical_pipeline.npy", np.asarray(control_trees, dtype=float))

    disease_summary, disease_failures = _run_disease_pipeline(output_dir, completed_patient_ids, args)
    failed_cases.extend(disease_failures)

    completed = [record for record in patient_records if record.get("status") == "completed"]
    summary = {
        "mode": "anatomical_lca_full_pipeline_mvp",
        "description": "Fresh anatomical control points passed through existing centerline, radius, disease, tube, and tight mesh stages.",
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "total_anatomical_trees_discovered": int(len(records)),
        "total_anatomical_trees_processed": int(len(completed)),
        "valid_centerline_outputs": int(sum(record["validations"]["centerline"] for record in completed)),
        "valid_radius_outputs": int(sum(record["validations"]["radius"] for record in completed)),
        "valid_tube_surface_outputs": int(sum(record["validations"]["tube_surface"] for record in completed)),
        "valid_tight_mesh_outputs": int(sum(record["validations"]["tight_mesh"] for record in completed)),
        "disease_cases_generated": disease_summary["cases_generated"],
        "valid_diseased_cases": disease_summary["valid_cases"],
        "invalid_diseased_cases": disease_summary["invalid_cases"],
        "valid_diseased_tight_mesh_cases": disease_summary["valid_tight_mesh_cases"],
        "failed_cases": failed_cases,
        "average_lad_downward_score": _average_metric(completed, "lad_downward_score"),
        "average_lcx_lateral_score": _average_metric(completed, "lcx_lateral_score"),
        "average_anatomical_score": _average_metric(completed, "anatomical_score"),
        "patients": patient_records,
        "disease": disease_summary,
        "limitations": [
            "MVP anatomical realism, not clinical reconstruction.",
            "No real heart surface fitting.",
            "No patient-specific apex/base/groove landmarks.",
            "Disease, tube surface, tight mesh, and hub connector logic are reused unchanged.",
        ],
    }
    _write_json(output_dir / "anatomical_pipeline_summary.json", summary)
    _write_json(
        output_dir / "anatomical_pipeline_validation.json",
        {
            "total_cases": len(patient_records),
            "valid_cases": len(completed),
            "failed_cases": failed_cases,
            "patients": patient_records,
        },
    )
    _write_summary_markdown(output_dir / "ANATOMICAL_PIPELINE_SUMMARY.md", summary)
    return summary


def _default_args(base_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        input_dir=output_path("dataset_lca_anatomical"),
        output_dir=output_path("dataset_lca_anatomical_pipeline"),
        metadata_dir=lca_generated_control_points_dir(),
        no_clean=False,
        lmca_points=150,
        lad_points=300,
        lcx_points=250,
        radius_adjacent_jump_threshold=0.2,
        tube_circle_points=24,
        minimum_radius_mm=DEFAULT_DISEASE_SETTINGS["minimum_radius_mm"],
        disease_adjacent_jump_threshold=DEFAULT_DISEASE_SETTINGS["adjacent_jump_threshold_mm"],
    )


def parse_args() -> argparse.Namespace:
    defaults = _default_args(Path.cwd())
    parser = argparse.ArgumentParser(description="Run full existing LCA pipeline on anatomical 27x3 control points.")
    parser.add_argument("--input-dir", type=Path, default=defaults.input_dir)
    parser.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    parser.add_argument("--metadata-dir", type=Path, default=defaults.metadata_dir)
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--lmca-points", type=int, default=defaults.lmca_points)
    parser.add_argument("--lad-points", type=int, default=defaults.lad_points)
    parser.add_argument("--lcx-points", type=int, default=defaults.lcx_points)
    parser.add_argument("--radius-adjacent-jump-threshold", type=float, default=defaults.radius_adjacent_jump_threshold)
    parser.add_argument("--tube-circle-points", type=int, default=defaults.tube_circle_points)
    parser.add_argument("--minimum-radius-mm", type=float, default=defaults.minimum_radius_mm)
    parser.add_argument("--disease-adjacent-jump-threshold", type=float, default=defaults.disease_adjacent_jump_threshold)
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_anatomical_records(args.input_dir)
    if len(records) == 0:
        raise FileNotFoundError(f"No anatomical patient records found in {args.input_dir}")
    summary = run_pipeline_records(records, args.output_dir, args)
    print(f"Processed anatomical patients: {summary['total_anatomical_trees_processed']}")
    print(f"Disease cases generated: {summary['disease_cases_generated']}")
    print(f"Valid diseased tight mesh cases: {summary['valid_diseased_tight_mesh_cases']}")
    print(f"Summary: {args.output_dir / 'ANATOMICAL_PIPELINE_SUMMARY.md'}")


if __name__ == "__main__":
    main()
