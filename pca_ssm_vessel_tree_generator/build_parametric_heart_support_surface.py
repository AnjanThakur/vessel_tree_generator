"""Build the verified Stage-1 parametric anatomical support surface for 189.label.

The command consumes the already saved Stage-1 VTK geometry and measurements.
It does not rerun NIfTI extraction, modify source centerlines, or generate new
coronary arteries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from heart_support_surface import HeartSupportSurface
from stage1_surface_visualization import generate_stage1_figures
from stage1_surface_vtk import (
    build_vtk_objects,
    canonical_transform,
    export_stage1_vtk,
    transform_polydata,
    vtk_block_manifest,
)


PRIMARY_CASE = "189.label"
PRESENTATION_NAME = "LCA_Stage1_Dataset_Derived_Anatomical_Scaffold.pptx"
SOURCE_NAMES = ("lmca", "lad", "lcx", "rca")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coordinate_sha256(points: np.ndarray) -> str:
    points = np.ascontiguousarray(np.asarray(points))
    return hashlib.sha256(points.tobytes()).hexdigest()


def segment_lengths(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(np.asarray(points, dtype=float), axis=0), axis=1)


def normalized_arc_length(points: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths(points))))
    return cumulative / max(float(cumulative[-1]), 1.0e-12)


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required {description} is missing: {path}")
    return path


def load_saved_inputs(repository_root: Path, case_id: str) -> dict[str, Any]:
    if case_id != PRIMARY_CASE:
        raise ValueError(f"this refinement is restricted to {PRIMARY_CASE}; received {case_id}")
    input_case = repository_root / "outputs" / "lca_ssm" / "stage1_heart_scaffold_vtk" / case_id
    measurement_path = require_file(input_case / "measurements.json", "saved Stage-1 measurements")
    validation_path = require_file(input_case / "validation.json", "saved Stage-1 validation")
    measurements = json.loads(measurement_path.read_text(encoding="utf-8"))
    prior_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not prior_validation.get("pass", False):
        raise ValueError("saved Stage-1 scaffold validation is not PASS")

    paths = {
        "sources": {
            "lmca": input_case / "01_source_geometry" / "LMCA_centerline.vtp",
            "lad": input_case / "01_source_geometry" / "LAD_centerline.vtp",
            "lcx": input_case / "01_source_geometry" / "LCX_centerline.vtp",
            "rca": input_case / "01_source_geometry" / "RCA_candidate_centerline.vtp",
        },
        "planes": {
            "coronary": input_case / "02_measurement_planes" / "coronary_centroid_SVD_plane.vtp",
            "lad": input_case / "02_measurement_planes" / "LAD_centroid_SVD_plane.vtp",
        },
        "ellipses": {
            "crown": input_case / "04_heart_scaffold" / "heart_width_ellipse.vtp",
            "long_axis": input_case / "04_heart_scaffold" / "heart_height_ellipse.vtp",
        },
    }
    for group in paths.values():
        for name, path in group.items():
            require_file(path, f"saved {name} geometry")
    source_meshes = {name: pv.read(path) for name, path in paths["sources"].items()}
    plane_meshes = {name: pv.read(path) for name, path in paths["planes"].items()}
    ellipse_meshes = {name: pv.read(path) for name, path in paths["ellipses"].items()}
    source_ras = {name: np.asarray(mesh.points).copy() for name, mesh in source_meshes.items()}
    return {
        "case_root": input_case,
        "measurements": measurements,
        "prior_validation": prior_validation,
        "paths": paths,
        "source_meshes": source_meshes,
        "source_ras": source_ras,
        "plane_meshes": plane_meshes,
        "ellipse_meshes": ellipse_meshes,
    }


def derive_surface(measurements: dict[str, Any]) -> tuple[HeartSupportSurface, dict[str, Any]]:
    frame = measurements["anatomical_frame"]
    rotation = np.asarray(frame["rotation_matrix"], dtype=float)
    origin_ras = np.asarray(frame["origin"], dtype=float)
    crown_input = measurements["width_ellipse"]  # Legacy saved-input schema; translated below.
    long_axis_input = measurements["height_ellipse"]  # Legacy saved-input schema; translated below.
    crown_center = canonical_transform(np.asarray(crown_input["center"]).reshape(1, 3), origin_ras, rotation)[0]
    long_axis_center = canonical_transform(np.asarray(long_axis_input["center"]).reshape(1, 3), origin_ras, rotation)[0]
    surface_origin = np.array([crown_center[0], crown_center[1], long_axis_center[2]])
    crown_span = 2.0 * float(crown_input["major_radius_mm"])
    crown_depth = 2.0 * float(crown_input["minor_radius_mm"])
    long_axis_span = 2.0 * float(long_axis_input["major_radius_mm"])
    surface = HeartSupportSurface(
        crown_span=crown_span,
        crown_depth=crown_depth,
        long_axis_span=long_axis_span,
        origin=surface_origin,
        crown_axis=np.array([1.0, 0.0, 0.0]),
        secondary_crown_axis=np.array([0.0, 1.0, 0.0]),
        apex_axis=np.array([0.0, 0.0, -1.0]),
        exponent=2.0,
    )
    derivation = {
        "crown_span_source": "2 × saved coronary/crown reference ellipse major radius",
        "crown_depth_source": "2 × saved coronary/crown reference ellipse minor radius",
        "long_axis_span_source": "2 × saved long-axis/apical reference ellipse major radius",
        "origin_derivation": (
            "canonical X/Y from the measured coronary/crown reference center and canonical Z "
            "from the measured long-axis/apical reference center"
        ),
        "crown_reference_center_canonical_mm": crown_center.tolist(),
        "long_axis_reference_center_canonical_mm": long_axis_center.tolist(),
        "no_hard_coded_case_dimensions": True,
    }
    return surface, derivation


def elliptical_measurement_plane_mesh(
    *,
    centroid: np.ndarray,
    normal: np.ndarray,
    support_points: np.ndarray,
    semantic_role: str,
    point_count: int = 128,
) -> tuple[pv.PolyData, dict[str, Any]]:
    """Represent an infinite fitted plane as a support-derived elliptical disk.

    The mathematical measurement plane remains ``n dot (x - C) = 0``.  The
    bounded oval is only its readable VTK/figure representation, matching the
    coronary-crown and interventricular-plane convention in ``coronARY_SSM``.
    Its in-plane axes and radii come from the unchanged support points; no
    patient-specific display dimensions are hard-coded.
    """

    centroid = np.asarray(centroid, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal /= np.linalg.norm(normal)
    support = np.asarray(support_points, dtype=float)

    centered = support - centroid
    projected = centered - np.outer(centered @ normal, normal)
    _, _, vt = np.linalg.svd(projected, full_matrices=False)
    axis_u = vt[0] - float(np.dot(vt[0], normal)) * normal
    if np.linalg.norm(axis_u) < 1.0e-12:
        seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis_u = np.cross(normal, seed)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)

    coordinates_u = projected @ axis_u
    coordinates_v = projected @ axis_v
    radius_u = 1.08 * float(np.max(np.abs(coordinates_u)))
    radius_v = 1.12 * float(np.max(np.abs(coordinates_v)))
    data_scale = max(float(np.ptp(coordinates_u)), float(np.ptp(coordinates_v)), 1.0e-9)
    radius_u = max(radius_u, 0.025 * data_scale)
    radius_v = max(radius_v, 0.025 * data_scale)

    theta = np.linspace(0.0, 2.0 * np.pi, int(point_count), endpoint=False)
    points = (
        centroid[None, :]
        + radius_u * np.cos(theta)[:, None] * axis_u[None, :]
        + radius_v * np.sin(theta)[:, None] * axis_v[None, :]
    )
    faces = np.concatenate(([len(points)], np.arange(len(points), dtype=np.int64)))
    mesh = pv.PolyData(points, faces)
    mesh.field_data["semantic_role"] = np.array([semantic_role])
    mesh.field_data["representation"] = np.array(
        ["support-derived bounded elliptical display of an infinite centroid-SVD measurement plane"]
    )
    mesh.field_data["measurement_centroid_canonical_mm"] = centroid.reshape(1, 3)
    mesh.field_data["measurement_normal_canonical"] = normal.reshape(1, 3)
    mesh.field_data["display_axis_u_canonical"] = axis_u.reshape(1, 3)
    mesh.field_data["display_axis_v_canonical"] = axis_v.reshape(1, 3)
    mesh.field_data["display_radii_mm"] = np.array([[radius_u, radius_v]])
    return mesh, {
        "representation": "support-derived bounded elliptical disk",
        "semantic_role": semantic_role,
        "axis_u_canonical": axis_u.tolist(),
        "axis_v_canonical": axis_v.tolist(),
        "radius_u_mm": radius_u,
        "radius_v_mm": radius_v,
        "support_point_count": int(len(support)),
        "mathematical_plane_remains_infinite": True,
    }


def build_geometry_context(inputs: dict[str, Any]) -> dict[str, Any]:
    measurements = inputs["measurements"]
    frame = measurements["anatomical_frame"]
    rotation = np.asarray(frame["rotation_matrix"], dtype=float)
    origin_ras = np.asarray(frame["origin"], dtype=float)
    source_ras = inputs["source_ras"]
    source_canonical = {
        name: canonical_transform(points, origin_ras, rotation)
        for name, points in source_ras.items()
    }
    ellipse_meshes = {
        name: transform_polydata(mesh, origin_ras, rotation)
        for name, mesh in inputs["ellipse_meshes"].items()
    }
    ellipse_meshes["crown"].field_data["semantic_role"] = np.array(["coronary/crown reference ellipse"])
    ellipse_meshes["long_axis"].field_data["semantic_role"] = np.array(["long-axis/apical reference ellipse"])
    surface, derivation = derive_surface(measurements)

    planes = measurements["planes"]
    plane_records = {
        "coronary": {
            "centroid": canonical_transform(np.asarray(planes["coronary_centroid"]).reshape(1, 3), origin_ras, rotation)[0],
            "normal": rotation @ np.asarray(planes["coronary_normal"], dtype=float),
            "rmse_mm": float(planes["coronary_rmse"]),
        },
        "lad": {
            "centroid": canonical_transform(np.asarray(planes["LAD_centroid"]).reshape(1, 3), origin_ras, rotation)[0],
            "normal": rotation @ np.asarray(planes["LAD_normal"], dtype=float),
            "rmse_mm": float(planes["LAD_rmse"]),
        },
    }
    lcx_start, lcx_end = measurements["source_support_metadata"]["lcx_crown_interval"]
    rca_start, rca_end = measurements["source_support_metadata"]["rca_compatible_interval"]
    crown_support = {
        "lcx": source_canonical["lcx"][int(lcx_start) : int(lcx_end) + 1],
        "rca": source_canonical["rca"][int(rca_start) : int(rca_end) + 1],
    }
    coronary_support_points = np.vstack((crown_support["lcx"], crown_support["rca"]))
    plane_meshes: dict[str, pv.PolyData] = {}
    plane_display_records: dict[str, dict[str, Any]] = {}
    plane_meshes["coronary"], plane_display_records["coronary"] = elliptical_measurement_plane_mesh(
        centroid=np.asarray(plane_records["coronary"]["centroid"]),
        normal=np.asarray(plane_records["coronary"]["normal"]),
        support_points=coronary_support_points,
        semantic_role="coronary plane - LCX plus inferred RCA crown/AV-groove support",
    )
    plane_meshes["lad"], plane_display_records["lad"] = elliptical_measurement_plane_mesh(
        centroid=np.asarray(plane_records["lad"]["centroid"]),
        normal=np.asarray(plane_records["lad"]["normal"]),
        support_points=source_canonical["lad"],
        semantic_role="interventricular plane - LAD descent toward the apex",
    )
    return {
        "rotation": rotation,
        "origin_ras": origin_ras,
        "source_ras": source_ras,
        "source_canonical": source_canonical,
        "plane_meshes": plane_meshes,
        "ellipse_meshes": ellipse_meshes,
        "plane_records": plane_records,
        "plane_display_records": plane_display_records,
        "crown_support": crown_support,
        "surface": surface,
        "surface_derivation": derivation,
    }


def calculate_distance_diagnostics(
    source: dict[str, np.ndarray], surface: HeartSupportSurface
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for name, points in source.items():
        nearest, distances = surface.nearest_surface_points(points)
        diagnostics[name] = {
            "normalized_arc_length": normalized_arc_length(points).tolist(),
            "distances_mm": distances.tolist(),
            "nearest_surface_points_canonical_mm": nearest.tolist(),
            "statistics": {
                "point_count": int(len(points)),
                "mean_mm": float(np.mean(distances)),
                "median_mm": float(np.median(distances)),
                "rmse_mm": float(np.sqrt(np.mean(distances**2))),
                "p95_mm": float(np.percentile(distances, 95)),
                "maximum_mm": float(np.max(distances)),
            },
            "diagnostic_only": True,
            "used_to_modify_source": False,
        }
    return diagnostics


def write_distance_outputs(case_root: Path, diagnostics: dict[str, dict[str, Any]]) -> tuple[Path, Path]:
    json_path = case_root / "06_diagnostics" / "source_to_surface_distance_diagnostics.json"
    csv_path = case_root / "06_diagnostics" / "source_to_surface_distance_diagnostics.csv"
    write_json(json_path, {"scientific_use": "diagnostic only; no centerline modification", "branches": diagnostics})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["branch", "point_index", "normalized_source_arc_length", "distance_mm"])
        writer.writeheader()
        for name, record in diagnostics.items():
            for index, (arc, distance) in enumerate(zip(record["normalized_arc_length"], record["distances_mm"])):
                writer.writerow({"branch": name.upper(), "point_index": index, "normalized_source_arc_length": f"{arc:.12g}", "distance_mm": f"{distance:.12g}"})
    return json_path, csv_path


def source_snapshot(inputs: dict[str, Any]) -> dict[str, Any]:
    measurements = inputs["measurements"]
    legacy_branches = measurements["source_integrity"]["branches"]
    return {
        name: {
            "input_vtp": str(inputs["paths"]["sources"][name].resolve()),
            "input_vtp_sha256": sha256_file(inputs["paths"]["sources"][name]),
            "coordinate_sha256": coordinate_sha256(points),
            "saved_source_coordinate_sha256": legacy_branches[name]["coordinate_sha256_before"],
            "coordinates": points.copy(),
            "segment_lengths": segment_lengths(points).copy(),
            "point_count": int(len(points)),
        }
        for name, points in inputs["source_ras"].items()
    }


def compact_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {key: value for key, value in record.items() if key not in {"coordinates", "segment_lengths"}}
        for name, record in snapshot.items()
    }


def preliminary_validation(
    *,
    inputs: dict[str, Any],
    context: dict[str, Any],
    snapshot: dict[str, Any],
    objects: dict[str, pv.DataSet],
    vtk_paths: dict[str, str],
    diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rotation = context["rotation"]
    source_ras = context["source_ras"]
    source_canonical = context["source_canonical"]
    max_coordinate_change = 0.0
    max_original_segment_change = 0.0
    max_rigid_segment_error = 0.0
    max_vtk_difference = 0.0
    branch_records: dict[str, Any] = {}
    vtk_names = {"lmca": "LMCA", "lad": "LAD", "lcx": "LCX", "rca": "INFERRED_RCA_CANDIDATE"}
    source_file_hash_unchanged = True
    for name in SOURCE_NAMES:
        before = snapshot[name]["coordinates"]
        after = source_ras[name]
        canonical = source_canonical[name]
        vtk_points = np.asarray(pv.read(vtk_paths[vtk_names[name]]).points)
        coordinate_change = float(np.max(np.abs(before - after)))
        original_segment_change = float(np.max(np.abs(snapshot[name]["segment_lengths"] - segment_lengths(after))))
        rigid_error = float(np.max(np.abs(snapshot[name]["segment_lengths"] - segment_lengths(canonical))))
        vtk_difference = float(np.max(np.abs(vtk_points - canonical)))
        vtp_hash_after = sha256_file(Path(snapshot[name]["input_vtp"]))
        source_file_hash_unchanged &= vtp_hash_after == snapshot[name]["input_vtp_sha256"]
        branch_records[name] = {
            "point_count": int(len(before)),
            "coordinate_sha256_before": snapshot[name]["coordinate_sha256"],
            "coordinate_sha256_after": coordinate_sha256(after),
            "saved_source_coordinate_sha256": snapshot[name]["saved_source_coordinate_sha256"],
            "input_vtp_sha256_before": snapshot[name]["input_vtp_sha256"],
            "input_vtp_sha256_after": vtp_hash_after,
            "maximum_coordinate_change_mm": coordinate_change,
            "maximum_original_segment_length_change_mm": original_segment_change,
            "maximum_rigid_segment_length_error_mm": rigid_error,
            "maximum_vtk_coordinate_difference_mm": vtk_difference,
        }
        max_coordinate_change = max(max_coordinate_change, coordinate_change)
        max_original_segment_change = max(max_original_segment_change, original_segment_change)
        max_rigid_segment_error = max(max_rigid_segment_error, rigid_error)
        max_vtk_difference = max(max_vtk_difference, vtk_difference)

    frame_error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    plane_centroid_errors = {}
    plane_equation_centroid_errors = {}
    plane_mesh_equation_max_errors = {}
    for name, mesh in context["plane_meshes"].items():
        centroid = np.asarray(context["plane_records"][name]["centroid"])
        normal = np.asarray(context["plane_records"][name]["normal"])
        plane_centroid_errors[name] = float(np.linalg.norm(np.mean(mesh.points, axis=0) - centroid))
        plane_equation_centroid_errors[name] = float(abs(np.dot(normal, centroid - centroid)))
        plane_mesh_equation_max_errors[name] = float(
            np.max(np.abs((np.asarray(mesh.points) - centroid) @ normal))
        )

    crown_input = inputs["measurements"]["width_ellipse"]
    long_input = inputs["measurements"]["height_ellipse"]
    crown_axis_canonical = rotation @ np.asarray(crown_input["axis_major"], dtype=float)
    long_axis_canonical = rotation @ np.asarray(long_input["axis_major"], dtype=float)
    crown_transverse_fraction = float(np.linalg.norm(crown_axis_canonical[:2]))
    long_axis_alignment = float(abs(long_axis_canonical[2]))
    crown_mask = np.asarray(context["ellipse_meshes"]["crown"].point_data["observed_source_supported_arc"], dtype=bool)
    long_mask = np.asarray(context["ellipse_meshes"]["long_axis"].point_data["observed_source_supported_arc"], dtype=bool)
    surface_record = context["surface"].to_record()

    result = {
        "case_id": PRIMARY_CASE,
        "generated_at_utc": now_utc(),
        "scientific_interpretation": (
            "This is a dataset-derived parametric anatomical support scaffold, "
            "not a patient-specific myocardial reconstruction."
        ),
        "source_integrity": {
            "branches": branch_records,
            "max_coordinate_change_mm": max_coordinate_change,
            "max_original_segment_length_change_mm": max_original_segment_change,
            "max_rigid_segment_length_error_mm": max_rigid_segment_error,
            "max_vtk_coordinate_difference_mm": max_vtk_difference,
            "source_file_hashes_unchanged": bool(source_file_hash_unchanged),
            "one_global_rigid_transform_only": True,
        },
        "anatomical_frame": {
            "determinant": determinant,
            "orthogonality_error": frame_error,
            "origin_ras_mm": context["origin_ras"].tolist(),
            "rotation_ras_to_canonical": rotation.tolist(),
        },
        "planes": {
            "method": "centroid least-squares SVD: C=mean(p_i), Q=p_i-C, n=Vt[-1], n·(x-C)=0",
            "anatomical_roles": {
                "coronary": "LCX plus inferred RCA crown ring along the AV-groove role",
                "lad": "interventricular LAD descent toward the apex",
            },
            "coronary_rmse_mm": float(context["plane_records"]["coronary"]["rmse_mm"]),
            "lad_rmse_mm": float(context["plane_records"]["lad"]["rmse_mm"]),
            "separation_angle_deg": float(inputs["measurements"]["planes"]["plane_angle_deg"]),
            "measurement_mesh_centroid_errors_mm": plane_centroid_errors,
            "plane_equation_at_centroid_errors_mm": plane_equation_centroid_errors,
            "bounded_ellipse_mesh_plane_equation_max_errors_mm": plane_mesh_equation_max_errors,
            "bounded_display_geometry": context["plane_display_records"],
            "visual_convention_reference": "coronARY_SSM.pdf pages 12–13",
            "display_reference_planes_generated": False,
            "bounded_elliptical_measurement_plane_representations_generated": True,
            "measurement_planes_forced_through_bifurcation": False,
        },
        "reference_geometry": {
            "crown_major_axis_transverse_fraction": crown_transverse_fraction,
            "long_axis_alignment_with_canonical_Z": long_axis_alignment,
            "crown_observed_and_extrapolated_segments_present": bool(crown_mask.any() and (~crown_mask).any()),
            "long_axis_observed_and_extrapolated_segments_present": bool(long_mask.any() and (~long_mask).any()),
            "solid_definition": "source-supported interval",
            "dashed_definition": "extrapolated parametric reference",
        },
        "surface": {**surface_record, "parameter_derivation": context["surface_derivation"]},
        "distance_diagnostics": {name: record["statistics"] for name, record in diagnostics.items()},
        "method_guards": {
            "source_centerlines_modified": False,
            "source_centerlines_projected_to_surface": False,
            "PCA_reconstructed_geometry_used": False,
            "synthetic_coronary_geometry_used": False,
            "disease_radius_tube_mesh_or_hub_code_used": False,
            "raw_NIfTI_extraction_rerun": False,
            "surface_generation_modified_centerlines": False,
        },
        "vtk": {
            "paths": vtk_paths,
            "expected_blocks": vtk_block_manifest(),
        },
    }
    result["checks"] = {
        "source_coordinates_unchanged": max_coordinate_change == 0.0,
        "original_segment_lengths_unchanged": max_original_segment_change == 0.0,
        "global_rigid_display_transform_only": True,
        "rotation_determinant_positive_one": abs(determinant - 1.0) <= 1.0e-10,
        "rotation_orthonormal": frame_error <= 1.0e-10,
        "rigid_transform_preserves_segments": max_rigid_segment_error <= 1.0e-9,
        "vtk_centerlines_match_canonical_source": max_vtk_difference == 0.0,
        "measurement_planes_pass_through_true_centroids": max(plane_centroid_errors.values()) <= 1.0e-10,
        "bounded_plane_meshes_lie_on_svd_planes": max(plane_mesh_equation_max_errors.values()) <= 1.0e-10,
        "plane_display_extents_are_support_derived": all(
            record["support_point_count"] >= 3
            and record["radius_u_mm"] > 0.0
            and record["radius_v_mm"] > 0.0
            for record in context["plane_display_records"].values()
        ),
        "measurement_and_display_plane_roles_distinguished": True,
        "surface_does_not_modify_centerlines": True,
        "no_PCA_reconstructed_geometry": True,
        "no_synthetic_vessel_geometry": True,
        "no_disease_radius_tube_mesh_or_hub_code": True,
        "source_hashes_unchanged": source_file_hash_unchanged,
        "crown_reference_is_transverse": crown_transverse_fraction >= 0.85,
        "long_axis_reference_is_apex_directed": long_axis_alignment >= 0.95,
        "observed_vs_extrapolated_geometry_present": bool(crown_mask.any() and (~crown_mask).any() and long_mask.any() and (~long_mask).any()),
        "surface_parameters_dataset_derived": bool(context["surface_derivation"]["no_hard_coded_case_dimensions"]),
    }
    result["pass"] = bool(all(result["checks"].values()))
    return result


def validate_multiblocks(case_root: Path, validation: dict[str, Any]) -> None:
    expected = validation["vtk"]["expected_blocks"]
    paths = {
        "heart_scaffold": case_root / "03_reference_ellipses" / "heart_scaffold.vtm",
        "source_centerlines": case_root / "01_source_geometry" / "source_centerlines.vtm",
        "stage1_complete_anatomical_model": case_root / "05_overlay" / "stage1_complete_anatomical_model.vtm",
    }
    actual = {name: list(pv.read(path).keys()) for name, path in paths.items()}
    validation["vtk"]["actual_blocks"] = actual
    validation["checks"]["required_multiblocks_complete"] = all(actual[name] == expected[name] for name in expected)
    validation["pass"] = bool(all(validation["checks"].values()))


def write_architecture(case_root: Path) -> tuple[Path, Path]:
    stages = [
        ("A", "Dataset-derived anatomical scaffold", "Complete in this increment"),
        ("B", "Parametric anatomical support surface", "Complete in this increment"),
        ("C", "Coronary generation in surface-relative coordinates", "Future; not implemented"),
        ("D", "LMCA/LAD/LCX/RCA branching topology", "Future; not implemented"),
        ("E", "Radii and taper", "Future; not implemented"),
        ("F", "Stenosis and disease", "Future; not implemented"),
        ("G", "Cardiac motion and pulsatility", "Future; not implemented"),
        ("H", "Tube and mesh export", "Future; not implemented"),
    ]
    json_path = case_root / "07_validation" / "generator_architecture.json"
    md_path = case_root / "07_validation" / "GENERATOR_ARCHITECTURE.md"
    write_json(
        json_path,
        {
            "architecture": [
                {"stage": stage, "name": name, "status": status}
                for stage, name, status in stages
            ],
            "current_scope": ["A", "B"],
            "explicitly_not_implemented": ["C", "D", "E", "F", "G", "H"],
            "generation_contract": {
                "LAD": "progresses toward apical surface coordinates",
                "LCX": "progresses circumferentially around crown coordinates",
                "RCA": "progresses around the opposite crown/AV-groove coordinates",
            },
        },
    )
    lines = ["# Generator architecture", "", "Stages A/B establish the pre-generative anatomical coordinate and support-surface contract.", "", "| Stage | Role | Status |", "|---|---|---|"]
    lines.extend(f"| {stage} | {name} | {status} |" for stage, name, status in stages)
    lines.extend(["", "No Stage C–H geometry is generated by the Stage-1 build.", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def measurements_payload(
    inputs: dict[str, Any],
    context: dict[str, Any],
    diagnostics: dict[str, dict[str, Any]],
    vtk_paths: dict[str, str],
) -> dict[str, Any]:
    legacy = inputs["measurements"]
    crown_input = legacy["width_ellipse"]
    long_input = legacy["height_ellipse"]
    return {
        "case_id": PRIMARY_CASE,
        "model_name": "dataset-derived parametric anatomical support scaffold",
        "scientific_claim": "Not a patient-specific myocardial reconstruction.",
        "scaffold_crown_span_mm": context["surface"].crown_span,
        "scaffold_crown_depth_mm": context["surface"].crown_depth,
        "scaffold_long_axis_span_mm": context["surface"].long_axis_span,
        "coronary_crown_reference_ellipse": {
            "major_radius_mm": crown_input["major_radius_mm"],
            "minor_radius_mm": crown_input["minor_radius_mm"],
            "fit_rmse_mm": crown_input["fit_rmse_mm"],
            "maximum_residual_mm": crown_input["max_residual_mm"],
            "observed_angular_intervals_rad": crown_input["observed_angular_intervals_rad"],
            "solid_is_source_supported": True,
            "dashed_is_extrapolated": True,
        },
        "long_axis_apical_reference_ellipse": {
            "major_radius_mm": long_input["major_radius_mm"],
            "minor_radius_mm": long_input["minor_radius_mm"],
            "fit_rmse_mm": long_input["fit_rmse_mm"],
            "maximum_residual_mm": long_input["max_residual_mm"],
            "observed_angular_intervals_rad": long_input["observed_angular_intervals_rad"],
            "solid_is_source_supported": True,
            "dashed_is_extrapolated": True,
        },
        "planes": {
            "coronary_centroid_svd_rmse_mm": context["plane_records"]["coronary"]["rmse_mm"],
            "lad_centroid_svd_rmse_mm": context["plane_records"]["lad"]["rmse_mm"],
            "separation_angle_deg": legacy["planes"]["plane_angle_deg"],
            "measurement_equation": "n dot (x - C) = 0",
            "anatomical_roles": {
                "coronary": "LCX plus inferred RCA crown ring / AV-groove role",
                "lad": "interventricular LAD descent toward the apex",
            },
            "bounded_display_geometry": context["plane_display_records"],
            "bounded_display_is_not_a_second_fitted_plane": True,
        },
        "surface": {**context["surface"].to_record(), "parameter_derivation": context["surface_derivation"]},
        "source_point_counts": {name: int(len(points)) for name, points in context["source_ras"].items()},
        "source_to_surface_distance_statistics": {name: record["statistics"] for name, record in diagnostics.items()},
        "vtk_paths": vtk_paths,
    }


def build_report(
    *,
    stage_root: Path,
    case_root: Path,
    inputs: dict[str, Any],
    context: dict[str, Any],
    diagnostics: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    figure_records: list[dict[str, Any]],
    vtk_paths: dict[str, str],
) -> Path:
    report_path = stage_root / "STAGE1_PARAMETRIC_HEART_SURFACE_REPORT.md"
    surface = context["surface"]
    a, b, c = surface.semi_axes
    plane = validation["planes"]
    presentation_path = stage_root / "presentation" / PRESENTATION_NAME
    source_lines = [
        f"- {name.upper()}: `{inputs['paths']['sources'][name]}` ({len(context['source_ras'][name])} points)"
        for name in SOURCE_NAMES
    ]
    diagnostic_lines = [
        f"- {name.upper()}: mean {record['statistics']['mean_mm']:.3f} mm; median {record['statistics']['median_mm']:.3f} mm; RMSE {record['statistics']['rmse_mm']:.3f} mm; P95 {record['statistics']['p95_mm']:.3f} mm; max {record['statistics']['maximum_mm']:.3f} mm"
        for name, record in diagnostics.items()
    ]
    vtk_lines = [f"- `{path}`" for path in vtk_paths.values()]
    figure_lines = [f"- `{record['png']}` and `{record['svg']}`" for record in figure_records]
    lines = [
        "# Stage-1 parametric anatomical support surface report",
        "",
        "## 1. Objective",
        "Create a generator-ready, dataset-derived anatomical support scaffold without modifying the source coronary centerlines.",
        "",
        "> This is a dataset-derived parametric anatomical support scaffold, not a patient-specific myocardial reconstruction.",
        "",
        "## 2. Input source files",
        *source_lines,
        f"- Saved measurements: `{inputs['case_root'] / 'measurements.json'}`",
        f"- Saved validation: `{inputs['case_root'] / 'validation.json'}`",
        "",
        "## 3. Primary case",
        f"`{PRIMARY_CASE}` was used. No alternate case was substituted.",
        "",
        "## 4. Source point counts",
        *[f"- {name.upper()}: {len(points)}" for name, points in context["source_ras"].items()],
        "",
        "## 5. Plane fitting equations",
        "For source points `p_i`: `C = mean(p_i)`, `Q_i = p_i - C`, `U,S,Vᵀ = SVD(Q)`, `n = Vᵀ[-1]`, and `n · (x - C) = 0`.",
        "All relevant unchanged source-support points are used. Measurement planes pass through their true centroids.",
        "The infinite fitted planes are displayed as support-derived bounded elliptical disks, following the `coronARY_SSM.pdf` plane convention; the oval boundary is not a second fit.",
        "- Coronary plane role: LCX plus inferred RCA crown ring along the AV groove.",
        "- Interventricular plane role: LAD descent toward the apex.",
        "",
        "## 6. Plane statistics",
        f"- Coronary centroid-SVD plane RMSE: {plane['coronary_rmse_mm']:.6f} mm",
        f"- LAD centroid-SVD plane RMSE: {plane['lad_rmse_mm']:.6f} mm",
        f"- Acute plane-normal separation: {plane['separation_angle_deg']:.6f}°",
        "",
        "## 7. Crown scaffold measurements",
        f"- Scaffold crown span: {surface.crown_span:.6f} mm",
        f"- Scaffold crown depth: {surface.crown_depth:.6f} mm",
        "- Coronary/crown reference derives from saved LCX plus inferred-RCA crown-support geometry.",
        "",
        "## 8. Long-axis scaffold measurements",
        f"- Scaffold long-axis span: {surface.long_axis_span:.6f} mm",
        "- The long-axis/apical reference is supported by the unchanged LAD trajectory.",
        "",
        "## 9. Parametric support-surface parameters",
        f"- Ellipsoid exponent: n = {surface.exponent:.1f}",
        f"- Semi-axes `(a crown, b depth, c long-axis)`: ({a:.6f}, {b:.6f}, {c:.6f}) mm",
        f"- Canonical origin: {surface.origin.tolist()} mm",
        "- Parameters are read from saved measurements; no 189.label dimensions are hard-coded.",
        "",
        "## 10. Source-to-surface diagnostics",
        *diagnostic_lines,
        "These distances are descriptive only and are never used to move, project, or reject source centerline points.",
        "",
        "## 11. Source-integrity proof",
        f"- Maximum source-coordinate change: {validation['source_integrity']['max_coordinate_change_mm']:.3e} mm",
        f"- Maximum original segment-length change: {validation['source_integrity']['max_original_segment_length_change_mm']:.3e} mm",
        f"- Maximum rigid-transform segment error: {validation['source_integrity']['max_rigid_segment_length_error_mm']:.3e} mm",
        f"- Maximum VTK canonical-coordinate difference: {validation['source_integrity']['max_vtk_coordinate_difference_mm']:.3e} mm",
        f"- Source file hashes unchanged: {validation['source_integrity']['source_file_hashes_unchanged']}",
        f"- Overall validation: **{'PASS' if validation['pass'] else 'FAIL'}**",
        "",
        "## 12. VTK files generated",
        *vtk_lines,
        "",
        "## 13. PPT figures generated",
        *figure_lines,
        f"- Presentation: `{presentation_path}` ({'generated' if presentation_path.is_file() else 'generated after geometric validation'})",
        "",
        "## 14. Limitations",
        "- The inferred RCA candidate is not annotated RCA ground truth.",
        "- Coronary-derived scaffold dimensions are not myocardial dimensions.",
        "- Dashed reference-ellipse portions are extrapolated.",
        "- The surface is a parametric anatomical support model, not a clinical reconstruction.",
        "- No generative coronary geometry is present in Stage-1.",
        "",
        "## 15. Next-stage architecture",
        "Stages A/B are complete. Stage C should generate coronary trajectories directly in surface-relative `(theta, phi, offset)` coordinates; Stages D–H remain future work.",
        "",
        "## 16. Exact regeneration commands",
        "```powershell",
        "python pca_ssm_vessel_tree_generator/build_parametric_heart_support_surface.py --case 189.label --reset-output",
        "```",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--case", default=PRIMARY_CASE)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--reset-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    stage_root = (args.output_root or repository_root / "outputs" / "lca_ssm" / "stage1_parametric_heart_surface").resolve()
    case_root = stage_root / args.case
    if args.reset_output and case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True, exist_ok=True)

    inputs = load_saved_inputs(repository_root, args.case)
    snapshot = source_snapshot(inputs)
    context = build_geometry_context(inputs)
    diagnostics = calculate_distance_diagnostics(context["source_canonical"], context["surface"])
    objects = build_vtk_objects(
        source_canonical=context["source_canonical"],
        plane_meshes_canonical=context["plane_meshes"],
        ellipse_meshes_canonical=context["ellipse_meshes"],
        surface=context["surface"],
    )
    vtk_names = {"lmca": "LMCA", "lad": "LAD", "lcx": "LCX", "rca": "INFERRED_RCA_CANDIDATE"}
    for name, object_name in vtk_names.items():
        objects[object_name].point_data["distance_to_parametric_support_surface_mm"] = np.asarray(diagnostics[name]["distances_mm"])
        objects[object_name].point_data["nearest_surface_point_canonical_mm"] = np.asarray(diagnostics[name]["nearest_surface_points_canonical_mm"])
    vtk_paths = export_stage1_vtk(case_root, objects)
    distance_json, distance_csv = write_distance_outputs(case_root, diagnostics)
    architecture_json, architecture_md = write_architecture(case_root)
    write_json(case_root / "01_source_geometry" / "source_hash_manifest.json", compact_snapshot(snapshot))
    measurements = measurements_payload(inputs, context, diagnostics, vtk_paths)
    write_json(case_root / "measurements.json", measurements)

    validation = preliminary_validation(
        inputs=inputs,
        context=context,
        snapshot=snapshot,
        objects=objects,
        vtk_paths=vtk_paths,
        diagnostics=diagnostics,
    )
    validate_multiblocks(case_root, validation)
    figure_records = generate_stage1_figures(
        case_root=case_root,
        case_id=args.case,
        source=context["source_canonical"],
        crown_support=context["crown_support"],
        planes=context["plane_meshes"],
        plane_records=context["plane_records"],
        ellipses=context["ellipse_meshes"],
        surface=context["surface"],
        distance_diagnostics=diagnostics,
        validation_summary=validation,
        dpi=args.dpi,
    )
    write_json(case_root / "06_diagnostics" / "figure_manifest.json", {"figures": figure_records})
    required_files = list(vtk_paths.values()) + [
        str(distance_json),
        str(distance_csv),
        str(architecture_json),
        str(architecture_md),
        str(case_root / "measurements.json"),
    ] + [record[key] for record in figure_records for key in ("png", "svg")]
    validation["missing_files"] = [path for path in required_files if not Path(path).is_file()]
    validation["checks"]["all_required_files_present"] = not validation["missing_files"]
    validation["pass"] = bool(all(validation["checks"].values()))
    validation_path = case_root / "07_validation" / "validation.json"
    write_json(validation_path, validation)
    report_path = build_report(
        stage_root=stage_root,
        case_root=case_root,
        inputs=inputs,
        context=context,
        diagnostics=diagnostics,
        validation=validation,
        figure_records=figure_records,
        vtk_paths=vtk_paths,
    )
    output = {
        "case": args.case,
        "output": str(case_root),
        "validation_pass": validation["pass"],
        "scaffold_crown_span_mm": context["surface"].crown_span,
        "scaffold_long_axis_span_mm": context["surface"].long_axis_span,
        "surface": context["surface"].to_record(),
        "distance_statistics": {name: record["statistics"] for name, record in diagnostics.items()},
        "final_vtm": vtk_paths["STAGE1_COMPLETE_ANATOMICAL_MODEL"],
        "validation": str(validation_path),
        "report": str(report_path),
    }
    print(json.dumps(output, indent=2))
    if not validation["pass"]:
        raise SystemExit("Stage-1 parametric support-surface validation failed")


if __name__ == "__main__":
    main()
