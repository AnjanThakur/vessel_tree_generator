"""Export a dataset-derived two-ellipse Stage-1 heart scaffold to VTK.

The heart-width ellipse and heart-height ellipse are reference envelopes, not
coronary centerline replacements and not myocardial-surface reconstructions.
All VTK geometry is written in the original RAS coordinate system.  The one
anatomical rigid frame is used only for consistent diagnostic views.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from build_stage1_final_anatomical_model import (
    EPS,
    Stage1Config,
    analyze_case,
    array_sha256,
    circular_support_mask,
    classify_crown,
    compact_plane,
    curve_length,
    jsonable,
    local_geometry,
    normalize,
    select_crown_interval,
    sha256,
    sparse_full_ellipse,
    write_json,
)
from create_canonical_heart_evidence import load_case, segment_lengths
from lca_ssm_planes import fit_plane_svd
from validate_lcx_crown_trajectory import ValidationConfig


COLORS = {
    "lmca": "#222222",
    "lad": "#D62828",
    "lcx": "#1976B9",
    "rca": "#77589A",
    "width": "#00A6C7",
    "height": "#E87500",
    "coronary_plane": "#BFE3F2",
    "lad_plane": "#F4D4AA",
    "bifurcation": "#2CA02C",
    "crown_axis": "#0072B2",
    "apex_axis": "#D55E00",
}

FIGURE_STEMS = [
    "01_source_centerlines_only",
    "02_measurement_planes_and_source_support",
    "03_heart_scaffold_only",
    "04_source_plus_heart_scaffold_overlay",
    "05_front_orthographic_view",
    "06_superior_orthographic_view",
    "07_lateral_long_axis_view",
    "08_technical_measurement_summary",
]


@dataclass
class HeartScaffold:
    case_id: str
    origin: np.ndarray
    rotation_ras_to_anatomical: np.ndarray
    x_crown: np.ndarray
    y_axis: np.ndarray
    z_superior: np.ndarray
    coronary_plane: dict[str, Any]
    lad_plane: dict[str, Any]
    heart_width_ellipse: dict[str, Any]
    heart_height_ellipse: dict[str, Any]
    source_ras: dict[str, np.ndarray]
    crown_support_ras: dict[str, np.ndarray]
    source_support_metadata: dict[str, Any]

    @property
    def heart_width_mm(self) -> float:
        return 2.0 * float(self.heart_width_ellipse["major_radius_mm"])

    @property
    def heart_height_mm(self) -> float:
        return 2.0 * float(self.heart_height_ellipse["major_radius_mm"])

    def canonical(self, points: np.ndarray) -> np.ndarray:
        return (self.rotation_ras_to_anatomical @ (np.asarray(points) - self.origin).T).T


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_svd_plane(plane: dict[str, Any], source: str) -> dict[str, Any]:
    return compact_plane(plane, source)


def ellipse_parameter_angles(points_2d: np.ndarray, ellipse: dict[str, Any]) -> np.ndarray:
    theta = float(ellipse["orientation_rad"])
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    local = (points_2d - np.asarray(ellipse["center_2d_mm"])) @ rotation
    return np.arctan2(
        local[:, 1] / float(ellipse["semi_minor_axis_mm"]),
        local[:, 0] / float(ellipse["semi_major_axis_mm"]),
    )


def project_to_plane_2d(points: np.ndarray, plane: dict[str, Any]) -> np.ndarray:
    relative = np.asarray(points) - np.asarray(plane["centroid"])
    return np.column_stack((relative @ np.asarray(plane["basis_u"]), relative @ np.asarray(plane["basis_v"])))


def fit_heart_width_ellipse(
    rca_support: np.ndarray,
    lcx_support: np.ndarray,
    coronary_plane: dict[str, Any],
    *,
    rca_indices: tuple[int, int],
    lcx_indices: tuple[int, int],
    rca_full_length_mm: float,
    lcx_full_length_mm: float,
) -> dict[str, Any]:
    """Fit the transverse/crown heart-width envelope from coronary evidence."""
    source_points = np.vstack((rca_support, lcx_support))
    points_2d = project_to_plane_2d(source_points, coronary_plane)
    fit = sparse_full_ellipse(points_2d, "dataset-derived heart width / crown ellipse")
    theta = float(fit["orientation_rad"])
    major_axis = normalize(
        math.cos(theta) * np.asarray(coronary_plane["basis_u"])
        + math.sin(theta) * np.asarray(coronary_plane["basis_v"])
    )
    minor_axis = normalize(np.cross(np.asarray(coronary_plane["normal"]), major_axis))
    center = (
        np.asarray(coronary_plane["centroid"])
        + fit["center_2d_mm"][0] * np.asarray(coronary_plane["basis_u"])
        + fit["center_2d_mm"][1] * np.asarray(coronary_plane["basis_v"])
    )
    outline_ras = (
        np.asarray(coronary_plane["centroid"])
        + fit["outline_2d_mm"][:, 0, None] * np.asarray(coronary_plane["basis_u"])
        + fit["outline_2d_mm"][:, 1, None] * np.asarray(coronary_plane["basis_v"])
    )
    rca_2d = project_to_plane_2d(rca_support, coronary_plane)
    lcx_2d = project_to_plane_2d(lcx_support, coronary_plane)
    rca_angles = ellipse_parameter_angles(rca_2d, fit)
    lcx_angles = ellipse_parameter_angles(lcx_2d, fit)
    observed_mask, observed_intervals = circular_support_mask(
        np.asarray(fit["outline_angles_rad"]), [rca_angles, lcx_angles]
    )
    source_residuals_2d, _ = cKDTree(fit["outline_2d_mm"]).query(points_2d)
    plane_residuals = np.abs((source_points - np.asarray(coronary_plane["centroid"])) @ np.asarray(coronary_plane["normal"]))
    residuals_3d = np.sqrt(source_residuals_2d**2 + plane_residuals**2)
    selected_length = curve_length(rca_support) + curve_length(lcx_support)
    full_length = rca_full_length_mm + lcx_full_length_mm
    return {
        "semantic_name": "heart_width_ellipse",
        "interpretation": "transverse/circumferential heart-width crown reference",
        "center_ras_mm": center,
        "plane_normal_ras": np.asarray(coronary_plane["normal"]),
        "major_radius_mm": float(fit["semi_major_axis_mm"]),
        "minor_radius_mm": float(fit["semi_minor_axis_mm"]),
        "major_axis_ras": major_axis,
        "minor_axis_ras": minor_axis,
        "outline_ras_mm": outline_ras,
        "outline_parameter_rad": np.asarray(fit["outline_angles_rad"]),
        "observed_support_mask": observed_mask,
        "observed_angular_intervals_rad": observed_intervals,
        "source_point_count": int(len(source_points)),
        "source_arc_fraction": float(selected_length / max(full_length, EPS)),
        "source_intervals": {
            "inferred_rca_candidate": list(rca_indices),
            "resolved_lcx": list(lcx_indices),
        },
        "fit_method": "robust nonlinear full ellipse using every exact point in two continuous coronary-support intervals",
        "fit_rmse_mm": float(np.sqrt(np.mean(source_residuals_2d**2))),
        "max_residual_mm": float(np.max(source_residuals_2d)),
        "source_3d_descriptive_rmse_mm": float(np.sqrt(np.mean(residuals_3d**2))),
        "source_3d_descriptive_max_residual_mm": float(np.max(residuals_3d)),
        "inferred_rca_is_annotated_ground_truth": False,
        "ellipse_is_reference_only": True,
    }


def fit_heart_height_ellipse(
    lad: np.ndarray,
    bifurcation: np.ndarray,
    lad_plane: dict[str, Any],
) -> dict[str, Any]:
    """Fit a long-axis heart-height envelope with measured apex orientation."""
    normal = np.asarray(lad_plane["normal"])
    terminal_displacement = np.asarray(lad[-1]) - np.asarray(bifurcation)
    apex_direction = terminal_displacement - np.dot(terminal_displacement, normal) * normal
    apex_direction = normalize(apex_direction)
    z_superior = -apex_direction
    transverse = normalize(np.cross(normal, z_superior))
    centroid = np.asarray(lad_plane["centroid"])
    projected = np.asarray(lad) - np.outer((np.asarray(lad) - centroid) @ normal, normal)
    relative = projected - centroid
    x_long = relative @ z_superior
    y_transverse = relative @ transverse
    points_2d = np.column_stack((x_long, y_transverse))
    long_range = max(float(np.ptp(x_long)), 1.0)
    transverse_range = max(float(np.ptp(y_transverse)), 1.0)
    initial = np.array([
        0.5 * (float(np.min(x_long)) + float(np.max(x_long))),
        float(np.median(y_transverse)),
        math.log(max(0.55 * long_range, 1.0)),
        math.log(max(0.60 * transverse_range, 1.0)),
    ])
    lower = np.array([
        float(np.min(x_long) - 0.75 * long_range),
        float(np.min(y_transverse) - transverse_range),
        math.log(max(0.50 * long_range, 0.5)),
        math.log(max(0.25 * transverse_range, 0.5)),
    ])
    upper = np.array([
        float(np.max(x_long) + 0.75 * long_range),
        float(np.max(y_transverse) + transverse_range),
        math.log(1.5 * long_range),
        math.log(max(transverse_range, 1.0)),
    ])

    def residual(parameters: np.ndarray) -> np.ndarray:
        cx, cy, log_a, log_b = parameters
        a, b = np.exp([log_a, log_b])
        radial = np.sqrt(((x_long - cx) / a) ** 2 + ((y_transverse - cy) / b) ** 2)
        return (radial - 1.0) * math.sqrt(float(a * b))

    optimized = least_squares(
        residual,
        np.clip(initial, lower + 1.0e-8, upper - 1.0e-8),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=max(0.5, 0.02 * long_range),
        max_nfev=6000,
    )
    cx, cy, log_a, log_b = optimized.x
    a, b = float(math.exp(log_a)), float(math.exp(log_b))
    if a <= b:
        raise ValueError(
            "anatomical-frame or ellipse-parameter inconsistency: measured LAD spans do not yield a long-axis-dominant height ellipse"
        )
    center = centroid + cx * z_superior + cy * transverse
    angles = np.linspace(0.0, 2.0 * math.pi, 1440, endpoint=False)
    outline = center + (a * np.cos(angles))[:, None] * z_superior + (b * np.sin(angles))[:, None] * transverse
    projected_points = center + (x_long - cx)[:, None] * z_superior + (y_transverse - cy)[:, None] * transverse
    residuals_2d, _ = cKDTree(outline).query(projected_points)
    plane_residuals = np.abs((np.asarray(lad) - centroid) @ normal)
    residuals_3d = np.sqrt(residuals_2d**2 + plane_residuals**2)
    point_angles = np.unwrap(np.arctan2((y_transverse - cy) / b, (x_long - cx) / a))
    observed_mask, observed_intervals = circular_support_mask(angles, [point_angles])
    return {
        "semantic_name": "heart_height_ellipse",
        "interpretation": "superior-to-apex long-axis heart-height reference",
        "center_ras_mm": center,
        "plane_normal_ras": normal,
        "major_radius_mm": a,
        "minor_radius_mm": b,
        "major_axis_ras": z_superior,
        "minor_axis_ras": transverse,
        "outline_ras_mm": outline,
        "outline_parameter_rad": angles,
        "observed_support_mask": observed_mask,
        "observed_angular_intervals_rad": observed_intervals,
        "source_point_count": int(len(lad)),
        "source_arc_fraction": 1.0,
        "source_intervals": {"resolved_lad": [0, int(len(lad) - 1)]},
        "fit_method": "robust axis-constrained ellipse using all unchanged LAD points; major axis fixed to measured superior-apex direction, long radius bounded from measured longitudinal span, and minor radius bounded from measured transverse LAD span",
        "measured_longitudinal_source_span_mm": long_range,
        "measured_transverse_source_span_mm": transverse_range,
        "fit_rmse_mm": float(np.sqrt(np.mean(residuals_2d**2))),
        "max_residual_mm": float(np.max(residuals_2d)),
        "source_3d_descriptive_rmse_mm": float(np.sqrt(np.mean(residuals_3d**2))),
        "source_3d_descriptive_max_residual_mm": float(np.max(residuals_3d)),
        "optimizer_success": bool(optimized.success),
        "optimizer_message": str(optimized.message),
        "ellipse_is_reference_only": True,
    }


def build_anatomical_frame(
    width_ellipse: dict[str, Any],
    height_ellipse: dict[str, Any],
    lcx_support: np.ndarray,
    lad: np.ndarray,
    bifurcation: np.ndarray,
) -> dict[str, Any]:
    z_axis = normalize(np.asarray(height_ellipse["major_axis_ras"]))
    if float(np.dot(lad[-1] - bifurcation, z_axis)) > 0.0:
        z_axis = -z_axis
    width_axis = normalize(np.asarray(width_ellipse["major_axis_ras"]))
    x_axis = normalize(width_axis - float(np.dot(width_axis, z_axis)) * z_axis)
    crown_displacement = np.mean(lcx_support, axis=0) - bifurcation
    if float(np.dot(crown_displacement, x_axis)) < 0.0:
        x_axis = -x_axis
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    rotation = np.vstack((x_axis, y_axis, z_axis))
    return {
        "origin_ras_mm": np.asarray(bifurcation),
        "X_crown_ras": x_axis,
        "Y_ras": y_axis,
        "Z_superior_ras": z_axis,
        "apex_direction_ras": -z_axis,
        "rotation_ras_to_anatomical": rotation,
        "determinant": float(np.linalg.det(rotation)),
        "orthogonality_max_error": float(np.max(np.abs(rotation @ rotation.T - np.eye(3)))),
        "distal_lad_anatomical_z_mm": float(np.dot(lad[-1] - bifurcation, z_axis)),
        "width_major_axis_alignment_with_X": float(abs(np.dot(width_axis, x_axis))),
        "width_major_axis_alignment_with_Z": float(abs(np.dot(width_axis, z_axis))),
        "height_major_axis_alignment_with_Z": float(abs(np.dot(height_ellipse["major_axis_ras"], z_axis))),
        "width_height_axis_angle_deg": float(np.degrees(np.arccos(np.clip(abs(np.dot(width_axis, height_ellipse["major_axis_ras"])), 0.0, 1.0)))),
    }


def build_heart_scaffold(output_root: Path, case_id: str) -> tuple[HeartScaffold, dict[str, Any]]:
    config = Stage1Config()
    analysis = analyze_case(output_root, case_id, config)
    analysis["lcx_crown_validation"] = classify_crown(analysis, config)
    if analysis["resolution_status"] not in {"resolved_existing_assignment", "resolved_swapped_assignment"}:
        raise ValueError(f"{case_id}: not in resolved Stage-1 cohort ({analysis['resolution_status']})")
    if analysis["lcx_crown_validation"]["status"] != "crown_supported":
        raise ValueError(f"{case_id}: LCX crown evidence is {analysis['lcx_crown_validation']['status']}")
    neutral = analysis["_neutral"]
    lad_source = analysis["new_resolved_assignment"]["lad_source"]
    lcx_source = analysis["new_resolved_assignment"]["lcx_source"]
    lad, lcx = neutral[lad_source], neutral[lcx_source]
    raw = analysis["_case"]["raw"]
    rca = raw["rca"]
    bifurcation = np.asarray(raw["lmca"][-1])
    lcx_interval = analysis["lcx_crown_validation"]["interval"]
    lcx_start, lcx_end = int(lcx_interval["start_index"]), int(lcx_interval["end_index"])
    lcx_support = lcx[lcx_start:lcx_end + 1]
    coronary_plane = fit_plane_svd(
        np.vstack((rca, lcx_support)),
        name=f"{case_id} final coronary centroid-SVD plane",
    )
    lad_plane = fit_plane_svd(lad, name=f"{case_id} LAD centroid-SVD plane")
    rca_local = local_geometry(rca, analysis["_rca_plane"], config.tangent_window)
    rca_interval = select_crown_interval(
        rca_local,
        ValidationConfig(
            tangent_window=config.tangent_window,
            candidate_start_max_fraction=config.crown_max_start_fraction,
            candidate_min_arc_fraction=config.crown_min_arc_fraction,
        ),
    )
    rca_start, rca_end = int(rca_interval["start_index"]), int(rca_interval["end_index"])
    rca_support = rca[rca_start:rca_end + 1]
    width = fit_heart_width_ellipse(
        rca_support,
        lcx_support,
        coronary_plane,
        rca_indices=(rca_start, rca_end),
        lcx_indices=(lcx_start, lcx_end),
        rca_full_length_mm=curve_length(rca),
        lcx_full_length_mm=curve_length(lcx),
    )
    height = fit_heart_height_ellipse(lad, bifurcation, lad_plane)
    frame = build_anatomical_frame(width, height, lcx_support, lad, bifurcation)
    source = {"lmca": raw["lmca"], "lad": lad, "lcx": lcx, "rca": rca}
    scaffold = HeartScaffold(
        case_id=case_id,
        origin=bifurcation,
        rotation_ras_to_anatomical=np.asarray(frame["rotation_ras_to_anatomical"]),
        x_crown=np.asarray(frame["X_crown_ras"]),
        y_axis=np.asarray(frame["Y_ras"]),
        z_superior=np.asarray(frame["Z_superior_ras"]),
        coronary_plane=coronary_plane,
        lad_plane=lad_plane,
        heart_width_ellipse=width,
        heart_height_ellipse=height,
        source_ras=source,
        crown_support_ras={"rca": rca_support, "lcx": lcx_support},
        source_support_metadata={
            "resolved_assignment": analysis["new_resolved_assignment"],
            "assignment_confidence": analysis["assignment_scoring"]["confidence"],
            "lcx_crown_validation": analysis["lcx_crown_validation"]["status"],
            "lcx_crown_interval": [lcx_start, lcx_end],
            "rca_compatible_interval": [rca_start, rca_end],
            "inferred_rca_limitation": "Inferred disconnected component; not annotated RCA ground truth.",
        },
    )
    return scaffold, {"analysis": analysis, "frame": frame}


def polyline(points: np.ndarray, *, closed: bool = False) -> pv.PolyData:
    points = np.asarray(points, dtype=float)
    mesh = pv.PolyData(points)
    indexes = np.arange(len(points) + (1 if closed else 0), dtype=np.int64)
    if closed:
        indexes[-1] = 0
    mesh.lines = np.concatenate(([len(indexes)], indexes))
    return mesh


def bifurcation_point(point: np.ndarray) -> pv.PolyData:
    mesh = pv.PolyData(np.asarray(point, dtype=float).reshape(1, 3))
    mesh.verts = np.array([1, 0], dtype=np.int64)
    return mesh


def rectangular_plane(point: np.ndarray, plane: dict[str, Any], width: float, height: float) -> pv.PolyData:
    u, v = np.asarray(plane["basis_u"]), np.asarray(plane["basis_v"])
    p = np.asarray(point)
    corners = np.array([
        p - 0.5 * width * u - 0.5 * height * v,
        p + 0.5 * width * u - 0.5 * height * v,
        p + 0.5 * width * u + 0.5 * height * v,
        p - 0.5 * width * u + 0.5 * height * v,
    ])
    return pv.PolyData(corners, faces=np.array([4, 0, 1, 2, 3], dtype=np.int64))


def axis_line(start: np.ndarray, end: np.ndarray) -> pv.PolyData:
    return polyline(np.vstack((start, end)))


def save_polydata(path: Path, mesh: pv.PolyData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(path, binary=True)


def save_multiblock(path: Path, blocks: dict[str, pv.DataSet]) -> None:
    dataset = pv.MultiBlock()
    for name, block in blocks.items():
        dataset[name] = block
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(path, binary=True)


def vtk_objects(scaffold: HeartScaffold) -> dict[str, pv.DataSet]:
    extent = max(scaffold.heart_width_mm, scaffold.heart_height_mm)
    cor_width = 1.15 * scaffold.heart_width_mm
    cor_height = max(0.65 * scaffold.heart_width_mm, 20.0)
    lad_width = max(0.75 * scaffold.heart_width_mm, 20.0)
    lad_height = 1.10 * scaffold.heart_height_mm
    width_ellipse = polyline(scaffold.heart_width_ellipse["outline_ras_mm"], closed=True)
    width_ellipse.point_data["observed_source_supported_arc"] = scaffold.heart_width_ellipse["observed_support_mask"].astype(np.uint8)
    height_ellipse = polyline(scaffold.heart_height_ellipse["outline_ras_mm"], closed=True)
    height_ellipse.point_data["observed_source_supported_arc"] = scaffold.heart_height_ellipse["observed_support_mask"].astype(np.uint8)
    return {
        "SOURCE_LMCA": polyline(scaffold.source_ras["lmca"]),
        "SOURCE_LAD": polyline(scaffold.source_ras["lad"]),
        "SOURCE_LCX": polyline(scaffold.source_ras["lcx"]),
        "INFERRED_RCA_CANDIDATE": polyline(scaffold.source_ras["rca"]),
        "BIFURCATION": bifurcation_point(scaffold.origin),
        "CORONARY_MEASUREMENT_PLANE": rectangular_plane(scaffold.coronary_plane["centroid"], scaffold.coronary_plane, cor_width, cor_height),
        "LAD_MEASUREMENT_PLANE": rectangular_plane(scaffold.lad_plane["centroid"], scaffold.lad_plane, lad_width, lad_height),
        "CORONARY_DISPLAY_PLANE": rectangular_plane(scaffold.origin, scaffold.coronary_plane, cor_width, cor_height),
        "LAD_DISPLAY_PLANE": rectangular_plane(scaffold.origin, scaffold.lad_plane, lad_width, lad_height),
        "HEART_WIDTH_ELLIPSE": width_ellipse,
        "HEART_HEIGHT_ELLIPSE": height_ellipse,
        "CROWN_AXIS": axis_line(scaffold.origin - 0.55 * scaffold.heart_width_mm * scaffold.x_crown, scaffold.origin + 0.55 * scaffold.heart_width_mm * scaffold.x_crown),
        "APEX_AXIS": axis_line(scaffold.origin, scaffold.origin - 1.05 * scaffold.heart_height_mm * scaffold.z_superior),
    }


def export_vtk(root: Path, scaffold: HeartScaffold) -> dict[str, str]:
    objects = vtk_objects(scaffold)
    mapping = {
        "01_source_geometry/LMCA_centerline.vtp": "SOURCE_LMCA",
        "01_source_geometry/LAD_centerline.vtp": "SOURCE_LAD",
        "01_source_geometry/LCX_centerline.vtp": "SOURCE_LCX",
        "01_source_geometry/RCA_candidate_centerline.vtp": "INFERRED_RCA_CANDIDATE",
        "01_source_geometry/bifurcation.vtp": "BIFURCATION",
        "02_measurement_planes/coronary_centroid_SVD_plane.vtp": "CORONARY_MEASUREMENT_PLANE",
        "02_measurement_planes/LAD_centroid_SVD_plane.vtp": "LAD_MEASUREMENT_PLANE",
        "03_display_planes/coronary_plane_at_bifurcation.vtp": "CORONARY_DISPLAY_PLANE",
        "03_display_planes/LAD_plane_at_bifurcation.vtp": "LAD_DISPLAY_PLANE",
        "04_heart_scaffold/heart_width_ellipse.vtp": "HEART_WIDTH_ELLIPSE",
        "04_heart_scaffold/heart_height_ellipse.vtp": "HEART_HEIGHT_ELLIPSE",
        "04_heart_scaffold/crown_axis.vtp": "CROWN_AXIS",
        "04_heart_scaffold/apex_axis.vtp": "APEX_AXIS",
    }
    for relative, block_name in mapping.items():
        save_polydata(root / relative, objects[block_name])
    scaffold_blocks = {
        name: objects[name]
        for name in (
            "BIFURCATION", "CORONARY_DISPLAY_PLANE", "LAD_DISPLAY_PLANE",
            "HEART_WIDTH_ELLIPSE", "HEART_HEIGHT_ELLIPSE", "CROWN_AXIS", "APEX_AXIS",
        )
    }
    overlay_blocks = {name: objects[name] for name in (
        "SOURCE_LMCA", "SOURCE_LAD", "SOURCE_LCX", "INFERRED_RCA_CANDIDATE", "BIFURCATION",
        "CORONARY_MEASUREMENT_PLANE", "LAD_MEASUREMENT_PLANE",
        "HEART_WIDTH_ELLIPSE", "HEART_HEIGHT_ELLIPSE", "CROWN_AXIS", "APEX_AXIS",
    )}
    save_multiblock(root / "04_heart_scaffold/heart_scaffold.vtm", scaffold_blocks)
    save_multiblock(root / "05_overlay/source_plus_heart_scaffold.vtm", overlay_blocks)
    paths = {name: str((root / relative).resolve()) for relative, name in mapping.items()}
    paths["HEART_SCAFFOLD_MULTIBLOCK"] = str((root / "04_heart_scaffold/heart_scaffold.vtm").resolve())
    paths["SOURCE_PLUS_HEART_SCAFFOLD_MULTIBLOCK"] = str((root / "05_overlay/source_plus_heart_scaffold.vtm").resolve())
    return paths


def equal_3d(axis: Any, groups: Iterable[np.ndarray]) -> None:
    points = np.vstack([np.asarray(group) for group in groups if len(group)])
    low, high = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (low + high)
    radius = max(float(np.max(high - low)) * 0.55, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def plane_grid_canonical(scaffold: HeartScaffold, plane: dict[str, Any], point: np.ndarray, size_u: float, size_v: float) -> np.ndarray:
    grid_u = np.linspace(-0.5 * size_u, 0.5 * size_u, 7)
    grid_v = np.linspace(-0.5 * size_v, 0.5 * size_v, 7)
    u, v = np.meshgrid(grid_u, grid_v)
    ras = np.asarray(point) + u[..., None] * np.asarray(plane["basis_u"]) + v[..., None] * np.asarray(plane["basis_v"])
    return scaffold.canonical(ras.reshape(-1, 3)).reshape(ras.shape)


def configure_figures() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.grid": True, "grid.alpha": 0.15,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def save_figure(fig: plt.Figure, root: Path, stem: str, dpi: int) -> dict[str, Any]:
    directory = root / "06_diagnostic_figures"
    directory.mkdir(parents=True, exist_ok=True)
    png, svg = directory / f"{stem}.png", directory / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png.resolve()), "svg": str(svg.resolve()), "dpi": dpi}


def canonical_sets(scaffold: HeartScaffold) -> dict[str, np.ndarray]:
    return {
        **{name: scaffold.canonical(points) for name, points in scaffold.source_ras.items()},
        "width": scaffold.canonical(scaffold.heart_width_ellipse["outline_ras_mm"]),
        "height": scaffold.canonical(scaffold.heart_height_ellipse["outline_ras_mm"]),
        "rca_support": scaffold.canonical(scaffold.crown_support_ras["rca"]),
        "lcx_support": scaffold.canonical(scaffold.crown_support_ras["lcx"]),
    }


def draw_sources_3d(axis: Any, data: dict[str, np.ndarray]) -> None:
    for name, label in (("lmca", "LMCA"), ("lad", "LAD"), ("lcx", "LCX"), ("rca", "inferred RCA")):
        points = data[name]
        axis.plot(*points.T, color=COLORS[name], lw=3 if name != "rca" else 2, ls="--" if name == "rca" else "-", label=label)
    axis.scatter(0, 0, 0, color=COLORS["bifurcation"], edgecolor="black", s=65, zorder=10)


def draw_scaffold_3d(axis: Any, scaffold: HeartScaffold, data: dict[str, np.ndarray]) -> None:
    for name, label in (("width", "heart width / crown"), ("height", "heart height / long axis")):
        ellipse = scaffold.heart_width_ellipse if name == "width" else scaffold.heart_height_ellipse
        points = data[name]
        mask = np.asarray(ellipse["observed_support_mask"], dtype=bool)
        axis.plot(*points.T, color=COLORS[name], lw=1.7, ls="--", alpha=0.45)
        solid = points.copy(); solid[~mask] = np.nan
        axis.plot(*solid.T, color=COLORS[name], lw=3.2, label=label)
    axis.scatter(0, 0, 0, color=COLORS["bifurcation"], edgecolor="black", s=65)
    axis.quiver(0, 0, 0, 0.55 * scaffold.heart_width_mm, 0, 0, color=COLORS["crown_axis"], linewidth=2.5, arrow_length_ratio=0.08)
    axis.quiver(0, 0, 0, 0, 0, -0.65 * scaffold.heart_height_mm, color=COLORS["apex_axis"], linewidth=2.5, arrow_length_ratio=0.08)


def decorate_3d(axis: Any, title: str) -> None:
    axis.set_xlabel("Anatomical X: crown")
    axis.set_ylabel("Anatomical Y")
    axis.set_zlabel("Anatomical Z: superior (+)")
    axis.set_title(title, weight="bold")
    axis.view_init(elev=20, azim=-63)


def generate_figures(root: Path, scaffold: HeartScaffold, dpi: int) -> list[dict[str, Any]]:
    data = canonical_sets(scaffold)
    records = []

    fig = plt.figure(figsize=(13, 8)); ax = fig.add_subplot(111, projection="3d")
    draw_sources_3d(ax, data); equal_3d(ax, [data[key] for key in ("lmca", "lad", "lcx", "rca")]); decorate_3d(ax, "Source Centerlines Only — Exact Geometry")
    ax.legend(loc="upper left", fontsize=8); records.append(save_figure(fig, root, FIGURE_STEMS[0], dpi))

    fig = plt.figure(figsize=(13, 8)); ax = fig.add_subplot(111, projection="3d")
    cor_grid = plane_grid_canonical(scaffold, scaffold.coronary_plane, scaffold.coronary_plane["centroid"], 1.15 * scaffold.heart_width_mm, 0.70 * scaffold.heart_width_mm)
    lad_grid = plane_grid_canonical(scaffold, scaffold.lad_plane, scaffold.lad_plane["centroid"], 0.75 * scaffold.heart_width_mm, 1.10 * scaffold.heart_height_mm)
    ax.plot_surface(cor_grid[..., 0], cor_grid[..., 1], cor_grid[..., 2], color=COLORS["coronary_plane"], alpha=0.40)
    ax.plot_surface(lad_grid[..., 0], lad_grid[..., 1], lad_grid[..., 2], color=COLORS["lad_plane"], alpha=0.35)
    ax.scatter(*data["rca_support"].T, color=COLORS["rca"], s=5, label="coronary-plane RCA support")
    ax.scatter(*data["lcx_support"].T, color=COLORS["lcx"], s=5, label="coronary-plane LCX support")
    ax.scatter(*data["lad"].T, color=COLORS["lad"], s=4, label="LAD-plane source")
    equal_3d(ax, [cor_grid.reshape(-1, 3), lad_grid.reshape(-1, 3), data["rca_support"], data["lcx_support"], data["lad"]]); decorate_3d(ax, "Centroid-SVD Measurement Planes + Fitting Sources")
    ax.legend(loc="upper left", fontsize=7); records.append(save_figure(fig, root, FIGURE_STEMS[1], dpi))

    fig = plt.figure(figsize=(13, 8)); ax = fig.add_subplot(111, projection="3d")
    draw_scaffold_3d(ax, scaffold, data); equal_3d(ax, [data["width"], data["height"]]); decorate_3d(ax, "Dataset-Derived Two-Ellipse Anatomical Heart Scaffold")
    ax.legend(loc="upper left", fontsize=8); records.append(save_figure(fig, root, FIGURE_STEMS[2], dpi))

    fig = plt.figure(figsize=(13, 8)); ax = fig.add_subplot(111, projection="3d")
    draw_sources_3d(ax, data); draw_scaffold_3d(ax, scaffold, data); equal_3d(ax, [data[key] for key in ("lmca", "lad", "lcx", "rca", "width", "height")]); decorate_3d(ax, "Source Arteries + Heart Scaffold Overlay")
    ax.legend(loc="upper left", fontsize=7, ncol=2); records.append(save_figure(fig, root, FIGURE_STEMS[3], dpi))

    for stem, title, axes, labels in (
        (FIGURE_STEMS[4], "Front Orthographic View", (0, 2), ("X crown", "Z superior / apex")),
        (FIGURE_STEMS[5], "Superior Orthographic View", (0, 1), ("X crown", "Y")),
        (FIGURE_STEMS[6], "Lateral / Long-Axis View", (1, 2), ("Y", "Z superior / apex")),
    ):
        fig, ax = plt.subplots(figsize=(12, 8))
        for name in ("lmca", "lad", "lcx", "rca"):
            points = data[name]
            ax.plot(points[:, axes[0]], points[:, axes[1]], color=COLORS[name], lw=2.6 if name != "rca" else 1.8, ls="--" if name == "rca" else "-", label=name.upper() if name != "rca" else "inferred RCA")
        for name in ("width", "height"):
            points = data[name]
            ellipse = scaffold.heart_width_ellipse if name == "width" else scaffold.heart_height_ellipse
            mask = np.asarray(ellipse["observed_support_mask"], dtype=bool)
            ax.plot(points[:, axes[0]], points[:, axes[1]], color=COLORS[name], ls="--", alpha=0.4)
            solid = points.copy(); solid[~mask] = np.nan
            ax.plot(solid[:, axes[0]], solid[:, axes[1]], color=COLORS[name], lw=3, label=f"heart {name}")
        ax.scatter(0, 0, s=55, color=COLORS["bifurcation"], edgecolor="black")
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_aspect("equal", adjustable="datalim"); ax.set_title(title + "\nOne fixed anatomical frame; no per-view rotation", weight="bold")
        ax.legend(loc="best", fontsize=7, ncol=2); records.append(save_figure(fig, root, stem, dpi))

    fig, ax = plt.subplots(figsize=(14, 8)); ax.axis("off")
    rows = [
        ["Case", scaffold.case_id],
        ["Heart width", f"{scaffold.heart_width_mm:.2f} mm"],
        ["Heart height", f"{scaffold.heart_height_mm:.2f} mm"],
        ["Width ellipse radii", f"{scaffold.heart_width_ellipse['major_radius_mm']:.2f} / {scaffold.heart_width_ellipse['minor_radius_mm']:.2f} mm"],
        ["Height ellipse radii", f"{scaffold.heart_height_ellipse['major_radius_mm']:.2f} / {scaffold.heart_height_ellipse['minor_radius_mm']:.2f} mm"],
        ["Coronary-plane RMSE", f"{scaffold.coronary_plane['rms_residual_mm']:.3f} mm"],
        ["LAD-plane RMSE", f"{scaffold.lad_plane['rms_residual_mm']:.3f} mm"],
        ["Plane angle", f"{np.degrees(np.arccos(np.clip(abs(np.dot(scaffold.coronary_plane['normal'], scaffold.lad_plane['normal'])), 0, 1))):.2f}°"],
        ["Frame determinant", f"{np.linalg.det(scaffold.rotation_ras_to_anatomical):.12f}"],
        ["Interpretation", "Dataset-derived reference geometry; not myocardial surface reconstruction"],
    ]
    table = ax.table(cellText=rows, colLabels=["Measurement", "Value"], loc="center", cellLoc="left", colWidths=[0.32, 0.62])
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1, 1.7)
    ax.set_title("Technical Measurement Summary", fontsize=18, weight="bold", pad=20)
    records.append(save_figure(fig, root, FIGURE_STEMS[7], dpi))
    return records


def source_snapshot(output_root: Path, case_id: str, scaffold: HeartScaffold) -> dict[str, Any]:
    case = load_case(output_root, case_id)
    return {
        "source_file": str(case["paths"]["raw"]),
        "source_file_sha256": sha256(case["paths"]["raw"]),
        "arrays": {
            name: {
                "point_count": int(len(points)),
                "coordinate_sha256": array_sha256(points),
                "segment_lengths": segment_lengths(points).copy(),
            }
            for name, points in scaffold.source_ras.items()
        },
    }


def validate_export(
    output_root: Path,
    case_root: Path,
    scaffold: HeartScaffold,
    snapshot: dict[str, Any],
    vtk_paths: dict[str, str],
    figures: list[dict[str, Any]],
    frame: dict[str, Any],
) -> dict[str, Any]:
    reloaded_case = load_case(output_root, scaffold.case_id)
    assignment = scaffold.source_support_metadata["resolved_assignment"]
    saved_lad = reloaded_case["metadata"]["assignment"]["selected_lad_source"]
    neutral = {
        saved_lad: reloaded_case["raw"]["lad"],
        "branch_b" if saved_lad == "branch_a" else "branch_a": reloaded_case["raw"]["lcx"],
    }
    reloaded = {
        "lmca": reloaded_case["raw"]["lmca"],
        "lad": neutral[assignment["lad_source"]],
        "lcx": neutral[assignment["lcx_source"]],
        "rca": reloaded_case["raw"]["rca"],
    }
    branch_integrity = {}
    for name, before in scaffold.source_ras.items():
        after = reloaded[name]
        vtk_key = {"lmca": "SOURCE_LMCA", "lad": "SOURCE_LAD", "lcx": "SOURCE_LCX", "rca": "INFERRED_RCA_CANDIDATE"}[name]
        vtk_points = pv.read(vtk_paths[vtk_key]).points
        branch_integrity[name] = {
            "point_count": int(len(before)),
            "coordinate_sha256_before": snapshot["arrays"][name]["coordinate_sha256"],
            "coordinate_sha256_after": array_sha256(after),
            "hash_unchanged": snapshot["arrays"][name]["coordinate_sha256"] == array_sha256(after),
            "maximum_coordinate_change_mm": float(np.max(np.abs(before - after))),
            "maximum_segment_length_change_mm": float(np.max(np.abs(segment_lengths(before) - segment_lengths(after)))),
            "vtk_point_count": int(len(vtk_points)),
            "vtk_maximum_coordinate_difference_mm": float(np.max(np.abs(vtk_points - before))),
        }
    display_errors = {
        name: float(np.max(np.abs(segment_lengths(points) - segment_lengths(scaffold.canonical(points)))))
        for name, points in scaffold.source_ras.items()
    }
    width_alignment = frame["width_major_axis_alignment_with_X"]
    width_z = frame["width_major_axis_alignment_with_Z"]
    height_alignment = frame["height_major_axis_alignment_with_Z"]
    expected_scaffold_blocks = [
        "BIFURCATION", "CORONARY_DISPLAY_PLANE", "LAD_DISPLAY_PLANE",
        "HEART_WIDTH_ELLIPSE", "HEART_HEIGHT_ELLIPSE", "CROWN_AXIS", "APEX_AXIS",
    ]
    expected_overlay_blocks = [
        "SOURCE_LMCA", "SOURCE_LAD", "SOURCE_LCX", "INFERRED_RCA_CANDIDATE", "BIFURCATION",
        "CORONARY_MEASUREMENT_PLANE", "LAD_MEASUREMENT_PLANE",
        "HEART_WIDTH_ELLIPSE", "HEART_HEIGHT_ELLIPSE", "CROWN_AXIS", "APEX_AXIS",
    ]
    scaffold_vtm = pv.read(vtk_paths["HEART_SCAFFOLD_MULTIBLOCK"])
    overlay_vtm = pv.read(vtk_paths["SOURCE_PLUS_HEART_SCAFFOLD_MULTIBLOCK"])
    all_files = list(vtk_paths.values()) + [item[key] for item in figures for key in ("png", "svg")] + [str(case_root / "measurements.json")]
    result = {
        "case_id": scaffold.case_id,
        "source_integrity": {
            "branches": branch_integrity,
            "source_file_hash_unchanged": snapshot["source_file_sha256"] == sha256(Path(snapshot["source_file"])),
            "max_coordinate_change_mm": max(item["maximum_coordinate_change_mm"] for item in branch_integrity.values()),
            "max_segment_length_change_mm": max(item["maximum_segment_length_change_mm"] for item in branch_integrity.values()),
            "max_vtk_source_coordinate_difference_mm": max(item["vtk_maximum_coordinate_difference_mm"] for item in branch_integrity.values()),
            "max_rigid_display_segment_length_error_mm": max(display_errors.values()),
            "hashes_unchanged": all(item["hash_unchanged"] for item in branch_integrity.values()),
        },
        "anatomical_frame": {
            "determinant": frame["determinant"],
            "orthogonality_error": frame["orthogonality_max_error"],
            "distal_lad_anatomical_z_mm": frame["distal_lad_anatomical_z_mm"],
            "height_major_axis_alignment_with_Z": height_alignment,
            "width_major_axis_alignment_with_X": width_alignment,
            "width_major_axis_alignment_with_Z": width_z,
            "width_height_axis_angle_deg": frame["width_height_axis_angle_deg"],
        },
        "vtk": {
            "heart_scaffold_block_names": scaffold_vtm.keys(),
            "overlay_block_names": overlay_vtm.keys(),
            "expected_heart_scaffold_blocks": expected_scaffold_blocks,
            "expected_overlay_blocks": expected_overlay_blocks,
        },
        "missing_files": [path for path in all_files if not Path(path).is_file()],
        "method_guards": {
            "source_centerlines_modified": False,
            "PCA_or_synthetic_geometry_used": False,
            "measurement_planes_forced_through_bifurcation": False,
            "display_planes_are_parallel_bifurcation_copies": True,
            "one_shared_rigid_frame_for_all_figures": True,
            "per_branch_or_per_ellipse_display_rotation": False,
            "heart_scaffold_claimed_as_myocardial_surface": False,
        },
    }
    result["checks"] = {
        "source_coordinates_exact": result["source_integrity"]["max_coordinate_change_mm"] == 0.0,
        "source_segments_exact": result["source_integrity"]["max_segment_length_change_mm"] == 0.0,
        "vtk_source_coordinates_exact": result["source_integrity"]["max_vtk_source_coordinate_difference_mm"] == 0.0,
        "source_hashes_unchanged": result["source_integrity"]["hashes_unchanged"] and result["source_integrity"]["source_file_hash_unchanged"],
        "rigid_display_preserves_segments": result["source_integrity"]["max_rigid_display_segment_length_error_mm"] < 1.0e-9,
        "frame_right_handed": abs(frame["determinant"] - 1.0) < 1.0e-10,
        "frame_orthonormal": frame["orthogonality_max_error"] < 1.0e-10,
        "lad_apex_is_negative_Z": frame["distal_lad_anatomical_z_mm"] < 0.0,
        "height_is_long_axis_aligned": height_alignment >= 0.95,
        "width_is_crown_aligned": width_alignment >= 0.75 and width_z <= 0.66,
        "ellipses_have_distinct_primary_directions": frame["width_height_axis_angle_deg"] >= 50.0,
        "height_ellipse_is_long_axis_dominant": (
            scaffold.heart_height_ellipse["major_radius_mm"]
            / max(scaffold.heart_height_ellipse["minor_radius_mm"], EPS)
        ) >= 1.20,
        "scaffold_dimensions_are_positive": scaffold.heart_width_mm > 0.0 and scaffold.heart_height_mm > 0.0,
        "heart_scaffold_multiblock_complete": scaffold_vtm.keys() == expected_scaffold_blocks,
        "overlay_multiblock_complete": overlay_vtm.keys() == expected_overlay_blocks,
        "all_files_present": not result["missing_files"],
    }
    result["pass"] = bool(all(result["checks"].values()))
    result["scientific_interpretation"] = (
        "The heart scaffold is dataset-derived reference geometry. "
        "It does not modify the source coronary centerlines and is not a reconstructed myocardial surface."
    )
    return result


def measurements_payload(scaffold: HeartScaffold, frame: dict[str, Any], vtk_paths: dict[str, str]) -> dict[str, Any]:
    plane_angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(scaffold.coronary_plane["normal"], scaffold.lad_plane["normal"])), 0.0, 1.0))))

    def ellipse_record(ellipse: dict[str, Any]) -> dict[str, Any]:
        record = {
            "center": ellipse["center_ras_mm"],
            "plane_normal": ellipse["plane_normal_ras"],
            "major_radius_mm": ellipse["major_radius_mm"],
            "minor_radius_mm": ellipse["minor_radius_mm"],
            "axis_major": ellipse["major_axis_ras"],
            "axis_minor": ellipse["minor_axis_ras"],
            "source_point_count": ellipse["source_point_count"],
            "source_arc_fraction": ellipse["source_arc_fraction"],
            "source_intervals": ellipse["source_intervals"],
            "observed_angular_intervals_rad": ellipse["observed_angular_intervals_rad"],
            "fit_rmse_mm": ellipse["fit_rmse_mm"],
            "max_residual_mm": ellipse["max_residual_mm"],
            "source_3d_descriptive_rmse_mm": ellipse["source_3d_descriptive_rmse_mm"],
            "fit_method": ellipse["fit_method"],
        }
        for key in ("measured_longitudinal_source_span_mm", "measured_transverse_source_span_mm"):
            if key in ellipse:
                record[key] = ellipse[key]
        return record

    return {
        "case_id": scaffold.case_id,
        "model_name": "dataset-derived two-ellipse anatomical heart scaffold",
        "bifurcation_xyz": scaffold.origin,
        "heart_width_mm": scaffold.heart_width_mm,
        "heart_height_mm": scaffold.heart_height_mm,
        "width_ellipse": ellipse_record(scaffold.heart_width_ellipse),
        "height_ellipse": ellipse_record(scaffold.heart_height_ellipse),
        "anatomical_frame": {
            "origin": scaffold.origin,
            "X_crown": scaffold.x_crown,
            "Y": scaffold.y_axis,
            "Z_superior": scaffold.z_superior,
            "rotation_matrix": scaffold.rotation_ras_to_anatomical,
            "determinant": frame["determinant"],
            "orthogonality_error": frame["orthogonality_max_error"],
        },
        "planes": {
            "coronary_centroid": scaffold.coronary_plane["centroid"],
            "coronary_normal": scaffold.coronary_plane["normal"],
            "coronary_rmse": scaffold.coronary_plane["rms_residual_mm"],
            "LAD_centroid": scaffold.lad_plane["centroid"],
            "LAD_normal": scaffold.lad_plane["normal"],
            "LAD_rmse": scaffold.lad_plane["rms_residual_mm"],
            "plane_angle_deg": plane_angle,
            "measurement_equation": "n dot (x - C) = 0",
            "display_planes": "parallel copies translated to the LMCA bifurcation; not used for fit residuals",
        },
        "source_point_counts": {name: int(len(points)) for name, points in scaffold.source_ras.items()},
        "source_support_metadata": scaffold.source_support_metadata,
        "vtk_paths": vtk_paths,
        "limitations": [
            "The scaffold is a parametric reference envelope, not a true myocardial surface reconstruction.",
            "The inferred RCA candidate is not annotated RCA ground truth.",
            "Ellipse residuals are diagnostic and do not alter or reject source arteries.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "lca_ssm")
    parser.add_argument("--case", default="189.label")
    parser.add_argument("--export-root", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--reset-output", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_figures()
    output_root = args.output_root.resolve()
    export_root = (args.export_root or output_root / "stage1_heart_scaffold_vtk").resolve()
    case_root = export_root / args.case
    if args.reset_output and case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True, exist_ok=True)
    scaffold, context = build_heart_scaffold(output_root, args.case)
    snapshot = source_snapshot(output_root, args.case, scaffold)
    vtk_paths = export_vtk(case_root, scaffold)
    measurements = measurements_payload(scaffold, context["frame"], vtk_paths)
    write_json(case_root / "measurements.json", measurements)
    figures = generate_figures(case_root, scaffold, args.dpi)
    write_json(case_root / "06_diagnostic_figures/figure_manifest.json", {"figures": figures})
    validation = validate_export(
        output_root, case_root, scaffold, snapshot, vtk_paths, figures, context["frame"]
    )
    measurements["source_integrity"] = validation["source_integrity"]
    write_json(case_root / "measurements.json", measurements)
    write_json(case_root / "validation.json", validation)
    print(json.dumps(jsonable({
        "output": str(case_root),
        "case": scaffold.case_id,
        "heart_width_mm": scaffold.heart_width_mm,
        "heart_height_mm": scaffold.heart_height_mm,
        "width_radii_mm": [scaffold.heart_width_ellipse["major_radius_mm"], scaffold.heart_width_ellipse["minor_radius_mm"]],
        "height_radii_mm": [scaffold.heart_height_ellipse["major_radius_mm"], scaffold.heart_height_ellipse["minor_radius_mm"]],
        "validation": validation,
        "paraview_file": vtk_paths["SOURCE_PLUS_HEART_SCAFFOLD_MULTIBLOCK"],
    }), indent=2))
    if not validation["pass"]:
        raise SystemExit("heart-scaffold export validation failed; inspect validation.json")


if __name__ == "__main__":
    main()
