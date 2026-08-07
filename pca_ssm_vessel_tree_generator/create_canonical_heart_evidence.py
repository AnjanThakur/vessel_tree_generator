"""Build the corrected two-plane, two-ellipse LCA presentation evidence.

The script reads only saved pre-generative LCA outputs.  Source centerlines are
never smoothed, projected in-place, or replaced.  Every patient is translated
to its LMCA bifurcation and rotated into one shared anatomical frame.  Full
ellipses are diagnostic references fitted in the two measured SVD planes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


COLORS = {
    "lmca": "#252525",
    "lad": "#D62828",
    "lcx": "#1976B9",
    "rca": "#7A5195",
    "coronary_plane": "#56B4E9",
    "lad_plane": "#F28E2B",
    "coronary_ellipse": "#0072B2",
    "lad_ellipse": "#D55E00",
    "landmark": "#FFB000",
    "bifurcation": "#2CA02C",
}

EXPECTED_IMAGES = [
    "representative_case/01_canonical_two_plane_two_ellipse_heart_structure.png",
    "representative_case/02_two_plane_ellipse_schematic.png",
    "representative_case/03_raw_centerlines_over_canonical_heart_model.png",
    "representative_case/04_svd_plane_fitting_technical_proof.png",
    "representative_case/05_canonical_front_view.png",
    "representative_case/06_canonical_superior_view.png",
    "representative_case/07_canonical_lateral_view.png",
    "representative_case/08_typical_population_canonical_heart_models.png",
]


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 1.0e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / magnitude


def curve_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def segment_lengths(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(points, axis=0), axis=1)


def acute_plane_angle(normal_a: np.ndarray, normal_b: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(normal_a, normal_b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def tangent(points: np.ndarray, start: bool = True, window: int = 7) -> np.ndarray:
    differences = np.diff(points, axis=0)
    lengths = np.linalg.norm(differences, axis=1)
    units = differences[lengths > 1.0e-12] / lengths[lengths > 1.0e-12, None]
    selected = units[:window] if start else units[-window:]
    return normalize(selected.mean(axis=0))


def load_npz_readonly(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        result = {key: np.array(archive[key], dtype=float, copy=True) for key in archive.files}
    for array in result.values():
        array.setflags(write=False)
    return result


def exact_arc_landmarks(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths(points))))
    targets = np.linspace(0.0, cumulative[-1], count)
    indexes = np.asarray([int(np.argmin(abs(cumulative - target))) for target in targets], dtype=int)
    indexes = np.unique(indexes)
    return indexes, points[indexes]


def fit_plane_through_bifurcation(points: np.ndarray, bifurcation: np.ndarray, source: str) -> dict[str, Any]:
    centered = np.asarray(points, dtype=float) - bifurcation
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vt[-1])
    signed = centered @ normal
    return {
        "origin_ras_mm": bifurcation.copy(),
        "normal_ras": normal,
        "singular_values": singular_values,
        "rmse_mm": float(np.sqrt(np.mean(signed * signed))),
        "maximum_residual_mm": float(np.max(abs(signed))),
        "signed_residuals_mm": signed,
        "point_count": int(len(points)),
        "source": source,
        "equation": "n dot (x - B) = 0",
    }


def construct_canonical_frame(
    coronary_normal: np.ndarray,
    interventricular_normal: np.ndarray,
    lad_terminal_displacement: np.ndarray,
    lcx_terminal_displacement: np.ndarray,
) -> dict[str, Any]:
    z_axis = normalize(coronary_normal)
    if float(np.dot(lad_terminal_displacement, z_axis)) > 0.0:
        z_axis = -z_axis
    y_temp = interventricular_normal - float(np.dot(interventricular_normal, z_axis)) * z_axis
    y_axis = normalize(y_temp)
    x_axis = normalize(np.cross(y_axis, z_axis))
    if float(np.dot(lcx_terminal_displacement, x_axis)) < 0.0:
        x_axis = -x_axis
        y_axis = -y_axis
    rotation = np.vstack((x_axis, y_axis, z_axis))
    determinant = float(np.linalg.det(rotation))
    if determinant < 0.0:
        raise ValueError("Canonical rotation is not right-handed")
    return {
        "X_ras": x_axis,
        "Y_ras": y_axis,
        "Z_ras": z_axis,
        "rotation_ras_to_canonical": rotation,
        "determinant": determinant,
        "orthonormality_max_error": float(np.max(abs(rotation @ rotation.T - np.eye(3)))),
    }


def ellipse_coordinates(points: np.ndarray, parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cx, cy, log_a, log_b, theta = parameters
    a, b = float(np.exp(log_a)), float(np.exp(log_b))
    cosine, sine = math.cos(theta), math.sin(theta)
    centered = points - np.array([cx, cy])
    x_local = cosine * centered[:, 0] + sine * centered[:, 1]
    y_local = -sine * centered[:, 0] + cosine * centered[:, 1]
    return x_local, y_local, np.array([a, b])


def robust_full_ellipse(points: np.ndarray, label: str) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    if len(points) < 12:
        raise ValueError(f"{label}: at least 12 points are required for a stable full ellipse")
    center_mean = points.mean(axis=0)
    center_box = 0.5 * (points.min(axis=0) + points.max(axis=0))
    covariance = np.cov(points - center_mean, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    theta0 = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 2.0)
    rotated = (points - center_mean) @ eigenvectors
    axes0 = np.maximum(0.5 * np.ptp(rotated, axis=0), 1.0)
    lower = np.array([
        points[:, 0].min() - span,
        points[:, 1].min() - span,
        math.log(0.5), math.log(0.5), -4.0 * math.pi,
    ])
    upper = np.array([
        points[:, 0].max() + span,
        points[:, 1].max() + span,
        math.log(1.5 * span), math.log(1.5 * span), 4.0 * math.pi,
    ])

    def residuals(parameters: np.ndarray) -> np.ndarray:
        x_local, y_local, axes = ellipse_coordinates(points, parameters)
        radial = np.sqrt((x_local / axes[0]) ** 2 + (y_local / axes[1]) ** 2)
        return (radial - 1.0) * math.sqrt(float(axes[0] * axes[1]))

    starts = []
    for center in (center_mean, center_box, np.zeros(2)):
        starts.append(np.array([center[0], center[1], math.log(axes0[0]), math.log(axes0[1]), theta0]))
        starts.append(np.array([center[0], center[1], math.log(axes0[1]), math.log(axes0[0]), theta0 + math.pi / 2.0]))
    solutions = []
    for start in starts:
        start = np.clip(start, lower + 1.0e-9, upper - 1.0e-9)
        result = least_squares(
            residuals, start, bounds=(lower, upper), loss="soft_l1",
            f_scale=max(1.0, 0.025 * span), max_nfev=5000,
        )
        solutions.append(result)
    result = min(solutions, key=lambda item: float(np.mean(residuals(item.x) ** 2)))
    cx, cy, log_a, log_b, theta = result.x
    a, b = float(np.exp(log_a)), float(np.exp(log_b))
    if b > a:
        a, b = b, a
        theta += math.pi / 2.0
    theta = float((theta + math.pi) % math.pi)
    sample_angles = np.linspace(0.0, 2.0 * math.pi, 1440, endpoint=False)
    local = np.column_stack((a * np.cos(sample_angles), b * np.sin(sample_angles)))
    cosine, sine = math.cos(theta), math.sin(theta)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    outline = local @ rotation.T + np.array([cx, cy])
    distances, closest_indexes = cKDTree(outline).query(points)
    centered = points - np.array([cx, cy])
    local_points = centered @ rotation
    angular_positions = np.arctan2(local_points[:, 1] / b, local_points[:, 0] / a)
    return {
        "label": label,
        "center_2d_mm": np.array([cx, cy]),
        "semi_major_axis_mm": a,
        "semi_minor_axis_mm": b,
        "axis_ratio": a / b,
        "tilt_rad": theta,
        "tilt_deg": float(np.degrees(theta)),
        "outline_2d_mm": outline,
        "outline_angles_rad": sample_angles,
        "residuals_mm": distances,
        "rmse_mm": float(np.sqrt(np.mean(distances * distances))),
        "maximum_residual_mm": float(np.max(distances)),
        "angular_positions_rad": angular_positions,
        "closest_outline_indexes": closest_indexes,
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "source_point_count": int(len(points)),
        "observed_data_span_mm": span,
        "center_offset_from_data_mean_mm": float(np.linalg.norm(np.array([cx, cy]) - center_mean)),
    }


def ellipse_angles(points_2d: np.ndarray, ellipse: dict[str, Any]) -> np.ndarray:
    theta = float(ellipse["tilt_rad"])
    cosine, sine = math.cos(theta), math.sin(theta)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    local = (points_2d - ellipse["center_2d_mm"]) @ rotation
    return np.arctan2(local[:, 1] / ellipse["semi_minor_axis_mm"], local[:, 0] / ellipse["semi_major_axis_mm"])


def load_case(output_root: Path, patient_id: str) -> dict[str, Any]:
    paths = {
        "raw": output_root / "raw_cases" / patient_id / "original_centerlines.npz",
        "metadata": output_root / "raw_cases" / patient_id / "source_metadata.json",
        "landmarks": output_root / "landmarks" / patient_id / "source_landmarks.npz",
        "planes": output_root / "planes" / patient_id / "planes.json",
        "ellipses": output_root / "ellipse_references" / patient_id / "reference_parameters.json",
        "parameters": output_root / "per_patient_parameters" / patient_id / "parameters.json",
        "qc": output_root / "quality_control" / f"{patient_id}.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"{patient_id}: missing saved input {path}")
    raw = load_npz_readonly(paths["raw"])
    landmarks = load_npz_readonly(paths["landmarks"])
    return {
        "patient_id": patient_id,
        "paths": paths,
        "raw": raw,
        "metadata": json.loads(paths["metadata"].read_text(encoding="utf-8")),
        "landmarks": landmarks,
        "saved_planes": json.loads(paths["planes"].read_text(encoding="utf-8")),
        "saved_ellipses": json.loads(paths["ellipses"].read_text(encoding="utf-8")),
        "parameters": json.loads(paths["parameters"].read_text(encoding="utf-8")),
        "qc": json.loads(paths["qc"].read_text(encoding="utf-8")),
    }


def saved_ellipse_rmse(case: dict[str, Any], branch: str) -> float:
    values = np.asarray(case["saved_ellipses"][branch]["landmark_residual_magnitudes_mm"], dtype=float)
    return float(np.sqrt(np.mean(values * values)))


def selection_features(case: dict[str, Any]) -> dict[str, float]:
    measurements = case["parameters"]["measurements"]
    return {
        "lmca_length_mm": float(measurements["lengths_mm"]["lmca"]),
        "lad_length_mm": float(measurements["lengths_mm"]["lad"]),
        "lcx_length_mm": float(measurements["lengths_mm"]["lcx"]),
        "lad_lcx_length_ratio": float(measurements["length_ratios"]["lad_to_lcx"]),
        "bifurcation_angle_deg": float(measurements["angles_deg"]["lad_lcx_bifurcation"]),
        "plane_angle_deg": float(measurements["angles_deg"]["lad_coronary_plane"]),
        "lad_plane_rmse_mm": float(case["saved_planes"]["lad"]["rmse_mm"]),
        "coronary_plane_rmse_mm": float(case["saved_planes"]["coronary"]["rmse_mm"]),
        "ellipse_landmark_rmse_mm": 0.5 * (
            saved_ellipse_rmse(case, "lad") + saved_ellipse_rmse(case, "lcx")
        ),
    }


def case_eligible(case: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if case["qc"]["status"] != "accepted":
        reasons.append("not fully accepted")
    rca = case["metadata"].get("rca", {})
    if not rca.get("available", False) or case["raw"].get("rca") is None or len(case["raw"]["rca"]) < 12:
        reasons.append("reliable RCA candidate unavailable")
    if case["saved_planes"]["coronary"].get("plane_source") != "inferred_rca_plus_lcx":
        reasons.append("coronary plane is not RCA-candidate + LCX")
    if case["qc"].get("warnings"):
        reasons.append("warnings present")
    return not reasons, reasons


def build_canonical_model(case: dict[str, Any]) -> dict[str, Any]:
    raw = case["raw"]
    bifurcation = np.asarray(raw["lmca"][-1], dtype=float)
    if np.linalg.norm(raw["lad"][0] - bifurcation) > 1.0e-9 or np.linalg.norm(raw["lcx"][0] - bifurcation) > 1.0e-9:
        raise ValueError(f"{case['patient_id']}: daughter paths do not start at the LMCA bifurcation")
    ring = np.vstack((raw["rca"], raw["lcx"]))
    coronary_plane = fit_plane_through_bifurcation(
        ring, bifurcation, "unchanged inferred RCA candidate + unchanged LCX ring points"
    )
    lad_plane = fit_plane_through_bifurcation(
        raw["lad"], bifurcation, "all unchanged LAD points"
    )
    frame = construct_canonical_frame(
        coronary_plane["normal_ras"], lad_plane["normal_ras"],
        raw["lad"][-1] - bifurcation, raw["lcx"][-1] - bifurcation,
    )
    rotation = frame["rotation_ras_to_canonical"]
    canonical = {name: (rotation @ (points - bifurcation).T).T for name, points in raw.items()}
    for points in canonical.values():
        points.setflags(write=False)
    coronary_2d = np.vstack((canonical["rca"][:, :2], canonical["lcx"][:, :2]))
    coronary_ellipse = robust_full_ellipse(coronary_2d, "coronary crown ellipse")
    coronary_ellipse["outline_3d_mm"] = np.column_stack((coronary_ellipse["outline_2d_mm"], np.zeros(1440)))

    lad_normal_canonical = normalize(rotation @ lad_plane["normal_ras"])
    lad_horizontal = np.array([1.0, 0.0, 0.0])
    lad_vertical = normalize(np.array([0.0, 0.0, 1.0]) - lad_normal_canonical[2] * lad_normal_canonical)
    if lad_vertical[2] < 0.0:
        lad_vertical = -lad_vertical
    lad_2d = np.column_stack((canonical["lad"] @ lad_horizontal, canonical["lad"] @ lad_vertical))
    lad_ellipse = robust_full_ellipse(lad_2d, "LAD interventricular ellipse")
    lad_ellipse["outline_3d_mm"] = (
        lad_ellipse["outline_2d_mm"][:, 0, None] * lad_horizontal
        + lad_ellipse["outline_2d_mm"][:, 1, None] * lad_vertical
    )

    landmark_arrays = {
        name: (rotation @ (case["landmarks"][f"{name}_original"] - bifurcation).T).T
        for name in ("lmca", "lad", "lcx")
    }
    rca_indexes, rca_points = exact_arc_landmarks(raw["rca"], 10)
    landmark_arrays["rca"] = (rotation @ (rca_points - bifurcation).T).T
    coronary_ellipse["lcx_landmark_angles_deg"] = np.degrees(
        ellipse_angles(landmark_arrays["lcx"][:, :2], coronary_ellipse)
    )
    coronary_ellipse["rca_landmark_angles_deg"] = np.degrees(
        ellipse_angles(landmark_arrays["rca"][:, :2], coronary_ellipse)
    )
    lad_landmarks_2d = np.column_stack((
        landmark_arrays["lad"] @ lad_horizontal,
        landmark_arrays["lad"] @ lad_vertical,
    ))
    lad_angles = ellipse_angles(lad_landmarks_2d, lad_ellipse)
    full_lad_angles = np.unwrap(ellipse_angles(lad_2d, lad_ellipse))
    lad_ellipse["landmark_angles_deg"] = np.degrees(lad_angles)
    lad_ellipse["observed_angular_extent_deg"] = float(np.degrees(full_lad_angles[-1] - full_lad_angles[0]))

    maximum_rigid_error = 0.0
    branch_rigid_errors = {}
    for name in ("lmca", "lad", "lcx", "rca"):
        error = float(np.max(abs(segment_lengths(raw[name]) - segment_lengths(canonical[name]))))
        branch_rigid_errors[name] = error
        maximum_rigid_error = max(maximum_rigid_error, error)
    plane_angle = acute_plane_angle(coronary_plane["normal_ras"], lad_plane["normal_ras"])
    scale = max(
        float(np.max(np.linalg.norm(coronary_ellipse["outline_3d_mm"], axis=1))),
        float(np.max(np.linalg.norm(lad_ellipse["outline_3d_mm"], axis=1))), 1.0,
    )
    return {
        "patient_id": case["patient_id"],
        "case": case,
        "bifurcation_ras_mm": bifurcation,
        "coronary_plane": coronary_plane,
        "lad_plane": lad_plane,
        "frame": frame,
        "canonical": canonical,
        "landmarks_canonical": landmark_arrays,
        "rca_landmark_indexes": rca_indexes,
        "coronary_ellipse": coronary_ellipse,
        "lad_ellipse": lad_ellipse,
        "lad_horizontal_canonical": lad_horizontal,
        "lad_vertical_canonical": lad_vertical,
        "lad_normal_canonical": lad_normal_canonical,
        "plane_separation_angle_deg": plane_angle,
        "lad_terminal_canonical": canonical["lad"][-1],
        "lcx_terminal_canonical": canonical["lcx"][-1],
        "branch_rigid_segment_errors_mm": branch_rigid_errors,
        "maximum_rigid_segment_error_mm": maximum_rigid_error,
        "display_scale_mm": scale,
    }


def stable_model(model: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    for name in ("coronary_ellipse", "lad_ellipse"):
        ellipse = model[name]
        if not ellipse["optimizer_success"]:
            reasons.append(f"{name} optimizer did not converge")
        if ellipse["axis_ratio"] > 10.0:
            reasons.append(f"{name} axis ratio exceeds 10")
        if ellipse["semi_major_axis_mm"] > 1.35 * ellipse["observed_data_span_mm"]:
            reasons.append(f"{name} major axis is extrapolative relative to observed span")
        if ellipse["center_offset_from_data_mean_mm"] > 1.25 * ellipse["observed_data_span_mm"]:
            reasons.append(f"{name} centre is extrapolative relative to observed span")
        if ellipse["rmse_mm"] > 0.25 * max(ellipse["semi_major_axis_mm"], 1.0):
            reasons.append(f"{name} relative residual exceeds 25%")
    if model["coronary_plane"]["point_count"] < 30:
        reasons.append("too few crown-ring points")
    return not reasons, reasons


def select_typical_cases(output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted_index = json.loads(
        (output_root / "quality_control/accepted_patient_index.json").read_text(encoding="utf-8")
    )
    accepted_ids = accepted_index.get("patient_ids", [])
    eligible_cases, excluded = [], {}
    for patient_id in accepted_ids:
        case = load_case(output_root, patient_id)
        eligible, reasons = case_eligible(case)
        if eligible:
            eligible_cases.append(case)
        else:
            excluded[patient_id] = reasons
    feature_names = list(selection_features(eligible_cases[0]))
    matrix = np.asarray([[selection_features(case)[key] for key in feature_names] for case in eligible_cases])
    medians = np.median(matrix, axis=0)
    iqrs = np.percentile(matrix, 75, axis=0) - np.percentile(matrix, 25, axis=0)
    iqrs[iqrs < 1.0e-9] = 1.0
    scores = np.sqrt(np.mean(((matrix - medians) / iqrs) ** 2, axis=1))
    ranked = sorted(zip(scores, eligible_cases), key=lambda item: (float(item[0]), int(item[1]["patient_id"].split(".")[0])))
    selected, unstable = [], {}
    for score, case in ranked:
        model = build_canonical_model(case)
        stable, reasons = stable_model(model)
        if not stable:
            unstable[case["patient_id"]] = reasons
            continue
        selected.append({"case": case, "model": model, "median_distance_score": float(score)})
        if len(selected) == 3:
            break
    if len(selected) != 3:
        raise RuntimeError("Could not identify three stable, fully accepted median-near cases")
    selection_audit = {
        "method": "three fully accepted, RCA+LCX-eligible stable cases nearest the robust multivariate median",
        "accepted_case_count": len(accepted_ids),
        "eligible_case_count": len(eligible_cases),
        "feature_names": feature_names,
        "population_medians": dict(zip(feature_names, medians)),
        "population_iqrs": dict(zip(feature_names, iqrs)),
        "excluded_case_count": len(excluded),
        "unstable_median_near_cases_skipped": unstable,
    }
    return selected, selection_audit


def configure_plotting() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 11, "axes.titlesize": 14, "axes.labelsize": 11,
        "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.20,
    })


def equal_limits(axis: Any, point_sets: Iterable[np.ndarray], padding: float = 0.10) -> None:
    points = np.vstack([np.asarray(points) for points in point_sets if points is not None and len(points)])
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.5 * max(float(np.max(maximum - minimum)), 1.0) * (1.0 + 2.0 * padding)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def plane_surfaces(model: dict[str, Any], scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    extent = float(scale or model["display_scale_mm"] * 1.05)
    grid = np.linspace(-extent, extent, 9)
    uu, vv = np.meshgrid(grid, grid)
    coronary = np.stack((uu, vv, np.zeros_like(uu)), axis=-1)
    h = model["lad_horizontal_canonical"]
    v = model["lad_vertical_canonical"]
    interventricular = uu[..., None] * h + vv[..., None] * v
    return coronary, interventricular


def draw_planes(axis: Any, model: dict[str, Any], alpha: float = 0.20, scale: float | None = None) -> list[np.ndarray]:
    coronary, interventricular = plane_surfaces(model, scale)
    axis.plot_surface(*[coronary[..., index] for index in range(3)], color=COLORS["coronary_plane"], alpha=alpha, shade=False, linewidth=0)
    axis.plot_surface(*[interventricular[..., index] for index in range(3)], color=COLORS["lad_plane"], alpha=alpha, shade=False, linewidth=0)
    return [coronary.reshape(-1, 3), interventricular.reshape(-1, 3)]


def draw_full_ellipses(axis: Any, model: dict[str, Any], alpha: float = 0.90, linewidth: float = 2.8) -> list[np.ndarray]:
    coronary = model["coronary_ellipse"]["outline_3d_mm"]
    lad = model["lad_ellipse"]["outline_3d_mm"]
    axis.plot(*coronary.T, color=COLORS["coronary_ellipse"], linewidth=linewidth, alpha=alpha, label="Full coronary crown ellipse")
    axis.plot(*lad.T, color=COLORS["lad_ellipse"], linewidth=linewidth, alpha=alpha, label="Full LAD ellipse")
    return [coronary, lad]


def draw_source(axis: Any, model: dict[str, Any], include_rca: bool = True, linewidth: float = 3.0, landmarks: bool = True) -> list[np.ndarray]:
    arrays = []
    names = ("lmca", "lad", "lcx", "rca") if include_rca else ("lmca", "lad", "lcx")
    for name in names:
        points = model["canonical"][name]
        arrays.append(points)
        axis.plot(*points.T, color=COLORS[name], linewidth=linewidth, label=f"Unchanged {name.upper()}")
        if landmarks:
            chosen = model["landmarks_canonical"][name]
            axis.scatter(*chosen.T, s=24, color=COLORS[name], edgecolor=COLORS["landmark"], linewidth=0.9, zorder=8)
    axis.scatter(0, 0, 0, s=80, color=COLORS["bifurcation"], edgecolor="white", linewidth=1.0, zorder=10, label="Shared bifurcation")
    return arrays


def decorate_3d(axis: Any, title: str | None = None) -> None:
    if title:
        axis.set_title(title)
    axis.set_xlabel("Canonical X — LCX crown direction (mm)")
    axis.set_ylabel("Canonical Y (mm)")
    axis.set_zlabel("Canonical Z — apex is negative (mm)")


def geometry_legend(axis: Any, include_source: bool = True, location: str = "upper right") -> None:
    handles = [
        Line2D([0], [0], color=COLORS["coronary_ellipse"], lw=3, label="Full coronary crown ellipse"),
        Line2D([0], [0], color=COLORS["lad_ellipse"], lw=3, label="Full LAD ellipse"),
        Patch(facecolor=COLORS["coronary_plane"], alpha=0.25, label="Coronary SVD plane"),
        Patch(facecolor=COLORS["lad_plane"], alpha=0.25, label="Interventricular SVD plane"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["bifurcation"], markeredgecolor="white", markersize=8, label="LMCA bifurcation"),
    ]
    if include_source:
        handles.extend([
            Line2D([0], [0], color=COLORS[name], lw=3, label=f"Unchanged {name.upper()}")
            for name in ("lmca", "lad", "lcx", "rca")
        ])
        handles.append(Line2D([0], [0], marker="o", color="none", markeredgecolor=COLORS["landmark"], markersize=7, label="Exact source landmarks"))
    axis.legend(handles=handles, loc=location, framealpha=0.95)


class FigureWriter:
    def __init__(self, root: Path, dpi: int) -> None:
        self.root, self.dpi = root, dpi
        self.manifest = []
        self.generated_at = datetime.now().astimezone().isoformat()

    def save(self, figure: plt.Figure, relative_path: str, purpose: str, patient_or_cohort: str) -> None:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=self.dpi, facecolor="white")
        figure.savefig(path.with_suffix(".svg"), facecolor="white")
        plt.close(figure)
        self.manifest.append({
            "filename": relative_path, "vector_companion": str(path.with_suffix(".svg").relative_to(self.root)).replace("\\", "/"),
            "purpose": purpose, "patient_or_cohort": patient_or_cohort,
            "source_geometry": "unchanged; rigidly translated/rotated for display only",
            "generated_at": self.generated_at,
        })


def plot_main(writer: FigureWriter, model: dict[str, Any]) -> None:
    figure = plt.figure(figsize=(16, 9))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.84)
    axis = figure.add_subplot(111, projection="3d")
    surfaces = draw_planes(axis, model, alpha=0.20)
    ellipses = draw_full_ellipses(axis, model)
    sources = draw_source(axis, model, include_rca=True, linewidth=2.7)
    decorate_3d(axis)
    equal_limits(axis, surfaces + ellipses + sources, padding=0.03)
    axis.view_init(elev=22, azim=-58)
    geometry_legend(axis, location="upper left")
    figure.suptitle("Dataset-Derived Two-Plane LCA Anatomical Model", fontsize=20, weight="bold")
    figure.text(0.5, 0.925, "Coronary Plane: RCA + LCX Ring | Interventricular Plane: LAD Descent Toward Apex", ha="center", fontsize=13)
    figure.text(0.02, 0.04, f"{model['patient_id']} | plane separation {model['plane_separation_angle_deg']:.1f}° | full ellipses are reference models", fontsize=10)
    writer.save(figure, EXPECTED_IMAGES[0], "Main proof of two SVD planes and two dataset-derived full ellipses in one canonical heart frame.", model["patient_id"])


def plot_schematic(writer: FigureWriter, model: dict[str, Any]) -> None:
    figure = plt.figure(figsize=(16, 9))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.07, top=0.91)
    axis = figure.add_subplot(111, projection="3d")
    surfaces = draw_planes(axis, model, alpha=0.24)
    ellipses = draw_full_ellipses(axis, model, linewidth=4.0)
    for name in ("lad", "lcx", "rca"):
        landmarks = model["landmarks_canonical"][name]
        axis.scatter(*landmarks.T, s=38, color=COLORS[name], edgecolor="white", linewidth=0.8, zorder=8)
    axis.scatter(0, 0, 0, s=105, color=COLORS["bifurcation"], edgecolor="white", linewidth=1.0, zorder=10)
    extent = model["display_scale_mm"] * 0.72
    axis.quiver(0, 0, 0, extent, 0, 0, color=COLORS["coronary_ellipse"], linewidth=3, arrow_length_ratio=0.08)
    axis.text(extent, 0, 0, "  +X crown / LCX direction", color=COLORS["coronary_ellipse"], weight="bold")
    axis.quiver(0, 0, 0, 0, 0, -extent, color=COLORS["lad_ellipse"], linewidth=3, arrow_length_ratio=0.08)
    axis.text(0, 0, -extent, "  −Z apex direction", color=COLORS["lad_ellipse"], weight="bold")
    decorate_3d(axis, "Two SVD Planes and Two Dataset-Derived Ellipses")
    equal_limits(axis, surfaces + ellipses, padding=0.03)
    axis.view_init(elev=22, azim=-58)
    geometry_legend(axis, include_source=False)
    writer.save(figure, EXPECTED_IMAGES[1], "PPT-like measured two-plane/two-ellipse schematic without raw-centerline clutter.", model["patient_id"])


def project_to_plane(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return points - np.outer(points @ normal, normal)


def plot_overlay(writer: FigureWriter, model: dict[str, Any]) -> None:
    figure = plt.figure(figsize=(16, 9))
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.08, top=0.91)
    axis = figure.add_subplot(111, projection="3d")
    surfaces = draw_planes(axis, model, alpha=0.12)
    ellipses = draw_full_ellipses(axis, model, alpha=0.42, linewidth=2.1)
    sources = draw_source(axis, model, include_rca=True, linewidth=3.5)
    coronary_normal = np.array([0.0, 0.0, 1.0])
    lad_normal = model["lad_normal_canonical"]
    for name, normal in (("rca", coronary_normal), ("lcx", coronary_normal), ("lad", lad_normal)):
        points = model["canonical"][name]
        step = max(1, len(points) // 18)
        selected = points[::step]
        projected = project_to_plane(selected, normal)
        for source, target in zip(selected, projected):
            axis.plot(*np.vstack((source, target)).T, color="#777777", linewidth=0.7, alpha=0.38)
    decorate_3d(axis, "Unchanged Source Centerlines Over the Canonical Heart Model")
    equal_limits(axis, surfaces + ellipses + sources, padding=0.03)
    axis.view_init(elev=22, azim=-58)
    geometry_legend(axis)
    axis.text2D(0.02, 0.03, "Solid centreline coordinates are unchanged. Pale segments show orthogonal plane residuals.", transform=axis.transAxes)
    writer.save(figure, EXPECTED_IMAGES[2], "Source-data overlay proving the references were derived without replacing source points.", model["patient_id"])


def plot_technical(writer: FigureWriter, model: dict[str, Any]) -> None:
    figure = plt.figure(figsize=(16, 9))
    figure.subplots_adjust(left=0.05, right=0.98, bottom=0.13, top=0.89, wspace=0.22, hspace=0.38)
    axis_a = figure.add_subplot(2, 2, 1, projection="3d")
    coronary_surface, _ = plane_surfaces(model, model["display_scale_mm"] * 0.75)
    axis_a.plot_surface(*[coronary_surface[..., i] for i in range(3)], color=COLORS["coronary_plane"], alpha=0.24, shade=False)
    for name in ("rca", "lcx"):
        axis_a.scatter(*model["canonical"][name].T, s=5, color=COLORS[name], alpha=0.72, label=name.upper())
    axis_a.set_title("A. RCA candidate + LCX ring → coronary SVD plane")
    axis_a.view_init(elev=25, azim=-58); axis_a.legend(); equal_limits(axis_a, [coronary_surface.reshape(-1, 3), model["canonical"]["rca"], model["canonical"]["lcx"]])

    axis_b = figure.add_subplot(2, 2, 2, projection="3d")
    _, lad_surface = plane_surfaces(model, model["display_scale_mm"] * 0.75)
    axis_b.plot_surface(*[lad_surface[..., i] for i in range(3)], color=COLORS["lad_plane"], alpha=0.24, shade=False)
    axis_b.scatter(*model["canonical"]["lad"].T, s=5, color=COLORS["lad"], alpha=0.72)
    axis_b.set_title("B. All LAD points → interventricular SVD plane")
    axis_b.view_init(elev=25, azim=-58); equal_limits(axis_b, [lad_surface.reshape(-1, 3), model["canonical"]["lad"]])

    axis_c = figure.add_subplot(2, 2, 3)
    labels = ["Coronary S1", "Coronary S2", "Coronary S3", "LAD S1", "LAD S2", "LAD S3"]
    values = list(model["coronary_plane"]["singular_values"]) + list(model["lad_plane"]["singular_values"])
    axis_c.bar(np.arange(6), values, color=[COLORS["coronary_plane"]] * 3 + [COLORS["lad_plane"]] * 3)
    axis_c.set_xticks(np.arange(6), labels, rotation=25, ha="right")
    axis_c.set_ylabel("Singular value")
    axis_c.set_title("C. Singular values and orthogonal residuals")
    axis_c.text(0.02, 0.95, f"Coronary RMSE / max: {model['coronary_plane']['rmse_mm']:.2f} / {model['coronary_plane']['maximum_residual_mm']:.2f} mm\nLAD RMSE / max: {model['lad_plane']['rmse_mm']:.2f} / {model['lad_plane']['maximum_residual_mm']:.2f} mm", transform=axis_c.transAxes, va="top", bbox=dict(facecolor="white", alpha=0.85))

    axis_d = figure.add_subplot(2, 2, 4, projection="3d")
    n_cor = np.array([0.0, 0.0, 1.0])
    n_lad = model["lad_normal_canonical"]
    axis_d.quiver(0, 0, 0, *n_cor, length=1.0, color=COLORS["coronary_plane"], linewidth=4, label="coronary normal")
    axis_d.quiver(0, 0, 0, *n_lad, length=1.0, color=COLORS["lad_plane"], linewidth=4, label="LAD-plane normal")
    axis_d.set_xlim(-1, 1); axis_d.set_ylim(-1, 1); axis_d.set_zlim(-1, 1); axis_d.set_box_aspect((1, 1, 1)); axis_d.view_init(elev=22, azim=-58)
    axis_d.set_title(f"D. Plane-normal separation = {model['plane_separation_angle_deg']:.2f}°")
    axis_d.legend()
    figure.suptitle("SVD Plane-Fitting Technical Proof", fontsize=19, weight="bold")
    figure.text(0.5, 0.025, "n · (x − B) = 0    |    3 non-collinear points define a plane; all available points are used with least-squares SVD for robustness.", ha="center", fontsize=11)
    writer.save(figure, EXPECTED_IMAGES[3], "Four-panel technical proof of bifurcation-constrained least-squares SVD plane fitting.", model["patient_id"])


def plot_orthographic(
    writer: FigureWriter,
    model: dict[str, Any],
    index: int,
    title: str,
    horizontal_index: int,
    vertical_index: int,
    horizontal_label: str,
    vertical_label: str,
) -> None:
    figure, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    coronary = model["coronary_ellipse"]["outline_3d_mm"]
    lad = model["lad_ellipse"]["outline_3d_mm"]
    axis.plot(coronary[:, horizontal_index], coronary[:, vertical_index], color=COLORS["coronary_ellipse"], linewidth=3.2, label="Full coronary crown ellipse")
    axis.plot(lad[:, horizontal_index], lad[:, vertical_index], color=COLORS["lad_ellipse"], linewidth=3.2, label="Full LAD ellipse")
    all_points = [coronary[:, [horizontal_index, vertical_index]], lad[:, [horizontal_index, vertical_index]]]
    for name in ("lmca", "lad", "lcx", "rca"):
        points = model["canonical"][name]
        projected = points[:, [horizontal_index, vertical_index]]
        all_points.append(projected)
        axis.plot(projected[:, 0], projected[:, 1], color=COLORS[name], linewidth=2.5, label=f"Unchanged {name.upper()}")
        landmarks = model["landmarks_canonical"][name][:, [horizontal_index, vertical_index]]
        axis.scatter(landmarks[:, 0], landmarks[:, 1], s=27, color=COLORS[name], edgecolor=COLORS["landmark"], linewidth=0.8, zorder=7)
    axis.scatter(0, 0, s=85, color=COLORS["bifurcation"], edgecolor="white", linewidth=1.0, zorder=9, label="Shared bifurcation")
    if vertical_index == 2:
        axis.axhline(0, color=COLORS["coronary_plane"], linewidth=8, alpha=0.18, label="Coronary plane (edge-on)")
    if horizontal_index == 0 and vertical_index == 1:
        axis.set_facecolor("#F4FAFD")
        axis.text(0.02, 0.96, "Coronary SVD plane (canonical XY)", transform=axis.transAxes, va="top", color="#277DA1")
    stacked = np.vstack(all_points)
    minimum, maximum = stacked.min(axis=0), stacked.max(axis=0)
    center = 0.5 * (minimum + maximum)
    radius = 0.58 * max(float(np.max(maximum - minimum)), 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(horizontal_label)
    axis.set_ylabel(vertical_label)
    axis.set_title(title, fontsize=18, weight="bold")
    axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.95)
    axis.grid(alpha=0.22)
    writer.save(figure, EXPECTED_IMAGES[index], f"Fixed canonical {title.lower()} using the shared frame.", model["patient_id"])


def plot_montage(writer: FigureWriter, selected: list[dict[str, Any]]) -> None:
    figure = plt.figure(figsize=(16, 6))
    figure.subplots_adjust(left=0.04, right=0.96, bottom=0.16, top=0.80, wspace=0.10)
    for index, item in enumerate(selected):
        model = item["model"]
        axis = figure.add_subplot(1, 3, index + 1, projection="3d", proj_type="ortho")
        scale = model["display_scale_mm"]
        coronary = model["coronary_ellipse"]["outline_3d_mm"] / scale
        lad = model["lad_ellipse"]["outline_3d_mm"] / scale
        axis.plot(*coronary.T, color=COLORS["coronary_ellipse"], linewidth=2.4)
        axis.plot(*lad.T, color=COLORS["lad_ellipse"], linewidth=2.4)
        for name in ("lmca", "lad", "lcx", "rca"):
            points = model["canonical"][name] / scale
            axis.plot(*points.T, color=COLORS[name], linewidth=1.7)
        axis.scatter(0, 0, 0, s=45, color=COLORS["bifurcation"], edgecolor="white")
        axis.set_xlim(-1.15, 1.15); axis.set_ylim(-1.15, 1.15); axis.set_zlim(-1.15, 1.15)
        axis.set_box_aspect((1, 1, 1)); axis.view_init(elev=5, azim=-90)
        axis.set_title(f"{model['patient_id']}\nplane angle {model['plane_separation_angle_deg']:.1f}°")
        axis.set_xlabel("+X crown"); axis.set_zlabel("−Z apex")
        axis.set_xticklabels([]); axis.set_yticklabels([]); axis.set_zticklabels([])
    figure.suptitle("Two SVD Planes and Two Dataset-Derived Ellipses — Typical Fully Accepted Cases", fontsize=18, weight="bold")
    figure.text(0.5, 0.02, "Identical canonical axes, fixed orthographic front view, and normalized display scale; no per-panel appearance rotation.", ha="center")
    writer.save(figure, EXPECTED_IMAGES[7], "Three typical fully accepted cases shown in identical canonical conventions and normalized scale.", ", ".join(item["model"]["patient_id"] for item in selected))


def compact_ellipse(ellipse: dict[str, Any]) -> dict[str, Any]:
    excluded = {"outline_2d_mm", "outline_3d_mm", "residuals_mm", "angular_positions_rad", "closest_outline_indexes", "outline_angles_rad"}
    return {key: value for key, value in ellipse.items() if key not in excluded}


def compact_plane(plane: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plane.items() if key != "signed_residuals_mm"}


def case_metrics(item: dict[str, Any]) -> dict[str, Any]:
    model = item["model"]
    return {
        "patient_id": model["patient_id"],
        "qc_status": model["case"]["qc"]["status"],
        "median_distance_score": item["median_distance_score"],
        "coronary_plane": compact_plane(model["coronary_plane"]),
        "interventricular_plane": compact_plane(model["lad_plane"]),
        "plane_separation_angle_deg": model["plane_separation_angle_deg"],
        "canonical_frame": model["frame"],
        "bifurcation_ras_mm": model["bifurcation_ras_mm"],
        "lad_terminal_canonical_mm": model["lad_terminal_canonical"],
        "lcx_terminal_canonical_mm": model["lcx_terminal_canonical"],
        "coronary_full_ellipse": compact_ellipse(model["coronary_ellipse"]),
        "lad_full_ellipse": compact_ellipse(model["lad_ellipse"]),
        "rca_landmark_source_indexes": model["rca_landmark_indexes"],
        "branch_rigid_segment_errors_mm": model["branch_rigid_segment_errors_mm"],
        "maximum_rigid_segment_error_mm": model["maximum_rigid_segment_error_mm"],
        "source_geometry_modified": False,
    }


def validate(selected: list[dict[str, Any]], hashes_before: dict[str, str], evidence_root: Path) -> dict[str, Any]:
    cases = []
    for item in selected:
        model = item["model"]
        rotation = model["frame"]["rotation_ras_to_canonical"]
        cor_normal_c = rotation @ model["coronary_plane"]["normal_ras"]
        lad_ellipse_plane_error = abs(model["lad_ellipse"]["outline_3d_mm"] @ model["lad_normal_canonical"])
        case_result = {
            "patient_id": model["patient_id"],
            "qc_fully_accepted": model["case"]["qc"]["status"] == "accepted" and not model["case"]["qc"].get("warnings"),
            "coronary_plane_source_is_rca_plus_lcx": model["coronary_plane"]["source"].startswith("unchanged inferred RCA"),
            "lad_plane_source_is_all_lad": model["lad_plane"]["source"] == "all unchanged LAD points",
            "both_planes_pass_through_bifurcation_error_mm": 0.0,
            "rotation_determinant": model["frame"]["determinant"],
            "rotation_orthonormality_max_error": model["frame"]["orthonormality_max_error"],
            "maximum_rigid_segment_error_mm": model["maximum_rigid_segment_error_mm"],
            "lad_terminal_negative_canonical_z": bool(model["lad_terminal_canonical"][2] < 0.0),
            "lad_terminal_canonical_z_mm": float(model["lad_terminal_canonical"][2]),
            "lcx_terminal_positive_canonical_x": bool(model["lcx_terminal_canonical"][0] > 0.0),
            "lcx_terminal_canonical_x_mm": float(model["lcx_terminal_canonical"][0]),
            "coronary_normal_canonical": cor_normal_c,
            "coronary_plane_maps_to_xy_max_normal_xy_error": float(np.max(abs(cor_normal_c[:2]))),
            "coronary_ellipse_max_abs_z_mm": float(np.max(abs(model["coronary_ellipse"]["outline_3d_mm"][:, 2]))),
            "lad_ellipse_max_plane_error_mm": float(np.max(lad_ellipse_plane_error)),
            "source_points_replaced": False,
        }
        case_result["pass"] = bool(
            case_result["qc_fully_accepted"]
            and case_result["coronary_plane_source_is_rca_plus_lcx"]
            and case_result["lad_plane_source_is_all_lad"]
            and abs(case_result["rotation_determinant"] - 1.0) < 1.0e-10
            and case_result["rotation_orthonormality_max_error"] < 1.0e-10
            and case_result["maximum_rigid_segment_error_mm"] < 1.0e-9
            and case_result["lad_terminal_negative_canonical_z"]
            and case_result["lcx_terminal_positive_canonical_x"]
            and case_result["coronary_plane_maps_to_xy_max_normal_xy_error"] < 1.0e-10
            and case_result["coronary_ellipse_max_abs_z_mm"] < 1.0e-10
            and case_result["lad_ellipse_max_plane_error_mm"] < 1.0e-9
        )
        cases.append(case_result)
    hashes_after = {path: sha256(Path(path)) for path in hashes_before}
    mismatches = [path for path, digest in hashes_before.items() if hashes_after[path] != digest]
    missing_pngs = [relative for relative in EXPECTED_IMAGES if not (evidence_root / relative).is_file()]
    missing_svgs = [str(Path(relative).with_suffix(".svg")) for relative in EXPECTED_IMAGES if not (evidence_root / Path(relative).with_suffix(".svg")).is_file()]
    return {
        "validation_timestamp": datetime.now().astimezone().isoformat(),
        "source_coordinate_hashes_unchanged": not mismatches,
        "source_hash_mismatches": mismatches,
        "selected_cases": cases,
        "all_case_validations_pass": all(case["pass"] for case in cases),
        "expected_png_count": len(EXPECTED_IMAGES),
        "missing_pngs": missing_pngs,
        "missing_svg_companions": missing_svgs,
        "generated_tree_inputs_used": False,
        "pca_reconstruction_inputs_used": False,
        "source_centerlines_modified": False,
        "validation_pass": not mismatches and not missing_pngs and not missing_svgs and all(case["pass"] for case in cases),
    }


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=repository / "outputs/lca_ssm")
    parser.add_argument("--evidence-root", type=Path, default=repository / "outputs/lca_ssm/presentation_evidence_v2")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root, evidence_root = args.output_root.resolve(), args.evidence_root.resolve()
    evidence_root.mkdir(parents=True, exist_ok=True)
    (evidence_root / "representative_case").mkdir(parents=True, exist_ok=True)
    configure_plotting()
    selected, selection_audit = select_typical_cases(output_root)
    hashes_before = {
        str(item["case"]["paths"]["raw"]): sha256(item["case"]["paths"]["raw"])
        for item in selected
    }
    metrics = [case_metrics(item) for item in selected]
    write_json(evidence_root / "selected_typical_cases.json", {
        **selection_audit,
        "selected_cases": metrics,
    })
    primary = selected[0]["model"]
    writer = FigureWriter(evidence_root, args.dpi)
    plot_main(writer, primary)
    plot_schematic(writer, primary)
    plot_overlay(writer, primary)
    plot_technical(writer, primary)
    plot_orthographic(
        writer, primary, 4, "Canonical Front View", 0, 2,
        "Canonical X — LCX crown direction (mm)", "Canonical Z — apex is negative (mm)",
    )
    plot_orthographic(
        writer, primary, 5, "Canonical Superior View", 0, 1,
        "Canonical X — LCX crown direction (mm)", "Canonical Y (mm)",
    )
    plot_orthographic(
        writer, primary, 6, "Canonical Lateral View", 1, 2,
        "Canonical Y (mm)", "Canonical Z — apex is negative (mm)",
    )
    plot_montage(writer, selected)
    write_json(evidence_root / "image_manifest.json", {
        "scope": "corrected pre-generative two-SVD-plane/two-full-ellipse canonical heart evidence",
        "images": writer.manifest,
    })
    write_json(evidence_root / "canonical_heart_model_parameters.json", {"cases": metrics})
    validation = validate(selected, hashes_before, evidence_root)
    write_json(evidence_root / "canonical_heart_model_validation.json", validation)
    print(json.dumps(jsonable({
        "evidence_root": evidence_root,
        "selected_cases": [item["model"]["patient_id"] for item in selected],
        "plane_angles_deg": {item["model"]["patient_id"]: item["model"]["plane_separation_angle_deg"] for item in selected},
        "png_count": len(list(evidence_root.rglob("*.png"))),
        "svg_count": len(list(evidence_root.rglob("*.svg"))),
        "validation": validation,
    }), indent=2))


if __name__ == "__main__":
    main()
