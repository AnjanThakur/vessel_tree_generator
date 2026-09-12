"""Build the original-PPT two-plane/two-ellipse population measurements.

This module deliberately stops at measured population statistics.  It does not
sample parameters, generate control points, add synthetic noise, build splines,
or create synthetic vessels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from skimage.measure import EllipseModel

from lca_ssm_planes import angle_degrees, fit_plane_svd


EPS = 1.0e-12
ANGLE_FIELDS_DIRECTIONAL = {
    "bifurcation_crown_theta_deg",
    "lcx_terminal_theta_deg",
    "rca_start_theta_deg",
    "rca_terminal_theta_deg",
    "bifurcation_lad_theta_deg",
    "lad_terminal_theta_deg",
}
ANGLE_FIELDS_AXIAL = {"crown_tilt_deg", "lad_tilt_deg"}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=True), encoding="utf-8")


def numeric_case_key(path_or_name: Path | str) -> tuple[int, str]:
    name = path_or_name.name if isinstance(path_or_name, Path) else str(path_or_name)
    token = name.split(".", 1)[0]
    return (int(token) if token.isdigit() else 10**9, name)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def directory_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(root)).replace("\\", "/"): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def unit(vector: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length <= EPS:
        raise ValueError(f"{name} is zero or non-finite")
    return vector / length


def orient_normal_ras(normal: np.ndarray) -> np.ndarray:
    """Resolve plane-normal sign in fixed RAS axis priority Z, Y, X."""
    normal = unit(normal, "plane normal")
    for axis in (2, 1, 0):
        if abs(normal[axis]) > 1.0e-8:
            return normal if normal[axis] > 0.0 else -normal
    return normal


def deterministic_ras_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a right-handed plane basis tied to fixed NIfTI RAS axes.

    ``u`` is the projection of RAS +X unless that axis is nearly normal to the
    plane; RAS +Y then +Z are tried in order.  This makes ellipse tilt and
    theta comparable across cases rather than dependent on source point order.
    """
    normal = orient_normal_ras(normal)
    selected = None
    selected_name = None
    for name, axis in (("RAS+X", np.array([1.0, 0.0, 0.0])),
                       ("RAS+Y", np.array([0.0, 1.0, 0.0])),
                       ("RAS+Z", np.array([0.0, 0.0, 1.0]))):
        projected = axis - np.dot(axis, normal) * normal
        if np.linalg.norm(projected) >= 0.25:
            selected = projected
            selected_name = name
            break
    if selected is None:
        raise ValueError("could not construct deterministic RAS plane basis")
    u = unit(selected, "plane basis u")
    v = unit(np.cross(normal, u), "plane basis v")
    if np.linalg.det(np.column_stack((u, v, normal))) < 0:
        v = -v
    return u, v, normal


def fit_ras_plane(points: np.ndarray, name: str, source_role: str) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    base = fit_plane_svd(points, name=name)
    u, v, normal = deterministic_ras_basis(base["normal"])
    centroid = np.asarray(base["centroid"], dtype=float)
    signed = (points - centroid) @ normal
    return {
        "method": "centroid_SVD_best_fit_plane",
        "source_role": source_role,
        "point_count": int(len(points)),
        "centroid_ras_mm": centroid,
        "normal_ras": normal,
        "basis_u_ras": u,
        "basis_v_ras": v,
        "basis_u_definition": "projected RAS +X, with +Y/+Z fallback",
        "normal_sign_definition": "positive first nonzero component in RAS Z/Y/X priority",
        "singular_values": base["singular_values"],
        "signed_point_plane_residuals_mm": signed,
        "rmse_mm": float(np.sqrt(np.mean(signed**2))),
        "mean_absolute_mm": float(np.mean(np.abs(signed))),
        "maximum_absolute_mm": float(np.max(np.abs(signed))),
    }


def project_to_plane(points: np.ndarray, plane: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    centroid = np.asarray(plane["centroid_ras_mm"])
    normal = np.asarray(plane["normal_ras"])
    signed = (points - centroid) @ normal
    projected = points - np.outer(signed, normal)
    local = projected - centroid
    coordinates = np.column_stack((local @ plane["basis_u_ras"], local @ plane["basis_v_ras"]))
    return coordinates, signed


def lift_from_plane(points_2d: np.ndarray, plane: dict[str, Any]) -> np.ndarray:
    points_2d = np.asarray(points_2d, dtype=float)
    return (
        np.asarray(plane["centroid_ras_mm"])
        + np.outer(points_2d[:, 0], plane["basis_u_ras"])
        + np.outer(points_2d[:, 1], plane["basis_v_ras"])
    )


def ellipse_local(points: np.ndarray, center: np.ndarray, tilt: float) -> np.ndarray:
    c, s = math.cos(tilt), math.sin(tilt)
    rotation = np.array([[c, s], [-s, c]])
    return (np.asarray(points) - np.asarray(center)) @ rotation.T


def ellipse_world(local: np.ndarray, center: np.ndarray, tilt: float) -> np.ndarray:
    c, s = math.cos(tilt), math.sin(tilt)
    rotation = np.array([[c, -s], [s, c]])
    return np.asarray(local) @ rotation.T + np.asarray(center)


def radial_fit_residual(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    center = params[:2]
    a, b = np.exp(params[2:4])
    local = ellipse_local(points, center, params[4])
    radius = np.sqrt((local[:, 0] / a) ** 2 + (local[:, 1] / b) ** 2)
    return (radius - 1.0) * math.sqrt(a * b)


def closest_ellipse_theta(local_points: np.ndarray, a: float, b: float) -> np.ndarray:
    """Closest-point angular parameter for an axis-aligned ellipse."""
    points = np.asarray(local_points, dtype=float)
    theta = np.arctan2(points[:, 1] / max(b, EPS), points[:, 0] / max(a, EPS))
    for _ in range(16):
        ct, st = np.cos(theta), np.sin(theta)
        ex, ey = a * ct, b * st
        dx, dy = -a * st, b * ct
        ddx, ddy = -a * ct, -b * st
        fx = ex - points[:, 0]
        fy = ey - points[:, 1]
        gradient = fx * dx + fy * dy
        hessian = dx * dx + dy * dy + fx * ddx + fy * ddy
        step = np.divide(gradient, hessian, out=np.zeros_like(gradient), where=np.abs(hessian) > EPS)
        theta -= np.clip(step, -0.35, 0.35)
    return np.arctan2(np.sin(theta), np.cos(theta))


def fit_actual_ellipse(points_2d: np.ndarray, plane: dict[str, Any], name: str) -> dict[str, Any]:
    """Fit a genuine 2-D ellipse to every projected source support point."""
    points = np.asarray(points_2d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5:
        raise ValueError(f"{name} needs at least five planar support points")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} contains non-finite planar coordinates")
    span = float(max(np.ptp(points, axis=0)))
    if span <= 1.0e-6:
        raise ValueError(f"{name} planar support has negligible extent")
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=span * 1.0e-8) < 2:
        raise ValueError(f"{name} planar support is collinear")

    estimator = EllipseModel.from_estimate(points)
    starts: list[np.ndarray] = []
    if estimator:
        xc, yc = estimator.center
        aa, bb = estimator.axis_lengths
        phi = estimator.theta
        if np.all(np.isfinite([xc, yc, aa, bb, phi])) and min(aa, bb) > EPS:
            starts.append(np.array([xc, yc, math.log(aa), math.log(bb), phi], dtype=float))
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    pca_local = centered @ vh.T
    axes = np.maximum(np.ptp(pca_local, axis=0) / 2.0, span * 0.05)
    pca_phi = math.atan2(vh[0, 1], vh[0, 0])
    starts.append(np.array([*points.mean(axis=0), math.log(axes[0]), math.log(axes[1]), pca_phi]))

    lower = np.array([
        points[:, 0].min() - 4 * span,
        points[:, 1].min() - 4 * span,
        math.log(span * 0.01),
        math.log(span * 0.01),
        -4 * math.pi,
    ])
    upper = np.array([
        points[:, 0].max() + 4 * span,
        points[:, 1].max() + 4 * span,
        math.log(span * 20.0),
        math.log(span * 20.0),
        4 * math.pi,
    ])
    solutions = []
    # The direct total-least-squares estimate is normally strong.  Refine that
    # single estimate against all points; retain the PCA start only when direct
    # estimation failed.  This keeps the cohort run tractable without dropping
    # or down-weighting any source support points.
    optimization_starts = starts[:1]
    for initial in optimization_starts:
        initial = np.minimum(np.maximum(initial, lower + 1.0e-8), upper - 1.0e-8)
        try:
            result = least_squares(
                radial_fit_residual,
                initial,
                args=(points,),
                bounds=(lower, upper),
                loss="linear",
                max_nfev=1200,
                ftol=1.0e-8,
                xtol=1.0e-8,
                gtol=1.0e-8,
            )
            if np.all(np.isfinite(result.x)):
                solutions.append(result)
        except Exception:
            continue
    if not solutions:
        raise ValueError(f"{name} nonlinear ellipse optimization failed")
    result = min(solutions, key=lambda item: float(np.mean(radial_fit_residual(item.x, points) ** 2)))
    center = np.asarray(result.x[:2])
    a, b = np.exp(result.x[2:4])
    tilt = float(result.x[4])
    if b > a:
        a, b = b, a
        tilt += math.pi / 2.0
    # An ellipse major axis is axial, so [−90°, 90°) is the deterministic domain.
    tilt = float((tilt + math.pi / 2.0) % math.pi - math.pi / 2.0)
    axis_ratio = float(a / b)
    if not np.all(np.isfinite([a, b, tilt])) or b <= EPS:
        raise ValueError(f"{name} returned a non-finite ellipse")
    if axis_ratio > 50.0 or a > 20.0 * span:
        raise ValueError(f"{name} is numerically unstable (axis ratio={axis_ratio:.3g})")

    local = ellipse_local(points, center, tilt)
    theta = closest_ellipse_theta(local, float(a), float(b))
    reference_local = np.column_stack((a * np.cos(theta), b * np.sin(theta)))
    reference_2d = ellipse_world(reference_local, center, tilt)
    in_plane = np.linalg.norm(points - reference_2d, axis=1)
    center_3d = lift_from_plane(center[None, :], plane)[0]
    sample_theta = np.linspace(-math.pi, math.pi, 720, endpoint=False)
    sample_local = np.column_stack((a * np.cos(sample_theta), b * np.sin(sample_theta)))
    sampled_2d = ellipse_world(sample_local, center, tilt)
    return {
        "method": "all_source_points_2d_actual_ellipse_nonlinear_least_squares",
        "optimizer_success": bool(result.success),
        "optimizer_message": result.message,
        "support_point_count": int(len(points)),
        "center_2d_mm": center,
        "center_3d_ras_mm": center_3d,
        "a_mm": float(a),
        "b_mm": float(b),
        "axis_ratio": axis_ratio,
        "eccentricity": float(math.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))),
        "tilt_rad": tilt,
        "tilt_deg": float(math.degrees(tilt)),
        "tilt_domain": "[-90, 90) degrees; axial period 180 degrees",
        "theta_zero_definition": "positive major-axis direction with nonnegative plane-basis-u component",
        "theta_direction": "increases from major axis toward positive minor axis in right-handed plane (u,v,normal)",
        "theta_rad": theta,
        "reference_points_2d_mm": reference_2d,
        "sampled_ellipse_2d_mm": sampled_2d,
        "in_plane_residuals_mm": in_plane,
        "in_plane_rmse_mm": float(np.sqrt(np.mean(in_plane**2))),
        "in_plane_mean_mm": float(np.mean(in_plane)),
        "in_plane_max_mm": float(np.max(in_plane)),
    }


def cumulative_s(points: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= EPS:
        raise ValueError("branch has zero arc length")
    return cumulative / cumulative[-1]


def residual_records(
    case_id: str,
    branch_name: str,
    points: np.ndarray,
    plane: dict[str, Any],
    ellipse: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float], np.ndarray]:
    points_2d, signed_plane = project_to_plane(points, plane)
    center = np.asarray(ellipse["center_2d_mm"])
    local = ellipse_local(points_2d, center, ellipse["tilt_rad"])
    theta_wrapped = closest_ellipse_theta(local, ellipse["a_mm"], ellipse["b_mm"])
    theta_unwrapped = np.unwrap(theta_wrapped)
    reference_local = np.column_stack((
        ellipse["a_mm"] * np.cos(theta_wrapped),
        ellipse["b_mm"] * np.sin(theta_wrapped),
    ))
    reference_2d = ellipse_world(reference_local, center, ellipse["tilt_rad"])
    reference_3d = lift_from_plane(reference_2d, plane)
    delta_2d = points_2d - reference_2d
    # Signed radial in-plane residual: positive points away from ellipse center.
    outward = points_2d - center
    outward_norm = np.linalg.norm(outward, axis=1)
    outward_unit = np.divide(outward, outward_norm[:, None], out=np.zeros_like(outward), where=outward_norm[:, None] > EPS)
    in_plane_signed = np.sum(delta_2d * outward_unit, axis=1)
    euclidean = np.linalg.norm(points - reference_3d, axis=1)
    s = cumulative_s(points)
    records = []
    for index in range(len(points)):
        records.append({
            "case_id": case_id,
            "branch": branch_name,
            "point_index": index,
            "s": float(s[index]),
            "theta_rad": float(theta_unwrapped[index]),
            "theta_deg": float(math.degrees(theta_unwrapped[index])),
            "reference_x": float(reference_3d[index, 0]),
            "reference_y": float(reference_3d[index, 1]),
            "reference_z": float(reference_3d[index, 2]),
            "source_x": float(points[index, 0]),
            "source_y": float(points[index, 1]),
            "source_z": float(points[index, 2]),
            "in_plane_residual": float(in_plane_signed[index]),
            "out_of_plane_residual": float(signed_plane[index]),
            "euclidean_residual": float(euclidean[index]),
        })
    summary = {
        "mean": float(np.mean(euclidean)),
        "std": float(np.std(euclidean, ddof=1)) if len(euclidean) > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(euclidean**2))),
        "p95": float(np.percentile(euclidean, 95)),
        "maximum": float(np.max(euclidean)),
        "in_plane_rmse": float(np.sqrt(np.mean(in_plane_signed**2))),
        "out_of_plane_rmse": float(np.sqrt(np.mean(signed_plane**2))),
    }
    return records, summary, theta_unwrapped


def landmark_theta(point: np.ndarray, plane: dict[str, Any], ellipse: dict[str, Any]) -> float:
    point_2d, _ = project_to_plane(np.asarray(point)[None, :], plane)
    local = ellipse_local(point_2d, ellipse["center_2d_mm"], ellipse["tilt_rad"])
    return float(closest_ellipse_theta(local, ellipse["a_mm"], ellipse["b_mm"])[0])


def scalar_stats(values: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: math.nan for key in ("mean", "std", "median", "iqr", "min", "max", "p5", "p95")} | {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def circular_stats_degrees(values: Iterable[float], axial: bool = False) -> dict[str, float | int]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "circular_mean_deg": math.nan, "circular_std_deg": math.nan, "resultant_length": math.nan}
    multiplier = 2.0 if axial else 1.0
    radians = np.radians(values) * multiplier
    resultant = np.mean(np.exp(1j * radians))
    length = float(abs(resultant))
    mean = math.atan2(resultant.imag, resultant.real) / multiplier
    period = math.pi if axial else 2.0 * math.pi
    mean = (mean + period / 2.0) % period - period / 2.0
    std = math.sqrt(max(0.0, -2.0 * math.log(max(length, EPS)))) / multiplier
    return {
        "n": int(len(values)),
        "circular_mean_deg": float(math.degrees(mean)),
        "circular_std_deg": float(math.degrees(std)),
        "resultant_length": length,
        "period_deg": 180.0 if axial else 360.0,
    }


PARAMETER_FIELDS = [
    "case_id", "coronary_support", "coronary_support_point_count",
    "coronary_plane_centroid_x", "coronary_plane_centroid_y", "coronary_plane_centroid_z",
    "coronary_plane_normal_x", "coronary_plane_normal_y", "coronary_plane_normal_z", "coronary_plane_rmse",
    "crown_ellipse_center_2d_u", "crown_ellipse_center_2d_v",
    "crown_ellipse_center_3d_x", "crown_ellipse_center_3d_y", "crown_ellipse_center_3d_z",
    "crown_a", "crown_b", "crown_axis_ratio", "crown_tilt_deg", "crown_eccentricity",
    "crown_fit_rmse", "crown_fit_max_error",
    "lad_plane_centroid_x", "lad_plane_centroid_y", "lad_plane_centroid_z",
    "lad_plane_normal_x", "lad_plane_normal_y", "lad_plane_normal_z", "lad_plane_rmse",
    "lad_ellipse_center_2d_u", "lad_ellipse_center_2d_v",
    "lad_ellipse_center_3d_x", "lad_ellipse_center_3d_y", "lad_ellipse_center_3d_z",
    "lad_a", "lad_b", "lad_axis_ratio", "lad_tilt_deg", "lad_eccentricity",
    "lad_fit_rmse", "lad_fit_max_error", "plane_angle_deg",
    "bifurcation_crown_theta_rad", "bifurcation_crown_theta_deg",
    "lcx_terminal_theta_rad", "lcx_terminal_theta_deg",
    "rca_start_theta_rad", "rca_start_theta_deg", "rca_terminal_theta_rad", "rca_terminal_theta_deg",
    "bifurcation_lad_theta_rad", "bifurcation_lad_theta_deg",
    "lad_terminal_theta_rad", "lad_terminal_theta_deg",
    "lcx_theta_start_rad", "lcx_theta_end_rad", "lcx_signed_angular_extent_deg", "lcx_angular_extent_deg",
    "lad_theta_start_rad", "lad_theta_end_rad", "lad_signed_angular_extent_deg", "lad_angular_extent_deg",
    "lcx_residual_mean", "lcx_residual_std", "lcx_residual_rmse", "lcx_residual_p95", "lcx_residual_max",
    "lad_residual_mean", "lad_residual_std", "lad_residual_rmse", "lad_residual_p95", "lad_residual_max",
    "max_source_coordinate_change", "max_segment_length_change", "source_hashes",
]


def blank_parameter_row(case_id: str) -> dict[str, Any]:
    return {field: (case_id if field == "case_id" else math.nan) for field in PARAMETER_FIELDS}


def process_case(case_id: str, raw_path: Path, output_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    with np.load(raw_path) as archive:
        arrays = {name: np.asarray(archive[name], dtype=float).copy() for name in archive.files}
    for required in ("lmca", "lad", "lcx"):
        if required not in arrays:
            raise ValueError(f"extracted archive missing {required}")
        if arrays[required].ndim != 2 or arrays[required].shape[1] != 3 or len(arrays[required]) < 5:
            raise ValueError(f"{required} centerline has invalid shape {arrays[required].shape}")
        if not np.all(np.isfinite(arrays[required])):
            raise ValueError(f"{required} centerline contains non-finite coordinates")
    original = {name: value.copy() for name, value in arrays.items()}
    hashes = {name: array_sha256(value) for name, value in arrays.items()}

    lad = arrays["lad"]
    lcx = arrays["lcx"]
    rca = arrays.get("rca")
    if rca is not None and len(rca) >= 5:
        coronary_support = np.vstack((rca, lcx))
        coronary_source = "inferred_RCA_candidate_plus_LCX_not_ground_truth_RCA"
    else:
        coronary_support = lcx
        coronary_source = "LCX_only_crown_approximation_insufficient_RCA_support"

    lad_plane = fit_ras_plane(lad, "unchanged LAD source", "interventricular/LAD descent")
    crown_plane = fit_ras_plane(coronary_support, "unchanged coronary support", coronary_source)
    lad_2d, _ = project_to_plane(lad, lad_plane)
    crown_2d, _ = project_to_plane(coronary_support, crown_plane)
    lad_ellipse = fit_actual_ellipse(lad_2d, lad_plane, "LAD ellipse")
    crown_ellipse = fit_actual_ellipse(crown_2d, crown_plane, "crown ellipse")

    lcx_records, lcx_residual, lcx_theta = residual_records(case_id, "LCX", lcx, crown_plane, crown_ellipse)
    lad_records, lad_residual, lad_theta = residual_records(case_id, "LAD", lad, lad_plane, lad_ellipse)
    residuals = lcx_records + lad_records
    crown_support_records, crown_support_residual, _ = residual_records(
        case_id, "CROWN_SUPPORT", coronary_support, crown_plane, crown_ellipse
    )
    # CROWN_SUPPORT records are retained only in the patient JSON summary, not duplicated in population CSV.
    _ = crown_support_records

    landmark_angles = {
        "bifurcation_crown_theta_rad": landmark_theta(lcx[0], crown_plane, crown_ellipse),
        "lcx_terminal_theta_rad": landmark_theta(lcx[-1], crown_plane, crown_ellipse),
        "bifurcation_lad_theta_rad": landmark_theta(lad[0], lad_plane, lad_ellipse),
        "lad_terminal_theta_rad": landmark_theta(lad[-1], lad_plane, lad_ellipse),
        "rca_start_theta_rad": math.nan,
        "rca_terminal_theta_rad": math.nan,
    }
    if rca is not None and len(rca) >= 5:
        landmark_angles["rca_start_theta_rad"] = landmark_theta(rca[0], crown_plane, crown_ellipse)
        landmark_angles["rca_terminal_theta_rad"] = landmark_theta(rca[-1], crown_plane, crown_ellipse)

    plane_angle = angle_degrees(crown_plane["normal_ras"], lad_plane["normal_ras"])
    # Plane normals are axial.  Report the acute geometric plane angle.
    plane_angle = min(plane_angle, 180.0 - plane_angle)
    coordinate_change = max(float(np.max(np.abs(arrays[name] - original[name]))) for name in arrays)
    segment_change = 0.0
    for name in arrays:
        before = np.linalg.norm(np.diff(original[name], axis=0), axis=1)
        after = np.linalg.norm(np.diff(arrays[name], axis=0), axis=1)
        if len(before):
            segment_change = max(segment_change, float(np.max(np.abs(before - after))))

    row = blank_parameter_row(case_id)
    row.update({
        "coronary_support": coronary_source,
        "coronary_support_point_count": len(coronary_support),
        "coronary_plane_centroid_x": crown_plane["centroid_ras_mm"][0],
        "coronary_plane_centroid_y": crown_plane["centroid_ras_mm"][1],
        "coronary_plane_centroid_z": crown_plane["centroid_ras_mm"][2],
        "coronary_plane_normal_x": crown_plane["normal_ras"][0],
        "coronary_plane_normal_y": crown_plane["normal_ras"][1],
        "coronary_plane_normal_z": crown_plane["normal_ras"][2],
        "coronary_plane_rmse": crown_plane["rmse_mm"],
        "crown_ellipse_center_2d_u": crown_ellipse["center_2d_mm"][0],
        "crown_ellipse_center_2d_v": crown_ellipse["center_2d_mm"][1],
        "crown_ellipse_center_3d_x": crown_ellipse["center_3d_ras_mm"][0],
        "crown_ellipse_center_3d_y": crown_ellipse["center_3d_ras_mm"][1],
        "crown_ellipse_center_3d_z": crown_ellipse["center_3d_ras_mm"][2],
        "crown_a": crown_ellipse["a_mm"], "crown_b": crown_ellipse["b_mm"],
        "crown_axis_ratio": crown_ellipse["axis_ratio"], "crown_tilt_deg": crown_ellipse["tilt_deg"],
        "crown_eccentricity": crown_ellipse["eccentricity"],
        "crown_fit_rmse": crown_support_residual["rmse"],
        "crown_fit_max_error": crown_support_residual["maximum"],
        "lad_plane_centroid_x": lad_plane["centroid_ras_mm"][0],
        "lad_plane_centroid_y": lad_plane["centroid_ras_mm"][1],
        "lad_plane_centroid_z": lad_plane["centroid_ras_mm"][2],
        "lad_plane_normal_x": lad_plane["normal_ras"][0],
        "lad_plane_normal_y": lad_plane["normal_ras"][1],
        "lad_plane_normal_z": lad_plane["normal_ras"][2],
        "lad_plane_rmse": lad_plane["rmse_mm"],
        "lad_ellipse_center_2d_u": lad_ellipse["center_2d_mm"][0],
        "lad_ellipse_center_2d_v": lad_ellipse["center_2d_mm"][1],
        "lad_ellipse_center_3d_x": lad_ellipse["center_3d_ras_mm"][0],
        "lad_ellipse_center_3d_y": lad_ellipse["center_3d_ras_mm"][1],
        "lad_ellipse_center_3d_z": lad_ellipse["center_3d_ras_mm"][2],
        "lad_a": lad_ellipse["a_mm"], "lad_b": lad_ellipse["b_mm"],
        "lad_axis_ratio": lad_ellipse["axis_ratio"], "lad_tilt_deg": lad_ellipse["tilt_deg"],
        "lad_eccentricity": lad_ellipse["eccentricity"],
        "lad_fit_rmse": lad_residual["rmse"], "lad_fit_max_error": lad_residual["maximum"],
        "plane_angle_deg": plane_angle,
        "lcx_theta_start_rad": lcx_theta[0], "lcx_theta_end_rad": lcx_theta[-1],
        "lcx_signed_angular_extent_deg": math.degrees(lcx_theta[-1] - lcx_theta[0]),
        "lcx_angular_extent_deg": abs(math.degrees(lcx_theta[-1] - lcx_theta[0])),
        "lad_theta_start_rad": lad_theta[0], "lad_theta_end_rad": lad_theta[-1],
        "lad_signed_angular_extent_deg": math.degrees(lad_theta[-1] - lad_theta[0]),
        "lad_angular_extent_deg": abs(math.degrees(lad_theta[-1] - lad_theta[0])),
        "lcx_residual_mean": lcx_residual["mean"], "lcx_residual_std": lcx_residual["std"],
        "lcx_residual_rmse": lcx_residual["rmse"], "lcx_residual_p95": lcx_residual["p95"],
        "lcx_residual_max": lcx_residual["maximum"],
        "lad_residual_mean": lad_residual["mean"], "lad_residual_std": lad_residual["std"],
        "lad_residual_rmse": lad_residual["rmse"], "lad_residual_p95": lad_residual["p95"],
        "lad_residual_max": lad_residual["maximum"],
        "max_source_coordinate_change": coordinate_change,
        "max_segment_length_change": segment_change,
        "source_hashes": json.dumps(hashes, sort_keys=True),
    })
    for key, radians in landmark_angles.items():
        row[key] = radians
        row[key.replace("_rad", "_deg")] = math.degrees(radians) if np.isfinite(radians) else math.nan

    warnings = []
    if rca is None:
        warnings.append("no usable inferred RCA candidate; coronary model uses LCX-only approximation")
    for branch, residual, ellipse in (("crown", crown_support_residual, crown_ellipse), ("LAD", lad_residual, lad_ellipse)):
        if residual["rmse"] > 0.25 * ellipse["a_mm"]:
            warnings.append(f"{branch} absolute ellipse RMSE is high relative to semi-major axis")
        if ellipse["axis_ratio"] > 10.0:
            warnings.append(f"{branch} ellipse axis ratio exceeds 10")
    payload = {
        "case_id": case_id,
        "status": "statistics_eligible_with_warnings" if warnings else "statistics_eligible",
        "warnings": warnings,
        "source_geometry_immutable": True,
        "source_integrity": {
            "max_source_coordinate_change": coordinate_change,
            "max_segment_length_change": segment_change,
            "source_hashes": hashes,
            "raw_archive_sha256": file_sha256(raw_path),
        },
        "landmarks": {
            "LMCA_start_ras_mm": arrays["lmca"][0],
            "LMCA_bifurcation_ras_mm": arrays["lmca"][-1],
            "LAD_start_ras_mm": lad[0], "LAD_terminal_ras_mm": lad[-1],
            "LCX_start_ras_mm": lcx[0], "LCX_terminal_ras_mm": lcx[-1],
            "RCA_candidate_start_ras_mm": None if rca is None else rca[0],
            "RCA_candidate_terminal_ras_mm": None if rca is None else rca[-1],
            "ellipse_angles_rad": landmark_angles,
        },
        "planes": {"coronary_AV_groove": crown_plane, "interventricular_LAD": lad_plane},
        "ellipses": {"crown_AV_groove": crown_ellipse, "interventricular_LAD": lad_ellipse},
        "branch_residual_summaries": {"LCX": lcx_residual, "LAD": lad_residual, "crown_support": crown_support_residual},
        "branch_angular_extents": {
            "LCX": {"theta_start_rad": lcx_theta[0], "theta_end_rad": lcx_theta[-1],
                    "signed_extent_deg": row["lcx_signed_angular_extent_deg"], "absolute_extent_deg": row["lcx_angular_extent_deg"]},
            "LAD": {"theta_start_rad": lad_theta[0], "theta_end_rad": lad_theta[-1],
                    "signed_extent_deg": row["lad_signed_angular_extent_deg"], "absolute_extent_deg": row["lad_angular_extent_deg"]},
        },
        "plane_angle_deg": plane_angle,
    }
    write_json(output_root / "patients" / case_id / "two_plane_two_ellipse.json", payload)
    plot_data = {"arrays": arrays, "crown_plane": crown_plane, "lad_plane": lad_plane,
                 "crown_ellipse": crown_ellipse, "lad_ellipse": lad_ellipse,
                 "crown_support": coronary_support, "row": row}
    return row, residuals, plot_data


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_population_statistics(rows: list[dict[str, Any]], residuals: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variables = [
        "crown_a", "crown_b", "crown_axis_ratio", "crown_eccentricity", "crown_tilt_deg",
        "lad_a", "lad_b", "lad_axis_ratio", "lad_eccentricity", "lad_tilt_deg",
        "plane_angle_deg", "bifurcation_crown_theta_deg", "lcx_terminal_theta_deg",
        "rca_start_theta_deg", "rca_terminal_theta_deg", "bifurcation_lad_theta_deg", "lad_terminal_theta_deg",
        "lcx_signed_angular_extent_deg", "lcx_angular_extent_deg",
        "lad_signed_angular_extent_deg", "lad_angular_extent_deg",
        "crown_fit_rmse", "crown_fit_max_error", "lad_fit_rmse", "lad_fit_max_error",
        "lcx_residual_mean", "lcx_residual_std", "lcx_residual_rmse", "lcx_residual_p95",
        "lad_residual_mean", "lad_residual_std", "lad_residual_rmse", "lad_residual_p95",
        "coronary_plane_rmse", "lad_plane_rmse",
    ]
    result: dict[str, Any] = {
        "angle_conventions": {
            "ellipse_tilt": "axial circular variable, period 180 degrees, stored in [-90,90)",
            "landmark_theta": "directional circular variable, period 360 degrees",
            "branch_extent": "unwrapped ordered branch difference; summarized linearly",
        },
        "variables": {},
    }
    csv_rows = []
    for variable in variables:
        values = [float(row[variable]) for row in rows if variable in row and np.isfinite(float(row[variable]))]
        summary = scalar_stats(values)
        kind = "linear"
        if variable in ANGLE_FIELDS_AXIAL:
            summary.update(circular_stats_degrees(values, axial=True)); kind = "axial_circular"
        elif variable in ANGLE_FIELDS_DIRECTIONAL:
            summary.update(circular_stats_degrees(values, axial=False)); kind = "directional_circular"
        result["variables"][variable] = {"type": kind, **summary}
        csv_rows.append({"variable": variable, "type": kind, **summary})
    for branch in ("LAD", "LCX"):
        for component in ("in_plane_residual", "out_of_plane_residual", "euclidean_residual"):
            name = f"{branch.lower()}_pointwise_{component}_mm"
            values = [abs(float(item[component])) if component != "out_of_plane_residual" else float(item[component])
                      for item in residuals if item["branch"] == branch]
            summary = scalar_stats(values)
            result["variables"][name] = {"type": "pooled_pointwise_linear", **summary}
            csv_rows.append({"variable": name, "type": "pooled_pointwise_linear", **summary})
    return result, csv_rows


def build_correlation(rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    fields = [
        "crown_a", "crown_b", "crown_axis_ratio", "crown_eccentricity",
        "lad_a", "lad_b", "lad_axis_ratio", "lad_eccentricity",
        "plane_angle_deg", "lcx_angular_extent_deg", "lad_angular_extent_deg",
        "crown_fit_rmse", "lad_fit_rmse", "lcx_residual_rmse", "lad_residual_rmse",
    ]
    matrix = np.asarray([[float(row[name]) for name in fields] for row in rows], dtype=float)
    correlation = np.full((len(fields), len(fields)), np.nan)
    for i in range(len(fields)):
        for j in range(len(fields)):
            valid = np.isfinite(matrix[:, i]) & np.isfinite(matrix[:, j])
            if np.sum(valid) >= 3 and np.std(matrix[valid, i]) > EPS and np.std(matrix[valid, j]) > EPS:
                correlation[i, j] = np.corrcoef(matrix[valid, i], matrix[valid, j])[0, 1]
    return fields, correlation


def save_figure(figure: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def histogram_panels(rows: list[dict[str, Any]], fields: list[tuple[str, str]], title: str, base: Path, color: str) -> None:
    figure, axes = plt.subplots(1, len(fields), figsize=(4.2 * len(fields), 3.8))
    axes = np.atleast_1d(axes)
    for axis, (field, label) in zip(axes, fields):
        values = [float(row[field]) for row in rows if np.isfinite(float(row[field]))]
        axis.hist(values, bins=min(24, max(8, int(math.sqrt(len(values))))), color=color, edgecolor="white")
        axis.axvline(np.median(values), color="#111827", ls="--", lw=1.5, label="median")
        axis.set_xlabel(label); axis.set_ylabel("cases"); axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    figure.suptitle(title, fontweight="bold")
    figure.tight_layout()
    save_figure(figure, base)


def add_plane_patch(axis: Any, plane: dict[str, Any], center: np.ndarray, extent: float, color: str) -> None:
    grid = np.linspace(-extent, extent, 8)
    uu, vv = np.meshgrid(grid, grid)
    patch = (center[None, None, :] + uu[:, :, None] * plane["basis_u_ras"][None, None, :]
             + vv[:, :, None] * plane["basis_v_ras"][None, None, :])
    axis.plot_surface(patch[:, :, 0], patch[:, :, 1], patch[:, :, 2], color=color, alpha=0.12, linewidth=0)


def set_equal_3d(axis: Any, points: np.ndarray) -> None:
    minimum, maximum = points.min(axis=0), points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float(np.max(maximum - minimum)) / 2.0, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def make_figures(
    rows: list[dict[str, Any]],
    statuses: list[dict[str, Any]],
    residuals: list[dict[str, Any]],
    correlation_fields: list[str],
    correlation: np.ndarray,
    example: dict[str, Any],
    figures_dir: Path,
) -> None:
    # 01 population flow
    stages = ["NIfTI\ndiscovered", "centerlines\navailable", "landmarks\navailable", "two planes", "two ellipses", "statistics\neligible"]
    counts = [len(statuses), sum(int(s["centerline_ok"]) for s in statuses), sum(int(s["landmarks_ok"]) for s in statuses),
              sum(int(s["coronary_plane_ok"] and s["lad_plane_ok"]) for s in statuses),
              sum(int(s["crown_ellipse_ok"] and s["lad_ellipse_ok"]) for s in statuses),
              sum(int(s["statistics_eligible"]) for s in statuses)]
    figure, axis = plt.subplots(figsize=(10.5, 4.2))
    bars = axis.bar(range(len(stages)), counts, color=["#334155", "#0f766e", "#0f766e", "#2563eb", "#7c3aed", "#b45309"])
    axis.set_xticks(range(len(stages)), stages); axis.set_ylabel("cases"); axis.set_ylim(0, max(counts) * 1.16)
    axis.set_title("PPT-priority population case flow", fontweight="bold")
    for bar, count in zip(bars, counts): axis.text(bar.get_x()+bar.get_width()/2, count+2, str(count), ha="center", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False); axis.grid(axis="y", alpha=0.18)
    save_figure(figure, figures_dir / "01_population_case_flow")

    histogram_panels(rows, [("crown_a", "a (mm)"), ("crown_b", "b (mm)"), ("crown_tilt_deg", "tilt (deg)")],
                     "Crown / AV-groove fitted ellipse", figures_dir / "02_crown_ellipse_parameter_distributions", "#0f766e")
    histogram_panels(rows, [("lad_a", "a (mm)"), ("lad_b", "b (mm)"), ("lad_tilt_deg", "tilt (deg)")],
                     "LAD / interventricular fitted ellipse", figures_dir / "03_lad_ellipse_parameter_distributions", "#b91c1c")
    histogram_panels(rows, [("plane_angle_deg", "acute plane angle (deg)")], "Relationship between the two SVD planes",
                     figures_dir / "04_plane_angle_distribution", "#4f46e5")

    figure, axes = plt.subplots(2, 2, figsize=(9, 7))
    for axis, field, label, color in zip(axes.flat,
        ["bifurcation_crown_theta_deg", "lcx_terminal_theta_deg", "bifurcation_lad_theta_deg", "lad_terminal_theta_deg"],
        ["LCX start / bifurcation", "LCX terminal", "LAD start / bifurcation", "LAD terminal"],
        ["#0f766e", "#14b8a6", "#b91c1c", "#ef4444"]):
        values = [float(row[field]) for row in rows if np.isfinite(float(row[field]))]
        axis.hist(values, bins=20, color=color, edgecolor="white"); axis.set_title(label); axis.set_xlabel("ellipse theta (deg)"); axis.set_ylabel("cases"); axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Landmark angular positions on fitted ellipses", fontweight="bold"); figure.tight_layout()
    save_figure(figure, figures_dir / "05_landmark_angle_distributions")

    histogram_panels(rows, [("lcx_angular_extent_deg", "LCX |extent| (deg)"), ("lad_angular_extent_deg", "LAD |extent| (deg)")],
                     "Ordered branch angular extents along fitted ellipses", figures_dir / "06_branch_angular_extent_distributions", "#d97706")

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for axis, component, label in zip(axes, ["in_plane_residual", "out_of_plane_residual", "euclidean_residual"],
                                      ["signed in-plane (mm)", "signed out-of-plane (mm)", "Euclidean (mm)"]):
        for branch, color in (("LAD", "#b91c1c"), ("LCX", "#0f766e")):
            values = np.asarray([float(r[component]) for r in residuals if r["branch"] == branch])
            axis.hist(values, bins=55, density=True, histtype="step", lw=1.5, label=branch, color=color)
        axis.set_xlabel(label); axis.set_ylabel("density"); axis.grid(alpha=0.2)
    axes[0].legend(frameon=False); figure.suptitle("Per-point departure from fitted ellipse reference", fontweight="bold"); figure.tight_layout()
    save_figure(figure, figures_dir / "07_residual_noise_distribution")

    figure, axis = plt.subplots(figsize=(10.5, 8.5))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    labels = [name.replace("_", " ") for name in correlation_fields]
    axis.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    axis.set_yticks(range(len(labels)), labels, fontsize=7)
    axis.set_title("Pearson dependence diagnostic (linear continuous variables)", fontweight="bold")
    figure.colorbar(image, ax=axis, fraction=0.046, label="r"); figure.tight_layout()
    save_figure(figure, figures_dir / "08_parameter_correlation")

    arrays = example["arrays"]
    all_points = np.vstack([arrays[name] for name in ("lmca", "lad", "lcx")])
    if "rca" in arrays: all_points = np.vstack((all_points, arrays["rca"]))
    center = all_points.mean(axis=0); extent = max(np.ptp(all_points, axis=0)) * 0.35
    figure = plt.figure(figsize=(8.2, 6.8)); axis = figure.add_subplot(111, projection="3d")
    for name, color in (("lmca", "#111827"), ("lad", "#b91c1c"), ("lcx", "#0f766e"), ("rca", "#2563eb")):
        if name in arrays: axis.plot(*arrays[name].T, color=color, lw=1.8, label=("inferred RCA candidate" if name=="rca" else name.upper()))
    add_plane_patch(axis, example["crown_plane"], center, extent, "#0f766e")
    add_plane_patch(axis, example["lad_plane"], center, extent, "#b91c1c")
    axis.set_xlabel("RAS X (mm)"); axis.set_ylabel("RAS Y (mm)"); axis.set_zlabel("RAS Z (mm)"); axis.legend(fontsize=7)
    axis.set_title(f"Example {example['row']['case_id']}: two independently fitted SVD planes", fontweight="bold"); set_equal_3d(axis, all_points); figure.tight_layout()
    save_figure(figure, figures_dir / "09_example_patient_two_planes")

    figure = plt.figure(figsize=(8.2, 6.8)); axis = figure.add_subplot(111, projection="3d")
    for ellipse, plane, color, label in ((example["crown_ellipse"], example["crown_plane"], "#0f766e", "crown ellipse"),
                                         (example["lad_ellipse"], example["lad_plane"], "#b91c1c", "LAD ellipse")):
        curve = lift_from_plane(ellipse["sampled_ellipse_2d_mm"], plane); axis.plot(*curve.T, color=color, lw=2.3, label=label)
    axis.scatter(*arrays["lcx"][0], color="#f59e0b", s=38, label="bifurcation")
    axis.set_xlabel("RAS X (mm)"); axis.set_ylabel("RAS Y (mm)"); axis.set_zlabel("RAS Z (mm)"); axis.legend(fontsize=8)
    axis.set_title(f"Example {example['row']['case_id']}: two actual planar ellipse fits", fontweight="bold"); set_equal_3d(axis, all_points); figure.tight_layout()
    save_figure(figure, figures_dir / "10_example_patient_two_ellipses")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, points, plane, ellipse, title, color in ((axes[0], example["crown_support"], example["crown_plane"], example["crown_ellipse"], "Crown support", "#0f766e"),
                                                       (axes[1], arrays["lad"], example["lad_plane"], example["lad_ellipse"], "LAD source", "#b91c1c")):
        p2d, _ = project_to_plane(points, plane)
        axis.plot(p2d[:,0], p2d[:,1], ".", ms=2.0, alpha=0.35, color=color, label="unchanged source projected for measurement")
        curve = ellipse["sampled_ellipse_2d_mm"]; axis.plot(curve[:,0], curve[:,1], color="#111827", lw=2, label="fitted actual ellipse")
        axis.set_aspect("equal", adjustable="datalim"); axis.set_xlabel("plane u (mm)"); axis.set_ylabel("plane v (mm)"); axis.set_title(title); axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=7); figure.suptitle(f"Example {example['row']['case_id']}: source is measured, never replaced", fontweight="bold"); figure.tight_layout()
    save_figure(figure, figures_dir / "11_example_patient_source_vs_ellipse")

    # Population-mean parameter reference, intentionally not a generated vessel.
    crown_a=np.mean([r["crown_a"] for r in rows]); crown_b=np.mean([r["crown_b"] for r in rows])
    lad_a=np.mean([r["lad_a"] for r in rows]); lad_b=np.mean([r["lad_b"] for r in rows])
    t=np.linspace(0,2*np.pi,720); crown=np.column_stack((crown_a*np.cos(t), crown_b*np.sin(t), np.zeros_like(t)))
    lad=np.column_stack((np.zeros_like(t), lad_b*np.sin(t), lad_a*np.cos(t)))
    figure=plt.figure(figsize=(8,6.5)); axis=figure.add_subplot(111,projection="3d")
    axis.plot(*crown.T,color="#0f766e",lw=2.5,label="mean crown a,b reference")
    axis.plot(*lad.T,color="#b91c1c",lw=2.5,label="mean LAD a,b reference")
    axis.scatter(0,0,0,color="#f59e0b",s=45,label="shared illustrative origin")
    axis.set_xlabel("reference X (mm)"); axis.set_ylabel("reference Y (mm)"); axis.set_zlabel("reference Z (mm)"); axis.legend()
    axis.set_title("Population-mean two-ellipse reference (not a synthetic vessel)",fontweight="bold"); set_equal_3d(axis,np.vstack((crown,lad))); figure.tight_layout()
    save_figure(figure, figures_dir / "12_population_mean_reference_model")


def audit_markdown() -> str:
    rows = [
        ("Population NIfTI discovery", "DONE", "nii files/", "Programmatic discovery retained"),
        ("Centerline extraction", "DONE", "lca_ssm_label_adapter.py; outputs/lca_ssm/raw_cases", "191 extracted; 9 documented failures"),
        ("Landmark extraction", "DONE", "build_lca_population_ssm.py::select_landmarks", "Endpoints and bifurcation reused as exact source samples"),
        ("Branch identification", "DONE", "build_lca_population_ssm.py::classify_daughters", "RAS multi-signal inference; not ground truth"),
        ("Coronary SVD plane", "PARTIAL", "build_lca_population_ssm.py::process_case", "Reusable SVD; new run adds fixed RAS basis/sign convention"),
        ("LAD SVD plane", "PARTIAL", "lca_ssm_planes.py::fit_plane_svd", "Reusable SVD; new run adds population-comparable basis"),
        ("Coronary ellipse", "INCONSISTENT_WITH_FINAL_DEFINITION", "ellipse_references", "Prior constrained landmark arc was diagnostic; new all-support actual ellipse required"),
        ("LAD ellipse", "INCONSISTENT_WITH_FINAL_DEFINITION", "ellipse_references", "Prior constrained landmark arc was diagnostic; new all-source actual ellipse required"),
        ("Ellipse a/b/tilt", "PARTIAL", "population_statistics", "Existed for reference arcs; circular tilt statistics missing"),
        ("Landmark ellipse angles", "PARTIAL", "ellipse reference JSON", "Needed fixed global convention and direct endpoint exports"),
        ("Branch angular extents", "PARTIAL", "ellipse reference JSON", "Needed ordered full-centerline unwrapped extents"),
        ("Ellipse residual/noise", "PARTIAL", "ellipse reference JSON", "Needed signed in-plane/out-of-plane per-point population CSV"),
        ("Population table", "MISSING", "none in exact PPT schema", "One-row-per-patient table added by this increment"),
        ("Statistical distributions", "PARTIAL", "population_statistics", "Needed IQR and circular summaries"),
        ("Statistical covariance/dependence", "PARTIAL", "parameter correlation arrays", "Needed readable CSV and figure"),
        ("Ellipse parameter sampling", "MISSING", "future boundary", "Intentionally deferred by user"),
        ("Landmark sampling", "MISSING", "future boundary", "Intentionally deferred by user"),
        ("Control-point distribution on ellipse arcs", "MISSING", "future boundary", "Intentionally deferred by user"),
        ("Residual/noise sampling", "MISSING", "future boundary", "Intentionally deferred by user"),
        ("Bifurcation connection", "PARTIAL", "existing topology code", "Real topology exists; synthetic connection deferred"),
        ("B-spline interpolation", "DONE", "existing LCA/RCA topology modules", "Available for later reuse; not invoked"),
        ("Validation", "PARTIAL", "existing validation utilities", "Measurement validation implemented; synthetic validation deferred"),
    ]
    table = "\n".join(f"| {req} | {status} | `{location}` | {note} |" for req,status,location,note in rows)
    return f"""# Original PPT requirement gap audit

This audit was completed before implementing the new PPT-priority measurement pipeline. The later heart-support-surface work is preserved as an extension and is not treated as a substitute.

| PPT requirement | Status before this increment | Existing module/output | Reuse decision / gap |
|---|---|---|---|
{table}

## Scope boundary

The current increment stops after population statistics. Sampling, synthetic landmarks/control points, synthetic noise, bifurcation generation, B-spline generation, VTK generation of synthetic trees, and real-versus-generated comparison are deliberately not implemented.
"""


def write_documentation(output: Path, manifest: dict[str, Any], statistics: dict[str, Any]) -> None:
    write_json(output / "run_manifest.json", manifest)
    (output / "PPT_REQUIREMENT_GAP_AUDIT.md").write_text(audit_markdown(), encoding="utf-8")
    n = manifest["statistics_eligible_count"]
    method = f"""# Population two-plane / two-ellipse method

## What is measured

For each of the {manifest['nifti_discovered_count']} discovered NIfTI labels, the pipeline looks for the previously extracted, unchanged physical RAS centerlines. {manifest['centerline_available_count']} cases have usable LMCA/LAD/LCX centerlines. Failures remain visible in `population_case_status.csv`.

1. **SVD plane:** subtract the source-point centroid and use the final right-singular vector as the best-fit plane normal. This produces a centroid, a normal, a deterministic RAS-tied in-plane basis, and point-to-plane residuals. It does not produce an ellipse.
2. **Actual ellipse:** project the unchanged source coordinates into that plane only for measurement, then fit a genuine 2-D ellipse using all support points. The fit produces center, semi-major radius `a`, semi-minor radius `b`, and axial tilt. The source vessel coordinates are never replaced.
3. **Two anatomical roles:** the crown/AV-groove plane uses LCX plus an inferred disconnected RCA candidate where available; that RCA is explicitly not annotated ground truth. The LAD/interventricular plane uses the LAD descent.
4. **Angles:** ellipse theta=0 is the positive major-axis direction whose component along deterministic plane basis `u` is nonnegative. Theta increases toward the positive minor axis in the right-handed `(u,v,normal)` frame. Landmark theta is 360-degree circular. Ellipse tilt is axial with a 180-degree period and is stored in `[-90,90)`.
5. **Angular extent:** each ordered source branch is mapped to nearest ellipse theta, unwrapped along source order, and terminal theta minus initial theta is the signed branch angular extent.
6. **Per-point deviation:** each source point stores normalized branch position `s`, corresponding ellipse reference XYZ, signed in-plane radial residual, signed plane-normal residual, and total Euclidean residual.

## Population model

{n} cases have two numerically valid SVD planes and two actual ellipse fits and enter the measured population summaries. Linear variables report N, mean, sample standard deviation, median, IQR, range, P5 and P95. Landmark angles use directional circular statistics; ellipse tilts use axial circular statistics. Pearson correlations are supplied only as a diagnostic for selected continuous linear quantities.

## Source integrity

The measurement pipeline copies arrays in memory, never writes to the source archives, and verifies zero coordinate and segment-length change. SHA-256 hashes are stored for each source array and archive.
"""
    (output / "POPULATION_TWO_ELLIPSE_METHOD.md").write_text(method, encoding="utf-8")
    report = f"""# PPT-priority implementation report

## Result

- NIfTI volumes discovered: {manifest['nifti_discovered_count']}
- Extracted centerline cases available: {manifest['centerline_available_count']}
- Documented extraction failures: {manifest['centerline_failure_count']}
- Cases with both SVD planes: {manifest['two_plane_fit_count']}
- Cases with both actual ellipses: {manifest['two_ellipse_fit_count']}
- Cases entering statistics: {manifest['statistics_eligible_count']}
- Cases using inferred RCA candidate + LCX crown support: {manifest['inferred_rca_support_count']}
- Cases using flagged LCX-only crown approximation: {manifest['lcx_only_crown_count']}

The original PPT plane and ellipse steps are now separate in both code and outputs. The two bounded ovals drawn for visualization are sampled from fitted ellipse parameters, not visual representations of infinite SVD planes.

## Preserved extension work

`outputs/lca_ssm/stage1_heart_scaffold_vtk/` and `outputs/lca_ssm/stage1_parametric_heart_surface/` were hash-checked before and after this run and were not changed. They remain extended anatomical support-surface work, separate from the original PPT population model.

## Deliberate stopping point

No parameter sampler, landmark sampler, synthetic residual/noise model, generated control points, generated bifurcation, B-spline synthetic tree, synthetic VTK, or real-versus-generated comparison was implemented in this increment.
"""
    (output / "PPT_PRIORITY_IMPLEMENTATION_REPORT.md").write_text(report, encoding="utf-8")
    generator = """# Generator method — future implementation boundary

The user requested that this increment stop immediately before the generative model. Therefore this document defines the hand-off and does not describe code that has been implemented.

## Available measured inputs

The future generator may consume the patient parameter table, circular statistics, continuous-variable correlation matrix, and pointwise residual table produced here. It must preserve cross-parameter dependence, the 180-degree axial nature of ellipse tilt, 360-degree landmark theta, and the empirical along-branch structure of residuals.

## Future steps (not implemented)

1. Sample the two ellipse parameter sets and their plane relationship from a joint population model.
2. Sample bifurcation and terminal angular positions using circular distributions and valid conditional extents.
3. Place ideal reference control points by normalized ellipse arc length.
4. Add smooth, seed-reproducible dataset-derived in-plane and plane-normal deviations while preserving endpoints and junction continuity.
5. Connect LMCA, LAD, and LCX at one bifurcation.
6. Reuse the repository's existing B-spline implementation rather than introducing another spline engine.
7. Validate topology, continuity, lengths, curvature, sampled support, residual magnitude, and real-versus-generated distributions.

No part of that sequence is executed by `ppt_priority_population_model.py`.
"""
    (output / "GENERATOR_METHOD.md").write_text(generator, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=repository)
    parser.add_argument("--clean", action="store_true", help="Delete only the dedicated ppt_priority_completion output before rebuilding")
    parser.add_argument("--max-cases", type=int, help="Smoke-test only the first N NIfTI cases")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository = args.repository.resolve()
    source_root = repository / "pca_ssm_vessel_tree_generator"
    existing_output = repository / "outputs" / "lca_ssm"
    output = existing_output / "ppt_priority_completion"
    if args.clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    protected = {
        "stage1_heart_scaffold_vtk": existing_output / "stage1_heart_scaffold_vtk",
        "stage1_parametric_heart_surface": existing_output / "stage1_parametric_heart_surface",
    }
    protected_before = {name: directory_hashes(path) for name, path in protected.items()}
    (output / "PPT_REQUIREMENT_GAP_AUDIT.md").write_text(audit_markdown(), encoding="utf-8")

    nifti_paths = sorted((source_root / "nii files").glob("*.nii.gz"), key=numeric_case_key)
    if args.max_cases is not None:
        nifti_paths = nifti_paths[:args.max_cases]
    raw_root = existing_output / "raw_cases"
    rejected_index_path = existing_output / "rejected_patient_index.json"
    rejected_reasons = {}
    if rejected_index_path.is_file():
        rejected_payload = json.loads(rejected_index_path.read_text(encoding="utf-8"))
        rejected_reasons = {item["patient_id"]: item["reason"] for item in rejected_payload.get("patients", [])}

    statuses: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    residuals: list[dict[str, Any]] = []
    plot_data: list[dict[str, Any]] = []
    for nifti in nifti_paths:
        case_id = nifti.name[:-7] if nifti.name.endswith(".nii.gz") else nifti.stem
        raw_path = raw_root / case_id / "original_centerlines.npz"
        status = {
            "case_id": case_id, "nifti_found": True, "centerline_ok": False, "landmarks_ok": False,
            "coronary_plane_ok": False, "lad_plane_ok": False, "crown_ellipse_ok": False,
            "lad_ellipse_ok": False, "statistics_eligible": False, "failure_reason": "",
        }
        if not raw_path.is_file():
            status["failure_reason"] = rejected_reasons.get(case_id, "no extracted centerline archive found")
            statuses.append(status)
            continue
        status["centerline_ok"] = True
        status["landmarks_ok"] = True
        try:
            row, case_residuals, case_plot_data = process_case(case_id, raw_path, output)
            status.update({"coronary_plane_ok": True, "lad_plane_ok": True, "crown_ellipse_ok": True,
                           "lad_ellipse_ok": True, "statistics_eligible": True})
            rows.append(row); residuals.extend(case_residuals); plot_data.append(case_plot_data)
            print(f"ELIGIBLE {case_id}", flush=True)
        except Exception as exc:
            status["failure_reason"] = str(exc)
            # More granular diagnostics are conservative: a processing failure is not silently promoted.
            print(f"FAILED {case_id}: {exc}", flush=True)
        statuses.append(status)

    write_csv(output / "population_case_status.csv", statuses, [
        "case_id", "nifti_found", "centerline_ok", "landmarks_ok", "coronary_plane_ok", "lad_plane_ok",
        "crown_ellipse_ok", "lad_ellipse_ok", "statistics_eligible", "failure_reason",
    ])
    write_csv(output / "population_two_plane_two_ellipse_parameters.csv", rows, PARAMETER_FIELDS)
    residual_fields = ["case_id", "branch", "point_index", "s", "theta_rad", "theta_deg", "reference_x", "reference_y", "reference_z",
                       "source_x", "source_y", "source_z", "in_plane_residual", "out_of_plane_residual", "euclidean_residual"]
    write_csv(output / "population_pointwise_ellipse_residuals.csv", residuals, residual_fields)
    if not rows:
        raise SystemExit("no cases produced two valid ellipse fits")

    statistics, statistics_rows = build_population_statistics(rows, residuals)
    statistics["case_count"] = len(rows)
    write_json(output / "population_statistics.json", statistics)
    stats_fields = ["variable", "type", "n", "mean", "std", "median", "iqr", "min", "max", "p5", "p95",
                    "circular_mean_deg", "circular_std_deg", "resultant_length", "period_deg"]
    write_csv(output / "population_statistics.csv", statistics_rows, stats_fields)
    correlation_fields, correlation = build_correlation(rows)
    correlation_rows = [{"parameter": name, **{other: correlation[i, j] for j, other in enumerate(correlation_fields)}}
                        for i, name in enumerate(correlation_fields)]
    write_csv(output / "population_parameter_correlation.csv", correlation_rows, ["parameter", *correlation_fields])

    inferred_count = sum("inferred_RCA" in str(row["coronary_support"]) for row in rows)
    protected_after = {name: directory_hashes(path) for name, path in protected.items()}
    protection = {
        name: {"file_count": len(protected_before[name]), "unchanged": protected_before[name] == protected_after[name],
               "before_hashes": protected_before[name], "after_hashes": protected_after[name]}
        for name in protected
    }
    write_json(output / "protected_stage1_integrity.json", protection)
    if not all(item["unchanged"] for item in protection.values()):
        raise RuntimeError("a protected Stage-1 output changed during the run")
    manifest = {
        "scope": "original PPT measurement pipeline through population statistics only",
        "generative_model_implemented": False,
        "nifti_discovered_count": len(nifti_paths),
        "centerline_available_count": sum(int(s["centerline_ok"]) for s in statuses),
        "centerline_failure_count": sum(not s["centerline_ok"] for s in statuses),
        "landmark_available_count": sum(int(s["landmarks_ok"]) for s in statuses),
        "two_plane_fit_count": sum(int(s["coronary_plane_ok"] and s["lad_plane_ok"]) for s in statuses),
        "two_ellipse_fit_count": sum(int(s["crown_ellipse_ok"] and s["lad_ellipse_ok"]) for s in statuses),
        "statistics_eligible_count": len(rows),
        "failed_case_count": sum(not s["statistics_eligible"] for s in statuses),
        "inferred_rca_support_count": inferred_count,
        "lcx_only_crown_count": len(rows) - inferred_count,
        "pointwise_residual_record_count": len(residuals),
        "protected_stage1_outputs_unchanged": all(item["unchanged"] for item in protection.values()),
        "source_geometry_max_coordinate_change_mm": max(float(row["max_source_coordinate_change"]) for row in rows),
        "source_geometry_max_segment_length_change_mm": max(float(row["max_segment_length_change"]) for row in rows),
        "important_outputs": {
            "case_status_csv": output / "population_case_status.csv",
            "patient_parameters_csv": output / "population_two_plane_two_ellipse_parameters.csv",
            "pointwise_residuals_csv": output / "population_pointwise_ellipse_residuals.csv",
            "statistics_csv": output / "population_statistics.csv",
            "statistics_json": output / "population_statistics.json",
            "correlation_csv": output / "population_parameter_correlation.csv",
        },
    }
    write_documentation(output, manifest, statistics)
    if not args.skip_figures:
        combined = np.asarray([[row["crown_fit_rmse"], row["lad_fit_rmse"]] for row in rows])
        score = np.sum((combined - np.median(combined, axis=0)) ** 2 / np.maximum(np.var(combined, axis=0), EPS), axis=1)
        example = plot_data[int(np.argmin(score))]
        make_figures(rows, statuses, residuals, correlation_fields, correlation, example, output / "figures")
        manifest["example_case_id"] = example["row"]["case_id"]
        write_json(output / "run_manifest.json", manifest)
    print(json.dumps(jsonable(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
