# LCA_topology_generator/control_points.py

from typing import Dict, Tuple, Any
import numpy as np

from .parameters import (
    BRANCH_STATS,
    CONTROL_POINTS_PER_BRANCH,
    sample_truncated_normal,
)


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("Cannot normalize zero vector.")
    return v / norm


def polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """
    Convert a small set of anatomical anchor points into fixed number of control points.
    This avoids manually placing all 20 points.
    """
    points = np.asarray(points, dtype=float)

    distances = np.zeros(len(points))
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))

    total = distances[-1]
    if total <= 1e-9:
        raise ValueError("Cannot resample polyline with zero length.")

    target = np.linspace(0.0, total, n)

    result = np.zeros((n, 3), dtype=float)
    for dim in range(3):
        result[:, dim] = np.interp(target, distances, points[:, dim])

    return result


def vector_at_angle(base_dir: np.ndarray, angle_deg: float, preferred_side: np.ndarray) -> np.ndarray:
    """
    Create a vector that forms angle_deg with base_dir.
    preferred_side controls the side toward which new vector opens.
    """
    base_dir = normalize(base_dir)
    preferred_side = normalize(preferred_side)

    perpendicular = preferred_side - np.dot(preferred_side, base_dir) * base_dir
    perpendicular = normalize(perpendicular)

    angle_rad = np.deg2rad(angle_deg)
    new_dir = np.cos(angle_rad) * base_dir + np.sin(angle_rad) * perpendicular
    return normalize(new_dir)


def generate_lmca_control_points(
    rng: np.random.Generator,
    origin: np.ndarray,
    length_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LMCA is generated as short root trunk from coronary origin to bifurcation.
    """
    slight_y = rng.normal(0.0, 0.25)
    slight_z = rng.normal(0.0, 0.15)

    bifurcation = np.array([length_mm, slight_y, slight_z], dtype=float)

    # Small smooth offsets to avoid perfectly straight artificial trunk.
    anchors = np.array([
        origin,
        origin + 0.30 * (bifurcation - origin) + np.array([0.0, rng.normal(0, 0.12), rng.normal(0, 0.08)]),
        origin + 0.65 * (bifurcation - origin) + np.array([0.0, rng.normal(0, 0.12), rng.normal(0, 0.08)]),
        bifurcation,
    ])

    control_points = resample_polyline(anchors, CONTROL_POINTS_PER_BRANCH)

    # Enforce exact connection endpoint
    control_points[0] = origin
    control_points[-1] = bifurcation

    return control_points, bifurcation


def generate_lad_control_points(
    rng: np.random.Generator,
    bifurcation: np.ndarray,
    length_mm: float,
    wraparound: bool,
) -> np.ndarray:
    """
    LAD starts from LMCA bifurcation and descends toward apex.
    """
    lad_dir = normalize(np.array([
        0.28 + rng.normal(0, 0.04),
        -0.10 + rng.normal(0, 0.05),
        -0.95 + rng.normal(0, 0.04),
    ]))

    # Side vector for smooth curvature
    curve_side = normalize(np.array([0.0, -1.0, 0.0]))

    distal_extra = np.array([0.0, 0.0, 0.0])
    if wraparound:
        distal_extra = np.array([
            rng.normal(2.0, 1.0),
            rng.normal(1.5, 0.8),
            rng.normal(-3.0, 1.0),
        ])

    anchors = np.array([
        bifurcation,
        bifurcation + 0.20 * length_mm * lad_dir + 1.5 * curve_side,
        bifurcation + 0.45 * length_mm * lad_dir + rng.normal(0, 1.0) * curve_side,
        bifurcation + 0.70 * length_mm * lad_dir - 1.0 * curve_side,
        bifurcation + 1.00 * length_mm * lad_dir + distal_extra,
    ])

    control_points = resample_polyline(anchors, CONTROL_POINTS_PER_BRANCH)
    control_points[0] = bifurcation

    return control_points


def generate_lcx_control_points(
    rng: np.random.Generator,
    bifurcation: np.ndarray,
    lad_control_points: np.ndarray,
    length_mm: float,
    target_angle_deg: float,
) -> np.ndarray:
    """
    LCX starts from same LMCA bifurcation and curves laterally.
    It is generated relative to LAD direction using target LAD-LCX angle.
    """
    lad_initial_dir = normalize(lad_control_points[3] - lad_control_points[0])

    # Open LCX away from LAD toward lateral side
    preferred_lateral_side = np.array([0.0, 1.0, 0.0])
    lcx_dir = vector_at_angle(lad_initial_dir, target_angle_deg, preferred_lateral_side)

    lateral_sweep = normalize(np.array([0.0, 1.0, -0.10]))
    posterior_curve = normalize(np.array([-0.15, 0.75, -0.20]))

    anchors = np.array([
        bifurcation,
        bifurcation + 0.18 * length_mm * lcx_dir + 2.0 * lateral_sweep,
        bifurcation + 0.42 * length_mm * lcx_dir + 5.0 * lateral_sweep,
        bifurcation + 0.70 * length_mm * lcx_dir + 4.0 * posterior_curve,
        bifurcation + 1.00 * length_mm * lcx_dir + rng.normal(0, 1.5) * posterior_curve,
    ])

    control_points = resample_polyline(anchors, CONTROL_POINTS_PER_BRANCH)
    control_points[0] = bifurcation

    return control_points


def generate_one_lca_candidate(
    rng: np.random.Generator,
    tree_id: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Generate one connected LCA candidate:
    LMCA -> LAD + LCX
    """
    lmca_length = sample_truncated_normal(rng, BRANCH_STATS["LMCA_LENGTH"])
    lad_length = sample_truncated_normal(rng, BRANCH_STATS["LAD_LENGTH"])
    lcx_length = sample_truncated_normal(rng, BRANCH_STATS["LCX_LENGTH"])
    target_angle = sample_truncated_normal(rng, BRANCH_STATS["LAD_LCX_ANGLE"])

    lmca_diameter = sample_truncated_normal(rng, BRANCH_STATS["LMCA_DIAMETER"])
    lad_diameter = sample_truncated_normal(rng, BRANCH_STATS["LAD_DIAMETER"])
    lcx_diameter = sample_truncated_normal(rng, BRANCH_STATS["LCX_DIAMETER"])

    # 86% wrap-around LAD from literature.
    wraparound_lad = bool(rng.random() < 0.86)

    origin = np.array([0.0, 0.0, 0.0], dtype=float)

    lmca_cp, bifurcation = generate_lmca_control_points(
        rng=rng,
        origin=origin,
        length_mm=lmca_length,
    )

    lad_cp = generate_lad_control_points(
        rng=rng,
        bifurcation=bifurcation,
        length_mm=lad_length,
        wraparound=wraparound_lad,
    )

    lcx_cp = generate_lcx_control_points(
        rng=rng,
        bifurcation=bifurcation,
        lad_control_points=lad_cp,
        length_mm=lcx_length,
        target_angle_deg=target_angle,
    )

    branches = {
        "LMCA": lmca_cp,
        "LAD": lad_cp,
        "LCX": lcx_cp,
    }

    metadata = {
        "tree_id": tree_id,
        "pattern": "bifurcation",
        "units": "mm",
        "lmca_length_target_mm": lmca_length,
        "lad_length_target_mm": lad_length,
        "lcx_length_target_mm": lcx_length,
        "lad_lcx_angle_target_deg": target_angle,
        "lmca_diameter_mm": lmca_diameter,
        "lad_diameter_mm": lad_diameter,
        "lcx_diameter_mm": lcx_diameter,
        "lmca_radius_mm": lmca_diameter / 2.0,
        "lad_radius_mm": lad_diameter / 2.0,
        "lcx_radius_mm": lcx_diameter / 2.0,
        "wraparound_lad": wraparound_lad,
    }

    return branches, metadata