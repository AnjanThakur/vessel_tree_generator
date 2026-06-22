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


def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
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


def curved_path_from_formula(
    start: np.ndarray,
    direction: np.ndarray,
    length_mm: float,
    side_vector: np.ndarray,
    vertical_curve_vector: np.ndarray,
    side_strength: float,
    vertical_strength: float,
    n_anchors: int = 80,
) -> np.ndarray:
    """
    Creates a smooth curved path using anatomical direction + controlled curvature.
    This gives more realistic shapes than straight anchor lines.
    """
    direction = normalize(direction)
    side_vector = normalize(side_vector)
    vertical_curve_vector = normalize(vertical_curve_vector)

    t_values = np.linspace(0.0, 1.0, n_anchors)
    points = []

    for t in t_values:
        base = start + (length_mm * t * direction)

        # Smooth side curvature: zero at start/end, max near middle
        side_curve = side_strength * length_mm * np.sin(np.pi * t) * side_vector

        # Progressive curvature: small near start, stronger distally
        vertical_curve = vertical_strength * length_mm * (t ** 2) * vertical_curve_vector

        points.append(base + side_curve + vertical_curve)

    return np.array(points)


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
    LMCA is short, so it should remain mostly straight with only mild curvature.
    """
    slight_y = rng.normal(0.0, 0.35)
    slight_z = rng.normal(0.0, 0.20)

    bifurcation = np.array([length_mm, slight_y, slight_z], dtype=float)

    direction = normalize(bifurcation - origin)
    side_vector = np.array([0.0, 1.0, 0.0])
    vertical_vector = np.array([0.0, 0.0, -1.0])

    anchors = curved_path_from_formula(
        start=origin,
        direction=direction,
        length_mm=length_mm,
        side_vector=side_vector,
        vertical_curve_vector=vertical_vector,
        side_strength=rng.uniform(0.005, 0.015),
        vertical_strength=rng.uniform(-0.005, 0.005),
        n_anchors=50,
    )

    # Scale endpoint exactly to bifurcation direction
    anchors = origin + (anchors - anchors[0])
    anchors[-1] = bifurcation

    control_points = resample_polyline(anchors, CONTROL_POINTS_PER_BRANCH)
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
    It should not be a straight vertical line; it should have mild anatomical curvature.
    """
    lad_dir = normalize(np.array([
        0.30 + rng.normal(0, 0.04),     # forward
        -0.12 + rng.normal(0, 0.05),    # slight medial/lateral shift
        -0.94 + rng.normal(0, 0.04),    # strong downward direction
    ]))

    side_vector = normalize(np.array([
        rng.normal(0.10, 0.05),
        -1.0,
        rng.normal(0.05, 0.03),
    ]))

    vertical_curve_vector = normalize(np.array([
        0.25,
        0.05,
        -1.0,
    ]))

    anchors = curved_path_from_formula(
        start=bifurcation,
        direction=lad_dir,
        length_mm=length_mm,
        side_vector=side_vector,
        vertical_curve_vector=vertical_curve_vector,
        side_strength=rng.uniform(0.015, 0.045),
        vertical_strength=rng.uniform(0.015, 0.035),
        n_anchors=100,
    )

    if wraparound:
        # Distal wrap-around effect near the apex
        t_values = np.linspace(0.0, 1.0, len(anchors))
        wrap_vector = normalize(np.array([
            rng.uniform(0.20, 0.45),
            rng.uniform(0.15, 0.35),
            rng.uniform(-0.20, 0.05),
        ]))

        for i, t in enumerate(t_values):
            if t > 0.70:
                strength = ((t - 0.70) / 0.30) ** 2
                anchors[i] += strength * rng.uniform(4.0, 8.0) * wrap_vector

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
    LCX starts from the same LMCA bifurcation and sweeps laterally.
    It must curve more than LAD because circumflex means it curves around.
    """
    lad_initial_dir = normalize(lad_control_points[3] - lad_control_points[0])

    preferred_lateral_side = np.array([0.0, 1.0, 0.0])
    lcx_initial_dir = vector_at_angle(
        lad_initial_dir,
        target_angle_deg,
        preferred_lateral_side,
    )

    lateral_vector = normalize(np.array([
        rng.uniform(-0.10, 0.15),
        1.0,
        rng.uniform(-0.15, 0.05),
    ]))

    posterior_curve_vector = normalize(np.array([
        rng.uniform(-0.35, -0.10),
        rng.uniform(0.50, 0.90),
        rng.uniform(-0.25, -0.05),
    ]))

    t_values = np.linspace(0.0, 1.0, 120)
    anchors = []

    for t in t_values:
        base = bifurcation + length_mm * t * lcx_initial_dir

        # Use t^2 so initial tangent remains close to target angle
        lateral_sweep = rng.uniform(0.18, 0.32) * length_mm * (t ** 2) * lateral_vector

        # Curve around distally
        posterior_sweep = rng.uniform(0.08, 0.18) * length_mm * (t ** 2.2) * posterior_curve_vector

        # Mild smooth wave, but small enough not to break angle
        wave = rng.uniform(0.005, 0.015) * length_mm * np.sin(np.pi * t) * lateral_vector

        anchors.append(base + lateral_sweep + posterior_sweep + wave)

    anchors = np.array(anchors)

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