# LCA_topology_generator/control_points.py

from typing import Dict, Tuple, Any
import numpy as np

from .parameters import (
    BRANCH_STATS,
    CONTROL_POINTS_PER_BRANCH,
    ENGINEERING_SHAPE_PARAMETERS,
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


def smoothstep_after(t: float, start: float) -> float:
    if t <= start:
        return 0.0

    value = (t - start) / max(1.0 - start, 1e-12)
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def generate_lmca_control_points(
    rng: np.random.Generator,
    origin: np.ndarray,
    length_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LMCA is short, so it should remain mostly straight with only mild curvature.
    """
    shape = ENGINEERING_SHAPE_PARAMETERS["LMCA"]

    slight_y = rng.normal(0.0, shape["bifurcation_y_std_mm"])
    slight_z = rng.normal(0.0, shape["bifurcation_z_std_mm"])

    bifurcation = np.array([length_mm, slight_y, slight_z], dtype=float)

    direction = normalize(bifurcation - origin)
    side_vector = np.array(shape["side_vector"], dtype=float)
    vertical_vector = np.array(shape["vertical_vector"], dtype=float)

    anchors = curved_path_from_formula(
        start=origin,
        direction=direction,
        length_mm=length_mm,
        side_vector=side_vector,
        vertical_curve_vector=vertical_vector,
        side_strength=rng.uniform(*shape["side_strength_range"]),
        vertical_strength=rng.uniform(*shape["vertical_strength_range"]),
        n_anchors=shape["anchor_count"],
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
    shape = ENGINEERING_SHAPE_PARAMETERS["LAD"]

    direction_mean = np.array(shape["direction_mean"], dtype=float)
    direction_std = np.array(shape["direction_std"], dtype=float)
    lad_dir = normalize(direction_mean + rng.normal(0.0, direction_std))

    side_vector = normalize(np.array([
        rng.normal(shape["side_vector_x_mean"], shape["side_vector_x_std"]),
        shape["side_vector_y"],
        rng.normal(shape["side_vector_z_mean"], shape["side_vector_z_std"]),
    ], dtype=float))

    vertical_curve_vector = normalize(np.array(
        shape["vertical_curve_vector"],
        dtype=float,
    ))

    anchors = curved_path_from_formula(
        start=bifurcation,
        direction=lad_dir,
        length_mm=length_mm,
        side_vector=side_vector,
        vertical_curve_vector=vertical_curve_vector,
        side_strength=rng.uniform(*shape["side_strength_range"]),
        vertical_strength=rng.uniform(*shape["vertical_strength_range"]),
        n_anchors=shape["anchor_count"],
    )

    if wraparound:
        # Distal wrap-around effect near the apex
        t_values = np.linspace(0.0, 1.0, len(anchors))
        wrap_vector = normalize(np.array([
            rng.uniform(*shape["wraparound_vector_x_range"]),
            rng.uniform(*shape["wraparound_vector_y_range"]),
            rng.uniform(*shape["wraparound_vector_z_range"]),
        ]))

        for i, t in enumerate(t_values):
            if t > shape["wraparound_start_s"]:
                distal_span = 1.0 - shape["wraparound_start_s"]
                strength = ((t - shape["wraparound_start_s"]) / distal_span) ** 2
                anchors[i] += (
                    strength
                    * rng.uniform(*shape["wraparound_strength_range_mm"])
                    * wrap_vector
                )

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
    LCX starts from the same LMCA bifurcation and follows a circumflex-like course.
    It should move laterally, but not shoot too far outward.
    """
    shape = ENGINEERING_SHAPE_PARAMETERS["LCX"]

    lad_initial_dir = normalize(lad_control_points[3] - lad_control_points[0])

    preferred_lateral_side = np.array(shape["preferred_lateral_side"], dtype=float)
    lcx_initial_dir = vector_at_angle(
        lad_initial_dir,
        target_angle_deg,
        preferred_lateral_side,
    )

    lateral_vector = normalize(np.array(shape["lateral_vector"], dtype=float))
    posterior_curve_vector = normalize(np.array(
        shape["posterior_curve_vector"],
        dtype=float,
    ))

    t_values = np.linspace(0.0, 1.0, shape["anchor_count"])
    anchors = []

    lateral_strength = rng.uniform(*shape["lateral_strength_range"])
    posterior_strength = rng.uniform(*shape["posterior_strength_range"])
    downward_strength = rng.uniform(*shape["downward_strength_range"])

    for t in t_values:
        base = bifurcation + length_mm * t * lcx_initial_dir
        sweep_s = smoothstep_after(t, shape["sweep_start_s"])

        # Delay circumflex shaping so the proximal tangent preserves target angle.
        lateral_sweep = (
            lateral_strength
            * length_mm
            * (sweep_s ** shape["lateral_exponent"])
            * lateral_vector
        )

        # Distal curve to make it circumflex-like
        posterior_sweep = (
            posterior_strength
            * length_mm
            * (sweep_s ** shape["posterior_exponent"])
            * posterior_curve_vector
        )

        # Slight downward course
        downward_sweep = (
            downward_strength
            * length_mm
            * (sweep_s ** shape["downward_exponent"])
            * np.array([0.0, 0.0, -1.0])
        )

        # Small natural wave, also delayed to avoid biasing the measured angle.
        wave = (
            rng.uniform(*shape["wave_strength_range"])
            * length_mm
            * np.sin(np.pi * sweep_s)
            * lateral_vector
        )

        anchors.append(base + lateral_sweep + posterior_sweep + downward_sweep + wave)

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

    wraparound_lad = bool(
        rng.random() < ENGINEERING_SHAPE_PARAMETERS["LAD"]["wraparound_probability"]
    )

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
