import math

import numpy as np

from .tortuosity import calculate_path_length


BRANCH_SLICES = {
    "LMCA": slice(0, 5),
    "LAD": slice(5, 17),
    "LCX": slice(17, 27),
}


DEFAULT_VALIDATION_CONFIG = {
    "min_control_path_length_mm": {
        "LMCA": 5.0,
        "LAD": 30.0,
        "LCX": 20.0,
    },
    "min_unique_control_points": {
        "LMCA": 3,
        "LAD": 6,
        "LCX": 5,
    },
    "max_bifurcation_gap_mm": 1e-6,
    "lad_lcx_angle_range_deg": [30.0, 140.0],
    "tortuosity_range": {
        "LMCA": [1.0, 1.5],
        "LAD": [1.0, 1.8],
        "LCX": [1.0, 2.0],
    },
    "max_spline_to_control_length_ratio": 1.25,
}


def _as_json_float(value):
    if value is None:
        return None
    return float(value)


def _unique_point_count(points: np.ndarray, decimals: int = 6) -> int:
    rounded = np.round(np.asarray(points, dtype=float), decimals=decimals)
    return int(len(np.unique(rounded, axis=0)))


def _branch_angle_deg(lad_points: np.ndarray, lcx_points: np.ndarray):
    lad_vec = lad_points[-1] - lad_points[0]
    lcx_vec = lcx_points[-1] - lcx_points[0]
    lad_norm = np.linalg.norm(lad_vec)
    lcx_norm = np.linalg.norm(lcx_vec)
    if lad_norm <= 1e-9 or lcx_norm <= 1e-9:
        return None

    cosine = np.dot(lad_vec, lcx_vec) / (lad_norm * lcx_norm)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(math.degrees(math.acos(cosine)))


def validate_lca_tree(control_tree: np.ndarray, centerlines: dict, metrics: dict, config: dict = None) -> dict:
    """
    Validates a dataset-derived LCA tree.

    Returns a JSON-serializable report with blocking errors and non-blocking warnings.
    """
    config = config or DEFAULT_VALIDATION_CONFIG
    errors = []
    warnings = []
    branch_reports = {}

    for branch_name, branch_slice in BRANCH_SLICES.items():
        control_points = np.asarray(control_tree[branch_slice], dtype=float)
        centerline = np.asarray(centerlines[branch_name], dtype=float)
        control_path_length = calculate_path_length(control_points)
        centerline_path_length = calculate_path_length(centerline)
        unique_points = _unique_point_count(control_points)
        tortuosity = metrics[branch_name]["tortuosity"]
        spline_ratio = (
            centerline_path_length / control_path_length
            if control_path_length > 1e-9
            else None
        )

        min_length = config["min_control_path_length_mm"][branch_name]
        min_unique = config["min_unique_control_points"][branch_name]
        tort_min, tort_max = config["tortuosity_range"][branch_name]

        if control_path_length < min_length:
            errors.append(
                f"{branch_name} control path length {control_path_length:.3f} mm is below {min_length:.3f} mm"
            )
        if unique_points < min_unique:
            errors.append(
                f"{branch_name} has {unique_points} unique control points; expected at least {min_unique}"
            )
        if tortuosity is None:
            errors.append(f"{branch_name} tortuosity is undefined")
        elif not (tort_min <= tortuosity <= tort_max):
            warnings.append(
                f"{branch_name} tortuosity {tortuosity:.3f} is outside [{tort_min:.3f}, {tort_max:.3f}]"
            )
        if spline_ratio is not None and spline_ratio > config["max_spline_to_control_length_ratio"]:
            warnings.append(
                f"{branch_name} spline/control path ratio {spline_ratio:.3f} exceeds "
                f"{config['max_spline_to_control_length_ratio']:.3f}"
            )

        branch_reports[branch_name] = {
            "control_path_length_mm": float(control_path_length),
            "centerline_path_length_mm": float(centerline_path_length),
            "unique_control_points": int(unique_points),
            "tortuosity": _as_json_float(tortuosity),
            "spline_to_control_length_ratio": _as_json_float(spline_ratio),
        }

    lad_gap = float(np.linalg.norm(control_tree[5] - control_tree[4]))
    lcx_gap = float(np.linalg.norm(control_tree[17] - control_tree[4]))
    max_gap = config["max_bifurcation_gap_mm"]
    if lad_gap > max_gap:
        errors.append(f"LAD start gap from LMCA endpoint is {lad_gap:.6f} mm")
    if lcx_gap > max_gap:
        errors.append(f"LCX start gap from LMCA endpoint is {lcx_gap:.6f} mm")

    branch_angle = _branch_angle_deg(centerlines["LAD"], centerlines["LCX"])
    angle_min, angle_max = config["lad_lcx_angle_range_deg"]
    if branch_angle is None:
        errors.append("LAD-LCX angle is undefined")
    elif not (angle_min <= branch_angle <= angle_max):
        warnings.append(
            f"LAD-LCX angle {branch_angle:.3f} deg is outside [{angle_min:.3f}, {angle_max:.3f}]"
        )

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "bifurcation_gaps_mm": {
            "LAD_to_LMCA": lad_gap,
            "LCX_to_LMCA": lcx_gap,
        },
        "lad_lcx_angle_deg": _as_json_float(branch_angle),
        "branches": branch_reports,
    }
