import numpy as np


DEFAULT_STATIC_RADIUS_MODEL = {
    "units": "mm",
    "description": "MVP distance-based coronary radius taper; no disease, motion, pulsatility, or random sampling.",
    "model": "distance_based_branch_type_with_lmca_cube_law",
    "branch_types": {
        "parent_trunk": {
            "description": "Short parent segment feeding a bifurcation.",
            "taper_rate_per_mm": 0.0025,
        },
        "distributing": {
            "description": "Main coronary distributing artery with gradual taper.",
            "taper_rate_per_mm": 0.0035,
        },
        "delivering": {
            "description": "Smaller delivering branch with faster taper.",
            "taper_rate_per_mm": 0.0065,
        },
    },
    "lmca_bifurcation": {
        "method": "cube_law",
        "formula": "lmca_distal = (lad_proximal^3 + lcx_proximal^3)^(1/3)",
        "cube_law_relative_tolerance": 0.08,
    },
    "branches": {
        "LMCA": {
            "proximal_radius_mm": 2.0,
            "branch_type": "parent_trunk",
        },
        "LAD": {
            "proximal_radius_mm": 1.3,
            "branch_type": "distributing",
        },
        "LCX": {
            "proximal_radius_mm": 1.2,
            "branch_type": "distributing",
        },
    },
}


def cumulative_arc_length(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    if len(points) == 1:
        return np.zeros(1, dtype=float)

    distances = np.zeros(len(points), dtype=float)
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    return distances


def arc_length_fraction(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    if len(points) == 1:
        return np.zeros(1, dtype=float)

    distances = cumulative_arc_length(points)
    total = distances[-1]
    if total <= 1e-9:
        return np.zeros(len(points), dtype=float)
    return distances / total


def distance_based_taper_radius(
    centerline: np.ndarray,
    proximal_radius_mm: float,
    taper_rate_per_mm: float,
) -> np.ndarray:
    """
    Creates a distance-based exponential taper profile.

    radius(distance) = proximal_radius * exp(-k * distance)
    """
    if proximal_radius_mm <= 0:
        raise ValueError("Proximal radius must be positive.")
    if taper_rate_per_mm < 0:
        raise ValueError("Taper rate must be non-negative.")

    distance = cumulative_arc_length(centerline)
    return proximal_radius_mm * np.exp(-taper_rate_per_mm * distance)


def transition_taper_to_distal_radius(
    centerline: np.ndarray,
    proximal_radius_mm: float,
    distal_radius_mm: float,
    taper_rate_per_mm: float,
) -> np.ndarray:
    """
    Creates a smooth distance-shaped profile that exactly reaches a target distal radius.

    This is used for LMCA so its distal radius can be set by the LAD/LCX
    cube-law estimate while still tapering according to branch distance.
    """
    if proximal_radius_mm <= 0 or distal_radius_mm <= 0:
        raise ValueError("Radii must be positive.")
    if distal_radius_mm > proximal_radius_mm:
        raise ValueError("Distal radius must be less than or equal to proximal radius.")
    if taper_rate_per_mm < 0:
        raise ValueError("Taper rate must be non-negative.")

    distance = cumulative_arc_length(centerline)
    total = distance[-1] if len(distance) else 0.0
    if total <= 1e-9 or abs(proximal_radius_mm - distal_radius_mm) <= 1e-12:
        return np.full(len(distance), proximal_radius_mm, dtype=float)
    if taper_rate_per_mm <= 1e-12:
        fraction = distance / total
        return proximal_radius_mm - (proximal_radius_mm - distal_radius_mm) * fraction

    exponential = np.exp(-taper_rate_per_mm * distance)
    distal_exponential = np.exp(-taper_rate_per_mm * total)
    shape = (exponential - distal_exponential) / (1.0 - distal_exponential)
    return distal_radius_mm + (proximal_radius_mm - distal_radius_mm) * shape


def cube_law_parent_radius(child_radius_1_mm: float, child_radius_2_mm: float) -> float:
    if child_radius_1_mm <= 0 or child_radius_2_mm <= 0:
        raise ValueError("Child radii must be positive.")
    return float((child_radius_1_mm ** 3 + child_radius_2_mm ** 3) ** (1.0 / 3.0))


def _branch_taper_rate(radius_model: dict, branch_name: str) -> tuple:
    branch_params = radius_model["branches"][branch_name]
    branch_type = branch_params.get("branch_type", "distributing")
    branch_types = radius_model.get("branch_types", {})
    if "taper_rate_per_mm" in branch_params:
        taper_rate = float(branch_params["taper_rate_per_mm"])
    else:
        taper_rate = float(branch_types.get(branch_type, {}).get("taper_rate_per_mm", 0.0035))
    if taper_rate < 0:
        raise ValueError(f"{branch_name} taper_rate_per_mm must be non-negative.")
    return branch_type, taper_rate


def add_radius_to_centerline(centerline: np.ndarray, radius_mm: np.ndarray) -> np.ndarray:
    centerline = np.asarray(centerline, dtype=float)
    radius_mm = np.asarray(radius_mm, dtype=float)
    if centerline.ndim != 2 or centerline.shape[1] != 3:
        raise ValueError("Centerline must have shape (N, 3).")
    if radius_mm.shape != (len(centerline),):
        raise ValueError("Radius profile must have shape (N,).")
    return np.column_stack([centerline, radius_mm])


def build_lca_radius_tree(centerlines: dict, radius_model: dict = None) -> tuple:
    """
    Adds static radius tapering to LMCA, LAD, and LCX centerlines.

    Returns `(centerlines_with_radius, metadata)`, where each branch array has
    shape `(N, 4)` with columns `[x, y, z, radius_mm]`.
    """
    radius_model = radius_model or DEFAULT_STATIC_RADIUS_MODEL
    centerlines_with_radius = {}
    metadata = {
        "units": radius_model.get("units", "mm"),
        "description": radius_model.get("description", ""),
        "model": radius_model.get("model", "distance_based_branch_type_with_lmca_cube_law"),
        "branch_types": radius_model.get("branch_types", {}),
        "lmca_bifurcation": dict(radius_model.get("lmca_bifurcation", {})),
        "metadata_source": radius_model.get("metadata_source"),
        "branch_radius_source": radius_model.get("branch_radius_source", {}),
        "branches": {},
    }

    proximal_radii = {
        branch_name: float(radius_model["branches"][branch_name]["proximal_radius_mm"])
        for branch_name in ["LMCA", "LAD", "LCX"]
    }
    lmca_cube_distal = cube_law_parent_radius(proximal_radii["LAD"], proximal_radii["LCX"])

    for branch_name in ["LMCA", "LAD", "LCX"]:
        params = radius_model["branches"][branch_name]
        proximal_radius_mm = proximal_radii[branch_name]
        branch_type, taper_rate = _branch_taper_rate(radius_model, branch_name)
        distance = cumulative_arc_length(centerlines[branch_name])

        if branch_name == "LMCA":
            distal_radius_mm = lmca_cube_distal
            if distal_radius_mm > proximal_radius_mm:
                raise ValueError(
                    "LMCA proximal radius must be greater than or equal to the "
                    "cube-law distal radius implied by LAD/LCX proximal radii."
                )
            radius = transition_taper_to_distal_radius(
                centerlines[branch_name],
                proximal_radius_mm=proximal_radius_mm,
                distal_radius_mm=distal_radius_mm,
                taper_rate_per_mm=taper_rate,
            )
            method = "cube_law_distal_transition"
        else:
            radius = distance_based_taper_radius(
                centerlines[branch_name],
                proximal_radius_mm=proximal_radius_mm,
                taper_rate_per_mm=taper_rate,
            )
            distal_radius_mm = float(radius[-1])
            method = "distance_exponential"

        centerlines_with_radius[branch_name] = add_radius_to_centerline(centerlines[branch_name], radius)
        metadata["branches"][branch_name] = {
            **params,
            "branch_type": branch_type,
            "taper_rate_per_mm": taper_rate,
            "taper_method": method,
            "proximal_radius_mm": proximal_radius_mm,
            "distal_radius_mm": distal_radius_mm,
            "arc_length_mm": float(distance[-1]) if len(distance) else 0.0,
            "num_points": int(len(radius)),
            "min_radius_mm": float(np.min(radius)),
            "max_radius_mm": float(np.max(radius)),
            "mean_radius_mm": float(np.mean(radius)),
            "is_monotonic_nonincreasing": bool(np.all(np.diff(radius) <= 1e-9)),
        }
    metadata["lmca_bifurcation"]["computed_lmca_distal_radius_mm"] = float(lmca_cube_distal)

    return centerlines_with_radius, metadata


def validate_lca_radius_tree(
    centerlines_with_radius: dict,
    adjacent_jump_threshold_mm: float = 0.2,
    cube_law_relative_tolerance: float = DEFAULT_STATIC_RADIUS_MODEL["lmca_bifurcation"]["cube_law_relative_tolerance"],
) -> dict:
    errors = []
    warnings = []
    branch_reports = {}

    for branch_name, points in centerlines_with_radius.items():
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 4:
            errors.append(f"{branch_name} radius centerline must have shape (N, 4)")
            continue

        radius = points[:, 3]
        radius_steps = np.diff(radius)
        if np.any(radius <= 0):
            errors.append(f"{branch_name} contains non-positive radius values")
        if not np.all(np.isfinite(radius)):
            errors.append(f"{branch_name} contains NaN or Inf radius values")
        if len(radius) != len(points):
            errors.append(f"{branch_name} radius length does not match centerline length")
        if radius[0] < radius[-1]:
            errors.append(f"{branch_name} proximal radius is smaller than distal radius")
        if not np.all(radius_steps <= 1e-9):
            warnings.append(f"{branch_name} radius is not monotonic nonincreasing")
        if len(radius_steps) > 0 and np.max(np.abs(radius_steps)) > adjacent_jump_threshold_mm:
            warnings.append(
                f"{branch_name} has an adjacent radius jump larger than {adjacent_jump_threshold_mm:.2f} mm"
            )

        branch_reports[branch_name] = {
            "num_points": int(len(points)),
            "proximal_radius_mm": float(radius[0]),
            "distal_radius_mm": float(radius[-1]),
            "min_radius_mm": float(np.min(radius)),
            "max_radius_mm": float(np.max(radius)),
            "max_step_mm": float(np.max(np.abs(radius_steps))) if len(radius_steps) > 0 else 0.0,
            "adjacent_jump_threshold_mm": float(adjacent_jump_threshold_mm),
            "passes_adjacent_jump_check": bool(
                len(radius_steps) == 0 or np.max(np.abs(radius_steps)) <= adjacent_jump_threshold_mm
            ),
            "monotonic_nonincreasing": bool(np.all(np.diff(radius) <= 1e-9)),
        }

    if all(branch in centerlines_with_radius for branch in ["LMCA", "LAD", "LCX"]):
        lmca_proximal = centerlines_with_radius["LMCA"][0, 3]
        lmca_distal = centerlines_with_radius["LMCA"][-1, 3]
        lad_proximal = centerlines_with_radius["LAD"][0, 3]
        lcx_proximal = centerlines_with_radius["LCX"][0, 3]
        if not lmca_proximal > lad_proximal:
            errors.append("LMCA proximal radius must be greater than LAD proximal radius")
        if not lmca_proximal > lcx_proximal:
            errors.append("LMCA proximal radius must be greater than LCX proximal radius")
        cube_law_distal = cube_law_parent_radius(lad_proximal, lcx_proximal)
        cube_law_relative_error = abs(lmca_distal - cube_law_distal) / max(cube_law_distal, 1e-9)
        if cube_law_relative_error > cube_law_relative_tolerance:
            errors.append(
                "LMCA distal radius is not compatible with LAD/LCX proximal radii "
                f"under cube-law tolerance {cube_law_relative_tolerance:.3f}"
            )
        branch_reports["bifurcation"] = {
            "lmca_distal_radius_mm": float(lmca_distal),
            "lad_proximal_radius_mm": float(lad_proximal),
            "lcx_proximal_radius_mm": float(lcx_proximal),
            "cube_law_lmca_distal_radius_mm": float(cube_law_distal),
            "cube_law_relative_error": float(cube_law_relative_error),
            "cube_law_relative_tolerance": float(cube_law_relative_tolerance),
            "passes_cube_law_check": bool(cube_law_relative_error <= cube_law_relative_tolerance),
        }

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "adjacent_jump_threshold_mm": adjacent_jump_threshold_mm,
        "cube_law_relative_tolerance": cube_law_relative_tolerance,
        "branches": branch_reports,
    }
