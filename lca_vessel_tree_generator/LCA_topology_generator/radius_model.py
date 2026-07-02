import numpy as np


DEFAULT_STATIC_RADIUS_MODEL = {
    "units": "mm",
    "description": "MVP static coronary radius taper; no disease, motion, pulsatility, or random sampling.",
    "branches": {
        "LMCA": {
            "proximal_radius_mm": 2.0,
            "distal_fraction": 0.85,
            "taper_exponent": 1.0,
        },
        "LAD": {
            "proximal_radius_mm": 1.3,
            "distal_fraction": 0.55,
            "taper_exponent": 1.0,
        },
        "LCX": {
            "proximal_radius_mm": 1.2,
            "distal_fraction": 0.55,
            "taper_exponent": 1.0,
        },
    },
}


def arc_length_fraction(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=float)
    if len(points) == 1:
        return np.zeros(1, dtype=float)

    distances = np.zeros(len(points), dtype=float)
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
    total = distances[-1]
    if total <= 1e-9:
        return np.zeros(len(points), dtype=float)
    return distances / total


def static_taper_radius(
    centerline: np.ndarray,
    proximal_radius_mm: float,
    distal_radius_mm: float,
    taper_exponent: float = 1.0,
) -> np.ndarray:
    """
    Creates a monotonic static taper radius profile along a centerline.

    `taper_exponent=1.0` preserves the original MVP linear taper. Values above
    or below 1.0 provide configurable non-linear tapering without adding random
    sampling, disease, motion, or pulsatility.
    """
    if proximal_radius_mm <= 0 or distal_radius_mm <= 0:
        raise ValueError("Radii must be positive.")
    if distal_radius_mm > proximal_radius_mm:
        raise ValueError("Distal radius must be less than or equal to proximal radius.")
    if taper_exponent <= 0:
        raise ValueError("Taper exponent must be positive.")

    s = arc_length_fraction(centerline)
    return proximal_radius_mm - (proximal_radius_mm - distal_radius_mm) * np.power(s, taper_exponent)


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
        "metadata_source": radius_model.get("metadata_source"),
        "branch_radius_source": radius_model.get("branch_radius_source", {}),
        "branches": {},
    }

    for branch_name in ["LMCA", "LAD", "LCX"]:
        params = radius_model["branches"][branch_name]
        proximal_radius_mm = float(params["proximal_radius_mm"])
        distal_radius_mm = float(params.get(
            "distal_radius_mm",
            proximal_radius_mm * float(params["distal_fraction"]),
        ))
        taper_exponent = float(params.get("taper_exponent", 1.0))
        radius = static_taper_radius(
            centerlines[branch_name],
            proximal_radius_mm=proximal_radius_mm,
            distal_radius_mm=distal_radius_mm,
            taper_exponent=taper_exponent,
        )
        centerlines_with_radius[branch_name] = add_radius_to_centerline(centerlines[branch_name], radius)
        metadata["branches"][branch_name] = {
            **params,
            "proximal_radius_mm": proximal_radius_mm,
            "distal_radius_mm": distal_radius_mm,
            "taper_exponent": taper_exponent,
            "num_points": int(len(radius)),
            "min_radius_mm": float(np.min(radius)),
            "max_radius_mm": float(np.max(radius)),
            "mean_radius_mm": float(np.mean(radius)),
            "is_monotonic_nonincreasing": bool(np.all(np.diff(radius) <= 1e-9)),
        }

    return centerlines_with_radius, metadata


def validate_lca_radius_tree(
    centerlines_with_radius: dict,
    adjacent_jump_threshold_mm: float = 0.2,
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
        if lmca_distal < lad_proximal:
            errors.append("LMCA distal radius must be greater than or equal to LAD proximal radius")
        if lmca_distal < lcx_proximal:
            errors.append("LMCA distal radius must be greater than or equal to LCX proximal radius")
        branch_reports["bifurcation"] = {
            "lmca_distal_radius_mm": float(lmca_distal),
            "lad_proximal_radius_mm": float(lad_proximal),
            "lcx_proximal_radius_mm": float(lcx_proximal),
            "lmca_distal_gte_lad_proximal": bool(lmca_distal >= lad_proximal),
            "lmca_distal_gte_lcx_proximal": bool(lmca_distal >= lcx_proximal),
        }

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "adjacent_jump_threshold_mm": adjacent_jump_threshold_mm,
        "branches": branch_reports,
    }
