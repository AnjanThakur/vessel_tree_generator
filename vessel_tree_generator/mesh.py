import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=float)
    return vector / norm


def _initial_normal(tangent: np.ndarray) -> np.ndarray:
    axes = np.eye(3)
    axis = axes[np.argmin(np.abs(axes @ tangent))]
    normal = axis - np.dot(axis, tangent) * tangent
    return _unit(normal)


def _frame_vectors(points: np.ndarray) -> tuple:
    tangents = np.gradient(points, axis=0)
    tangents = np.array([_unit(tangent) for tangent in tangents])

    normals = np.zeros_like(points)
    binormals = np.zeros_like(points)
    normals[0] = _initial_normal(tangents[0])
    binormals[0] = _unit(np.cross(tangents[0], normals[0]))

    for idx in range(1, len(points)):
        tangent = tangents[idx]
        previous_normal = normals[idx - 1]
        normal = previous_normal - np.dot(previous_normal, tangent) * tangent
        if np.linalg.norm(normal) <= 1e-9:
            normal = _initial_normal(tangent)
        normals[idx] = _unit(normal)
        binormals[idx] = _unit(np.cross(tangent, normals[idx]))

    return normals, binormals


def centerline_radius_to_surface(
    centerline_with_radius: np.ndarray,
    num_circle_points: int = 24,
) -> np.ndarray:
    """
    Sweeps circular cross-sections along an Nx4 [x, y, z, radius] centerline.

    Returns an array with shape (N, num_circle_points, 3).
    """
    centerline_with_radius = np.asarray(centerline_with_radius, dtype=float)
    if centerline_with_radius.ndim != 2 or centerline_with_radius.shape[1] != 4:
        raise ValueError("Expected centerline_with_radius shape (N, 4).")
    if len(centerline_with_radius) < 2:
        raise ValueError("At least two centerline points are required for tube generation.")
    if num_circle_points < 6:
        raise ValueError("num_circle_points must be at least 6.")

    points = centerline_with_radius[:, :3]
    radius = centerline_with_radius[:, 3]
    if np.any(~np.isfinite(centerline_with_radius)):
        raise ValueError("Centerline/radius contains NaN or Inf values.")
    if np.any(radius <= 0):
        raise ValueError("Radius values must be positive.")

    normals, binormals = _frame_vectors(points)
    theta = np.linspace(0.0, 2.0 * np.pi, num_circle_points, endpoint=False)
    circle = (
        np.cos(theta)[None, :, None] * normals[:, None, :]
        + np.sin(theta)[None, :, None] * binormals[:, None, :]
    )
    return points[:, None, :] + radius[:, None, None] * circle


def build_lca_tube_surfaces(
    centerlines_with_radius: dict,
    num_circle_points: int = 24,
) -> tuple:
    surfaces = {}
    metadata = {
        "surface_shape": "N x num_circle_points x 3",
        "num_circle_points": int(num_circle_points),
        "branches": {},
    }

    for branch_name in ["LMCA", "LAD", "LCX"]:
        surface = centerline_radius_to_surface(
            centerlines_with_radius[branch_name],
            num_circle_points=num_circle_points,
        )
        surfaces[branch_name] = surface
        metadata["branches"][branch_name] = {
            "shape": list(surface.shape),
            "num_centerline_points": int(surface.shape[0]),
            "num_circle_points": int(surface.shape[1]),
        }

    return surfaces, metadata


def validate_lca_tube_surfaces(surfaces: dict, centerlines_with_radius: dict) -> dict:
    errors = []
    warnings = []
    branch_reports = {}

    for branch_name in ["LMCA", "LAD", "LCX"]:
        if branch_name not in surfaces:
            errors.append(f"{branch_name} surface is missing")
            continue
        if branch_name not in centerlines_with_radius:
            errors.append(f"{branch_name} centerline radius array is missing")
            continue

        surface = np.asarray(surfaces[branch_name], dtype=float)
        centerline = np.asarray(centerlines_with_radius[branch_name], dtype=float)
        if surface.ndim != 3 or surface.shape[2] != 3:
            errors.append(f"{branch_name} surface must have shape (N, C, 3)")
            continue
        if surface.shape[0] != len(centerline):
            errors.append(f"{branch_name} surface length does not match centerline length")
        if not np.all(np.isfinite(surface)):
            errors.append(f"{branch_name} surface contains NaN or Inf values")
        if surface.shape[1] < 6:
            warnings.append(f"{branch_name} has very few circle samples")

        branch_reports[branch_name] = {
            "shape": list(surface.shape),
            "matches_centerline_length": bool(surface.shape[0] == len(centerline)),
            "is_finite": bool(np.all(np.isfinite(surface))),
        }

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "branches": branch_reports,
    }
