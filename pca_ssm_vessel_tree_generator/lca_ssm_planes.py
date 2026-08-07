"""Deterministic plane fitting and projection helpers for the LCA SSM."""

from __future__ import annotations

import numpy as np


EPS = 1.0e-12


def as_points(points: np.ndarray, *, name: str = "points", minimum: int = 3) -> np.ndarray:
    """Return a validated ``(N, 3)`` floating-point point cloud."""
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {array.shape}")
    if len(array) < minimum:
        raise ValueError(f"{name} requires at least {minimum} points; got {len(array)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite coordinates")
    return array


def canonicalize_direction(vector: np.ndarray) -> np.ndarray:
    """Normalize a direction and choose a deterministic sign."""
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError("cannot normalize a zero or non-finite direction")
    vector = vector / norm
    dominant = int(np.argmax(np.abs(vector)))
    if vector[dominant] < 0.0:
        vector = -vector
    return vector


def build_plane_basis(
    normal: np.ndarray, preferred_direction: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic in-plane axes ``u`` and ``v``."""
    normal = canonicalize_direction(normal)
    if preferred_direction is not None:
        candidate = np.asarray(preferred_direction, dtype=float)
        candidate = candidate - np.dot(candidate, normal) * normal
        if np.linalg.norm(candidate) > EPS:
            u = candidate / np.linalg.norm(candidate)
        else:
            preferred_direction = None
    if preferred_direction is None:
        axes = np.eye(3)
        reference = axes[int(np.argmin(np.abs(axes @ normal)))]
        u = reference - np.dot(reference, normal) * normal
        u /= np.linalg.norm(u)
        u = canonicalize_direction(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return u, v


def fit_plane_svd(points: np.ndarray, *, name: str = "points") -> dict:
    """Fit a least-squares plane by SVD and report geometric residuals."""
    points = as_points(points, name=name, minimum=3)
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    scale = max(float(singular_values[0]), EPS)
    rank = int(np.sum(singular_values > scale * 1.0e-8))
    if rank < 2:
        raise ValueError(f"{name} is collinear; a stable plane cannot be fitted")
    normal = canonicalize_direction(vh[-1])
    preferred = points[-1] - points[0]
    u, v = build_plane_basis(normal, preferred)
    signed = centered @ normal
    residuals = np.abs(signed)
    return {
        "centroid": centroid,
        "normal": normal,
        "basis_u": u,
        "basis_v": v,
        "singular_values": singular_values,
        "rank": rank,
        "signed_distances_mm": signed,
        "point_residuals_mm": residuals,
        "rms_residual_mm": float(np.sqrt(np.mean(signed**2))),
        "mean_residual_mm": float(np.mean(residuals)),
        "max_residual_mm": float(np.max(residuals)),
    }


def fit_plane_through_point_svd(
    points: np.ndarray,
    origin: np.ndarray,
    *,
    preferred_direction: np.ndarray,
    normal_reference: np.ndarray | None = None,
    contained_direction: np.ndarray | None = None,
    name: str = "points",
) -> dict:
    """Fit an SVD plane constrained to pass through a supplied 3-D point.

    Unlike :func:`fit_plane_svd`, the SVD is performed about ``origin`` rather
    than the data centroid.  ``preferred_direction`` establishes a shared
    in-plane ``u`` orientation, while ``normal_reference`` resolves the normal
    sign consistently across multiple planes.
    """
    points = as_points(points, name=name, minimum=3)
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("plane origin must be a finite 3-D point")
    local = points - origin
    constraint_direction = None
    if contained_direction is not None:
        constraint_direction = np.asarray(contained_direction, dtype=float)
        if constraint_direction.shape != (3,) or np.linalg.norm(constraint_direction) <= EPS:
            raise ValueError("contained_direction must be a non-zero 3-D direction")
        constraint_direction = constraint_direction / np.linalg.norm(constraint_direction)
        axes = np.eye(3)
        reference = axes[int(np.argmin(np.abs(axes @ constraint_direction)))]
        perpendicular_u = reference - np.dot(reference, constraint_direction) * constraint_direction
        perpendicular_u /= np.linalg.norm(perpendicular_u)
        perpendicular_v = np.cross(constraint_direction, perpendicular_u)
        perpendicular_v /= np.linalg.norm(perpendicular_v)
        normal_space = np.column_stack((perpendicular_u, perpendicular_v))
        _, singular_values, vh = np.linalg.svd(local @ normal_space, full_matrices=False)
        normal = normal_space @ vh[-1]
    else:
        _, singular_values, vh = np.linalg.svd(local, full_matrices=False)
        normal = vh[-1]
    scale = max(float(singular_values[0]), EPS)
    rank = int(np.sum(singular_values > scale * 1.0e-8))
    minimum_rank = 1 if constraint_direction is not None else 2
    if rank < minimum_rank:
        raise ValueError(f"{name} is collinear; a stable constrained plane cannot be fitted")
    normal = normal / np.linalg.norm(normal)
    if normal_reference is not None:
        reference = np.asarray(normal_reference, dtype=float)
        if reference.shape != (3,) or np.linalg.norm(reference) <= EPS:
            raise ValueError("normal_reference must be a non-zero 3-D direction")
        if np.dot(normal, reference) < 0.0:
            normal = -normal
    else:
        normal = canonicalize_direction(normal)
    u, v = build_plane_basis(normal, preferred_direction)
    # build_plane_basis canonicalizes the normal sign internally; restore the
    # requested normal orientation and right-handed (u, v, normal) frame.
    if np.dot(np.cross(u, v), normal) < 0.0:
        v = -v
    signed = local @ normal
    residuals = np.abs(signed)
    return {
        "centroid": origin.copy(),
        "origin": origin.copy(),
        "data_centroid": points.mean(axis=0),
        "normal": normal,
        "basis_u": u,
        "basis_v": v,
        "singular_values": singular_values,
        "rank": rank,
        "signed_distances_mm": signed,
        "point_residuals_mm": residuals,
        "rms_residual_mm": float(np.sqrt(np.mean(signed**2))),
        "mean_residual_mm": float(np.mean(residuals)),
        "max_residual_mm": float(np.max(residuals)),
        "origin_plane_distance_mm": float(abs(np.dot(origin - origin, normal))),
        "contained_direction": constraint_direction,
        "contained_direction_plane_dot": (
            None if constraint_direction is None else float(abs(np.dot(constraint_direction, normal)))
        ),
        "basis_handedness": float(np.linalg.det(np.column_stack((u, v, normal)))),
    }


def project_points_to_plane(points: np.ndarray, plane: dict) -> tuple[np.ndarray, np.ndarray]:
    """Project 3-D points into a fitted plane's 2-D coordinates and back to 3-D."""
    points = as_points(points, minimum=1)
    centered = points - np.asarray(plane["centroid"])
    normal = np.asarray(plane["normal"])
    projected_3d = points - np.outer(centered @ normal, normal)
    projected_centered = projected_3d - np.asarray(plane["centroid"])
    projected_2d = np.column_stack(
        (projected_centered @ plane["basis_u"], projected_centered @ plane["basis_v"])
    )
    return projected_2d, projected_3d


def lift_plane_points(points_2d: np.ndarray, plane: dict) -> np.ndarray:
    """Lift 2-D plane coordinates back into 3-D."""
    points_2d = np.asarray(points_2d, dtype=float)
    if points_2d.ndim != 2 or points_2d.shape[1] != 2:
        raise ValueError(f"points_2d must have shape (N, 2); got {points_2d.shape}")
    return (
        np.asarray(plane["centroid"])
        + np.outer(points_2d[:, 0], plane["basis_u"])
        + np.outer(points_2d[:, 1], plane["basis_v"])
    )


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Return the unsigned angle between two non-zero vectors in degrees."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= EPS or not np.isfinite(denominator):
        raise ValueError("angle requires two finite, non-zero vectors")
    cosine = float(np.clip(np.dot(first, second) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))
