"""Exact surface parameterization (u, v, offset) and local deviation vector calculation."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

EPS = 1.0e-12


def ellipsoid_point(u: float, v: float, a: float, b: float, c: float) -> np.ndarray:
    """Return 3D surface point on ellipsoid for parameters (u, v) in cardiac frame.

    Design Doc §4.3:
    x = a * sin(v) * cos(u)
    y = b * sin(v) * sin(u)
    z = c * cos(v)
    """
    x = a * math.sin(v) * math.cos(u)
    y = b * math.sin(v) * math.sin(u)
    z = c * math.cos(v)
    return np.array([x, y, z], dtype=float)


def ellipsoid_normal(u: float, v: float, a: float, b: float, c: float) -> np.ndarray:
    """Return outward unit normal vector on ellipsoid for parameters (u, v)."""
    nx = math.sin(v) * math.cos(u) / a
    ny = math.sin(v) * math.sin(u) / b
    nz = math.cos(v) / c
    n = np.array([nx, ny, nz], dtype=float)
    norm = float(np.linalg.norm(n))
    return n / max(norm, EPS)


def ellipsoid_tangent_u(u: float, v: float, a: float, b: float, c: float) -> np.ndarray:
    """Return unit circumferential tangent vector along u direction."""
    # The common sin(v) factor vanishes at both poles even though the limiting
    # circumferential direction remains well defined for the stored azimuth.
    # Remove that scale before normalization so the local basis does not
    # become singular for points whose z/c projection is clipped to a pole.
    del v, c
    tx = -a * math.sin(u)
    ty = b * math.cos(u)
    tz = 0.0
    t = np.array([tx, ty, tz], dtype=float)
    norm = float(np.linalg.norm(t))
    if norm <= EPS:
        raise ValueError("ellipsoid circumferential tangent is singular")
    return t / norm


def ellipsoid_tangent_v(u: float, v: float, a: float, b: float, c: float) -> np.ndarray:
    """Return unit longitudinal tangent vector along v direction (base to apex)."""
    tx = a * math.cos(v) * math.cos(u)
    ty = b * math.cos(v) * math.sin(u)
    tz = -c * math.sin(v)
    t = np.array([tx, ty, tz], dtype=float)
    norm = float(np.linalg.norm(t))
    if norm <= EPS:
        return np.array([0.0, 0.0, -1.0], dtype=float)
    return t / norm


def parameterize_point_angular_radial(
    cardiac_pt: np.ndarray, a: float, b: float, c: float
) -> tuple[float, float, float, np.ndarray]:
    """Map cardiac XYZ to angular/radial ellipsoid coordinates and a local residual.

    This exactly reconstructable angular parameterization is not a Euclidean
    nearest-point-on-ellipsoid optimization.

    Design Doc §4.4 & §5.3.1:
    - v: polar angle [0, pi] (base v=0 to apex v=pi) -> acos(clip(z/c, -1, 1))
    - u: azimuthal angle [-pi, pi) -> atan2(y/b, x/a)
    - offset: signed distance along outward normal
    - deviation: 3D vector [dev_x (tangent_u), dev_y (tangent_v), dev_z (normal)]
    """
    x, y, z = cardiac_pt
    v = math.acos(float(np.clip(z / c, -1.0, 1.0)))
    u = math.atan2(y / b, x / a)

    surf = ellipsoid_point(u, v, a, b, c)
    normal = ellipsoid_normal(u, v, a, b, c)
    tang_u = ellipsoid_tangent_u(u, v, a, b, c)
    tang_v = ellipsoid_tangent_v(u, v, a, b, c)

    residual = cardiac_pt - surf
    # On a triaxial ellipsoid the two unit tangents are generally not
    # orthogonal to each other.  Independent dot products therefore do not
    # recover the coefficients of the local basis and visibly distort a path
    # when it is reconstructed.  Solve the complete local basis instead.  The
    # normal remains orthogonal to both tangents, while least-squares also
    # handles the numerically delicate pole neighbourhoods safely.
    local_basis = np.column_stack((tang_u, tang_v, normal))
    deviation_vector, _, rank, _ = np.linalg.lstsq(local_basis, residual, rcond=None)
    if rank < 3 or not np.all(np.isfinite(deviation_vector)):
        raise ValueError("ellipsoid local basis is singular or non-finite")
    deviation_vector = np.asarray(deviation_vector, dtype=float)
    offset = float(deviation_vector[2])

    return u, v, offset, deviation_vector


def project_point_to_surface(
    cardiac_pt: np.ndarray, a: float, b: float, c: float
) -> tuple[float, float, float, np.ndarray]:
    """Compatibility alias for angular/radial surface parameterization.

    Historical callers retain this public name. It must not be interpreted as
    a Euclidean nearest-surface-point solver.
    """
    return parameterize_point_angular_radial(cardiac_pt, a, b, c)


def project_centerline_to_surface(
    cardiac_points: np.ndarray, a: float, b: float, c: float
) -> dict[str, np.ndarray]:
    """Project a full vessel centerline to surface coordinates (u, v, offset, deviations)."""
    cardiac_points = np.asarray(cardiac_points, dtype=float)
    n_pts = len(cardiac_points)

    u_arr = np.zeros(n_pts, dtype=float)
    v_arr = np.zeros(n_pts, dtype=float)
    offset_arr = np.zeros(n_pts, dtype=float)
    dev_arr = np.zeros((n_pts, 3), dtype=float)

    for i in range(n_pts):
        u, v, off, dev = project_point_to_surface(cardiac_points[i], a, b, c)
        u_arr[i] = u
        v_arr[i] = v
        offset_arr[i] = off
        dev_arr[i] = dev

    return {
        "u": u_arr,
        "v": v_arr,
        "offset": offset_arr,
        "deviations": dev_arr,
    }
