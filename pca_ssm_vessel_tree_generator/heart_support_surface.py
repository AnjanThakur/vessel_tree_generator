"""Reusable ellipsoidal anatomical support-surface parameterization.

This module models a dataset-derived reference scaffold.  It is not a
patient-specific myocardial reconstruction and it never modifies coronary
centerlines.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from scipy.optimize import minimize


EPS = 1.0e-12


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise ValueError("axis vector must have non-zero length")
    return vector / norm


@dataclass
class HeartSupportSurface:
    """Ellipsoidal anatomical support surface in one canonical rigid frame.

    Parameters are full spans in millimetres.  The baseline exponent is two,
    yielding a standard ellipsoid.  The named axes are expressed in the same
    canonical coordinate system as ``origin``.
    """

    crown_span: float
    crown_depth: float
    long_axis_span: float
    origin: np.ndarray
    crown_axis: np.ndarray
    secondary_crown_axis: np.ndarray
    apex_axis: np.ndarray
    exponent: float = 2.0

    def __post_init__(self) -> None:
        self.origin = np.asarray(self.origin, dtype=float).reshape(3)
        self.crown_axis = _unit(self.crown_axis)
        self.secondary_crown_axis = _unit(self.secondary_crown_axis)
        self.apex_axis = _unit(self.apex_axis)
        if min(self.crown_span, self.crown_depth, self.long_axis_span) <= 0.0:
            raise ValueError("all support-surface spans must be positive")
        if abs(float(self.exponent) - 2.0) > 1.0e-12:
            raise ValueError("the validated Stage-1 baseline currently supports exponent n=2 only")
        frame = self.frame_matrix
        error = float(np.max(np.abs(frame @ frame.T - np.eye(3))))
        if error > 1.0e-10 or float(np.linalg.det(frame)) <= 0.0:
            raise ValueError("surface axes must form a right-handed orthonormal frame")

    @property
    def semi_axes(self) -> np.ndarray:
        return 0.5 * np.array(
            [self.crown_span, self.crown_depth, self.long_axis_span], dtype=float
        )

    @property
    def frame_matrix(self) -> np.ndarray:
        """Rows map canonical XYZ vectors into surface-local coordinates."""
        superior_axis = -self.apex_axis
        return np.vstack((self.crown_axis, self.secondary_crown_axis, superior_axis))

    def _local_to_xyz(self, local: np.ndarray) -> np.ndarray:
        local = np.asarray(local, dtype=float)
        return self.origin + local @ self.frame_matrix

    def _xyz_to_local(self, point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float)
        return (point - self.origin) @ self.frame_matrix.T

    def surface_point(self, theta: Any, phi: Any) -> np.ndarray:
        """Return surface XYZ for azimuth ``theta`` and latitude ``phi``."""
        theta, phi = np.broadcast_arrays(
            np.asarray(theta, dtype=float), np.asarray(phi, dtype=float)
        )
        a, b, c = self.semi_axes
        local = np.stack(
            (
                a * np.cos(phi) * np.cos(theta),
                b * np.cos(phi) * np.sin(theta),
                c * np.sin(phi),
            ),
            axis=-1,
        )
        return self._local_to_xyz(local)

    def surface_normal(self, theta: Any, phi: Any) -> np.ndarray:
        """Return outward unit normals for surface coordinates."""
        points = self.surface_point(theta, phi)
        local = self._xyz_to_local(points)
        gradient_local = local / (self.semi_axes**2)
        gradient_xyz = gradient_local @ self.frame_matrix
        norm = np.linalg.norm(gradient_xyz, axis=-1, keepdims=True)
        return gradient_xyz / np.maximum(norm, EPS)

    def xyz_to_surface_coordinates(self, point: np.ndarray) -> tuple[float, float, float]:
        """Return ``(theta, phi, radial_scale)`` for an arbitrary XYZ point.

        ``radial_scale`` equals one on the ellipsoid, is less than one inside,
        and is greater than one outside.  It is a dimensionless diagnostic and
        is not an Euclidean distance.
        """
        local = self._xyz_to_local(np.asarray(point, dtype=float).reshape(3))
        scaled = local / self.semi_axes
        radial = float(np.linalg.norm(scaled))
        if radial <= EPS:
            return 0.0, 0.0, 0.0
        theta = float(np.arctan2(scaled[1], scaled[0]))
        phi = float(np.arctan2(scaled[2], np.hypot(scaled[0], scaled[1])))
        return theta, phi, radial

    def surface_coordinates_to_xyz(self, theta: Any, phi: Any) -> np.ndarray:
        return self.surface_point(theta, phi)

    @cached_property
    def _coarse_candidates(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        theta = np.linspace(-np.pi, np.pi, 72, endpoint=False)
        phi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 37)
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        points = self.surface_point(theta_grid, phi_grid).reshape(-1, 3)
        return points, theta_grid.ravel(), phi_grid.ravel()

    def nearest_surface_point(self, point: np.ndarray) -> np.ndarray:
        """Numerically compute the Euclidean nearest point on the ellipsoid."""
        point = np.asarray(point, dtype=float).reshape(3)
        coarse_points, coarse_theta, coarse_phi = self._coarse_candidates
        distance_squared = np.sum((coarse_points - point) ** 2, axis=1)
        candidate_indices = np.argpartition(distance_squared, 4)[:4]
        best_point = coarse_points[candidate_indices[0]]
        best_distance = float(distance_squared[candidate_indices[0]])

        def objective(parameters: np.ndarray) -> float:
            candidate = self.surface_point(parameters[0], parameters[1])
            return float(np.sum((candidate - point) ** 2))

        for index in candidate_indices:
            result = minimize(
                objective,
                np.array([coarse_theta[index], coarse_phi[index]], dtype=float),
                method="L-BFGS-B",
                bounds=((-np.pi, np.pi), (-0.5 * np.pi, 0.5 * np.pi)),
                options={"ftol": 1.0e-14, "gtol": 1.0e-10, "maxiter": 200},
            )
            candidate = self.surface_point(result.x[0], result.x[1])
            candidate_distance = float(np.sum((candidate - point) ** 2))
            if candidate_distance < best_distance:
                best_point = candidate
                best_distance = candidate_distance
        return np.asarray(best_point, dtype=float)

    def distance_to_surface(self, point: np.ndarray) -> float:
        point = np.asarray(point, dtype=float).reshape(3)
        return float(np.linalg.norm(point - self.nearest_surface_point(point)))

    def nearest_surface_points(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        nearest = np.vstack([self.nearest_surface_point(point) for point in points])
        distances = np.linalg.norm(points - nearest, axis=1)
        return nearest, distances

    def mesh_arrays(
        self, n_theta: int = 180, n_phi: int = 91
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Return non-degenerate triangulated surface arrays."""
        if n_theta < 12 or n_phi < 7:
            raise ValueError("surface mesh resolution is too low")
        theta = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
        interior_phi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_phi)[1:-1]
        theta_grid, phi_grid = np.meshgrid(theta, interior_phi)
        ring_points = self.surface_point(theta_grid, phi_grid).reshape(-1, 3)
        bottom = self.surface_point(0.0, -0.5 * np.pi).reshape(1, 3)
        top = self.surface_point(0.0, 0.5 * np.pi).reshape(1, 3)
        vertices = np.vstack((bottom, ring_points, top))
        bottom_index = 0
        first_ring = 1
        ring_count = len(interior_phi)
        top_index = len(vertices) - 1
        triangles: list[list[int]] = []
        for column in range(n_theta):
            nxt = (column + 1) % n_theta
            triangles.append([bottom_index, first_ring + nxt, first_ring + column])
        for ring in range(ring_count - 1):
            lower = first_ring + ring * n_theta
            upper = lower + n_theta
            for column in range(n_theta):
                nxt = (column + 1) % n_theta
                triangles.append([lower + column, lower + nxt, upper + nxt])
                triangles.append([lower + column, upper + nxt, upper + column])
        last_ring = first_ring + (ring_count - 1) * n_theta
        for column in range(n_theta):
            nxt = (column + 1) % n_theta
            triangles.append([top_index, last_ring + column, last_ring + nxt])
        faces = np.asarray(triangles, dtype=np.int64)
        theta_values = np.concatenate(([-np.pi], theta_grid.ravel(), [-np.pi]))
        phi_values = np.concatenate(
            (([-0.5 * np.pi]), phi_grid.ravel(), ([0.5 * np.pi]))
        )
        normals = self.surface_normal(theta_values, phi_values)
        metadata = {
            "theta_rad": theta_values,
            "phi_rad": phi_values,
            "surface_normal": normals,
        }
        return vertices, faces, metadata

    def to_record(self) -> dict[str, Any]:
        a, b, c = self.semi_axes
        return {
            "model": "ellipsoidal dataset-derived anatomical support surface",
            "scientific_claim": (
                "Dataset-derived parametric anatomical support scaffold; "
                "not a patient-specific myocardial reconstruction."
            ),
            "exponent_n": float(self.exponent),
            "origin_canonical_mm": self.origin.tolist(),
            "crown_span_mm": float(self.crown_span),
            "crown_depth_mm": float(self.crown_depth),
            "long_axis_span_mm": float(self.long_axis_span),
            "semi_axes_mm": {"a_crown": float(a), "b_depth": float(b), "c_long_axis": float(c)},
            "axes_canonical": {
                "crown_axis": self.crown_axis.tolist(),
                "secondary_crown_axis": self.secondary_crown_axis.tolist(),
                "apex_axis": self.apex_axis.tolist(),
            },
            "implicit_equation": "(x/a)^2 + (y/b)^2 + (z/c)^2 = 1 in surface-local coordinates",
        }
