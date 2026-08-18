"""B-spline coronary paths evaluated through Person 1's ellipsoid API."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator

from surface_relative.surface_projection import (
    ellipsoid_normal,
    ellipsoid_point,
    ellipsoid_tangent_u,
    ellipsoid_tangent_v,
    project_point_to_surface,
)

from generation.landmark_sampler import SurfaceLandmark, wrap_angle
from generation.parameter_sampler import EllipsoidParameters


@dataclass
class SurfacePath:
    """One generated path in surface coordinates and cardiac XYZ."""

    name: str
    u: np.ndarray
    v: np.ndarray
    normal_offset: np.ndarray
    local_deviation: np.ndarray
    points: np.ndarray
    u_control_points: np.ndarray
    v_control_points: np.ndarray


def shortest_angular_delta(start: float, end: float) -> float:
    return wrap_angle(end - start)


def open_uniform_knots(n_control: int, degree: int) -> np.ndarray:
    if n_control < 2:
        raise ValueError("a B-spline requires at least two control points")
    if not 1 <= degree < n_control:
        raise ValueError(f"degree must be in [1, n_control); got {degree}")
    interior_count = n_control - degree - 1
    interior = (
        np.arange(1, interior_count + 1, dtype=float) / (interior_count + 1)
        if interior_count > 0
        else np.empty(0, dtype=float)
    )
    return np.concatenate((np.zeros(degree + 1), interior, np.ones(degree + 1)))


def evaluate_bspline_controls(control_values: np.ndarray, sample_count: int) -> np.ndarray:
    control_values = np.asarray(control_values, dtype=float)
    if control_values.ndim != 1 or len(control_values) < 2:
        raise ValueError("control values must be a one-dimensional array with at least two entries")
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    degree = min(3, len(control_values) - 1)
    spline = BSpline(open_uniform_knots(len(control_values), degree), control_values, degree, extrapolate=False)
    values = np.asarray(spline(np.linspace(0.0, 1.0, sample_count)), dtype=float)
    values[0] = control_values[0]
    values[-1] = control_values[-1]
    return values


def interpolate_bspline_samples(sample_values: np.ndarray, sample_count: int) -> np.ndarray:
    """Shape-preservingly interpolate an empirical fixed trajectory.

    The public name is retained for compatibility with the first Person-2
    increment.  A global interpolating B-spline can overshoot sparse coronary
    samples and create hooks that were not present in the real path.  PCHIP is
    a smooth piecewise-cubic interpolant that passes through every fixed sample
    without overshooting each coordinate interval.
    """
    sample_values = np.asarray(sample_values, dtype=float)
    if sample_values.ndim != 1 or len(sample_values) < 2:
        raise ValueError("sample values must be one-dimensional with at least two entries")
    source = np.linspace(0.0, 1.0, len(sample_values))
    target = np.linspace(0.0, 1.0, sample_count)
    values = np.asarray(PchipInterpolator(source, sample_values)(target), dtype=float)
    values[0] = sample_values[0]
    values[-1] = sample_values[-1]
    return values


def surface_coordinate_to_point(
    u: float,
    v: float,
    normal_offset: float,
    ellipsoid: EllipsoidParameters,
    deviation: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate one point using the shared Person 1 surface basis."""
    local = np.zeros(3, dtype=float) if deviation is None else np.asarray(deviation, dtype=float).reshape(3)
    surface = ellipsoid_point(u, v, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    tangent_u = ellipsoid_tangent_u(u, v, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    tangent_v = ellipsoid_tangent_v(u, v, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    normal = ellipsoid_normal(u, v, ellipsoid.a, ellipsoid.b, ellipsoid.c)
    return surface + local[0] * tangent_u + local[1] * tangent_v + (normal_offset + local[2]) * normal


def reconstruct_surface_path(
    u: np.ndarray,
    v: np.ndarray,
    normal_offset: np.ndarray,
    local_deviation: np.ndarray,
    ellipsoid: EllipsoidParameters,
) -> np.ndarray:
    arrays = [np.asarray(value, dtype=float) for value in (u, v, normal_offset)]
    count = len(arrays[0])
    if any(value.shape != (count,) for value in arrays):
        raise ValueError("u, v, and normal_offset must have identical one-dimensional shapes")
    deviations = np.asarray(local_deviation, dtype=float)
    if deviations.shape != (count, 3):
        raise ValueError(f"local_deviation must have shape {(count, 3)}; got {deviations.shape}")
    return np.vstack([
        surface_coordinate_to_point(arrays[0][i], arrays[1][i], arrays[2][i], ellipsoid, deviations[i])
        for i in range(count)
    ])


class SurfacePathGenerator:
    """Generate smooth paths from anatomically named surface landmarks."""

    def __init__(self, ellipsoid: EllipsoidParameters, rng: np.random.Generator):
        self.ellipsoid = ellipsoid
        self.rng = rng

    def generate(
        self,
        name: str,
        start: SurfaceLandmark,
        end: SurfaceLandmark,
        *,
        control_point_count: int,
        sample_count: int,
        tortuosity_strength_rad: float = 0.0,
        obliquity_rad: float = 0.0,
        local_deviation: np.ndarray | None = None,
        control_uvo: np.ndarray | None = None,
        control_points_xyz: np.ndarray | None = None,
    ) -> SurfacePath:
        if control_uvo is not None and control_points_xyz is not None:
            raise ValueError("supply either control_uvo or control_points_xyz, not both")
        if control_uvo is not None:
            empirical = np.asarray(control_uvo, dtype=float)
            if empirical.ndim != 2 or empirical.shape[1] != 3 or len(empirical) < 2:
                raise ValueError(f"control_uvo must have shape (n>=2, 3); got {empirical.shape}")
            control_point_count = len(empirical)
        if control_points_xyz is not None:
            empirical_xyz = np.asarray(control_points_xyz, dtype=float)
            if (
                empirical_xyz.ndim != 2
                or empirical_xyz.shape[1] != 3
                or len(empirical_xyz) < 2
                or not np.all(np.isfinite(empirical_xyz))
            ):
                raise ValueError(
                    f"control_points_xyz must have shape (n>=2, 3); got {empirical_xyz.shape}"
                )
            control_point_count = len(empirical_xyz)
        if control_point_count < 2:
            raise ValueError("control_point_count must be at least two")
        if tortuosity_strength_rad < 0.0:
            raise ValueError("tortuosity_strength_rad cannot be negative")
        deviations = (
            np.zeros((sample_count, 3), dtype=float)
            if local_deviation is None
            else np.asarray(local_deviation, dtype=float)
        )
        if deviations.shape != (sample_count, 3) or not np.all(np.isfinite(deviations)):
            raise ValueError(
                f"local_deviation must have shape {(sample_count, 3)}; got {deviations.shape}"
            )

        if control_points_xyz is not None:
            # Surface azimuth becomes undefined at the ellipsoid poles.  Do
            # not interpolate u/v there: interpolate the exact matched cardiac
            # control points directly, then project the smooth baseline back
            # to surface coordinates for metadata and local PCA innovation.
            baseline_points = np.column_stack([
                interpolate_bspline_samples(empirical_xyz[:, axis], sample_count)
                for axis in range(3)
            ])
            projected = [
                project_point_to_surface(
                    point, self.ellipsoid.a, self.ellipsoid.b, self.ellipsoid.c
                )
                for point in baseline_points
            ]
            u_path = np.unwrap(np.asarray([values[0] for values in projected], dtype=float))
            v_path = np.asarray([values[1] for values in projected], dtype=float)
            offset_path = np.asarray([values[2] for values in projected], dtype=float)
            baseline_deviation = np.vstack([values[3] for values in projected])
            baseline_deviation[:, 2] = 0.0
            combined_deviation = baseline_deviation + deviations
            points = reconstruct_surface_path(
                u_path, v_path, offset_path, combined_deviation, self.ellipsoid
            )
            control_projected = [
                project_point_to_surface(
                    point, self.ellipsoid.a, self.ellipsoid.b, self.ellipsoid.c
                )
                for point in empirical_xyz
            ]
            return SurfacePath(
                name=name,
                u=u_path,
                v=v_path,
                normal_offset=offset_path,
                local_deviation=combined_deviation,
                points=points,
                u_control_points=np.unwrap(np.asarray([values[0] for values in control_projected])),
                v_control_points=np.asarray([values[1] for values in control_projected]),
            )

        t_control = np.linspace(0.0, 1.0, control_point_count)
        unwrapped_end = start.u + shortest_angular_delta(start.u, end.u)
        if control_uvo is None:
            u_control = start.u + (unwrapped_end - start.u) * t_control
            v_control = start.v + (end.v - start.v) * t_control
            offset_control = start.offset + (end.offset - start.offset) * t_control
        else:
            u_control = np.unwrap(empirical[:, 0].copy())
            shift = start.u - u_control[0]
            u_control += shift
            v_control = empirical[:, 1].copy()
            offset_control = empirical[:, 2].copy()

        envelope = 4.0 * t_control * (1.0 - t_control)
        if obliquity_rad:
            u_control += float(obliquity_rad) * np.sin(math.pi * t_control) * envelope
        if tortuosity_strength_rad:
            for frequency in (1, 2, 3):
                amplitude_u = tortuosity_strength_rad * float(self.rng.uniform(0.25, 0.75)) / frequency
                amplitude_v = tortuosity_strength_rad * float(self.rng.uniform(0.20, 0.60)) / frequency
                phase_u = float(self.rng.uniform(0.0, 2.0 * math.pi))
                phase_v = float(self.rng.uniform(0.0, 2.0 * math.pi))
                u_control += amplitude_u * np.sin(frequency * math.pi * t_control + phase_u) * envelope
                v_control += amplitude_v * np.sin(frequency * math.pi * t_control + phase_v) * envelope

        u_control[0], u_control[-1] = start.u, unwrapped_end
        v_control[0], v_control[-1] = start.v, end.v
        offset_control[0], offset_control[-1] = start.offset, end.offset
        v_control = np.clip(v_control, 0.01, math.pi - 0.01)
        evaluator = interpolate_bspline_samples if control_uvo is not None else evaluate_bspline_controls
        u_path = evaluator(u_control, sample_count)
        v_path = np.clip(evaluator(v_control, sample_count), 0.01, math.pi - 0.01)
        offset_path = evaluator(offset_control, sample_count)
        points = reconstruct_surface_path(u_path, v_path, offset_path, deviations, self.ellipsoid)
        return SurfacePath(
            name=name,
            u=u_path,
            v=v_path,
            normal_offset=offset_path,
            local_deviation=deviations,
            points=points,
            u_control_points=u_control,
            v_control_points=v_control,
        )
