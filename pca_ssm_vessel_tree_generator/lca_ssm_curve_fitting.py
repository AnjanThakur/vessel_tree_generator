"""Curve fitting helpers for LCX ellipse arcs and LAD guide curves."""

from __future__ import annotations

import numpy as np


EPS = 1.0e-12
try:
    TRAPEZOID = np.trapezoid
except AttributeError:  # NumPy < 2.0
    TRAPEZOID = np.trapz


def cumulative_parameter(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return normalized cumulative arc position and polyline length."""
    points = np.asarray(points, dtype=float)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= EPS:
        raise ValueError("curve has zero arc length")
    return cumulative / total, total


def fit_quadratic_guide(points_2d: np.ndarray) -> dict:
    """Fit a path-order-aware quadratic guide curve in a plane."""
    points_2d = np.asarray(points_2d, dtype=float)
    if points_2d.ndim != 2 or points_2d.shape[1] != 2 or len(points_2d) < 3:
        raise ValueError("quadratic guide requires at least three 2-D points")
    t, path_length = cumulative_parameter(points_2d)
    degree = min(2, len(np.unique(t)) - 1)
    if degree < 1:
        raise ValueError("quadratic guide parameter is degenerate")
    coefficients = np.vstack(
        [np.polyfit(t, points_2d[:, axis], degree) for axis in range(2)]
    )
    fitted = np.column_stack(
        [np.polyval(coefficients[axis], t) for axis in range(2)]
    )
    residuals = np.linalg.norm(points_2d - fitted, axis=1)
    derivatives = np.column_stack(
        [np.polyval(np.polyder(coefficients[axis]), t) for axis in range(2)]
    )
    speed = np.linalg.norm(derivatives, axis=1)
    guide_length = float(TRAPEZOID(speed, t))
    return {
        "method": f"polynomial_degree_{degree}",
        "parameter_positions": t,
        "coefficients_x": coefficients[0],
        "coefficients_y": coefficients[1],
        "fitted_points_2d": fitted,
        "point_residuals_mm": residuals,
        "rms_residual_mm": float(np.sqrt(np.mean(residuals**2))),
        "mean_residual_mm": float(np.mean(residuals)),
        "max_residual_mm": float(np.max(residuals)),
        "projected_polyline_length_mm": path_length,
        "guide_length_mm": guide_length,
    }


def _ellipse_coordinates(points: np.ndarray, center: np.ndarray, theta: float) -> np.ndarray:
    rotation = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )
    return (points - center) @ rotation.T


def _ellipse_model(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    center = params[:2]
    a, b = np.exp(params[2:4])
    local = _ellipse_coordinates(points, center, params[4])
    return np.sqrt((local[:, 0] / a) ** 2 + (local[:, 1] / b) ** 2) - 1.0


def _ellipse_arc_length(a: float, b: float, start: float, end: float) -> float:
    samples = np.linspace(start, end, max(64, int(abs(end - start) * 64)))
    speed = np.sqrt((a * np.sin(samples)) ** 2 + (b * np.cos(samples)) ** 2)
    return float(abs(TRAPEZOID(speed, samples)))


def fit_lcx_ellipse_arc(points_2d: np.ndarray) -> dict:
    """Fit an ordered LCX ellipse arc, falling back to a PCA quadratic arc."""
    points = np.asarray(points_2d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5:
        raise ValueError("LCX ellipse fitting requires at least five 2-D points")

    centered = points - points.mean(axis=0)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    if singular[0] <= EPS:
        raise ValueError("LCX projected points have zero extent")
    initial_theta = float(np.arctan2(vh[0, 1], vh[0, 0]))
    local = _ellipse_coordinates(points, points.mean(axis=0), initial_theta)
    initial_axes = np.maximum(np.ptp(local, axis=0) / 2.0, singular[0] * 0.02)
    span = max(float(np.ptp(points, axis=0).max()), 1.0e-3)

    failure = None
    try:
        from scipy.optimize import least_squares

        lower = np.array(
            [points[:, 0].min() - 3 * span, points[:, 1].min() - 3 * span,
             np.log(span * 0.02), np.log(span * 0.02), -4 * np.pi]
        )
        upper = np.array(
            [points[:, 0].max() + 3 * span, points[:, 1].max() + 3 * span,
             np.log(span * 20.0), np.log(span * 20.0), 4 * np.pi]
        )
        initial = np.array(
            [*points.mean(axis=0), np.log(initial_axes[0]), np.log(initial_axes[1]), initial_theta]
        )
        optimized = least_squares(
            _ellipse_model, initial, args=(points,), bounds=(lower, upper),
            loss="soft_l1", f_scale=0.05, max_nfev=5000,
        )
        center = optimized.x[:2]
        a, b = np.exp(optimized.x[2:4])
        theta = float(optimized.x[4])
        if b > a:
            a, b = b, a
            theta += np.pi / 2.0
        theta = float((theta + np.pi) % (2 * np.pi) - np.pi)
        local = _ellipse_coordinates(points, center, theta)
        angles = np.unwrap(np.arctan2(local[:, 1] / b, local[:, 0] / a))
        fitted_local = np.column_stack((a * np.cos(angles), b * np.sin(angles)))
        inverse = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        fitted = fitted_local @ inverse.T + center
        residuals = np.linalg.norm(points - fitted, axis=1)
        axis_ratio = float(a / b)
        rms = float(np.sqrt(np.mean(residuals**2)))
        if (not optimized.success or not np.all(np.isfinite(fitted)) or axis_ratio > 25.0
                or rms > 0.35 * span):
            raise ValueError(
                f"unstable ellipse (success={optimized.success}, axis_ratio={axis_ratio:.3g}, "
                f"rms/span={rms/span:.3g})"
            )
        return {
            "method": "robust_nonlinear_ellipse",
            "fallback_reason": None,
            "center_2d": center,
            "semi_major_axis_mm": float(a),
            "semi_minor_axis_mm": float(b),
            "axis_ratio": axis_ratio,
            "orientation_rad": theta,
            "orientation_deg": float(np.degrees(theta)),
            "angular_positions_rad": angles,
            "arc_start_rad": float(angles[0]),
            "arc_end_rad": float(angles[-1]),
            "angular_extent_rad": float(abs(angles[-1] - angles[0])),
            "angular_extent_deg": float(np.degrees(abs(angles[-1] - angles[0]))),
            "arc_length_mm": _ellipse_arc_length(a, b, angles[0], angles[-1]),
            "fitted_points_2d": fitted,
            "point_residuals_mm": residuals,
            "rms_residual_mm": rms,
            "mean_residual_mm": float(np.mean(residuals)),
            "max_residual_mm": float(np.max(residuals)),
        }
    except Exception as exc:  # scipy missing or an unstable partial-arc ellipse
        failure = str(exc)

    guide = fit_quadratic_guide(points)
    local = points @ vh.T
    axes = np.maximum(np.ptp(local, axis=0) / 2.0, EPS)
    return {
        "method": "pca_quadratic_arc_fallback",
        "fallback_reason": failure,
        "center_2d": points.mean(axis=0),
        "semi_major_axis_mm": float(max(axes)),
        "semi_minor_axis_mm": float(min(axes)),
        "axis_ratio": float(max(axes) / min(axes)),
        "orientation_rad": initial_theta,
        "orientation_deg": float(np.degrees(initial_theta)),
        "angular_positions_rad": np.pi * guide["parameter_positions"],
        "arc_start_rad": 0.0,
        "arc_end_rad": float(np.pi),
        "angular_extent_rad": float(np.pi),
        "angular_extent_deg": 180.0,
        "arc_length_mm": guide["guide_length_mm"],
        "fitted_points_2d": guide["fitted_points_2d"],
        "point_residuals_mm": guide["point_residuals_mm"],
        "rms_residual_mm": guide["rms_residual_mm"],
        "mean_residual_mm": guide["mean_residual_mm"],
        "max_residual_mm": guide["max_residual_mm"],
        "guide_coefficients_x": guide["coefficients_x"],
        "guide_coefficients_y": guide["coefficients_y"],
    }


def _unit_vector_2d(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (2,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite 2-D vector")
    length = float(np.linalg.norm(vector))
    if length <= EPS:
        raise ValueError(f"{name} must be non-zero")
    return vector / length


def _endpoint_constrained_affine_arc(
    start: np.ndarray,
    end: np.ndarray,
    shape_vector: np.ndarray,
    angular_extent: float,
    parameter_positions: np.ndarray,
    *,
    dense_samples: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate an affine ellipse arc with analytically fixed endpoints.

    A non-singular affine image of the unit circle is an ellipse.  Starting at
    unit-circle angle zero removes an irrelevant phase degree of freedom.  The
    affine matrix is parameterized so the first and last points are exactly
    ``start`` and ``end``; ``shape_vector`` supplies its two remaining degrees
    of freedom.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    shape_vector = np.asarray(shape_vector, dtype=float)
    positions = np.asarray(parameter_positions, dtype=float)
    chord = end - start
    q = np.array([np.cos(angular_extent) - 1.0, np.sin(angular_extent)])
    q_perpendicular = np.array([-q[1], q[0]])
    denominator = float(np.dot(q, q))
    if denominator <= EPS:
        raise ValueError("ellipse angular extent is degenerate")
    affine = (
        np.outer(chord, q) + np.outer(shape_vector, q_perpendicular)
    ) / denominator

    dense_parameter = np.linspace(0.0, 1.0, dense_samples)
    dense_angles = angular_extent * dense_parameter
    dense_unit = np.column_stack((np.cos(dense_angles) - 1.0, np.sin(dense_angles)))
    dense_curve = start + dense_unit @ affine.T
    dense_lengths = np.linalg.norm(np.diff(dense_curve, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(dense_lengths)))
    total_length = float(cumulative[-1])
    if not np.isfinite(total_length) or total_length <= EPS:
        raise ValueError("ellipse arc has zero or non-finite length")
    cumulative /= total_length
    angles = np.interp(positions, cumulative, dense_angles)
    unit = np.column_stack((np.cos(angles) - 1.0, np.sin(angles)))
    fitted = start + unit @ affine.T
    return fitted, dense_curve, affine, angles


def fit_constrained_ellipse_arc(
    points_2d: np.ndarray,
    initial_tangent_2d: np.ndarray,
    *,
    name: str = "branch",
    tangent_weight: float = 0.20,
    maximum_axis_ratio: float = 30.0,
    allow_axis_ratio_guard_exceedance: bool = False,
) -> dict:
    """Fit an ordered ellipse interval with exact start and terminal points.

    The source path's cumulative arc position establishes point correspondence.
    Multiple positive and negative angular extents are optimized.  Endpoints
    remain exact by construction, while a weighted residual keeps the fitted
    initial tangent close to the observed projected tangent.
    """
    points = np.asarray(points_2d, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5:
        raise ValueError(f"{name} constrained ellipse requires at least five 2-D points")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} projected points contain NaN or infinite values")
    initial_tangent = _unit_vector_2d(initial_tangent_2d, name=f"{name} initial tangent")
    parameter_positions, path_length = cumulative_parameter(points)
    start = points[0]
    end = points[-1]
    chord = end - start
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= EPS:
        raise ValueError(f"{name} projected endpoints coincide")
    span = max(float(np.ptp(points, axis=0).max()), chord_length, 1.0)

    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError("scipy is required for constrained ellipse optimization") from exc

    best = None
    optimization_messages = []
    lower_delta = 0.30 * np.pi
    upper_delta = 1.95 * np.pi

    for sign in (-1.0, 1.0):
        for initial_magnitude in (0.5 * np.pi, np.pi, 1.5 * np.pi):
            initial_delta = sign * initial_magnitude
            q = np.array([np.cos(initial_delta) - 1.0, np.sin(initial_delta)])
            denominator = float(np.dot(q, q))
            # Initialize the free shape vector so the start derivative follows
            # the observed tangent at approximately the observed arc speed.
            desired_derivative = initial_tangent * (path_length / initial_magnitude)
            q_x, q_y = q
            safe_q_x = q_x if abs(q_x) > 1.0e-4 else np.copysign(1.0e-4, q_x or 1.0)
            initial_shape = (
                desired_derivative * denominator - chord * q_y
            ) / safe_q_x
            initial = np.array([*initial_shape, np.log(initial_magnitude)])

            def residual_vector(parameters: np.ndarray) -> np.ndarray:
                delta = sign * np.exp(parameters[2])
                try:
                    fitted, _, affine, _ = _endpoint_constrained_affine_arc(
                        start, end, parameters[:2], delta, parameter_positions
                    )
                    start_derivative = delta * (affine @ np.array([0.0, 1.0]))
                    fitted_tangent = _unit_vector_2d(
                        start_derivative, name="fitted ellipse tangent"
                    )
                    singular_values = np.linalg.svd(affine, compute_uv=False)
                    axis_ratio = singular_values[0] / max(singular_values[1], EPS)
                    tangent_residual = (
                        fitted_tangent - initial_tangent
                    ) * (tangent_weight * span * np.sqrt(len(points)))
                    ratio_penalty = np.array([
                        max(0.0, np.log(axis_ratio / maximum_axis_ratio)) * span
                    ])
                    return np.concatenate(
                        ((fitted - points).ravel(), tangent_residual, ratio_penalty)
                    )
                except (ValueError, np.linalg.LinAlgError):
                    return np.full(points.size + 3, span * 1.0e3)

            try:
                optimized = least_squares(
                    residual_vector,
                    initial,
                    bounds=(
                        [-5.0 * span, -5.0 * span, np.log(lower_delta)],
                        [5.0 * span, 5.0 * span, np.log(upper_delta)],
                    ),
                    loss="soft_l1",
                    f_scale=1.0,
                    max_nfev=5000,
                )
                delta = sign * np.exp(optimized.x[2])
                fitted, sampled, affine, angles = _endpoint_constrained_affine_arc(
                    start, end, optimized.x[:2], delta, parameter_positions
                )
                residuals = np.linalg.norm(fitted - points, axis=1)
                rmse = float(np.sqrt(np.mean(residuals**2)))
                derivative = delta * (affine @ np.array([0.0, 1.0]))
                fitted_tangent = _unit_vector_2d(derivative, name="fitted ellipse tangent")
                tangent_angle = float(np.degrees(np.arccos(np.clip(
                    np.dot(fitted_tangent, initial_tangent), -1.0, 1.0
                ))))
                singular_values = np.linalg.svd(affine, compute_uv=False)
                axis_ratio = float(singular_values[0] / max(singular_values[1], EPS))
                score = rmse + 0.05 * span * tangent_angle / 180.0
                candidate = {
                    "score": score,
                    "optimized": optimized,
                    "delta": delta,
                    "fitted": fitted,
                    "sampled": sampled,
                    "affine": affine,
                    "angles": angles,
                    "residuals": residuals,
                    "rmse": rmse,
                    "fitted_tangent": fitted_tangent,
                    "tangent_angle": tangent_angle,
                    "singular_values": singular_values,
                    "axis_ratio": axis_ratio,
                }
                if best is None or candidate["score"] < best["score"]:
                    best = candidate
            except Exception as exc:  # retain other starts before failing
                optimization_messages.append(str(exc))

    if best is None:
        detail = "; ".join(optimization_messages[-3:])
        raise ValueError(f"{name} constrained ellipse optimization failed: {detail}")
    if not best["optimized"].success:
        raise ValueError(
            f"{name} constrained ellipse optimization failed "
            f"(success={best['optimized'].success})"
        )
    if (
        best["axis_ratio"] > maximum_axis_ratio
        and not allow_axis_ratio_guard_exceedance
    ):
        raise ValueError(
            f"{name} constrained ellipse is unstable "
            f"(success={best['optimized'].success}, axis_ratio={best['axis_ratio']:.3g})"
        )

    affine = best["affine"]
    left_vectors, axes, phase_rotation = np.linalg.svd(affine)
    center = start - affine[:, 0]
    orientation = float(np.arctan2(left_vectors[1, 0], left_vectors[0, 0]))
    dense_lengths = np.linalg.norm(np.diff(best["sampled"], axis=0), axis=1)
    arc_length = float(np.sum(dense_lengths))
    endpoint_error = float(np.linalg.norm(best["sampled"][-1] - end))
    start_error = float(np.linalg.norm(best["sampled"][0] - start))
    return {
        "method": "endpoint_exact_affine_ellipse_arc",
        "center_2d": center,
        "semi_major_axis_mm": float(axes[0]),
        "semi_minor_axis_mm": float(axes[1]),
        "axis_ratio": best["axis_ratio"],
        "axis_ratio_guard": float(maximum_axis_ratio),
        "axis_ratio_guard_exceeded": bool(best["axis_ratio"] > maximum_axis_ratio),
        "orientation_rad": orientation,
        "orientation_deg": float(np.degrees(orientation)),
        "affine_matrix": affine,
        "phase_rotation": phase_rotation,
        "parameter_positions": parameter_positions,
        "angular_positions_rad": best["angles"],
        "arc_start_rad": 0.0,
        "arc_end_rad": float(best["delta"]),
        "angular_extent_rad": float(abs(best["delta"])),
        "angular_extent_deg": float(np.degrees(abs(best["delta"]))),
        "signed_angular_extent_deg": float(np.degrees(best["delta"])),
        "arc_length_mm": arc_length,
        "projected_polyline_length_mm": path_length,
        "fitted_points_2d": best["fitted"],
        "sampled_arc_2d": best["sampled"],
        "point_residuals_mm": best["residuals"],
        "rms_residual_mm": best["rmse"],
        "normalized_rms_residual": float(best["rmse"] / path_length),
        "mean_residual_mm": float(np.mean(best["residuals"])),
        "max_residual_mm": float(np.max(best["residuals"])),
        "initial_tangent_2d": initial_tangent,
        "fitted_initial_tangent_2d": best["fitted_tangent"],
        "initial_tangent_error_deg": best["tangent_angle"],
        "start_endpoint_error_mm": start_error,
        "terminal_endpoint_error_mm": endpoint_error,
        "optimizer_success": bool(best["optimized"].success),
        "optimizer_message": str(best["optimized"].message),
        "optimizer_cost": float(best["optimized"].cost),
    }


def discrete_curve_metrics(points: np.ndarray) -> dict:
    """Measure length, tortuosity, directions, and discrete curvature."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError("curve metrics require at least two 3-D points")
    segments = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    if np.any(segment_lengths <= EPS):
        nonzero = segment_lengths > EPS
        segments = segments[nonzero]
        segment_lengths = segment_lengths[nonzero]
    if not len(segment_lengths):
        raise ValueError("curve contains no non-zero segments")
    tangents = segments / segment_lengths[:, None]
    path_length = float(segment_lengths.sum())
    chord = points[-1] - points[0]
    chord_length = float(np.linalg.norm(chord))
    if chord_length <= EPS:
        raise ValueError("curve endpoints coincide")
    window = max(1, min(5, len(tangents)))
    initial = tangents[:window].mean(axis=0)
    distal = tangents[-window:].mean(axis=0)
    initial /= np.linalg.norm(initial)
    distal /= np.linalg.norm(distal)
    direction = chord / chord_length
    if len(tangents) > 1:
        turning = np.arccos(np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0))
        mean_curvature = float(np.sum(turning) / path_length)
        total_turning = float(np.sum(turning))
    else:
        mean_curvature = 0.0
        total_turning = 0.0
    return {
        "length_mm": path_length,
        "chord_length_mm": chord_length,
        "tortuosity": float(path_length / chord_length),
        "overall_direction": direction,
        "initial_tangent": initial,
        "distal_tangent": distal,
        "initial_direction_score": float(np.dot(initial, direction)),
        "distal_direction_score": float(np.dot(distal, direction)),
        "total_turning_rad": total_turning,
        "mean_curvature_per_mm": mean_curvature,
    }
