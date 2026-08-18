"""Shared coronary-course measurements and anatomical role acceptance.

The rules in this module are geometric quality-control criteria for the
repository's common cardiac frame.  They are not clinical thresholds.  The
same implementation is deliberately used to screen real training cases and
generated trees so a visually implausible daughter assignment cannot pass by
being compared with an unresolved source case.
"""

from __future__ import annotations

from typing import Any

import numpy as np


EPS = 1.0e-12


def arc_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def resample_arc_length(points: np.ndarray, sample_count: int = 80) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
        raise ValueError(f"centerline must have shape (n>=2, 3); got {points.shape}")
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if cumulative[-1] <= EPS:
        return np.repeat(points[:1], sample_count, axis=0)
    source = cumulative / cumulative[-1]
    target = np.linspace(0.0, 1.0, sample_count)
    return np.column_stack([np.interp(target, source, points[:, axis]) for axis in range(3)])


def _initial_unit_direction(points: np.ndarray, fraction: float = 0.08) -> np.ndarray:
    sampled = resample_arc_length(points, 80)
    index = max(1, int(round(fraction * (len(sampled) - 1))))
    vector = sampled[index] - sampled[0]
    return vector / max(float(np.linalg.norm(vector)), EPS)


def coronary_course_metrics(lad: np.ndarray, lcx: np.ndarray) -> dict[str, float]:
    """Measure the LAD/LCX roles in the common cardiac frame.

    In this frame negative Z is apex-directed, while displacement in the XY
    plane represents lateral/circumferential travel around the cardiac crown.
    """
    lad = np.asarray(lad, dtype=float)
    lcx = np.asarray(lcx, dtype=float)
    lad_sampled = resample_arc_length(lad)
    lcx_sampled = resample_arc_length(lcx)
    lad_segments = np.diff(lad_sampled, axis=0)
    lcx_segments = np.diff(lcx_sampled, axis=0)
    lad_segment_lengths = np.maximum(np.linalg.norm(lad_segments, axis=1), EPS)
    lcx_segment_lengths = np.maximum(np.linalg.norm(lcx_segments, axis=1), EPS)
    lad_length = arc_length(lad)
    lcx_length = arc_length(lcx)
    bifurcation_z = float(0.5 * (lad[0, 2] + lcx[0, 2]))
    lad_drop = bifurcation_z - float(lad[-1, 2])
    lcx_drop = bifurcation_z - float(lcx[-1, 2])
    lad_terminal_xy = float(np.linalg.norm(lad[-1, :2] - lad[0, :2]))
    lcx_terminal_xy = float(np.linalg.norm(lcx[-1, :2] - lcx[0, :2]))
    lad_initial = _initial_unit_direction(lad)
    lcx_initial = _initial_unit_direction(lcx)
    return {
        "LAD_length_mm": lad_length,
        "LCX_length_mm": lcx_length,
        "LAD_minus_LCX_length_mm": lad_length - lcx_length,
        "LAD_inferior_displacement_mm": lad_drop,
        "LCX_inferior_displacement_mm": lcx_drop,
        "LAD_minus_LCX_inferior_displacement_mm": lad_drop - lcx_drop,
        "LCX_to_LAD_inferior_displacement_ratio": lcx_drop / max(lad_drop, EPS),
        "LAD_inferior_displacement_to_length_ratio": lad_drop / max(lad_length, EPS),
        "LCX_inferior_span_to_length_ratio": (
            float(np.ptp(lcx_sampled[:, 2])) / max(lcx_length, EPS)
        ),
        "LAD_descending_segment_fraction": float(np.mean(lad_segments[:, 2] <= 0.0)),
        "LAD_horizontal_path_fraction": float(
            np.mean(np.linalg.norm(lad_segments[:, :2], axis=1) / lad_segment_lengths)
        ),
        "LCX_horizontal_path_fraction": float(
            np.mean(np.linalg.norm(lcx_segments[:, :2], axis=1) / lcx_segment_lengths)
        ),
        "LAD_terminal_lateral_fraction": lad_terminal_xy / max(lad_length, EPS),
        "LCX_terminal_lateral_fraction": lcx_terminal_xy / max(lcx_length, EPS),
        "LAD_terminal_lateral_displacement_mm": lad_terminal_xy,
        "LCX_terminal_lateral_displacement_mm": lcx_terminal_xy,
        "LCX_to_LAD_lateral_displacement_ratio": (
            lcx_terminal_xy / max(lad_terminal_xy, EPS)
        ),
        "LAD_initial_inferior_direction_fraction": float(-lad_initial[2]),
        "LCX_initial_horizontal_direction_fraction": float(np.linalg.norm(lcx_initial[:2])),
    }


def anatomical_role_acceptance(metrics: dict[str, float]) -> dict[str, Any]:
    """Apply transparent hard criteria for unambiguous LAD/LCX roles."""
    checks = {
        "lad_has_minimum_inferior_reach": metrics["LAD_inferior_displacement_mm"] >= 25.0,
        "lad_is_dominant_descending_branch": (
            metrics["LAD_minus_LCX_inferior_displacement_mm"] >= 10.0
        ),
        "lcx_is_not_apex_descending_branch": (
            metrics["LCX_to_LAD_inferior_displacement_ratio"] <= 0.65
        ),
        "lad_has_longitudinal_course": (
            metrics["LAD_inferior_displacement_to_length_ratio"] >= 0.25
        ),
        "lad_descent_is_sustained": metrics["LAD_descending_segment_fraction"] >= 0.70,
        "lcx_has_horizontal_crown_course": metrics["LCX_horizontal_path_fraction"] >= 0.65,
        "lcx_inferior_span_is_limited": metrics["LCX_inferior_span_to_length_ratio"] <= 0.50,
        "lcx_is_more_lateral_than_lad": (
            metrics["LCX_terminal_lateral_fraction"]
            >= metrics["LAD_terminal_lateral_fraction"] + 0.03
        ),
        "lcx_has_substantial_absolute_lateral_reach": (
            metrics["LCX_terminal_lateral_displacement_mm"] >= 40.0
            and metrics["LCX_to_LAD_lateral_displacement_ratio"] >= 0.70
        ),
    }
    failed = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "accepted": not failed,
        "checks": {name: bool(passed) for name, passed in checks.items()},
        "failed_checks": failed,
        "policy": {
            "coordinate_frame": "common cardiac frame; negative Z is apex-directed",
            "criterion_type": "repository geometric anatomy QC; not a clinical threshold",
            "unresolved_assignments_may_train": False,
        },
    }
