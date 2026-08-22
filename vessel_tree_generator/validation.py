"""End-to-end validation for submission-ready 4D LCA cases."""

from __future__ import annotations

from typing import Any

import numpy as np


BRANCH_ORDER = ("LMCA", "LAD", "LCX")


def _topology_errors(branches: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "LMCA_to_LAD_mm": float(np.linalg.norm(branches["LMCA"][-1, :3] - branches["LAD"][0, :3])),
        "LMCA_to_LCX_mm": float(np.linalg.norm(branches["LMCA"][-1, :3] - branches["LCX"][0, :3])),
        "LAD_to_LCX_mm": float(np.linalg.norm(branches["LAD"][0, :3] - branches["LCX"][0, :3])),
    }


def validate_4d_case(case: dict[str, Any]) -> dict[str, Any]:
    """Validate topology, temporal correspondence, geometry, and radius data."""
    errors: list[str] = []
    warnings: list[str] = []
    frames = case.get("frames", [])
    phase_values = np.asarray(case.get("phase_values", []), dtype=float)
    times = np.asarray(case.get("time_seconds", []), dtype=float)
    reference = case.get("reference", {})

    if not frames:
        errors.append("case contains no cardiac frames")
    if len(phase_values) != len(frames):
        errors.append("phase_values count does not match frame count")
    if len(times) != len(frames):
        errors.append("time_seconds count does not match frame count")
    if len(phase_values) and (
        not np.all(np.isfinite(phase_values))
        or np.any(phase_values < 0.0)
        or np.any(phase_values > 1.0)
    ):
        errors.append("cardiac phases must be finite and lie in [0, 1]")
    if len(times) and (not np.all(np.isfinite(times)) or np.any(times < 0.0)):
        errors.append("time samples must be finite and non-negative")

    reference_shapes: dict[str, tuple[int, ...]] = {}
    for name in BRANCH_ORDER:
        points = np.asarray(reference.get("branches", {}).get(name, []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 4 or len(points) < 2:
            errors.append(f"reference {name} must have shape (N>=2, 4)")
            continue
        if not np.all(np.isfinite(points)) or np.any(points[:, 3] <= 0.0):
            errors.append(f"reference {name} contains invalid coordinates or radii")
        reference_shapes[name] = points.shape

    topology_by_phase: list[dict[str, float]] = []
    maximum_displacement = 0.0
    maximum_radius_change = 0.0
    for frame_index, frame in enumerate(frames):
        branches = frame.get("branches", {})
        if not set(BRANCH_ORDER).issubset(branches):
            errors.append(f"frame {frame_index} is missing a mandatory LCA branch")
            continue
        valid_frame_branches: dict[str, np.ndarray] = {}
        for name in BRANCH_ORDER:
            values = np.asarray(branches[name], dtype=float)
            if values.shape != reference_shapes.get(name):
                errors.append(
                    f"frame {frame_index}/{name} shape {values.shape} does not match reference {reference_shapes.get(name)}"
                )
                continue
            if not np.all(np.isfinite(values)):
                errors.append(f"frame {frame_index}/{name} contains NaN or Inf")
            if np.any(values[:, 3] <= 0.0):
                errors.append(f"frame {frame_index}/{name} contains non-positive radii")
            valid_frame_branches[name] = values
            reference_values = np.asarray(reference["branches"][name], dtype=float)
            maximum_displacement = max(
                maximum_displacement,
                float(np.max(np.linalg.norm(values[:, :3] - reference_values[:, :3], axis=1))),
            )
            maximum_radius_change = max(
                maximum_radius_change,
                float(np.max(np.abs(values[:, 3] / reference_values[:, 3] - 1.0))),
            )
        if len(valid_frame_branches) == len(BRANCH_ORDER):
            topology = _topology_errors(valid_frame_branches)
            topology_by_phase.append(topology)
            if max(topology.values()) > 1.0e-9:
                errors.append(f"frame {frame_index} does not preserve the exact shared bifurcation")

    phase_zero_indices = np.flatnonzero(np.isclose(phase_values, 0.0, atol=1.0e-12))
    phase_zero_identity_error = None
    if len(phase_zero_indices) and frames and len(reference_shapes) == len(BRANCH_ORDER):
        zero_frame = frames[int(phase_zero_indices[0])]["branches"]
        phase_zero_identity_error = max(
            float(np.max(np.abs(np.asarray(zero_frame[name])[:, :3] - np.asarray(reference["branches"][name])[:, :3])))
            for name in BRANCH_ORDER
        )
        if phase_zero_identity_error > 1.0e-9:
            errors.append(f"phase-0 geometry differs from the static reference by {phase_zero_identity_error:.3e} mm")

    maximum_step = 0.0
    maximum_closing_step = 0.0
    if len(frames) > 1 and len(reference_shapes) == len(BRANCH_ORDER):
        for previous, current in zip(frames[:-1], frames[1:]):
            for name in BRANCH_ORDER:
                maximum_step = max(
                    maximum_step,
                    float(np.max(np.linalg.norm(
                        np.asarray(current["branches"][name])[:, :3]
                        - np.asarray(previous["branches"][name])[:, :3],
                        axis=1,
                    ))),
                )
        for name in BRANCH_ORDER:
            maximum_closing_step = max(
                maximum_closing_step,
                float(np.max(np.linalg.norm(
                    np.asarray(frames[0]["branches"][name])[:, :3]
                    - np.asarray(frames[-1]["branches"][name])[:, :3],
                    axis=1,
                ))),
            )

    closed_cycle_requested = bool(
        len(phase_values) > 1
        and np.isclose(phase_values[0], 0.0, atol=1.0e-12)
        and np.isclose(phase_values[-1], 1.0, atol=1.0e-12)
    )
    if closed_cycle_requested and maximum_closing_step > 1.0e-9:
        errors.append(
            f"phase-1 geometry does not close to phase 0 ({maximum_closing_step:.3e} mm)"
        )

    amplitude = float(case.get("pulsatility", {}).get("amplitude", 0.0))
    if maximum_radius_change > amplitude + 1.0e-8:
        errors.append(
            f"radius variation {maximum_radius_change:.6f} exceeds configured pulsatility {amplitude:.6f}"
        )
    if maximum_displacement > 16.0:
        warnings.append(
            f"maximum displacement is {maximum_displacement:.2f} mm; above the charter's approximate 16 mm LCA guidance"
        )

    disease = case.get("disease", {})
    if not disease.get("validation", {}).get("is_valid", False):
        errors.append("static disease validation did not pass")

    return {
        "is_valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "mandatory_lca_branches_present": len(reference_shapes) == len(BRANCH_ORDER),
            "exact_shared_bifurcation_all_phases": bool(
                topology_by_phase and all(max(item.values()) <= 1.0e-9 for item in topology_by_phase)
            ),
            "phase_zero_matches_static_reference": phase_zero_identity_error is None or phase_zero_identity_error <= 1.0e-9,
            "all_radii_positive_and_finite": not any("radii" in error or "NaN" in error for error in errors),
            "temporal_point_correspondence_preserved": not any("shape" in error for error in errors),
            "requested_full_cycle_closes_exactly": not closed_cycle_requested or maximum_closing_step <= 1.0e-9,
            "disease_validation_pass": bool(disease.get("validation", {}).get("is_valid", False)),
        },
        "metrics": {
            "frame_count": len(frames),
            "phase_zero_identity_error_mm": phase_zero_identity_error,
            "maximum_reference_displacement_mm": maximum_displacement,
            "maximum_consecutive_phase_displacement_mm": maximum_step,
            "maximum_cycle_closing_displacement_mm": maximum_closing_step,
            "maximum_radius_variation_fraction": maximum_radius_change,
            "maximum_topology_error_mm": max(
                (max(item.values()) for item in topology_by_phase), default=None
            ),
        },
        "validation_scope": (
            "engineering topology, numerical, temporal-correspondence, disease, and radius checks; "
            "not clinical validation"
        ),
    }
