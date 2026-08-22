"""Phase-conditioned coronary radius and stenosis-compliance model."""

from __future__ import annotations

from typing import Any

import numpy as np

from pca_ssm_vessel_tree_generator.motion.contraction_curve import contraction_curve


def validate_pulsatility(amplitude: float, stenosis_compliance_factor: float) -> None:
    if not np.isfinite(amplitude) or not 0.0 <= amplitude <= 0.10:
        raise ValueError("pulsatility amplitude must lie in [0, 0.10]")
    if not np.isfinite(stenosis_compliance_factor) or not 0.0 <= stenosis_compliance_factor <= 1.0:
        raise ValueError("stenosis_compliance_factor must lie in [0, 1]")


def phase_radius(
    baseline_radius: np.ndarray,
    diseased_radius: np.ndarray,
    phase: float,
    *,
    amplitude: float = 0.03,
    stenosis_compliance_factor: float = 0.35,
    peak_phase: float = 0.60,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return time-varying radii with reduced pulsatility inside lesions.

    The supplied diseased radius is the end-diastolic reference radius. The
    independent pulse response peaks in early diastole (default phase 0.60),
    after the mechanical contraction peak. At the
    maximum lesion location its amplitude is multiplied by
    ``stenosis_compliance_factor``; transitions follow the continuous lesion
    reduction profile. This is a scalar research approximation to the reduced
    cyclic distensibility measured in atherosclerotic coronary segments.
    """
    validate_pulsatility(amplitude, stenosis_compliance_factor)
    phase = float(phase)
    if not np.isfinite(phase) or not 0.0 <= phase <= 1.0:
        raise ValueError("phase must lie in [0, 1]")
    baseline = np.asarray(baseline_radius, dtype=float)
    diseased = np.asarray(diseased_radius, dtype=float)
    if baseline.shape != diseased.shape or baseline.ndim != 1:
        raise ValueError("baseline_radius and diseased_radius must be matching 1D arrays")
    if np.any(baseline <= 0.0) or np.any(diseased <= 0.0):
        raise ValueError("radii must be positive")

    reduction = np.clip(1.0 - diseased / baseline, 0.0, 1.0)
    maximum_reduction = float(np.max(reduction)) if len(reduction) else 0.0
    lesion_weight = reduction / maximum_reduction if maximum_reduction > 1.0e-12 else np.zeros_like(reduction)
    local_amplitude = amplitude * (1.0 - (1.0 - stenosis_compliance_factor) * lesion_weight)
    pulse_scale = float(contraction_curve(phase, peak_phase=peak_phase))
    radius = diseased * (1.0 + local_amplitude * pulse_scale)
    if np.any(radius <= 0.0) or not np.all(np.isfinite(radius)):
        raise RuntimeError("pulsatility produced invalid radii")
    return radius, {
        "phase": phase,
        "pulse_scale": pulse_scale,
        "pulse_peak_phase": float(peak_phase),
        "healthy_amplitude": float(amplitude),
        "stenosis_compliance_factor": float(stenosis_compliance_factor),
        "minimum_local_amplitude": float(np.min(local_amplitude)),
        "maximum_local_amplitude": float(np.max(local_amplitude)),
        "maximum_disease_reduction_fraction": maximum_reduction,
        "reference_phase": "end_diastole",
        "peak_response": "early_diastolic_lumen_expansion",
        "physiology_scope": "parametric phase-offset response; not patient-specific mechanics",
    }
