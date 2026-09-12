"""Contraction curve s(phi) and ellipsoid deformation logic for Batch 6."""

from __future__ import annotations

import math
import numpy as np

EPS = 1.0e-12


def contraction_curve(phase: float, peak_phase: float = 0.35) -> float:
    """Smooth periodic contraction function (Design Doc §7.2).

    0 at end-diastole (phase=0), 1 at peak systole (~0.35), 0 at end-diastole (phase=1).
    """
    phase = float(phase) % 1.0
    if phase < peak_phase:
        return float(math.sin(math.pi * phase / (2.0 * peak_phase)))
    else:
        return float(math.cos(math.pi * (phase - peak_phase) / (2.0 * (1.0 - peak_phase))))


def deform_ellipsoid(
    a: float,
    b: float,
    c: float,
    u: float | np.ndarray,
    v: float | np.ndarray,
    phase: float,
    radial_amplitude: float = 0.15,
    longitudinal_amplitude: float = 0.10,
    torsion_amplitude_deg: float = 10.0,
    peak_phase: float = 0.35,
) -> tuple[float, float, float, float | np.ndarray, float | np.ndarray]:
    """Deform ellipsoid semi-axes and shift azimuthal coordinate u at cardiac phase (Design Doc §7.3)."""
    s = contraction_curve(phase, peak_phase=peak_phase)

    # 1. Radial contraction (a, b shrink)
    radial_factor = 1.0 - radial_amplitude * s
    a_def = a * radial_factor
    b_def = b * radial_factor

    # 2. Longitudinal shortening (c shrinks)
    long_factor = 1.0 - longitudinal_amplitude * s
    c_def = c * long_factor

    # 3. Torsion: base (v near 0) rotates more than apex (v near pi)
    torsion_rad = torsion_amplitude_deg * math.pi / 180.0 * s
    u_def = u + torsion_rad * (1.0 - v / math.pi)

    return a_def, b_def, c_def, u_def, v
