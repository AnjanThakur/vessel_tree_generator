"""B-spline curve fitting in (u,v) parameter space, tortuosity perturbation, and obliquity drift for Batch 5."""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from scipy.interpolate import splprep, splev

EPS = 1.0e-12


def surface_spline(u_ctrl: np.ndarray, v_ctrl: np.ndarray, num_eval: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Fit smooth 2D cubic B-splines to u(t) and v(t) control points in parameter space (Design Doc §6.4)."""
    u_c = np.asarray(u_ctrl, dtype=float)
    v_c = np.asarray(v_ctrl, dtype=float)
    n_pts = len(u_c)

    if n_pts < 2:
        return u_c.copy(), v_c.copy()

    if n_pts == 2:
        t = np.linspace(0.0, 1.0, num_eval)
        u_eval = u_c[0] + (u_c[1] - u_c[0]) * t
        v_eval = v_c[0] + (v_c[1] - v_c[0]) * t
        return u_eval, v_eval

    k_degree = min(3, n_pts - 1)
    pts_2d = np.vstack([u_c, v_c])

    try:
        tck, _ = splprep(pts_2d, k=k_degree, s=0.0)
        u_fine = np.linspace(0.0, 1.0, num_eval)
        u_eval, v_eval = splev(u_fine, tck)
        return np.asarray(u_eval, dtype=float), np.asarray(v_eval, dtype=float)
    except Exception:
        # Fallback to linear interpolation
        t = np.linspace(0.0, 1.0, num_eval)
        u_eval = np.interp(t, np.linspace(0.0, 1.0, n_pts), u_c)
        v_eval = np.interp(t, np.linspace(0.0, 1.0, n_pts), v_c)
        return u_eval, v_eval


def add_tortuosity(
    u_ctrl: np.ndarray, v_ctrl: np.ndarray, strength: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb control points with low-frequency sinusoidal modes (Design Doc §6.4).

    Modulated by envelope E(t) = 4t(1-t) ensuring zero perturbation at endpoints.
    """
    u_c = np.asarray(u_ctrl, dtype=float).copy()
    v_c = np.asarray(v_ctrl, dtype=float).copy()
    n = len(u_c)
    if n <= 2 or abs(strength) <= EPS:
        return u_c, v_c

    t = np.linspace(0.0, 1.0, n)
    envelope = 4.0 * t * (1.0 - t)  # Zero at endpoints, peaks in middle

    u_perturb = np.zeros(n, dtype=float)
    v_perturb = np.zeros(n, dtype=float)

    for k in [1, 2, 3]:
        amp_u = strength * rng.uniform(0.3, 1.0)
        amp_v = strength * rng.uniform(0.3, 1.0)
        phase_u = rng.uniform(0.0, 2.0 * np.pi)
        phase_v = rng.uniform(0.0, 2.0 * np.pi)

        u_perturb += amp_u * np.sin(k * np.pi * t + phase_u) * envelope
        v_perturb += amp_v * np.sin(k * np.pi * t + phase_v) * envelope

    return u_c + u_perturb, v_c + v_perturb


def add_obliquity(u_ctrl: np.ndarray, obliquity_angle: float) -> np.ndarray:
    """Add linear drift in u from start to end (Design Doc §6.4): u_ctrl += obliquity_angle * t."""
    u_c = np.asarray(u_ctrl, dtype=float).copy()
    n = len(u_c)
    if n < 2 or abs(obliquity_angle) <= EPS:
        return u_c

    t = np.linspace(0.0, 1.0, n)
    return u_c + obliquity_angle * t


def generate_vessel_uv(
    vessel_name: str,
    start_uv: tuple[float, float],
    end_uv: tuple[float, float],
    num_ctrl_pts: int,
    tort_stats: dict[str, Any],
    obliquity_stats: dict[str, Any],
    rng: np.random.Generator,
    num_eval: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate vessel path control points and evaluated smooth B-splines in (u, v) parameter space.

    Returns:
        u_ctrl, v_ctrl: (num_ctrl_pts,) control points after obliquity + tortuosity
        u_path, v_path: (num_eval,) evaluated B-spline path
    """
    t = np.linspace(0.0, 1.0, num_ctrl_pts)
    u_ctrl = start_uv[0] + (end_uv[0] - start_uv[0]) * t
    v_ctrl = start_uv[1] + (end_uv[1] - start_uv[1]) * t

    # 1. Add obliquity
    obliq_info = obliquity_stats.get(vessel_name, {})
    ob_mean = obliq_info.get("mean", 0.0)
    ob_std = obliq_info.get("std", 0.1)
    if math.isnan(ob_mean):
        ob_mean = 0.0
    if math.isnan(ob_std):
        ob_std = 0.1

    obliquity = float(rng.normal(ob_mean, max(ob_std, 1.0e-3)))
    obliquity = float(np.clip(obliquity, ob_mean - 2.0 * ob_std, ob_mean + 2.0 * ob_std)) * 0.05
    u_ctrl = add_obliquity(u_ctrl, obliquity)

    # 2. Add tortuosity
    tort_info = tort_stats.get(vessel_name, {})
    t_mean = tort_info.get("mean", 0.5)
    t_std = tort_info.get("std", 0.2)
    if math.isnan(t_mean):
        t_mean = 0.5
    if math.isnan(t_std):
        t_std = 0.2

    strength = float(rng.normal(t_mean, max(t_std, 1.0e-3)))
    strength = float(np.clip(strength, 0.0, t_mean + 2.0 * t_std)) * 0.05
    u_ctrl, v_ctrl = add_tortuosity(u_ctrl, v_ctrl, strength, rng)

    # 3. Fix endpoints
    u_ctrl[0], v_ctrl[0] = start_uv
    u_ctrl[-1], v_ctrl[-1] = end_uv

    # 4. Evaluate B-spline
    u_path, v_path = surface_spline(u_ctrl, v_ctrl, num_eval=num_eval)

    return u_ctrl, v_ctrl, u_path, v_path
