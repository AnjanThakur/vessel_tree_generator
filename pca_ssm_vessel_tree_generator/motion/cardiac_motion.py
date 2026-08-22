"""4D Cardiac Motion Deformation applier for Batch 6."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any
import numpy as np

try:
    from surface_relative.surface_projection import (
        project_point_to_surface,
        ellipsoid_point,
        ellipsoid_normal,
        ellipsoid_tangent_u,
        ellipsoid_tangent_v,
    )
except ImportError:
    from pca_ssm_vessel_tree_generator.surface_relative.surface_projection import (
        project_point_to_surface,
        ellipsoid_point,
        ellipsoid_normal,
        ellipsoid_tangent_u,
        ellipsoid_tangent_v,
    )

try:
    from motion.contraction_curve import deform_ellipsoid, contraction_curve
except ImportError:
    from pca_ssm_vessel_tree_generator.motion.contraction_curve import deform_ellipsoid, contraction_curve

EPS = 1.0e-12


def apply_cardiac_motion_to_tree(
    tree_data: dict[str, Any],
    num_phases: int = 10,
    radial_amplitude: float = 0.15,
    longitudinal_amplitude: float = 0.10,
    torsion_amplitude_deg: float = 10.0,
    peak_phase: float = 0.35,
    phase_values: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Deform reference 3D synthetic coronary tree across num_phases cardiac phases (Design Doc §7.4 & §7.5)."""
    if phase_values is None:
        if num_phases < 1:
            raise ValueError("num_phases must be at least one")
        phases = np.linspace(0.0, 1.0, num_phases, endpoint=False, dtype=float)
    else:
        phases = np.asarray(tuple(phase_values), dtype=float)
        if phases.ndim != 1 or len(phases) == 0:
            raise ValueError("phase_values must be a non-empty one-dimensional sequence")
        if not np.all(np.isfinite(phases)) or np.any(phases < 0.0) or np.any(phases > 1.0):
            raise ValueError("every cardiac phase must be finite and lie in [0, 1]")
        num_phases = int(len(phases))

    a0 = float(tree_data["ellipsoid_params"]["a_mm"])
    b0 = float(tree_data["ellipsoid_params"]["b_mm"])
    c0 = float(tree_data["ellipsoid_params"]["c_mm"])

    vessels_3d = tree_data["vessels_3d"]
    side_branches_ref = tree_data.get("side_branches", [])

    # 1. Project reference vessels to parameter space (u_0, v_0) and deviation vectors [dev_x, dev_y, dev_z]
    vessel_param_data = {}
    for vname, pts in vessels_3d.items():
        pts = np.asarray(pts, dtype=float)
        n_pts = len(pts)

        u_arr = np.zeros(n_pts, dtype=float)
        v_arr = np.zeros(n_pts, dtype=float)
        off_arr = np.zeros(n_pts, dtype=float)
        dev_arr = np.zeros((n_pts, 3), dtype=float)

        for i in range(n_pts):
            u_i, v_i, off_i, dev_vec = project_point_to_surface(pts[i], a0, b0, c0)
            u_arr[i] = u_i
            v_arr[i] = v_i
            off_arr[i] = off_i
            dev_arr[i] = dev_vec

        vessel_param_data[vname] = {
            "u": u_arr,
            "v": v_arr,
            "offset": off_arr,
            "deviations": dev_arr,
            "ref_points": pts,
        }

    # 2. Project reference side branches to parameter space (u_sb, v_sb)
    side_branches_param_data = []
    for sb in side_branches_ref:
        sb_pts = np.asarray(sb["points"], dtype=float)
        n_sb = len(sb_pts)
        u_sb = np.zeros(n_sb, dtype=float)
        v_sb = np.zeros(n_sb, dtype=float)
        off_sb = np.zeros(n_sb, dtype=float)
        dev_sb = np.zeros((n_sb, 3), dtype=float)

        for i in range(n_sb):
            u_i, v_i, off_i, dev_vec = project_point_to_surface(sb_pts[i], a0, b0, c0)
            u_sb[i] = u_i
            v_sb[i] = v_i
            off_sb[i] = off_i
            dev_sb[i] = dev_vec

        side_branches_param_data.append({
            "parent": sb.get("parent", "LAD"),
            "attachment_index": sb.get("attachment_index", 0),
            "u": u_sb,
            "v": v_sb,
            "offset": off_sb,
            "deviations": dev_sb,
        })

    phase_values = [float(p) for p in phases]
    frames = []

    for p_idx, phase in enumerate(phase_values):
        s_val = contraction_curve(phase, peak_phase=peak_phase)
        frame_vessels = {}

        # Deform major vessel centerlines (§7.4)
        for vname, p_data in vessel_param_data.items():
            u0_arr = p_data["u"]
            v0_arr = p_data["v"]
            devs = p_data["deviations"]
            n_pts = len(u0_arr)

            a_d, b_d, c_d, u_d, v_d = deform_ellipsoid(
                a0, b0, c0, u0_arr, v0_arr, phase,
                radial_amplitude=radial_amplitude,
                longitudinal_amplitude=longitudinal_amplitude,
                torsion_amplitude_deg=torsion_amplitude_deg,
                peak_phase=peak_phase,
            )

            pts_def = np.zeros((n_pts, 3), dtype=float)
            for i in range(n_pts):
                surf = ellipsoid_point(u_d[i], v_d[i], a_d, b_d, c_d)
                tang_u = ellipsoid_tangent_u(u_d[i], v_d[i], a_d, b_d, c_d)
                tang_v = ellipsoid_tangent_v(u_d[i], v_d[i], a_d, b_d, c_d)
                normal = ellipsoid_normal(u_d[i], v_d[i], a_d, b_d, c_d)

                dx, dy, dz = devs[i]
                pts_def[i] = surf + dx * tang_u + dy * tang_v + dz * normal

            frame_vessels[vname] = pts_def

        # Off-surface/ostial displacement is already represented by the local
        # deviation recovered during projection. Adding it again would
        # double-count the offset and break exact phase-0 identity.

        # Enforce LMCA bifurcation snapping continuity across all phases (§7.5)
        bif_point = frame_vessels["LMCA"][-1].copy()
        frame_vessels["LAD"][0] = bif_point
        frame_vessels["LCX"][0] = bif_point

        # Deform side branches (§7.5)
        frame_side_branches = []
        for sb_p in side_branches_param_data:
            u_sb0 = sb_p["u"]
            v_sb0 = sb_p["v"]
            devs_sb = sb_p["deviations"]
            n_sb = len(u_sb0)

            a_d, b_d, c_d, u_sbd, v_sbd = deform_ellipsoid(
                a0, b0, c0, u_sb0, v_sb0, phase,
                radial_amplitude=radial_amplitude,
                longitudinal_amplitude=longitudinal_amplitude,
                torsion_amplitude_deg=torsion_amplitude_deg,
                peak_phase=peak_phase,
            )

            sb_pts_def = np.zeros((n_sb, 3), dtype=float)
            for i in range(n_sb):
                surf = ellipsoid_point(u_sbd[i], v_sbd[i], a_d, b_d, c_d)
                tang_u = ellipsoid_tangent_u(u_sbd[i], v_sbd[i], a_d, b_d, c_d)
                tang_v = ellipsoid_tangent_v(u_sbd[i], v_sbd[i], a_d, b_d, c_d)
                normal = ellipsoid_normal(u_sbd[i], v_sbd[i], a_d, b_d, c_d)

                dx, dy, dz = devs_sb[i]
                sb_pts_def[i] = surf + dx * tang_u + dy * tang_v + dz * normal

            frame_side_branches.append({
                "parent": sb_p["parent"],
                "attachment_index": sb_p["attachment_index"],
                "points": sb_pts_def,
            })

        frames.append({
            "phase_index": p_idx,
            "phase": phase,
            "contraction_scale_s": s_val,
            "ellipsoid_params": {"a_mm": a_d, "b_mm": b_d, "c_mm": c_d},
            "vessels_3d": frame_vessels,
            "side_branches": frame_side_branches,
        })

    return {
        "tree_id": tree_data.get("tree_id", "synthetic_tree_000"),
        "num_phases": num_phases,
        "phase_values": phase_values,
        "reference_phase": 0.0,
        "reference_ellipsoid_params": {"a_mm": a0, "b_mm": b0, "c_mm": c0},
        "motion_parameters": {
            "radial_amplitude": radial_amplitude,
            "longitudinal_amplitude": longitudinal_amplitude,
            "torsion_amplitude_deg": torsion_amplitude_deg,
            "peak_phase": peak_phase,
        },
        "source_metadata": tree_data.get("source_metadata", {}),
        "frames": frames,
    }
