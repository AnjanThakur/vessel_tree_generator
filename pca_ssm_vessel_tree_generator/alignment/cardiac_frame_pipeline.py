"""Cardiac Frame Pipeline Coordinator for Batch 2.

Bridges Batch-1 scanner outputs to Batch-2 plane fitting, authoritative cardiac frame computation,
rigid coordinate transformation, and standardized output export.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from pca_ssm_vessel_tree_generator.alignment.plane_fitting import (
    fit_coronary_plane,
    compute_iv_normal,
)
from surface_relative.cardiac_frame import (
    compute_cardiac_frame,
    transform_to_cardiac_frame,
    validate_cardiac_frame,
)


def process_patient_cardiac_frame(
    centerlines_scanner: dict[str, np.ndarray],
    landmarks_scanner: dict[str, np.ndarray],
    patient_id: str,
    rca_resolved: bool,
    rca_unresolved_reason: str = "",
) -> dict[str, Any]:
    """Execute Batch-2 per-patient plane fitting, cardiac frame computation, and rigid transformation.

    Returns diagnostic result payload including plane QC, frame parameters, transformed data, and validation.
    """
    if not rca_resolved:
        reason = f"Skipped: RCA unresolved in Batch 1 ({rca_unresolved_reason})"
        return {
            "patient_id": patient_id,
            "batch2_status": "rca_unresolved",
            "batch2_passed": False,
            "rejection_reason": reason,
            "rejection_reasons": [reason],
        }

    rca_pts = centerlines_scanner.get("RCA")
    lcx_pts = centerlines_scanner.get("LCX")
    lad_pts = centerlines_scanner.get("LAD")
    lmca_pts = centerlines_scanner.get("LMCA")
    lca_ost = landmarks_scanner.get("lca_ostium")

    if rca_pts is None or lcx_pts is None or lad_pts is None or lmca_pts is None or lca_ost is None:
        return {
            "patient_id": patient_id,
            "batch2_status": "missing_required_inputs",
            "batch2_passed": False,
            "rejection_reasons": ["Missing required centerline or landmark inputs"],
        }

    try:
        # 1. Fit Coronary Plane (SVD on RCA+LCX with LAD descent sign correction)
        cor_fit = fit_coronary_plane(rca_pts, lcx_pts, lad_pts)

        # 2. Derive IV Plane Normal (unit(coronary_normal x LAD_dir))
        iv_fit = compute_iv_normal(cor_fit["coronary_normal"], lad_pts)

        # 3. Compute Authoritative Rigid Cardiac Frame via frozen surface_relative/cardiac_frame.py
        origin, R_total, frame_meta = compute_cardiac_frame(
            coronary_normal=cor_fit["coronary_normal"],
            iv_normal=iv_fit["iv_normal"],
            coronary_centroid=cor_fit["coronary_centroid"],
            lad_centerline=lad_pts,
            lca_ostium=lca_ost,
        )
    except Exception as exc:
        return {
            "patient_id": patient_id,
            "batch2_status": "alignment_failed",
            "batch2_passed": False,
            "rejection_reasons": [f"Plane fitting or cardiac frame computation exception: {exc}"],
        }

    # 4. Transform Centerlines & Landmarks Rigidly
    centerlines_cardiac = {
        name: transform_to_cardiac_frame(pts, origin, R_total)
        for name, pts in centerlines_scanner.items()
        if pts is not None and len(pts) > 0
    }

    landmarks_cardiac = {
        name: transform_to_cardiac_frame(coord, origin, R_total)
        for name, coord in landmarks_scanner.items()
        if coord is not None and len(coord) == 3
    }

    # 5. Authoritative Individual Cardiac Frame Validation
    frame_val = validate_cardiac_frame(
        origin=origin,
        R_total=R_total,
        branches_raw={"rca": rca_pts, "lcx": lcx_pts, "lad": lad_pts, "lmca": lmca_pts},
    )

    # 6. Additional Invariant & Anatomical Residual Checks
    lad_cardiac = centerlines_cardiac["LAD"]
    lad_z_change_mm = float(lad_cardiac[-1][2] - lad_cardiac[0][2])
    lad_apex_aligned = lad_z_change_mm < 0.0

    rca_cardiac = centerlines_cardiac["RCA"]
    lcx_cardiac = centerlines_cardiac["LCX"]
    ring_z = np.concatenate([rca_cardiac[:, 2], lcx_cardiac[:, 2]])
    ring_z_rms = float(np.sqrt(np.mean(ring_z**2)))
    ring_z_mean_abs = float(np.mean(np.abs(ring_z)))
    ring_z_max_abs = float(np.max(np.abs(ring_z)))

    # Canonical LCA angle check
    lca_ost_cardiac = landmarks_cardiac["lca_ostium"]
    radial_xy = float(math.hypot(lca_ost_cardiac[0], lca_ost_cardiac[1]))
    if radial_xy < 1.0e-6:
        canonical_lca_angle_deg = None
        lca_angle_defined = False
    else:
        lca_angle_rad = math.atan2(lca_ost_cardiac[1], lca_ost_cardiac[0])
        canonical_lca_angle_deg = math.degrees(lca_angle_rad)
        lca_angle_defined = True

    # Segment length preservation check (rigid transform invariant)
    length_preservation_errors = {}
    for name in ("LMCA", "LAD", "LCX", "RCA"):
        raw_p = centerlines_scanner.get(name)
        card_p = centerlines_cardiac.get(name)
        if raw_p is not None and card_p is not None and len(raw_p) >= 2:
            raw_lens = np.linalg.norm(np.diff(raw_p, axis=0), axis=1)
            card_lens = np.linalg.norm(np.diff(card_p, axis=0), axis=1)
            length_preservation_errors[name] = float(np.max(np.abs(raw_lens - card_lens)))

    max_length_preservation_error = max(length_preservation_errors.values()) if length_preservation_errors else 0.0
    length_preserved = max_length_preservation_error <= 1.0e-5

    batch2_passed = (
        frame_val["overall_pass"]
        and lad_apex_aligned
        and length_preserved
    )

    rejection_reasons = []
    if not frame_val["overall_pass"]:
        rejection_reasons.append("Failed authoritative cardiac frame validation checks")
    if not lad_apex_aligned:
        rejection_reasons.append(f"LAD does not descend toward negative Z (z_change = {lad_z_change_mm:.2f}mm)")
    if not length_preserved:
        rejection_reasons.append(f"Rigid transformation length error exceeds threshold ({max_length_preservation_error:.2e}mm)")

    return {
        "patient_id": patient_id,
        "batch2_status": "alignment_succeeded" if batch2_passed else "alignment_failed",
        "batch2_passed": batch2_passed,
        "rejection_reasons": rejection_reasons,
        "origin_scanner_ras_mm": origin.tolist(),
        "R_total": R_total.tolist(),
        "coronary_fit": cor_fit,
        "iv_fit": iv_fit,
        "frame_validation": frame_val,
        "transformed_centerlines": centerlines_cardiac,
        "transformed_landmarks": landmarks_cardiac,
        "residuals": {
            "lad_z_change_mm": lad_z_change_mm,
            "lad_apex_aligned": lad_apex_aligned,
            "ring_z_rms_mm": ring_z_rms,
            "ring_z_mean_abs_mm": ring_z_mean_abs,
            "ring_z_max_abs_mm": ring_z_max_abs,
            "canonical_lca_angle_deg": canonical_lca_angle_deg,
            "lca_angle_defined": lca_angle_defined,
            "max_length_preservation_error_mm": max_length_preservation_error,
            "length_preservation_by_vessel": length_preservation_errors,
        },
    }
