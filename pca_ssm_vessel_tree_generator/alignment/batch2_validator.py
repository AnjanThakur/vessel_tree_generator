"""Batch 2 Invariant & Anatomical Validation Module."""

from __future__ import annotations

from typing import Any
import numpy as np

EPS = 1.0e-12


def validate_batch2_invariants(
    centerlines_scanner: dict[str, np.ndarray],
    centerlines_cardiac: dict[str, np.ndarray],
    landmarks_scanner: dict[str, np.ndarray],
    landmarks_cardiac: dict[str, np.ndarray],
    R_total: np.ndarray,
    origin: np.ndarray,
) -> dict[str, Any]:
    """Validate strict rigid transformation invariants and anatomical alignment properties for Batch 2."""
    failures = []
    warnings = []

    # Check 1: Orthonormality and Right-Handedness
    det_val = float(np.linalg.det(R_total))
    if abs(det_val - 1.0) > 1.0e-5:
        failures.append(f"Determinant error: det(R_total) = {det_val:.6f} (expected +1.0)")

    ortho_err = float(np.max(np.abs(R_total @ R_total.T - np.eye(3))))
    if ortho_err > 1.0e-6:
        failures.append(f"Orthogonality error: max(abs(R @ R^T - I)) = {ortho_err:.2e}")

    # Check 2: Segment Length Preservation
    max_seg_err = 0.0
    for name in ("LMCA", "LAD", "LCX", "RCA"):
        raw_p = centerlines_scanner.get(name)
        card_p = centerlines_cardiac.get(name)
        if raw_p is not None and card_p is not None and len(raw_p) >= 2:
            raw_lens = np.linalg.norm(np.diff(raw_p, axis=0), axis=1)
            card_lens = np.linalg.norm(np.diff(card_p, axis=0), axis=1)
            err = float(np.max(np.abs(raw_lens - card_lens)))
            max_seg_err = max(max_seg_err, err)

    if max_seg_err > 1.0e-5:
        failures.append(f"Segment length preservation error ({max_seg_err:.2e}mm) exceeds threshold (1e-5mm)")

    # Check 3: Landmark Pairwise Distance Preservation
    lm_names = sorted([k for k, v in landmarks_scanner.items() if v is not None and len(v) == 3])
    max_dist_err = 0.0
    for i in range(len(lm_names)):
        for j in range(i + 1, len(lm_names)):
            k1, k2 = lm_names[i], lm_names[j]
            d_scanner = float(np.linalg.norm(landmarks_scanner[k1] - landmarks_scanner[k2]))
            d_cardiac = float(np.linalg.norm(landmarks_cardiac[k1] - landmarks_cardiac[k2]))
            dist_err = abs(d_scanner - d_cardiac)
            max_dist_err = max(max_dist_err, dist_err)

    if max_dist_err > 1.0e-5:
        failures.append(f"Landmark pairwise distance error ({max_dist_err:.2e}mm) exceeds threshold (1e-5mm)")

    # Check 4: LAD Descent Check
    lad_card = centerlines_cardiac.get("LAD")
    if lad_card is not None and len(lad_card) >= 2:
        lad_z_diff = float(lad_card[-1][2] - lad_card[0][2])
        if lad_z_diff >= 0.0:
            failures.append(f"LAD does not descend toward negative Z (z_diff = {lad_z_diff:.2f}mm)")

    # Check 5: Non-finite Coordinates
    all_pts = [v for v in centerlines_cardiac.values() if v is not None and len(v) > 0]
    all_pts.extend([v.reshape(1, 3) for v in landmarks_cardiac.values() if v is not None and len(v) == 3])
    if all_pts:
        concat = np.vstack(all_pts)
        if not np.all(np.isfinite(concat)):
            failures.append("Non-finite (NaN or Inf) coordinates in transformed cardiac frame payload")

    overall_pass = (len(failures) == 0)

    return {
        "overall_pass": overall_pass,
        "determinant": det_val,
        "orthogonality_error": ortho_err,
        "max_segment_length_error_mm": max_seg_err,
        "max_landmark_distance_error_mm": max_dist_err,
        "failures": failures,
        "warnings": warnings,
    }
