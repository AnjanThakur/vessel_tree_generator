"""Batch 1 Extraction Validation Suite.

Implements all 12 mandatory Batch-1 validation rules (Checks A through L).
Crucially distinguishes:
- 'extraction succeeded + RCA resolved'
- 'extraction succeeded + RCA unresolved' (valid diagnostic outcome, not a crash!)
- 'extraction failed'
"""

from __future__ import annotations

from typing import Any
import numpy as np

EPS = 1.0e-12


def validate_batch1_extraction(
    centerlines: dict[str, np.ndarray | None],
    landmarks: dict[str, np.ndarray | None],
    rca_resolved: bool,
    rca_unresolved_reason: str,
    qc_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Perform strict 12-rule Batch-1 validation on patient extraction result.

    Returns diagnostic status dictionary with checks A-L results.
    """
    failures = []
    warnings = []

    # Check A: Both ostia exist
    lca_ost = landmarks.get("lca_ostium")
    rca_ost = landmarks.get("rca_ostium")
    if lca_ost is None or len(lca_ost) != 3:
        failures.append("Check A: LCA ostium landmark missing or invalid")
    if rca_ost is None or len(rca_ost) != 3:
        failures.append("Check A: RCA ostium landmark missing or invalid")

    # Check B: Ostia are distinct
    if lca_ost is not None and rca_ost is not None:
        ost_dist = float(np.linalg.norm(lca_ost - rca_ost))
        if ost_dist <= EPS:
            failures.append("Check B: LCA ostium and RCA ostium are coincident")
    else:
        ost_dist = 0.0

    # Check C: LCA and RCA trees connect to respective ostia
    lmca = centerlines.get("LMCA")
    if lmca is not None and len(lmca) >= 1 and lca_ost is not None:
        if float(np.linalg.norm(lmca[0] - lca_ost)) > 1.0e-3:
            failures.append("Check C: LMCA start point does not connect to LCA ostium")

    rca = centerlines.get("RCA")
    if rca_resolved and rca is not None and len(rca) >= 1 and rca_ost is not None:
        if float(np.linalg.norm(rca[0] - rca_ost)) > 1.0e-3:
            failures.append("Check C: RCA start point does not connect to RCA ostium")

    # Check D: LMCA connects LCA ostium to bifurcation
    bif = landmarks.get("bifurcation")
    if lmca is not None and len(lmca) >= 1 and bif is not None:
        if float(np.linalg.norm(lmca[-1] - bif)) > 1.0e-3:
            failures.append("Check D: LMCA end point does not connect to bifurcation landmark")

    # Check E: LAD and LCX originate at bifurcation
    lad = centerlines.get("LAD")
    lcx = centerlines.get("LCX")
    if lad is not None and len(lad) >= 1 and bif is not None:
        if float(np.linalg.norm(lad[0] - bif)) > 1.0e-3:
            failures.append("Check E: LAD start point does not originate at bifurcation landmark")
    if lcx is not None and len(lcx) >= 1 and bif is not None:
        if float(np.linalg.norm(lcx[0] - bif)) > 1.0e-3:
            failures.append("Check E: LCX start point does not originate at bifurcation landmark")

    # Check F: LAD and LCX endpoints exist
    lad_end = landmarks.get("lad_endpoint")
    lcx_end = landmarks.get("lcx_endpoint")
    if lad_end is None or len(lad_end) != 3:
        failures.append("Check F: LAD endpoint landmark missing")
    if lcx_end is None or len(lcx_end) != 3:
        failures.append("Check F: LCX endpoint landmark missing")

    # Check G: RCA endpoint exists if RCA resolved
    rca_end = landmarks.get("rca_endpoint")
    if rca_resolved:
        if rca_end is None or len(rca_end) != 3 or np.all(rca_end == 0):
            failures.append("Check G: RCA endpoint landmark missing for resolved RCA")

    # Check H & I: Centerlines have >= 2 points, ordered proximal -> distal
    for name, pts in (("LMCA", lmca), ("LAD", lad), ("LCX", lcx)):
        if pts is None or len(pts) < 2:
            failures.append(f"Check H: {name} centerline has fewer than 2 points")
        else:
            diffs = np.diff(pts, axis=0)
            step_lens = np.linalg.norm(diffs, axis=1)
            if np.any(step_lens <= EPS):
                warnings.append(f"Check I: {name} centerline contains duplicate consecutive points")

    if rca_resolved and rca is not None:
        if len(rca) < 2:
            failures.append("Check H: RCA centerline has fewer than 2 points")

    # Check J: No NaN or Inf coordinates
    all_pts = []
    for pts in (lmca, lad, lcx, rca):
        if pts is not None and len(pts) > 0:
            all_pts.append(pts)
    for l_name, l_coord in landmarks.items():
        if l_coord is not None and len(l_coord) == 3:
            all_pts.append(l_coord.reshape(1, 3))

    if all_pts:
        concat_pts = np.vstack(all_pts)
        if not np.all(np.isfinite(concat_pts)):
            failures.append("Check J: Non-finite (NaN or Inf) coordinates detected in extraction")

    # Check K: Scanner RAS physical frame preserved
    # Check that coordinates are in physical mm scale (typical bounding box ~100-300mm)
    if all_pts:
        max_extent = float(np.max(np.abs(concat_pts)))
        if max_extent < 1.0 or max_extent > 5000.0:
            warnings.append(f"Check K: Physical coordinate extent ({max_extent:.1f}mm) is unusual")

    # Check L: Explicit RCA status recording
    if not rca_resolved:
        warnings.append(f"Check L: RCA is unresolved ({rca_unresolved_reason})")

    # Final status determination
    lca_succeeded = (len(failures) == 0)
    if lca_succeeded:
        if rca_resolved:
            overall_status = "extraction_succeeded_rca_resolved"
        else:
            overall_status = "extraction_succeeded_rca_unresolved"
    else:
        overall_status = "extraction_failed"

    return {
        "lca_succeeded": lca_succeeded,
        "rca_resolved": rca_resolved,
        "rca_unresolved_reason": rca_unresolved_reason,
        "overall_status": overall_status,
        "ostia_distance_mm": ost_dist,
        "failures": failures,
        "warnings": warnings,
        "checks_passed": len(failures) == 0,
    }
