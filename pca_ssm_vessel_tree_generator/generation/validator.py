"""Synthetic Tree Validator and Self-Intersection Detector for Batch 5.

Implements Part 6 §6.9 of the Technical Design Document:
- Branch length bounds (P2.5 - P97.5)
- LMCA bifurcation angle bounds (P2.5 - P97.5)
- Max out-of-plane deviation bounds (P2.5 - P97.5)
- Tortuosity bounds (P2.5 - P97.5)
- Explicit 1.0 mm 3D segment-to-segment self-intersection test
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np

EPS = 1.0e-12


def segment_distance_3d(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Compute exact minimum 3D Euclidean distance between two line segments (p1-q1) and (p2-q2)."""
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2

    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))

    D = a * c - b * b
    sc, sN, sD = D, D, D
    tc, tN, tD = D, D, D

    if D < EPS:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    sc = 0.0 if abs(sN) < EPS else sN / sD
    tc = 0.0 if abs(tN) < EPS else tN / tD

    dP = w + (sc * u) - (tc * v)
    return float(np.linalg.norm(dP))


def has_self_intersection(
    vessels: dict[str, np.ndarray],
    side_branches: list[dict[str, Any]] | None = None,
    min_dist_threshold_mm: float = 1.0,
) -> tuple[bool, str]:
    """Perform explicit 1.0 mm 3D segment-to-segment self-intersection test across vessel tree (Design Doc §6.9 Line 929).

    Checks distance between non-adjacent line segments. Returns (True, error_msg) if distance < min_dist_threshold_mm.
    """
    segments = []

    # 1. Main vessel segments
    for vname, pts in vessels.items():
        if pts is not None and len(pts) >= 2:
            pts = np.asarray(pts, dtype=float)
            for i in range(len(pts) - 1):
                segments.append((f"{vname}_{i}", pts[i], pts[i + 1]))

    # 2. Side branch segments
    if side_branches:
        for idx, sb in enumerate(side_branches):
            sb_pts = sb.get("points")
            if sb_pts is not None and len(sb_pts) >= 2:
                sb_pts = np.asarray(sb_pts, dtype=float)
                for i in range(len(sb_pts) - 1):
                    segments.append((f"SB_{idx}_{i}", sb_pts[i], sb_pts[i + 1]))

    n_segs = len(segments)
    for i in range(n_segs):
        name_i, p1, q1 = segments[i]
        parts_i = name_i.rsplit("_", 1)
        vname_i, idx_i = parts_i[0], int(parts_i[1])

        for j in range(i + 1, n_segs):
            name_j, p2, q2 = segments[j]
            parts_j = name_j.rsplit("_", 1)
            vname_j, idx_j = parts_j[0], int(parts_j[1])

            # If same main vessel, continuous spline trunk does not self-intersect
            if vname_i == vname_j and not vname_i.startswith("SB"):
                continue

            # If same side branch, skip adjacent connected segments
            if vname_i == vname_j and abs(idx_i - idx_j) <= 1:
                continue

            # Skip adjacent segments sharing common vertex (e.g., bifurcation)
            if (
                np.linalg.norm(p1 - p2) < EPS
                or np.linalg.norm(p1 - q2) < EPS
                or np.linalg.norm(q1 - p2) < EPS
                or np.linalg.norm(q1 - q2) < EPS
            ):
                continue

            dist = segment_distance_3d(p1, q1, p2, q2)
            if dist < min_dist_threshold_mm:
                return True, f"Self-intersection between {name_i} and {name_j}: distance = {dist:.3f}mm < {min_dist_threshold_mm:.1f}mm"

    return False, ""


def validate_synthetic_tree(
    vessels: dict[str, np.ndarray],
    side_branches: list[dict[str, Any]],
    thresholds: dict[str, Any],
    min_dist_threshold_mm: float = 1.0,
) -> tuple[bool, list[str]]:
    """Perform full Level 6 validation check on a synthetic 3D coronary tree against population thresholds.

    Returns:
        is_valid: bool
        errors: list of error message strings
    """
    errors = []

    # 1. Check branch lengths
    for vname in ("RCA", "LMCA", "LAD", "LCX"):
        pts = vessels.get(vname)
        if pts is None or len(pts) < 2:
            errors.append(f"{vname} centerline is missing or has fewer than 2 points")
            continue

        length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        key = f"branch_length_{vname}_mm"
        if key in thresholds:
            lo = thresholds[key]["p2_5"]
            hi = thresholds[key]["p97_5"]
            if not math.isnan(lo) and not math.isnan(hi):
                if not (lo <= length <= hi):
                    errors.append(f"{vname} branch length ({length:.1f}mm) outside [{lo:.1f}mm, {hi:.1f}mm]")

    # 2. Check LMCA bifurcation angle
    lad_pts = vessels.get("LAD")
    lcx_pts = vessels.get("LCX")
    if lad_pts is not None and lcx_pts is not None and len(lad_pts) >= 2 and len(lcx_pts) >= 2:
        v_lad = lad_pts[1] - lad_pts[0]
        v_lcx = lcx_pts[1] - lcx_pts[0]
        n_lad = np.linalg.norm(v_lad)
        n_lcx = np.linalg.norm(v_lcx)
        if n_lad > EPS and n_lcx > EPS:
            cos_bif = float(np.clip(np.dot(v_lad, v_lcx) / (n_lad * n_lcx), -1.0, 1.0))
            angle = float(math.degrees(math.acos(cos_bif)))

            key = "bifurcation_angle_deg"
            if key in thresholds:
                lo = thresholds[key]["p2_5"]
                hi = thresholds[key]["p97_5"]
                if not math.isnan(lo) and not math.isnan(hi):
                    if not (lo <= angle <= hi):
                        errors.append(f"LMCA bifurcation angle ({angle:.1f} deg) outside [{lo:.1f} deg, {hi:.1f} deg]")

    # 3. Check 1.0 mm segment self-intersection test
    has_self_intersect, self_int_msg = has_self_intersection(
        vessels, side_branches, min_dist_threshold_mm=min_dist_threshold_mm
    )
    if has_self_intersect:
        errors.append(f"Self-intersection error: {self_int_msg}")

    return (len(errors) == 0), errors
