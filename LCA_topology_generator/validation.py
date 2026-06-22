# LCA_topology_generator/validation.py

from typing import Dict, Any, Tuple
import numpy as np

from .parameters import (
    BRANCH_STATS,
    CONNECTION_TOLERANCE_MM,
    ANGLE_TOLERANCE_DEG,
)


def polyline_length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = v1 / max(np.linalg.norm(v1), 1e-12)
    v2 = v2 / max(np.linalg.norm(v2), 1e-12)
    value = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(value)))


def validate_lca_tree(
    branches: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    report = {
        "tree_id": metadata.get("tree_id"),
        "passed": True,
        "errors": [],
        "metrics": {},
    }

    lmca = branches["LMCA"]
    lad = branches["LAD"]
    lcx = branches["LCX"]

    # Shape validation
    for name, cp in branches.items():
        if cp.shape != (20, 3):
            report["passed"] = False
            report["errors"].append(f"{name} shape invalid: {cp.shape}, expected (20, 3)")

    # Connectivity validation
    bif = lmca[-1]
    lad_start_gap = float(np.linalg.norm(lad[0] - bif))
    lcx_start_gap = float(np.linalg.norm(lcx[0] - bif))

    report["metrics"]["lad_start_gap_mm"] = lad_start_gap
    report["metrics"]["lcx_start_gap_mm"] = lcx_start_gap

    if lad_start_gap > CONNECTION_TOLERANCE_MM:
        report["passed"] = False
        report["errors"].append("LAD does not start at LMCA bifurcation.")

    if lcx_start_gap > CONNECTION_TOLERANCE_MM:
        report["passed"] = False
        report["errors"].append("LCX does not start at LMCA bifurcation.")

    # Length validation
    lengths = {
        "LMCA": polyline_length(lmca),
        "LAD": polyline_length(lad),
        "LCX": polyline_length(lcx),
    }

    report["metrics"]["lengths_mm"] = lengths

    length_rules = {
        "LMCA": BRANCH_STATS["LMCA_LENGTH"],
        "LAD": BRANCH_STATS["LAD_LENGTH"],
        "LCX": BRANCH_STATS["LCX_LENGTH"],
    }

    for branch, length in lengths.items():
        stat = length_rules[branch]

        # Allow 15% tolerance because curved polyline can be slightly longer than target chord.
        lower = stat.min_value * 0.85
        upper = stat.max_value * 1.15

        if not (lower <= length <= upper):
            report["passed"] = False
            report["errors"].append(
                f"{branch} length out of accepted range: {length:.2f} mm"
            )

    # LAD-LCX angle validation near bifurcation
    lad_tangent = lad[3] - lad[0]
    lcx_tangent = lcx[3] - lcx[0]

    actual_angle = angle_between(lad_tangent, lcx_tangent)
    target_angle = metadata["lad_lcx_angle_target_deg"]

    report["metrics"]["lad_lcx_angle_actual_deg"] = actual_angle
    report["metrics"]["lad_lcx_angle_target_deg"] = target_angle

    if abs(actual_angle - target_angle) > ANGLE_TOLERANCE_DEG:
        report["passed"] = False
        report["errors"].append(
            f"Angle mismatch: target={target_angle:.2f}, actual={actual_angle:.2f}"
        )

    # Basic anatomical relation
    if lengths["LAD"] <= lengths["LCX"]:
        report["errors"].append(
            "Warning: LAD is not longer than LCX in this candidate."
        )

    return bool(report["passed"]), report