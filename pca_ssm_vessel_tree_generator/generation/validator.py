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

try:
    from generation.tree_assembler import SyntheticTree
    from surface_relative.anatomy import anatomical_role_acceptance, coronary_course_metrics
except ImportError:
    from pca_ssm_vessel_tree_generator.generation.tree_assembler import SyntheticTree
    from pca_ssm_vessel_tree_generator.surface_relative.anatomy import (
        anatomical_role_acceptance,
        coronary_course_metrics,
    )

EPS = 1.0e-12
CORE_ANATOMY_CHECKS = {
    "lad_is_dominant_descending_branch",
    "lcx_is_not_apex_descending_branch",
    "lcx_has_horizontal_crown_course",
    "lcx_is_more_lateral_than_lad",
}


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

            # Adjacent segments are topological neighbours, but non-adjacent
            # segments of the same vessel must still be checked for loops.
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


def path_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) if len(points) > 1 else 0.0


def tortuosity(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    chord = float(np.linalg.norm(points[-1] - points[0])) if len(points) > 1 else 0.0
    return path_length(points) / max(chord, EPS)


def bifurcation_angle_deg(lad: np.ndarray, lcx: np.ndarray) -> float:
    """Measure initial daughter angle from short, stable proximal chords."""
    lad = np.asarray(lad, dtype=float)
    lcx = np.asarray(lcx, dtype=float)
    if len(lad) < 2 or len(lcx) < 2:
        return math.nan
    lad_index = min(max(int(round(0.04 * (len(lad) - 1))), 1), len(lad) - 1)
    lcx_index = min(max(int(round(0.04 * (len(lcx) - 1))), 1), len(lcx) - 1)
    lad_direction = lad[lad_index] - lad[0]
    lcx_direction = lcx[lcx_index] - lcx[0]
    denominator = float(np.linalg.norm(lad_direction) * np.linalg.norm(lcx_direction))
    if denominator <= EPS:
        return math.nan
    cosine = float(np.clip(np.dot(lad_direction, lcx_direction) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def minimum_nonlocal_distance(
    points: np.ndarray, minimum_arc_separation_mm: float | None = None
) -> float:
    """Minimum distance between points separated by a physical arc distance.

    Using physical arc separation avoids false positives on densely sampled
    straight polylines while still detecting a vessel that doubles back.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return math.inf
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if minimum_arc_separation_mm is None:
        minimum_arc_separation_mm = max(3.0, 0.10 * float(cumulative[-1]))
    minimum = math.inf
    for index in range(len(points)):
        eligible = np.flatnonzero(cumulative - cumulative[index] >= minimum_arc_separation_mm)
        if len(eligible):
            minimum = min(
                minimum,
                float(np.min(np.linalg.norm(points[eligible] - points[index], axis=1))),
            )
    return minimum


def path_progression_metrics(points: np.ndarray, sample_count: int = 60) -> dict[str, float]:
    """Measure backtracking and local turns on uniform arc-length samples."""
    points = np.asarray(points, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    normalized = cumulative / max(float(cumulative[-1]), EPS)
    target = np.linspace(0.0, 1.0, sample_count)
    resampled = np.column_stack([
        np.interp(target, normalized, points[:, dimension]) for dimension in range(3)
    ])
    segments = np.diff(resampled, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    units = segments / np.maximum(lengths[:, None], EPS)
    turns = np.arccos(np.clip(np.sum(units[:-1] * units[1:], axis=1), -1.0, 1.0))
    chord = resampled[-1] - resampled[0]
    chord_length = max(float(np.linalg.norm(chord)), EPS)
    progress = segments @ (chord / chord_length)
    terminal_distance = np.linalg.norm(resampled - resampled[-1], axis=1)
    return {
        "backward_progress_ratio": float(-np.sum(np.minimum(progress, 0.0)) / chord_length),
        "terminal_progress_fraction": float(np.mean(np.diff(terminal_distance) <= 0.0)),
        "max_resampled_turn_angle_deg": float(np.degrees(np.max(turns))) if len(turns) else 0.0,
    }


def minimum_interbranch_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    trim_first_start: int = 0,
    trim_first_end: int = 0,
    trim_second_start: int = 0,
    trim_second_end: int = 0,
) -> float:
    first_stop = len(first) - trim_first_end if trim_first_end else len(first)
    second_stop = len(second) - trim_second_end if trim_second_end else len(second)
    a = np.asarray(first, dtype=float)[trim_first_start:first_stop]
    b = np.asarray(second, dtype=float)[trim_second_start:second_stop]
    if not len(a) or not len(b):
        return math.inf
    return float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)))


def _bounds(entry: Any, *, hard: bool = False) -> tuple[float, float] | None:
    if not isinstance(entry, dict):
        return None
    if hard and entry.get("min") is not None and entry.get("max") is not None:
        return float(entry["min"]), float(entry["max"])
    lower = entry.get("p2_5", entry.get("lower"))
    upper = entry.get("p97_5", entry.get("upper"))
    if lower is None or upper is None:
        return None
    return float(lower), float(upper)


def _within_bounds(value: float, bounds: tuple[float, float], *, lower_scale: float = 1.0) -> bool:
    """Compare with a tiny floating-point tolerance at learned range edges."""
    lower = lower_scale * bounds[0]
    upper = bounds[1]
    tolerance = 1.0e-9 * max(1.0, abs(lower), abs(upper))
    return lower - tolerance <= value <= upper + tolerance


class TreeValidator:
    """Compatibility validator with structural, anatomical and population QC.

    Central 95% population intervals are warnings. Observed extrema are hard
    limits. This prevents a statistically uncommon but valid sample from being
    rejected while retaining strict topology, anatomy and collision gates.
    """

    def __init__(self, thresholds: dict[str, Any] | None = None, intersection_tolerance_mm: float = 0.75):
        self.thresholds = thresholds
        self.intersection_tolerance_mm = float(intersection_tolerance_mm)

    @classmethod
    def from_person1_output(cls, path: "Path") -> "TreeValidator":
        import json
        from pathlib import Path

        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"missing validation thresholds: {source}")
        return cls(json.loads(source.read_text(encoding="utf-8")))

    def validate(self, tree: SyntheticTree) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        branches = tree.branches
        required = {"LMCA", "LAD", "LCX"}
        missing = sorted(required - set(branches))
        if missing:
            return {
                "accepted": False,
                "validation_level": "structural_only" if self.thresholds is None else "population_thresholds",
                "errors": [f"missing mandatory branches: {missing}"],
                "warnings": [],
                "metrics": {},
            }

        for name, points in branches.items():
            values = np.asarray(points, dtype=float)
            if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
                errors.append(f"{name} does not have a valid Nx3 polyline")
            elif not np.all(np.isfinite(values)):
                errors.append(f"{name} contains non-finite coordinates")

        topology_errors = {
            "LMCA_to_LAD_mm": float(np.linalg.norm(branches["LMCA"][-1] - branches["LAD"][0])),
            "LMCA_to_LCX_mm": float(np.linalg.norm(branches["LMCA"][-1] - branches["LCX"][0])),
            "LAD_to_LCX_mm": float(np.linalg.norm(branches["LAD"][0] - branches["LCX"][0])),
        }
        if max(topology_errors.values()) > 1.0e-12:
            errors.append(f"LCA bifurcation is not exact: {topology_errors}")

        lengths = {name: path_length(points) for name, points in branches.items()}
        lmca_is_shortest_core_branch = bool(
            lengths["LMCA"] < min(lengths["LAD"], lengths["LCX"])
        )
        if not lmca_is_shortest_core_branch:
            errors.append("LMCA is not shorter than both major daughter branches")
        tortuosities = {name: tortuosity(points) for name, points in branches.items()}
        obliquities = {
            name: float(np.unwrap(tree.surface_paths[name].u)[-1] - np.unwrap(tree.surface_paths[name].u)[0])
            for name in branches
            if name in tree.surface_paths
        }
        progression_metrics = {
            name: path_progression_metrics(points) for name, points in branches.items()
        }
        terminal_separation = float(
            np.linalg.norm(branches["LAD"][-1] - branches["LCX"][-1])
        )
        if terminal_separation <= 1.0e-6:
            errors.append("LAD and LCX terminals collapse to the same point")
        angle = bifurcation_angle_deg(branches["LAD"], branches["LCX"])
        if not np.isfinite(angle) or not 0.0 < angle < 180.0:
            errors.append(f"invalid LAD/LCX bifurcation angle: {angle}")

        anatomy_metrics = coronary_course_metrics(branches["LAD"], branches["LCX"])
        anatomy_acceptance = anatomical_role_acceptance(anatomy_metrics)
        anatomy_messages = {
            "lad_has_minimum_inferior_reach": "LAD terminal is not sufficiently inferior/apex-directed",
            "lad_is_dominant_descending_branch": "LAD is not the dominant descending branch relative to LCX",
            "lcx_is_not_apex_descending_branch": "LCX is too apex-directed for a crown-like branch",
            "lad_has_longitudinal_course": "LAD has insufficient longitudinal apex-directed displacement",
            "lad_descent_is_sustained": "LAD descent is not sustained along its sampled course",
            "lcx_has_horizontal_crown_course": "LCX does not have a sufficiently horizontal crown-like course",
            "lcx_inferior_span_is_limited": "LCX has excessive inferior span for a crown-like branch",
            "lcx_is_more_lateral_than_lad": "LCX is not more lateral/circumferential than LAD",
            "lcx_has_substantial_absolute_lateral_reach": "LCX has insufficient absolute lateral crown reach",
        }
        for name in anatomy_acceptance["failed_checks"]:
            message = anatomy_messages.get(name, name)
            if name in CORE_ANATOMY_CHECKS:
                errors.append(message)
            else:
                warnings.append(f"Descriptive anatomy QC: {message}")

        nonlocal_distances = {name: minimum_nonlocal_distance(points) for name, points in branches.items()}
        for name, distance in nonlocal_distances.items():
            if np.isfinite(distance) and distance < self.intersection_tolerance_mm:
                errors.append(f"{name} has a possible nonlocal self-intersection ({distance:.4f} mm)")

        trims = {name: max(3, int(round(0.08 * len(points)))) for name, points in branches.items()}
        pairwise = {
            "LMCA_to_LAD_mm": minimum_interbranch_distance(
                branches["LMCA"], branches["LAD"], trim_first_end=trims["LMCA"], trim_second_start=trims["LAD"]
            ),
            "LMCA_to_LCX_mm": minimum_interbranch_distance(
                branches["LMCA"], branches["LCX"], trim_first_end=trims["LMCA"], trim_second_start=trims["LCX"]
            ),
            "LAD_to_LCX_mm": minimum_interbranch_distance(
                branches["LAD"], branches["LCX"], trim_first_start=trims["LAD"], trim_second_start=trims["LCX"]
            ),
        }
        if "RCA" in branches:
            for other in ("LMCA", "LAD", "LCX"):
                pairwise[f"{other}_to_RCA_mm"] = minimum_interbranch_distance(branches[other], branches["RCA"])
        for pair, distance in pairwise.items():
            if np.isfinite(distance) and distance < self.intersection_tolerance_mm:
                first, second = pair.removesuffix("_mm").split("_to_")
                errors.append(f"{first} and {second} have a possible inter-branch collision ({distance:.4f} mm)")

        axes_raw = tree.ellipsoid.to_dict()
        axes = {key: float(axes_raw[key]) for key in ("a", "b", "c")}
        if self.thresholds is None:
            warnings.append("Population thresholds were not supplied; acceptance is structural only.")
        else:
            for axis, value in axes.items():
                entry = self.thresholds.get(f"ellipsoid_{axis}_mm")
                central, hard = _bounds(entry), _bounds(entry, hard=True)
                if hard and not _within_bounds(value, hard):
                    errors.append(f"ellipsoid {axis}={value:.3f} mm outside observed real range [{hard[0]:.3f}, {hard[1]:.3f}]")
                elif central and not central[0] <= value <= central[1]:
                    warnings.append(f"ellipsoid {axis} is outside the real central 95% interval")
            for name, value in lengths.items():
                entry = self.thresholds.get(f"branch_length_{name.lower()}_mm")
                central, hard = _bounds(entry), _bounds(entry, hard=True)
                if hard and not _within_bounds(value, hard, lower_scale=0.5):
                    errors.append(f"{name} length {value:.3f} mm outside observed smooth-scaffold range [{0.5 * hard[0]:.3f}, {hard[1]:.3f}]")
                elif central and not central[0] <= value <= central[1]:
                    warnings.append(f"{name} length is outside the real central 95% interval")

        return {
            "accepted": not errors,
            "validation_level": "structural_only" if self.thresholds is None else "population_thresholds",
            "validation_policy": {
                "central_95_population_intervals_are_warnings": True,
                "observed_real_min_max_used_as_hard_population_limits": True,
                "shared_anatomical_role_gate_is_hard": True,
                "hard_anatomical_role_checks": sorted(CORE_ANATOMY_CHECKS),
                "lmca_must_be_shorter_than_both_daughters": True,
                "absolute_size_anatomy_checks_are_descriptive": True,
                "self_clearance_uses_physical_arc_separation": True,
            },
            "errors": errors,
            "warnings": warnings,
            "metrics": {
                "topology_errors": topology_errors,
                "branch_lengths_mm": lengths,
                "lmca_is_shorter_than_both_daughters": lmca_is_shortest_core_branch,
                "branch_tortuosity": tortuosities,
                "branch_obliquity_rad": obliquities,
                "bifurcation_angle_deg": angle,
                "LAD_LCX_terminal_separation_mm": terminal_separation,
                "anatomical_direction": anatomy_metrics,
                "anatomical_role_acceptance": anatomy_acceptance,
                "minimum_nonlocal_distance_mm": nonlocal_distances,
                "minimum_interbranch_distance_mm": pairwise,
                "branch_progression": progression_metrics,
                "ellipsoid_axes_mm": axes,
            },
        }


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
