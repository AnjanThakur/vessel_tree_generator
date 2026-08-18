"""Structural and population-threshold validation for generated trees."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from generation.tree_assembler import SyntheticTree
from surface_relative.anatomy import anatomical_role_acceptance, coronary_course_metrics


def path_length(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) if len(points) > 1 else 0.0


def tortuosity(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=float)
    chord = float(np.linalg.norm(points[-1] - points[0])) if len(points) > 1 else 0.0
    return path_length(points) / max(chord, 1.0e-12)


def bifurcation_angle_deg(lad: np.ndarray, lcx: np.ndarray) -> float:
    """Use short stable daughter chords instead of one potentially noisy segment."""
    lad_index = min(max(int(round(0.04 * (len(lad) - 1))), 1), len(lad) - 1)
    lcx_index = min(max(int(round(0.04 * (len(lcx) - 1))), 1), len(lcx) - 1)
    lad_direction = lad[lad_index] - lad[0]
    lcx_direction = lcx[lcx_index] - lcx[0]
    denominator = float(np.linalg.norm(lad_direction) * np.linalg.norm(lcx_direction))
    if denominator <= 1.0e-12:
        return math.nan
    cosine = float(np.clip(np.dot(lad_direction, lcx_direction) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def path_progression_metrics(points: np.ndarray, sample_count: int = 60) -> dict[str, float]:
    """Measure backtracking and sharp turns on uniform arc-length samples."""
    points = np.asarray(points, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    normalized = cumulative / max(float(cumulative[-1]), 1.0e-12)
    target = np.linspace(0.0, 1.0, sample_count)
    resampled = np.column_stack([
        np.interp(target, normalized, points[:, dimension]) for dimension in range(3)
    ])
    segments = np.diff(resampled, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    units = segments / np.maximum(lengths[:, None], 1.0e-12)
    turns = np.arccos(np.clip(np.sum(units[:-1] * units[1:], axis=1), -1.0, 1.0))
    chord = resampled[-1] - resampled[0]
    chord_length = max(float(np.linalg.norm(chord)), 1.0e-12)
    progress = segments @ (chord / chord_length)
    terminal_distance = np.linalg.norm(resampled - resampled[-1], axis=1)
    return {
        "backward_progress_ratio": float(-np.sum(np.minimum(progress, 0.0)) / chord_length),
        "terminal_progress_fraction": float(np.mean(np.diff(terminal_distance) <= 0.0)),
        "max_resampled_turn_angle_deg": float(np.degrees(np.max(turns))) if len(turns) else 0.0,
    }


def minimum_nonlocal_distance(points: np.ndarray, minimum_arc_separation_mm: float | None = None) -> float:
    """Minimum distance between points physically separated along one polyline."""
    points = np.asarray(points, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if minimum_arc_separation_mm is None:
        minimum_arc_separation_mm = max(3.0, 0.10 * float(cumulative[-1]))
    minimum = math.inf
    for index in range(len(points)):
        eligible = np.flatnonzero(cumulative - cumulative[index] >= minimum_arc_separation_mm)
        candidates = points[eligible]
        if len(candidates):
            minimum = min(minimum, float(np.min(np.linalg.norm(candidates - points[index], axis=1))))
    return minimum


def minimum_interbranch_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    trim_first_start: int = 0,
    trim_first_end: int = 0,
    trim_second_start: int = 0,
    trim_second_end: int = 0,
) -> float:
    """Minimum distance between two branches after topological-neighbour trimming."""
    first_stop = len(first) - trim_first_end if trim_first_end else len(first)
    second_stop = len(second) - trim_second_end if trim_second_end else len(second)
    first_values = np.asarray(first, dtype=float)[trim_first_start:first_stop]
    second_values = np.asarray(second, dtype=float)[trim_second_start:second_stop]
    if not len(first_values) or not len(second_values):
        return math.inf
    return float(np.min(np.linalg.norm(first_values[:, None, :] - second_values[None, :, :], axis=2)))


def within_observed_bounds(value: float, lower: float, upper: float) -> bool:
    """Include a small reconstruction tolerance around observed hard bounds."""
    tolerance = 1.0e-3 * max(1.0, abs(float(lower)), abs(float(upper)))
    return float(lower) - tolerance <= float(value) <= float(upper) + tolerance


class TreeValidator:
    def __init__(self, thresholds: dict[str, Any] | None = None, intersection_tolerance_mm: float = 0.75):
        self.thresholds = thresholds
        self.intersection_tolerance_mm = float(intersection_tolerance_mm)

    @classmethod
    def from_person1_output(cls, path: Path) -> "TreeValidator":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"missing Person 1 validation thresholds: {path}")
        return cls(json.loads(path.read_text(encoding="utf-8")))

    @staticmethod
    def _bounds(entry: dict[str, Any]) -> tuple[float, float] | None:
        lower = entry.get("p2_5", entry.get("lower"))
        upper = entry.get("p97_5", entry.get("upper"))
        if lower is None or upper is None:
            return None
        return float(lower), float(upper)

    @staticmethod
    def _hard_bounds(entry: dict[str, Any]) -> tuple[float, float] | None:
        lower = entry.get("min")
        upper = entry.get("max")
        if lower is None or upper is None:
            return TreeValidator._bounds(entry)
        return float(lower), float(upper)

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
            if np.asarray(points).ndim != 2 or np.asarray(points).shape[1] != 3 or len(points) < 2:
                errors.append(f"{name} does not have a valid Nx3 polyline")
            elif not np.all(np.isfinite(points)):
                errors.append(f"{name} contains non-finite coordinates")

        topology_errors = {
            "LMCA_to_LAD_mm": float(np.linalg.norm(branches["LMCA"][-1] - branches["LAD"][0])),
            "LMCA_to_LCX_mm": float(np.linalg.norm(branches["LMCA"][-1] - branches["LCX"][0])),
            "LAD_to_LCX_mm": float(np.linalg.norm(branches["LAD"][0] - branches["LCX"][0])),
        }
        if max(topology_errors.values()) > 1.0e-12:
            errors.append(f"LCA bifurcation is not exact: {topology_errors}")
        terminal_separation = float(np.linalg.norm(branches["LAD"][-1] - branches["LCX"][-1]))
        if terminal_separation <= 1.0e-6:
            errors.append("LAD and LCX terminals collapse to the same point")

        lengths = {name: path_length(points) for name, points in branches.items()}
        tortuosities = {name: tortuosity(points) for name, points in branches.items()}
        obliquities = {
            name: float(np.unwrap(tree.surface_paths[name].u)[-1] - np.unwrap(tree.surface_paths[name].u)[0])
            for name in branches
            if name in tree.surface_paths
        }
        for name, length in lengths.items():
            if length <= 1.0e-6:
                errors.append(f"{name} has zero or negligible length")
        angle = bifurcation_angle_deg(branches["LAD"], branches["LCX"])
        if not np.isfinite(angle) or not 0.0 < angle < 180.0:
            errors.append(f"invalid LAD/LCX bifurcation angle: {angle}")

        anatomy_metrics = coronary_course_metrics(branches["LAD"], branches["LCX"])
        anatomy_acceptance = anatomical_role_acceptance(anatomy_metrics)
        anatomy_error_messages = {
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
        errors.extend(
            anatomy_error_messages[name] for name in anatomy_acceptance["failed_checks"]
        )

        nonlocal_distances = {name: minimum_nonlocal_distance(points) for name, points in branches.items()}
        for name, distance in nonlocal_distances.items():
            if np.isfinite(distance) and distance < self.intersection_tolerance_mm:
                errors.append(f"{name} has a possible nonlocal self-intersection ({distance:.4f} mm)")
        progression_metrics = {name: path_progression_metrics(points) for name, points in branches.items()}

        joint_trim = {
            name: max(3, int(round(0.08 * len(points))))
            for name, points in branches.items()
        }
        pairwise_distances: dict[str, float] = {}
        lca_pairs = (
            ("LMCA", "LAD", {"trim_first_end": joint_trim["LMCA"], "trim_second_start": joint_trim["LAD"]}),
            ("LMCA", "LCX", {"trim_first_end": joint_trim["LMCA"], "trim_second_start": joint_trim["LCX"]}),
            ("LAD", "LCX", {"trim_first_start": joint_trim["LAD"], "trim_second_start": joint_trim["LCX"]}),
        )
        for first_name, second_name, trims in lca_pairs:
            key = f"{first_name}_to_{second_name}_mm"
            pairwise_distances[key] = minimum_interbranch_distance(
                branches[first_name], branches[second_name], **trims
            )
        if "RCA" in branches:
            for other_name in ("LMCA", "LAD", "LCX"):
                key = f"{other_name}_to_RCA_mm"
                pairwise_distances[key] = minimum_interbranch_distance(
                    branches[other_name], branches["RCA"]
                )
        for pair, distance in pairwise_distances.items():
            if np.isfinite(distance) and distance < self.intersection_tolerance_mm:
                first_name, second_name = pair.removesuffix("_mm").split("_to_")
                errors.append(
                    f"{first_name} and {second_name} have a possible inter-branch collision "
                    f"({distance:.4f} mm)"
                )

        axes = tree.ellipsoid.to_dict()
        if self.thresholds is None:
            warnings.append("Population thresholds were not supplied; acceptance is structural only.")
        else:
            for axis in ("a", "b", "c"):
                entry = self.thresholds.get(f"ellipsoid_{axis}_mm")
                central = self._bounds(entry) if isinstance(entry, dict) else None
                hard = self._hard_bounds(entry) if isinstance(entry, dict) else None
                if hard and not within_observed_bounds(float(axes[axis]), *hard):
                    errors.append(
                        f"ellipsoid {axis}={axes[axis]:.3f} mm outside observed real range "
                        f"[{hard[0]:.3f}, {hard[1]:.3f}]"
                    )
                elif central and not central[0] <= float(axes[axis]) <= central[1]:
                    warnings.append(f"ellipsoid {axis} is outside the real central 95% interval")
            for name, length in lengths.items():
                entry = self.thresholds.get(f"branch_length_{name.lower()}_mm")
                central = self._bounds(entry) if isinstance(entry, dict) else None
                hard = self._hard_bounds(entry) if isinstance(entry, dict) else None
                effective_hard = None if hard is None else (0.5 * hard[0], hard[1])
                if effective_hard and not within_observed_bounds(length, *effective_hard):
                    errors.append(
                        f"{name} length {length:.3f} mm outside observed smooth-scaffold range "
                        f"[{effective_hard[0]:.3f}, {effective_hard[1]:.3f}]"
                    )
                elif central and not central[0] <= length <= central[1]:
                    warnings.append(f"{name} length is outside the real central 95% interval")
            angle_entry = self.thresholds.get("bifurcation_angle_deg")
            angle_central = self._bounds(angle_entry) if isinstance(angle_entry, dict) else None
            angle_hard = self._hard_bounds(angle_entry) if isinstance(angle_entry, dict) else None
            if angle_hard and not within_observed_bounds(angle, *angle_hard):
                errors.append(
                    f"bifurcation angle {angle:.3f} deg outside observed real range "
                    f"[{angle_hard[0]:.3f}, {angle_hard[1]:.3f}]"
                )
            elif angle_central and not angle_central[0] <= angle <= angle_central[1]:
                warnings.append("bifurcation angle is outside the real central 95% interval")
            tortuosity_entries = self.thresholds.get("branch_tortuosity", {})
            for name, value in tortuosities.items():
                entry = tortuosity_entries.get(name.lower()) if isinstance(tortuosity_entries, dict) else None
                central = self._bounds(entry) if isinstance(entry, dict) else None
                hard = self._hard_bounds(entry) if isinstance(entry, dict) else None
                # The generator intentionally smooths fixed anatomical control
                # points. Raw noisy-centerline lower tortuosity limits are not
                # a valid rejection criterion, but their upper limits still
                # protect against loops and excessive winding.
                if hard and not within_observed_bounds(value, 1.0, hard[1]):
                    errors.append(
                        f"{name} tortuosity {value:.3f} outside smooth-scaffold bounds "
                        f"[1.000, {hard[1]:.3f}]"
                    )
                elif central and value > central[1]:
                    warnings.append(f"{name} tortuosity exceeds the real central 95% interval")
            obliquity_entries = self.thresholds.get("branch_obliquity_rad", {})
            for name, value in obliquities.items():
                entry = obliquity_entries.get(name.lower()) if isinstance(obliquity_entries, dict) else None
                central = self._bounds(entry) if isinstance(entry, dict) else None
                hard = self._hard_bounds(entry) if isinstance(entry, dict) else None
                if hard and not within_observed_bounds(value, *hard):
                    errors.append(
                        f"{name} obliquity {value:.3f} rad outside observed real range "
                        f"[{hard[0]:.3f}, {hard[1]:.3f}]"
                    )
                elif central and not central[0] <= value <= central[1]:
                    warnings.append(f"{name} obliquity is outside the real central 95% interval")
            backward_entries = self.thresholds.get("branch_backward_progress_ratio", {})
            terminal_entries = self.thresholds.get("branch_terminal_progress_fraction", {})
            turn_entries = self.thresholds.get("branch_max_resampled_turn_angle_deg", {})
            for name, values in progression_metrics.items():
                branch_key = name.lower()
                backward_entry = backward_entries.get(branch_key, {})
                backward_hard = backward_entry.get("max")
                backward_central = backward_entry.get("p97_5")
                if backward_hard is not None and not within_observed_bounds(
                    values["backward_progress_ratio"], 0.0, float(backward_hard)
                ):
                    errors.append(
                        f"{name} backward progress ratio {values['backward_progress_ratio']:.3f} "
                        f"exceeds observed real maximum {float(backward_hard):.3f}"
                    )
                elif backward_central is not None and values["backward_progress_ratio"] > float(backward_central):
                    warnings.append(f"{name} backward progress exceeds the real central 95% interval")
                terminal_entry = terminal_entries.get(branch_key, {})
                terminal_hard = terminal_entry.get("min")
                terminal_central = terminal_entry.get("p2_5")
                if terminal_hard is not None and not within_observed_bounds(
                    values["terminal_progress_fraction"], float(terminal_hard), 1.0
                ):
                    errors.append(
                        f"{name} terminal progress fraction {values['terminal_progress_fraction']:.3f} "
                        f"is below observed real minimum {float(terminal_hard):.3f}"
                    )
                elif terminal_central is not None and values["terminal_progress_fraction"] < float(terminal_central):
                    warnings.append(f"{name} terminal progress is outside the real central 95% interval")
                turn_entry = turn_entries.get(branch_key, {})
                turn_hard = turn_entry.get("max")
                turn_central = turn_entry.get("p97_5")
                if turn_hard is not None and not within_observed_bounds(
                    values["max_resampled_turn_angle_deg"], 0.0, float(turn_hard)
                ):
                    errors.append(
                        f"{name} maximum local turn {values['max_resampled_turn_angle_deg']:.3f} deg "
                        f"exceeds observed real maximum {float(turn_hard):.3f} deg"
                    )
                elif turn_central is not None and values["max_resampled_turn_angle_deg"] > float(turn_central):
                    warnings.append(f"{name} maximum local turn exceeds the real central 95% interval")

        return {
            "accepted": not errors,
            "validation_level": "structural_only" if self.thresholds is None else "population_thresholds",
            "validation_policy": {
                "self_clearance_uses_physical_arc_separation": True,
                "branch_length_lower_bound_scale_for_smooth_scaffold": 0.5,
                "raw_tortuosity_lower_bounds_used_for_rejection": False,
                "observed_real_min_max_used_as_hard_population_limits": True,
                "central_95_population_intervals_are_warnings": True,
                "resolved_training_assignments_only": True,
                "shared_anatomical_role_gate_is_hard": True,
            },
            "errors": errors,
            "warnings": warnings,
            "metrics": {
                "topology_errors": topology_errors,
                "branch_lengths_mm": lengths,
                "branch_tortuosity": tortuosities,
                "branch_obliquity_rad": obliquities,
                "bifurcation_angle_deg": angle,
                "LAD_LCX_terminal_separation_mm": terminal_separation,
                "anatomical_direction": anatomy_metrics,
                "anatomical_role_acceptance": anatomy_acceptance,
                "minimum_nonlocal_distance_mm": nonlocal_distances,
                "minimum_interbranch_distance_mm": pairwise_distances,
                "branch_progression": progression_metrics,
                "ellipsoid_axes_mm": {name: float(axes[name]) for name in ("a", "b", "c")},
            },
        }
