"""Validate LCX crown/AV-groove behaviour from unchanged Stage-1 source paths.

The coronary reference plane is fitted to the inferred RCA candidate alone.
LCX is never used to construct that reference plane, and no ellipse participates
in measurement or interval selection.  Source arrays are loaded read-only and
may undergo one global rigid transform for visualisation only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from create_canonical_heart_evidence import (
    construct_canonical_frame,
    jsonable,
    load_case,
    normalize,
    segment_lengths,
    sha256,
    write_json,
)
from lca_ssm_planes import fit_plane_svd


EPS = 1.0e-12
RCA_LABEL = "Inferred RCA candidate"
RCA_LIMITATION = (
    "The RCA path is inferred from a disconnected source component and is not "
    "annotated RCA ground truth."
)
BRANCH_COLORS = {
    "lmca": "#202020",
    "lad": "#d62728",
    "lcx": "#1679b8",
    "rca": "#76539a",
}
PILOT_IMAGES = [
    "01_unchanged_3d_source_anatomy.png",
    "02_lcx_plane_proximity_vs_arc_length.png",
    "03_lcx_local_tangent_behavior.png",
    "04_lcx_angular_sweep.png",
    "05_lad_vs_lcx_role_comparison.png",
    "06_two_plane_source_geometry_proof.png",
]
POPULATION_IMAGES = [
    "representative_supported_cases.png",
    "representative_ambiguous_cases.png",
    "representative_failed_cases.png",
    "lcx_plane_rmse_distribution.png",
    "lcx_in_plane_tangent_fraction_distribution.png",
    "angular_sweep_distribution.png",
    "crown_interval_fraction_distribution.png",
    "lad_vs_lcx_role_separation_scatter.png",
]


@dataclass(frozen=True)
class ValidationConfig:
    tangent_window: int = 3
    candidate_start_max_fraction: float = 0.20
    candidate_min_arc_fraction: float = 0.25


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def cumulative_arc(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    lengths = segment_lengths(points)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= EPS:
        raise ValueError("source path has zero arc length")
    return cumulative, cumulative / total, total


def local_tangents(points: np.ndarray, window: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    tangents = np.zeros_like(points)
    for index in range(len(points)):
        lower = max(0, index - window)
        upper = min(len(points) - 1, index + window)
        vector = points[upper] - points[lower]
        magnitude = float(np.linalg.norm(vector))
        if magnitude <= EPS:
            if index + 1 < len(points):
                vector = points[index + 1] - points[index]
            else:
                vector = points[index] - points[index - 1]
            magnitude = float(np.linalg.norm(vector))
        if magnitude > EPS:
            tangents[index] = vector / magnitude
    return tangents


def local_curve_shape(
    points: np.ndarray, tangents: np.ndarray, cumulative: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    curvature = np.zeros(len(points), dtype=float)
    turning = np.zeros(len(points), dtype=float)
    for index in range(1, len(points) - 1):
        cosine = float(np.clip(np.dot(tangents[index - 1], tangents[index + 1]), -1.0, 1.0))
        angle = math.acos(cosine)
        turning[index] = math.degrees(angle)
        span = float(cumulative[index + 1] - cumulative[index - 1])
        curvature[index] = angle / max(span, EPS)
    if len(points) > 2:
        curvature[[0, -1]] = curvature[[1, -2]]
        turning[[0, -1]] = turning[[1, -2]]
    return curvature, turning


def angular_progression(angle_deg: np.ndarray, start: int = 0, end: int | None = None) -> dict[str, Any]:
    if end is None:
        end = len(angle_deg) - 1
    values = np.asarray(angle_deg[start:end + 1], dtype=float)
    changes = np.diff(values)
    positive = float(np.sum(changes[changes > 0.0]))
    negative = float(-np.sum(changes[changes < 0.0]))
    total_variation = positive + negative
    dominant_positive = positive >= negative
    dominant = positive if dominant_positive else negative
    reverse = negative if dominant_positive else positive
    reversal_mask = changes < 0.0 if dominant_positive else changes > 0.0
    reversal_runs: list[float] = []
    cursor = 0
    while cursor < len(changes):
        if not reversal_mask[cursor]:
            cursor += 1
            continue
        run = 0.0
        while cursor < len(changes) and reversal_mask[cursor]:
            run += abs(float(changes[cursor]))
            cursor += 1
        reversal_runs.append(run)
    return {
        "net_angular_sweep_deg": float(abs(values[-1] - values[0])) if len(values) else 0.0,
        "signed_angular_sweep_deg": float(values[-1] - values[0]) if len(values) else 0.0,
        "total_angular_variation_deg": total_variation,
        "angular_monotonic_fraction": float(dominant / total_variation) if total_variation > EPS else 0.0,
        "angular_reversal_count": int(len(reversal_runs)),
        "largest_angular_reversal_deg": float(max(reversal_runs, default=0.0)),
        "total_reverse_angular_travel_deg": reverse,
        "dominant_angular_direction": "increasing" if dominant_positive else "decreasing",
    }


def local_geometry(points: np.ndarray, plane: dict[str, Any], window: int) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    cumulative, normalized, total = cumulative_arc(points)
    tangents = local_tangents(points, window)
    normal = np.asarray(plane["normal"], dtype=float)
    centroid = np.asarray(plane["centroid"], dtype=float)
    signed = (points - centroid) @ normal
    tangent_normal = tangents @ normal
    out_fraction = np.abs(tangent_normal)
    in_plane_fraction = np.sqrt(np.maximum(0.0, 1.0 - tangent_normal**2))
    tangent_angle = np.degrees(np.arcsin(np.clip(out_fraction, 0.0, 1.0)))
    segments = np.diff(points, axis=0)
    segment_normal = segments @ normal
    segment_in_plane = np.linalg.norm(segments - np.outer(segment_normal, normal), axis=1)
    segment_out_plane = np.abs(segment_normal)
    cumulative_in_plane = np.concatenate(([0.0], np.cumsum(segment_in_plane)))
    cumulative_out_plane = np.concatenate(([0.0], np.cumsum(segment_out_plane)))
    relative = points - centroid
    plane_x = relative @ np.asarray(plane["basis_u"], dtype=float)
    plane_y = relative @ np.asarray(plane["basis_v"], dtype=float)
    angle_rad = np.unwrap(np.arctan2(plane_y, plane_x))
    angle_deg = np.degrees(angle_rad)
    angular_change = np.concatenate(([0.0], np.diff(angle_deg)))
    point_prefix_in = np.concatenate(([0.0], np.cumsum(in_plane_fraction)))
    point_prefix_out = np.concatenate(([0.0], np.cumsum(out_fraction)))
    point_prefix_residual_sq = np.concatenate(([0.0], np.cumsum(signed**2)))
    angle_steps = np.diff(angle_deg)
    angle_prefix_positive = np.concatenate(([0.0], np.cumsum(np.maximum(angle_steps, 0.0))))
    angle_prefix_negative = np.concatenate(([0.0], np.cumsum(np.maximum(-angle_steps, 0.0))))
    curvature, turning = local_curve_shape(points, tangents, cumulative)
    return {
        "points": points,
        "s_mm": cumulative,
        "s_normalized": normalized,
        "length_mm": total,
        "tangents": tangents,
        "plane_signed_distance_mm": signed,
        "plane_abs_distance_mm": np.abs(signed),
        "in_plane_fraction": in_plane_fraction,
        "out_of_plane_fraction": out_fraction,
        "tangent_plane_angle_deg": tangent_angle,
        "segment_in_plane_mm": segment_in_plane,
        "segment_out_of_plane_mm": segment_out_plane,
        "cumulative_in_plane_mm": cumulative_in_plane,
        "cumulative_out_of_plane_mm": cumulative_out_plane,
        "plane_x_mm": plane_x,
        "plane_y_mm": plane_y,
        "angular_coordinate_deg": angle_deg,
        "angular_change_deg": angular_change,
        "point_prefix_in_plane_fraction": point_prefix_in,
        "point_prefix_out_of_plane_fraction": point_prefix_out,
        "point_prefix_residual_squared_mm2": point_prefix_residual_sq,
        "angle_prefix_positive_deg": angle_prefix_positive,
        "angle_prefix_negative_deg": angle_prefix_negative,
        "curvature_per_mm": curvature,
        "local_turning_angle_deg": turning,
        "angular_progression": angular_progression(angle_deg),
    }


def percentile_ranks(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(values)) if len(values) > 1 else 0.5
    return ranks if higher_is_better else 1.0 - ranks


def candidate_interval_features(local: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    s = local["s_mm"]
    total_length = float(local["length_mm"])
    length = float(s[end] - s[start])
    point_count = end - start + 1
    in_plane_travel = float(local["cumulative_in_plane_mm"][end] - local["cumulative_in_plane_mm"][start])
    out_plane_travel = float(local["cumulative_out_of_plane_mm"][end] - local["cumulative_out_of_plane_mm"][start])
    mean_in = float((local["point_prefix_in_plane_fraction"][end + 1] - local["point_prefix_in_plane_fraction"][start]) / point_count)
    mean_out = float((local["point_prefix_out_of_plane_fraction"][end + 1] - local["point_prefix_out_of_plane_fraction"][start]) / point_count)
    residual_sum_sq = float(local["point_prefix_residual_squared_mm2"][end + 1] - local["point_prefix_residual_squared_mm2"][start])
    rmse = math.sqrt(max(residual_sum_sq / point_count, 0.0))
    positive = float(local["angle_prefix_positive_deg"][end] - local["angle_prefix_positive_deg"][start])
    negative = float(local["angle_prefix_negative_deg"][end] - local["angle_prefix_negative_deg"][start])
    total_variation = positive + negative
    net_sweep = float(abs(local["angular_coordinate_deg"][end] - local["angular_coordinate_deg"][start]))
    return {
        "start_index": int(start),
        "end_index": int(end),
        "start_s_normalized": float(local["s_normalized"][start]),
        "end_s_normalized": float(local["s_normalized"][end]),
        "arc_fraction": float(length / total_length),
        "length_mm": length,
        "mean_in_plane_fraction": mean_in,
        "mean_out_of_plane_fraction": mean_out,
        "in_plane_path_fraction": float(in_plane_travel / max(length, EPS)),
        "out_of_plane_path_fraction": float(out_plane_travel / max(length, EPS)),
        "plane_rmse_mm": rmse,
        "normalized_plane_rmse": float(rmse / total_length),
        "net_angular_sweep_deg": net_sweep,
        "angular_monotonic_fraction": float(max(positive, negative) / total_variation) if total_variation > EPS else 0.0,
    }


def select_crown_interval(local: dict[str, Any], config: ValidationConfig) -> dict[str, Any]:
    starts = np.flatnonzero(local["s_normalized"] <= config.candidate_start_max_fraction)
    candidates: list[dict[str, Any]] = []
    for start in starts:
        minimum_end_fraction = float(local["s_normalized"][start] + config.candidate_min_arc_fraction)
        minimum_end = int(np.searchsorted(local["s_normalized"], minimum_end_fraction, side="left"))
        for end in range(max(start + 2, minimum_end), len(local["points"])):
            candidates.append(candidate_interval_features(local, int(start), int(end)))
    if not candidates:
        raise ValueError("no continuous LCX interval satisfies the dimensionless search safeguards")
    weights = {
        "mean_in_plane_fraction": 0.18,
        "in_plane_path_fraction": 0.18,
        "normalized_plane_rmse": 0.18,
        "out_of_plane_path_fraction": 0.12,
        "net_angular_sweep_deg": 0.14,
        "angular_monotonic_fraction": 0.12,
        "arc_fraction": 0.05,
        "start_s_normalized": 0.03,
    }
    directions = {
        "mean_in_plane_fraction": True,
        "in_plane_path_fraction": True,
        "normalized_plane_rmse": False,
        "out_of_plane_path_fraction": False,
        "net_angular_sweep_deg": True,
        "angular_monotonic_fraction": True,
        "arc_fraction": True,
        "start_s_normalized": False,
    }
    score = np.zeros(len(candidates), dtype=float)
    for key, weight in weights.items():
        score += weight * percentile_ranks(
            np.asarray([candidate[key] for candidate in candidates]), directions[key]
        )
    best_index = int(np.argmax(score))
    selected = dict(candidates[best_index])
    point_slice = slice(selected["start_index"], selected["end_index"] + 1)
    residuals = local["plane_abs_distance_mm"][point_slice]
    selected.update({
        "median_in_plane_fraction": float(np.median(local["in_plane_fraction"][point_slice])),
        "median_out_of_plane_fraction": float(np.median(local["out_of_plane_fraction"][point_slice])),
        "plane_p95_mm": float(np.percentile(residuals, 95)),
        "plane_max_mm": float(np.max(residuals)),
        **angular_progression(
            local["angular_coordinate_deg"], selected["start_index"], selected["end_index"]
        ),
    })
    selected["multi_measurement_selection_score"] = float(score[best_index])
    selected["selected_indices_are_continuous"] = True
    selected["selected_point_indices"] = np.arange(
        selected["start_index"], selected["end_index"] + 1, dtype=int
    )
    excluded = []
    if selected["start_index"] > 0:
        excluded.append({
            "index_interval": [0, selected["start_index"] - 1],
            "reason": "continuous source prefix outside the highest-ranked multi-measurement interval",
        })
    if selected["end_index"] + 1 < len(local["points"]):
        excluded.append({
            "index_interval": [selected["end_index"] + 1, len(local["points"]) - 1],
            "reason": "continuous source suffix outside the highest-ranked multi-measurement interval",
        })
    selected["excluded_intervals"] = excluded
    selected["selection_method"] = {
        "reference_plane": "true centroid SVD plane from inferred RCA candidate only",
        "candidate_intervals": "all continuous source-index intervals beginning within the first 20% of LCX arc length and spanning at least 25%",
        "candidate_ranking": "within-case percentile ranks across multiple independent geometric measurements",
        "feature_weights": weights,
        "ellipse_used": False,
        "isolated_point_removal_used": False,
        "smoothing_used": False,
        "clinical_thresholds_used": False,
    }
    evidence_votes = {
        "in_plane_path_exceeds_out_of_plane_path": selected["in_plane_path_fraction"] > selected["out_of_plane_path_fraction"],
        "mean_in_plane_tangent_exceeds_mean_out_of_plane_tangent": selected["mean_in_plane_fraction"] > selected["mean_out_of_plane_fraction"],
        "net_angular_sweep_exceeds_total_reverse_travel": selected["net_angular_sweep_deg"] > selected["total_reverse_angular_travel_deg"],
        "net_angular_sweep_has_two_to_one_reversal_dominance": selected["net_angular_sweep_deg"] > 2.0 * selected["total_reverse_angular_travel_deg"],
        "continuous_interval_meets_declared_minimum_arc_fraction": selected["arc_fraction"] >= config.candidate_min_arc_fraction,
    }
    passed_votes = sum(bool(value) for value in evidence_votes.values())
    preliminary = "supported" if passed_votes >= 4 else "ambiguous" if passed_votes >= 3 else "not_supported"
    selected["pre_population_evidence_votes"] = evidence_votes
    selected["pre_population_support"] = preliminary
    selected["pre_population_rule"] = (
        "dimensionless geometric evidence votes used only for pilot interpretation; final cohort labels use population-derived QC thresholds"
    )
    return selected


def residual_metrics(values: np.ndarray, length_mm: float) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "plane_rmse_mm": float(np.sqrt(np.mean(values**2))),
        "plane_median_mm": float(np.median(values)),
        "plane_p95_mm": float(np.percentile(values, 95)),
        "plane_max_mm": float(np.max(values)),
        "normalized_plane_rmse_by_length": float(np.sqrt(np.mean(values**2)) / max(length_mm, EPS)),
    }


def branch_role_metrics(points: np.ndarray, plane: dict[str, Any], window: int) -> tuple[dict[str, Any], dict[str, Any]]:
    local = local_geometry(points, plane, window)
    displacement = points[-1] - points[0]
    inferior_reach = float(max(0.0, points[0, 2] - np.min(points[:, 2])))
    inferior_terminal = float(max(0.0, -displacement[2]))
    length = float(local["length_mm"])
    in_travel = float(local["cumulative_in_plane_mm"][-1])
    out_travel = float(local["cumulative_out_of_plane_mm"][-1])
    angular = local["angular_progression"]
    metrics = {
        "point_count": int(len(points)),
        "length_mm": length,
        "total_length_mm": length,
        "inferior_reach_ras_z_mm": inferior_reach,
        "inferior_terminal_displacement_ras_z_mm": inferior_terminal,
        "downward_apical_dominance": float(inferior_reach / max(length, EPS)),
        "terminal_displacement_ras_mm": displacement,
        "terminal_displacement_magnitude_mm": float(np.linalg.norm(displacement)),
        "crown_plane_in_plane_travel_mm": in_travel,
        "out_of_plane_travel_mm": out_travel,
        "in_plane_path_fraction": float(in_travel / max(length, EPS)),
        "out_of_plane_path_fraction": float(out_travel / max(length, EPS)),
        "mean_in_plane_tangent_fraction": float(np.mean(local["in_plane_fraction"])),
        "median_in_plane_tangent_fraction": float(np.median(local["in_plane_fraction"])),
        "mean_in_plane_fraction": float(np.mean(local["in_plane_fraction"])),
        "median_in_plane_fraction": float(np.median(local["in_plane_fraction"])),
        **residual_metrics(local["plane_abs_distance_mm"], length),
        **angular,
        "angular_sweep_deg": angular["net_angular_sweep_deg"],
        "curvature_median_per_mm": float(np.median(local["curvature_per_mm"])),
        "curvature_p95_per_mm": float(np.percentile(local["curvature_per_mm"], 95)),
        "maximum_local_turning_angle_deg": float(np.max(local["local_turning_angle_deg"])),
    }
    return metrics, local


def assignment_evidence(lad: dict[str, Any], lcx: dict[str, Any]) -> dict[str, Any]:
    comparisons = {
        "LAD_has_greater_inferior_reach": lad["inferior_reach_ras_z_mm"] > lcx["inferior_reach_ras_z_mm"],
        "LAD_has_greater_inferior_terminal_displacement": lad["inferior_terminal_displacement_ras_z_mm"] > lcx["inferior_terminal_displacement_ras_z_mm"],
        "LAD_has_greater_downward_apical_dominance": lad["downward_apical_dominance"] > lcx["downward_apical_dominance"],
        "LCX_has_greater_in_plane_path_fraction": lcx["in_plane_path_fraction"] > lad["in_plane_path_fraction"],
        "LCX_has_greater_mean_in_plane_tangent_fraction": lcx["mean_in_plane_tangent_fraction"] > lad["mean_in_plane_tangent_fraction"],
        "LCX_has_lower_normalized_plane_residual": lcx["normalized_plane_rmse_by_length"] < lad["normalized_plane_rmse_by_length"],
        "LCX_has_greater_net_angular_sweep": lcx["net_angular_sweep_deg"] > lad["net_angular_sweep_deg"],
    }
    wins = sum(bool(value) for value in comparisons.values())
    lcx_main_apex = bool(
        lcx["inferior_reach_ras_z_mm"] > lad["inferior_reach_ras_z_mm"]
        and lcx["total_length_mm"] > lad["total_length_mm"]
    )
    consistent = bool(wins >= 5 and not lcx_main_apex)
    return {
        "method": "seven independent raw-role comparisons; no automatic branch swap",
        "comparisons": comparisons,
        "expected_role_comparison_wins": int(wins),
        "comparison_count": len(comparisons),
        "assignment_confidence": float(wins / len(comparisons)),
        "assignment_consistent": consistent,
        "lcx_behaves_as_main_apex_branch": lcx_main_apex,
        "assignment_warning": None if consistent else "current LAD/LCX assignment is geometrically questionable; manual review required",
    }


def compact_plane(plane: dict[str, Any], source: str) -> dict[str, Any]:
    residuals = np.asarray(plane["point_residuals_mm"], dtype=float)
    singular = np.asarray(plane["singular_values"], dtype=float)
    return {
        "source": source,
        "equation": "n dot (x - C) = 0",
        "centroid": plane["centroid"],
        "normal": plane["normal"],
        "basis_u": plane["basis_u"],
        "basis_v": plane["basis_v"],
        "singular_values": singular,
        "s3_over_s2": float(singular[2] / max(singular[1], EPS)),
        "s3_over_s1": float(singular[2] / max(singular[0], EPS)),
        "rmse_mm": float(plane["rms_residual_mm"]),
        "median_residual_mm": float(np.median(residuals)),
        "p95_residual_mm": float(np.percentile(residuals, 95)),
        "max_residual_mm": float(plane["max_residual_mm"]),
        "point_count": int(len(residuals)),
    }


def case_analysis(case: dict[str, Any], config: ValidationConfig) -> dict[str, Any]:
    raw = case["raw"]
    required = ("lmca", "lad", "lcx")
    if any(name not in raw for name in required):
        raise ValueError(f"{case['patient_id']}: saved source arrays are incomplete")
    source_file = case["paths"]["raw"]
    source_before = {
        "file_sha256": sha256(source_file),
        "array_sha256": {name: array_sha256(points) for name, points in raw.items()},
        "point_counts": {name: int(len(points)) for name, points in raw.items()},
        "segment_lengths": {name: segment_lengths(points).copy() for name, points in raw.items()},
        "arrays_read_only": all(not points.flags.writeable for points in raw.values()),
    }
    rca_metadata = case["metadata"].get("rca", {})
    if not rca_metadata.get("available", False) or raw.get("rca") is None:
        return {
            "case": case["patient_id"],
            "qc_status": case["qc"].get("status"),
            "reference_plane_status": "reference_plane_unavailable",
            "reference_plane_reason": "saved inferred RCA candidate is unavailable",
            "source_before": source_before,
            "warnings": ["reference_plane_unavailable"],
        }
    if len(raw["rca"]) < 3:
        return {
            "case": case["patient_id"],
            "qc_status": case["qc"].get("status"),
            "reference_plane_status": "reference_plane_unavailable",
            "reference_plane_reason": "inferred RCA candidate has fewer than three points",
            "source_before": source_before,
            "warnings": ["reference_plane_unavailable"],
        }
    rca_plane = fit_plane_svd(raw["rca"], name=f"{case['patient_id']} inferred RCA candidate")
    lad_plane = fit_plane_svd(raw["lad"], name=f"{case['patient_id']} unchanged LAD")
    lad_metrics, lad_local = branch_role_metrics(raw["lad"], rca_plane, config.tangent_window)
    lcx_metrics, lcx_local = branch_role_metrics(raw["lcx"], rca_plane, config.tangent_window)
    interval = select_crown_interval(lcx_local, config)
    assignment = assignment_evidence(lad_metrics, lcx_metrics)
    saved_assignment = case["metadata"].get("assignment", {})
    current_assignment_record = {
        "method": saved_assignment.get("method"),
        "selected_lad_source": saved_assignment.get("selected_lad_source"),
        "selected_lcx_source": saved_assignment.get("selected_lcx_source"),
        "adapter_lad_source": saved_assignment.get("adapter_lad_source"),
        "agrees_with_adapter": saved_assignment.get("agrees_with_adapter"),
        "saved_assignment_score": saved_assignment.get("score"),
        "saved_assignment_margin": saved_assignment.get("margin"),
        "identities_changed_by_this_validator": False,
    }
    bifurcation = np.asarray(raw["lmca"][-1], dtype=float)
    frame = construct_canonical_frame(
        rca_plane["normal"], lad_plane["normal"],
        raw["lad"][-1] - bifurcation, raw["lcx"][-1] - bifurcation,
    )
    rotation = np.asarray(frame["rotation_ras_to_canonical"], dtype=float)
    canonical = {name: (rotation @ (points - bifurcation).T).T for name, points in raw.items()}
    rigid_errors = {
        name: float(np.max(np.abs(segment_lengths(points) - segment_lengths(canonical[name]))))
        for name, points in raw.items()
    }
    warnings = []
    if not assignment["assignment_consistent"]:
        warnings.append(assignment["assignment_warning"])
    if interval["pre_population_support"] != "supported":
        warnings.append(f"pilot LCX crown evidence is {interval['pre_population_support']} before population QC")
    return {
        "case": case["patient_id"],
        "qc_status": case["qc"].get("status"),
        "source_geometry_modified": False,
        "reference_plane_status": "available",
        "reference_plane": compact_plane(rca_plane, "inferred_RCA_candidate_only"),
        "reference_plane_limitation": RCA_LIMITATION,
        "lad_plane": compact_plane(lad_plane, "all_unchanged_LAD_source_points"),
        "plane_separation_deg": float(np.degrees(np.arccos(np.clip(abs(np.dot(rca_plane["normal"], lad_plane["normal"])), 0.0, 1.0)))),
        "lcx": {**lcx_metrics, **{f"crown_interval_{key}": value for key, value in interval.items() if key not in {"selected_point_indices", "selection_method", "excluded_intervals", "pre_population_evidence_votes"}}},
        "lcx_crown_interval": interval,
        "lad_vs_lcx": {
            "current_saved_assignment": current_assignment_record,
            "lad": lad_metrics, "lcx": lcx_metrics, **assignment,
        },
        "assignment_consistent": assignment["assignment_consistent"],
        "assignment_confidence": assignment["assignment_confidence"],
        "assignment_warning": assignment["assignment_warning"],
        "lcx_crown_support": interval["pre_population_support"],
        "classification_basis": "pre-population dimensionless engineering evidence votes",
        "warnings": warnings,
        "source_before": source_before,
        "rigid_frame": frame,
        "maximum_rigid_segment_length_error_mm": max(rigid_errors.values()),
        "rigid_segment_length_errors_mm": rigid_errors,
        "_case": case,
        "_raw": raw,
        "_rca_plane": rca_plane,
        "_lad_plane": lad_plane,
        "_lcx_local": lcx_local,
        "_lad_local": lad_local,
        "_canonical": canonical,
    }


def validate_source_after(analysis: dict[str, Any], output_root: Path) -> dict[str, Any]:
    patient_id = analysis["case"]
    if "_raw" not in analysis:
        source_path = output_root / "raw_cases" / patient_id / "original_centerlines.npz"
        return {
            "source_file_sha256_unchanged": sha256(source_path) == analysis["source_before"]["file_sha256"],
            "reference_plane_available": False,
        }
    reloaded = load_case(output_root, patient_id)["raw"]
    branch_results = {}
    for name, before in analysis["_raw"].items():
        after = reloaded[name]
        branch_results[name] = {
            "point_count_before": int(len(before)),
            "point_count_after": int(len(after)),
            "point_count_unchanged": len(before) == len(after),
            "coordinate_hash_before": analysis["source_before"]["array_sha256"][name],
            "coordinate_hash_after": array_sha256(after),
            "coordinate_hash_unchanged": array_sha256(after) == analysis["source_before"]["array_sha256"][name],
            "maximum_coordinate_difference_mm": float(np.max(np.abs(before - after))),
            "maximum_segment_length_difference_mm": float(np.max(np.abs(segment_lengths(before) - segment_lengths(after)))),
        }
    source_file = output_root / "raw_cases" / patient_id / "original_centerlines.npz"
    result = {
        "source_file": str(source_file),
        "source_file_sha256_before": analysis["source_before"]["file_sha256"],
        "source_file_sha256_after": sha256(source_file),
        "source_file_sha256_unchanged": sha256(source_file) == analysis["source_before"]["file_sha256"],
        "source_arrays_loaded_read_only": analysis["source_before"]["arrays_read_only"],
        "branches": branch_results,
        "maximum_rigid_segment_length_error_mm": analysis["maximum_rigid_segment_length_error_mm"],
        "source_arrays_overwritten": False,
        "single_global_rigid_transform_used_for_visualization_only": True,
    }
    result["pass"] = bool(
        result["source_file_sha256_unchanged"]
        and result["source_arrays_loaded_read_only"]
        and all(item["point_count_unchanged"] and item["coordinate_hash_unchanged"] for item in branch_results.values())
        and result["maximum_rigid_segment_length_error_mm"] < 1.0e-9
    )
    return result


def compact_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    excluded = {key for key in analysis if key.startswith("_") or key == "source_before"}
    return {key: value for key, value in analysis.items() if key not in excluded}


def write_local_csv(path: Path, analysis: dict[str, Any]) -> None:
    local = analysis["_lcx_local"]
    interval = analysis["lcx_crown_interval"]
    start, end = interval["start_index"], interval["end_index"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index", "s_mm", "s_normalized", "x", "y", "z",
        "plane_signed_distance_mm", "plane_abs_distance_mm",
        "tangent_x", "tangent_y", "tangent_z",
        "in_plane_fraction", "out_of_plane_fraction", "tangent_plane_angle_deg",
        "cumulative_in_plane_path_mm", "cumulative_out_of_plane_displacement_mm",
        "curvature", "local_turning_angle_deg", "angular_coordinate_deg",
        "angular_change_deg", "inside_candidate_crown_interval",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, point in enumerate(local["points"]):
            tangent = local["tangents"][index]
            writer.writerow({
                "index": index,
                "s_mm": local["s_mm"][index],
                "s_normalized": local["s_normalized"][index],
                "x": point[0], "y": point[1], "z": point[2],
                "plane_signed_distance_mm": local["plane_signed_distance_mm"][index],
                "plane_abs_distance_mm": local["plane_abs_distance_mm"][index],
                "tangent_x": tangent[0], "tangent_y": tangent[1], "tangent_z": tangent[2],
                "in_plane_fraction": local["in_plane_fraction"][index],
                "out_of_plane_fraction": local["out_of_plane_fraction"][index],
                "tangent_plane_angle_deg": local["tangent_plane_angle_deg"][index],
                "cumulative_in_plane_path_mm": local["cumulative_in_plane_mm"][index],
                "cumulative_out_of_plane_displacement_mm": local["cumulative_out_of_plane_mm"][index],
                "curvature": local["curvature_per_mm"][index],
                "local_turning_angle_deg": local["local_turning_angle_deg"][index],
                "angular_coordinate_deg": local["angular_coordinate_deg"][index],
                "angular_change_deg": local["angular_change_deg"][index],
                "inside_candidate_crown_interval": start <= index <= end,
            })


def plane_surface_canonical(
    plane: dict[str, Any], rotation: np.ndarray, bifurcation: np.ndarray, extent: float
) -> np.ndarray:
    center = rotation @ (np.asarray(plane["centroid"]) - bifurcation)
    u = rotation @ np.asarray(plane["basis_u"])
    v = rotation @ np.asarray(plane["basis_v"])
    grid = np.linspace(-extent, extent, 9)
    aa, bb = np.meshgrid(grid, grid)
    return center + aa[..., None] * u + bb[..., None] * v


def equal_3d(axis: Any, sets: Iterable[np.ndarray]) -> None:
    points = np.vstack([np.asarray(item).reshape(-1, 3) for item in sets])
    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    radius = 0.55 * max(float(np.max(np.ptp(points, axis=0))), 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def save_figure(figure: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=dpi, facecolor="white")
    plt.close(figure)


def configure_plotting() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 10,
        "legend.fontsize": 8.5, "axes.grid": True, "grid.alpha": 0.18,
    })


def plot_pilot_source_3d(path: Path, analysis: dict[str, Any], dpi: int) -> None:
    figure = plt.figure(figsize=(16, 9))
    figure.subplots_adjust(left=0.03, right=0.78, bottom=0.10, top=0.84)
    axis = figure.add_subplot(111, projection="3d")
    canonical = analysis["_canonical"]
    raw = analysis["_raw"]
    rotation = np.asarray(analysis["rigid_frame"]["rotation_ras_to_canonical"])
    bifurcation = raw["lmca"][-1]
    extent = 0.36 * max(float(np.ptp(np.vstack(list(canonical.values())), axis=0).max()), 1.0)
    cor_surface = plane_surface_canonical(analysis["_rca_plane"], rotation, bifurcation, extent)
    lad_surface = plane_surface_canonical(analysis["_lad_plane"], rotation, bifurcation, extent)
    axis.plot_surface(*[cor_surface[..., index] for index in range(3)], color="#70b7dd", alpha=0.18, shade=False)
    axis.plot_surface(*[lad_surface[..., index] for index in range(3)], color="#f2ae72", alpha=0.18, shade=False)
    sets = [cor_surface, lad_surface]
    for name in ("lmca", "lad", "lcx", "rca"):
        points = canonical[name]
        label = RCA_LABEL if name == "rca" else name.upper()
        axis.plot(*points.T, color=BRANCH_COLORS[name], linewidth=3.0, label=label)
        sets.append(points)
    axis.scatter(0, 0, 0, color="#2ca02c", s=90, edgecolor="white", label="LMCA bifurcation")
    axis.view_init(elev=22, azim=-58)
    equal_3d(axis, sets)
    axis.set_xlabel("Global rigid canonical X (mm)")
    axis.set_ylabel("Global rigid canonical Y (mm)")
    axis.set_zlabel("Global rigid canonical Z (mm)")
    figure.suptitle("118.label — Independent LCX Crown-Trajectory Validation", fontsize=20, weight="bold")
    figure.text(0.5, 0.885, "Coronary reference plane derived from inferred RCA candidate only", ha="center", fontsize=12)
    figure.legend(loc="center left", bbox_to_anchor=(0.79, 0.52), framealpha=0.96)
    figure.text(0.5, 0.035, f"Unchanged source paths; one global rigid display transform only. {RCA_LIMITATION}", ha="center", fontsize=8.8)
    save_figure(figure, path, dpi)


def interval_span(axis: Any, analysis: dict[str, Any]) -> None:
    interval = analysis["lcx_crown_interval"]
    axis.axvspan(interval["start_s_normalized"], interval["end_s_normalized"], color="#2ca02c", alpha=0.14, label="Highest-ranked continuous candidate (not acceptance)")


def plot_pilot_proximity(path: Path, analysis: dict[str, Any], dpi: int) -> None:
    local = analysis["_lcx_local"]
    figure, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    figure.subplots_adjust(left=0.08, right=0.97, bottom=0.10, top=0.88, hspace=0.23)
    axes[0].plot(local["s_normalized"], local["plane_signed_distance_mm"], color="#455a64", linewidth=2)
    axes[0].axhline(0, color="#111111", linewidth=1)
    axes[0].set_ylabel("Signed distance (mm)")
    axes[0].set_title("Signed distance to RCA-only centroid SVD plane")
    axes[1].plot(local["s_normalized"], local["plane_abs_distance_mm"], color=BRANCH_COLORS["lcx"], linewidth=2.5)
    axes[1].set_ylabel("Absolute distance (mm)")
    axes[1].set_xlabel("Normalized unchanged LCX source arc length")
    axes[1].set_title("Absolute plane proximity (raw measurement; no residual-only point rejection)")
    for axis in axes:
        interval_span(axis, analysis)
        axis.legend(loc="best")
    figure.suptitle("118.label — LCX Plane Proximity Along Preserved Source Order", fontsize=18, weight="bold")
    save_figure(figure, path, dpi)


def plot_pilot_tangents(path: Path, analysis: dict[str, Any], dpi: int) -> None:
    local = analysis["_lcx_local"]
    figure, axis = plt.subplots(figsize=(16, 8))
    figure.subplots_adjust(left=0.08, right=0.90, bottom=0.12, top=0.86)
    axis.plot(local["s_normalized"], local["in_plane_fraction"], color="#0878b9", linewidth=2.7, label="In-plane tangent fraction")
    axis.plot(local["s_normalized"], local["out_of_plane_fraction"], color="#d95f02", linewidth=2.2, label="Out-of-plane tangent fraction")
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Normalized unchanged LCX source arc length")
    axis.set_ylabel("Unit-tangent component fraction")
    angle_axis = axis.twinx()
    angle_axis.plot(local["s_normalized"], local["tangent_plane_angle_deg"], color="#6a3d9a", alpha=0.62, linewidth=1.6, label="Tangent-to-plane angle")
    angle_axis.set_ylabel("Tangent-to-plane angle (degrees)")
    interval_span(axis, analysis)
    handles = axis.lines + angle_axis.lines + [Line2D([0], [0], color="#2ca02c", lw=8, alpha=0.18, label="Highest-ranked continuous candidate (not acceptance)")]
    axis.legend(handles=handles, loc="upper left", framealpha=0.96)
    axis.set_title(f"118.label — LCX Local Tangent Behaviour (centered source window k={analysis['lcx_crown_interval']['selection_method'].get('tangent_window', 3)})", fontsize=18, weight="bold")
    save_figure(figure, path, dpi)


def plot_pilot_angular(path: Path, analysis: dict[str, Any], dpi: int) -> None:
    local = analysis["_lcx_local"]
    interval = analysis["lcx_crown_interval"]
    start, end = interval["start_index"], interval["end_index"]
    figure, axis = plt.subplots(figsize=(12, 10))
    figure.subplots_adjust(left=0.10, right=0.96, bottom=0.10, top=0.88)
    axis.plot(local["plane_x_mm"], local["plane_y_mm"], color=BRANCH_COLORS["lcx"], alpha=0.25, linewidth=2.3, label="Full unchanged LCX projection")
    axis.plot(local["plane_x_mm"][start:end + 1], local["plane_y_mm"][start:end + 1], color=BRANCH_COLORS["lcx"], linewidth=4.2, label="Highest-ranked continuous candidate (not acceptance)")
    order_indexes = np.unique(np.linspace(0, len(local["points"]) - 2, 10).astype(int))
    for index in order_indexes:
        axis.annotate("", xy=(local["plane_x_mm"][index + 1], local["plane_y_mm"][index + 1]), xytext=(local["plane_x_mm"][index], local["plane_y_mm"][index]), arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
    axis.scatter(local["plane_x_mm"][0], local["plane_y_mm"][0], s=85, color="#2ca02c", edgecolor="white", label="LCX start / bifurcation")
    axis.scatter(local["plane_x_mm"][-1], local["plane_y_mm"][-1], s=85, color="#111111", marker="s", label="LCX terminal")
    axis.scatter(0, 0, s=70, color=BRANCH_COLORS["rca"], marker="x", label="RCA-only plane centroid / angular origin")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("RCA-only plane coordinate u (mm)")
    axis.set_ylabel("RCA-only plane coordinate v (mm)")
    axis.set_title("118.label — Unchanged LCX Angular Sweep in the Independent Coronary Plane", fontsize=17, weight="bold")
    axis.legend(loc="best", framealpha=0.96)
    figure.text(0.5, 0.035, "No ellipse defines the reference, interval, or claimed crown behaviour.", ha="center", fontsize=10)
    save_figure(figure, path, dpi)


def plot_pilot_role_comparison(path: Path, analysis: dict[str, Any], dpi: int) -> None:
    lad = analysis["lad_vs_lcx"]["lad"]
    lcx = analysis["lad_vs_lcx"]["lcx"]
    figure = plt.figure(figsize=(16, 9))
    grid = figure.add_gridspec(2, 1, height_ratios=(1.15, 1.0), hspace=0.28, left=0.05, right=0.97, bottom=0.08, top=0.88)
    table_axis = figure.add_subplot(grid[0]); table_axis.axis("off")
    rows = [
        ("Total length (mm)", lad["total_length_mm"], lcx["total_length_mm"]),
        ("Inferior/apex reach in RAS-Z (mm)", lad["inferior_reach_ras_z_mm"], lcx["inferior_reach_ras_z_mm"]),
        ("Crown-plane in-plane travel (mm)", lad["crown_plane_in_plane_travel_mm"], lcx["crown_plane_in_plane_travel_mm"]),
        ("Out-of-plane travel (mm)", lad["out_of_plane_travel_mm"], lcx["out_of_plane_travel_mm"]),
        ("Terminal displacement (mm)", lad["terminal_displacement_magnitude_mm"], lcx["terminal_displacement_magnitude_mm"]),
        ("Mean in-plane tangent fraction", lad["mean_in_plane_tangent_fraction"], lcx["mean_in_plane_tangent_fraction"]),
        ("Normalized plane RMSE", lad["normalized_plane_rmse_by_length"], lcx["normalized_plane_rmse_by_length"]),
        ("Net angular sweep (deg)", lad["net_angular_sweep_deg"], lcx["net_angular_sweep_deg"]),
    ]
    table = table_axis.table(cellText=[[name, f"{a:.4f}", f"{b:.4f}"] for name, a, b in rows], colLabels=["Raw measurement", "Assigned LAD", "Assigned LCX"], loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 1.7)
    bar_axis = figure.add_subplot(grid[1])
    labels = ["Apical dominance", "In-plane path", "In-plane tangent", "Angular sweep share"]
    lad_values = [
        lad["downward_apical_dominance"], lad["in_plane_path_fraction"],
        lad["mean_in_plane_tangent_fraction"], lad["net_angular_sweep_deg"] / max(lad["net_angular_sweep_deg"] + lcx["net_angular_sweep_deg"], EPS),
    ]
    lcx_values = [
        lcx["downward_apical_dominance"], lcx["in_plane_path_fraction"],
        lcx["mean_in_plane_tangent_fraction"], lcx["net_angular_sweep_deg"] / max(lad["net_angular_sweep_deg"] + lcx["net_angular_sweep_deg"], EPS),
    ]
    x = np.arange(len(labels)); width = 0.36
    bar_axis.bar(x - width / 2, lad_values, width, color=BRANCH_COLORS["lad"], label="Assigned LAD")
    bar_axis.bar(x + width / 2, lcx_values, width, color=BRANCH_COLORS["lcx"], label="Assigned LCX")
    bar_axis.set_xticks(x, labels); bar_axis.set_ylabel("Dimensionless measurement"); bar_axis.legend()
    figure.suptitle(f"118.label — Are the Current LAD/LCX Roles Supported?  assignment_consistent={analysis['assignment_consistent']}", fontsize=18, weight="bold")
    save_figure(figure, path, dpi)


def plot_pilot_two_plane(path: Path, analysis: dict[str, Any], dpi: int) -> None:
    figure = plt.figure(figsize=(16, 9))
    figure.subplots_adjust(left=0.03, right=0.78, bottom=0.10, top=0.88)
    axis = figure.add_subplot(111, projection="3d")
    canonical, raw = analysis["_canonical"], analysis["_raw"]
    rotation = np.asarray(analysis["rigid_frame"]["rotation_ras_to_canonical"])
    bifurcation = raw["lmca"][-1]
    extent = 0.34 * max(float(np.ptp(np.vstack(list(canonical.values())), axis=0).max()), 1.0)
    surfaces = [
        (plane_surface_canonical(analysis["_rca_plane"], rotation, bifurcation, extent), "#70b7dd"),
        (plane_surface_canonical(analysis["_lad_plane"], rotation, bifurcation, extent), "#f2ae72"),
    ]
    sets = []
    for surface, color in surfaces:
        axis.plot_surface(*[surface[..., index] for index in range(3)], color=color, alpha=0.20, shade=False)
        sets.append(surface)
    for name in ("lmca", "lad", "lcx", "rca"):
        points = canonical[name]; sets.append(points)
        axis.plot(*points.T, color=BRANCH_COLORS[name], linewidth=3.4, label=RCA_LABEL if name == "rca" else f"Unchanged {name.upper()}")
    axis.scatter(0, 0, 0, s=95, color="#2ca02c", edgecolor="white", label="LMCA bifurcation")
    axis.view_init(elev=18, azim=-56); equal_3d(axis, sets)
    axis.set_xlabel("Global rigid X (mm)"); axis.set_ylabel("Global rigid Y (mm)"); axis.set_zlabel("Global rigid Z (mm)")
    axis.set_title("118.label — Two Independent SVD Planes with Unchanged Source Geometry", fontsize=18, weight="bold")
    figure.legend(loc="center left", bbox_to_anchor=(0.79, 0.52), framealpha=0.96)
    figure.text(0.5, 0.035, "No LCX ellipse or geometry modification is shown. Imperfections remain visible.", ha="center")
    save_figure(figure, path, dpi)


def render_pilot(root: Path, analysis: dict[str, Any], dpi: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    plot_pilot_source_3d(root / PILOT_IMAGES[0], analysis, dpi)
    plot_pilot_proximity(root / PILOT_IMAGES[1], analysis, dpi)
    plot_pilot_tangents(root / PILOT_IMAGES[2], analysis, dpi)
    plot_pilot_angular(root / PILOT_IMAGES[3], analysis, dpi)
    plot_pilot_role_comparison(root / PILOT_IMAGES[4], analysis, dpi)
    plot_pilot_two_plane(root / PILOT_IMAGES[5], analysis, dpi)
    write_local_csv(root / "118.label_lcx_local_geometry.csv", analysis)


def population_row(analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis.get("reference_plane_status") != "available":
        return {
            "case": analysis["case"], "qc_status": analysis.get("qc_status"),
            "reference_plane_status": analysis.get("reference_plane_status"),
            "lcx_crown_support": "reference_plane_unavailable",
            "assignment_questionable": True,
        }
    lcx = analysis["lcx"]
    interval = analysis["lcx_crown_interval"]
    lad = analysis["lad_vs_lcx"]["lad"]
    return {
        "case": analysis["case"],
        "qc_status": analysis["qc_status"],
        "reference_plane_status": analysis["reference_plane_status"],
        "inferred_rca_candidate_point_count": analysis["reference_plane"]["point_count"],
        "rca_only_plane_rmse_mm": analysis["reference_plane"]["rmse_mm"],
        "rca_only_plane_max_residual_mm": analysis["reference_plane"]["max_residual_mm"],
        "rca_only_plane_s3_over_s2": analysis["reference_plane"]["s3_over_s2"],
        "lad_plane_rmse_mm": analysis["lad_plane"]["rmse_mm"],
        "plane_separation_deg": analysis["plane_separation_deg"],
        "lcx_point_count": lcx["point_count"],
        "lcx_length_mm": lcx["total_length_mm"],
        "lcx_plane_rmse_mm": lcx["plane_rmse_mm"],
        "lcx_normalized_plane_rmse": lcx["normalized_plane_rmse_by_length"],
        "lcx_mean_in_plane_tangent_fraction": lcx["mean_in_plane_tangent_fraction"],
        "lcx_full_in_plane_path_fraction": lcx["in_plane_path_fraction"],
        "lcx_full_net_angular_sweep_deg": lcx["net_angular_sweep_deg"],
        "crown_interval_start_index": interval["start_index"],
        "crown_interval_end_index": interval["end_index"],
        "crown_interval_start_s_normalized": interval["start_s_normalized"],
        "crown_interval_end_s_normalized": interval["end_s_normalized"],
        "crown_interval_arc_fraction": interval["arc_fraction"],
        "crown_interval_length_mm": interval["length_mm"],
        "crown_interval_plane_rmse_mm": interval["plane_rmse_mm"],
        "crown_interval_normalized_plane_rmse": interval["normalized_plane_rmse"],
        "crown_interval_mean_in_plane_fraction": interval["mean_in_plane_fraction"],
        "crown_interval_in_plane_path_fraction": interval["in_plane_path_fraction"],
        "crown_interval_out_of_plane_path_fraction": interval["out_of_plane_path_fraction"],
        "crown_interval_angular_sweep_deg": interval["net_angular_sweep_deg"],
        "crown_interval_angular_monotonic_fraction": interval["angular_monotonic_fraction"],
        "crown_interval_angular_reversal_count": interval["angular_reversal_count"],
        "crown_interval_largest_reversal_deg": interval["largest_angular_reversal_deg"],
        "lad_downward_apical_dominance": lad["downward_apical_dominance"],
        "lcx_downward_apical_dominance": lcx["downward_apical_dominance"],
        "lad_in_plane_path_fraction": lad["in_plane_path_fraction"],
        "assignment_consistent": analysis["assignment_consistent"],
        "assignment_confidence": analysis["assignment_confidence"],
        "assignment_questionable": not analysis["assignment_consistent"],
        "pre_population_support": interval["pre_population_support"],
        "source_hash_unchanged": None,
        "maximum_rigid_segment_length_error_mm": analysis["maximum_rigid_segment_length_error_mm"],
    }


def robust_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    return {
        "count": int(len(values)), "median": median, "scaled_mad": mad,
        "p05": float(np.percentile(values, 5)), "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)), "p95": float(np.percentile(values, 95)),
    }


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - median)))
    if scale <= EPS:
        scale = max(float(np.std(values)), EPS)
    return np.clip((values - median) / scale, -3.0, 3.0)


def classify_population(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row["reference_plane_status"] == "available"]
    feature_specs = {
        "crown_interval_mean_in_plane_fraction": 1.0,
        "crown_interval_in_plane_path_fraction": 1.0,
        "crown_interval_out_of_plane_path_fraction": -1.0,
        "crown_interval_normalized_plane_rmse": -1.0,
        "crown_interval_angular_monotonic_fraction": 1.0,
        "crown_interval_arc_fraction": 1.0,
        "crown_interval_angular_sweep_deg": 1.0,
    }
    score = np.zeros(len(available), dtype=float)
    feature_statistics = {}
    for key, direction in feature_specs.items():
        values = np.asarray([float(row[key]) for row in available])
        transformed = np.log1p(values) if key == "crown_interval_angular_sweep_deg" else values
        score += direction * robust_z(transformed) / len(feature_specs)
        feature_statistics[key] = robust_summary(values)
    lower = float(np.percentile(score, 10))
    supported = float(np.median(score))
    for row, value in zip(available, score):
        row["population_crown_support_score"] = float(value)
        if value >= supported:
            row["lcx_crown_support"] = "supported"
        elif value >= lower:
            row["lcx_crown_support"] = "ambiguous"
        else:
            row["lcx_crown_support"] = "not_supported"
    for row in rows:
        if row["reference_plane_status"] != "available":
            row["population_crown_support_score"] = None
            row["lcx_crown_support"] = "reference_plane_unavailable"
    return {
        "policy_name": "population-derived QC thresholds",
        "clinical_thresholds_used": False,
        "score_method": "mean signed robust-z across seven independent interval measurements",
        "supported_threshold": supported,
        "ambiguous_lower_threshold": lower,
        "not_supported_rule": "score below accepted-cohort 10th percentile",
        "ambiguous_rule": "score from accepted-cohort 10th percentile to median",
        "supported_rule": "score at or above accepted-cohort median",
        "feature_statistics": feature_statistics,
        "score_statistics": robust_summary(score),
    }


def write_population_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)


def representative_ids(rows: list[dict[str, Any]], status: str, count: int = 3) -> list[str]:
    candidates = [row for row in rows if row["lcx_crown_support"] == status]
    if not candidates:
        return []
    values = np.asarray([float(row["population_crown_support_score"]) for row in candidates])
    target = float(np.median(values))
    candidates.sort(key=lambda row: (abs(float(row["population_crown_support_score"]) - target), row["case"]))
    return [row["case"] for row in candidates[:count]]


def plot_population_representatives(
    path: Path, analyses: dict[str, dict[str, Any]], ids: list[str], title: str, dpi: int
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    figure.subplots_adjust(left=0.04, right=0.98, bottom=0.14, top=0.80, wspace=0.22)
    if not ids:
        for axis in axes:
            axis.axis("off")
        axes[1].text(0.5, 0.5, "No cases in this population-derived class", ha="center", va="center", transform=axes[1].transAxes)
    for axis, patient_id in zip(axes, ids):
        analysis = analyses[patient_id]
        local = analysis["_lcx_local"]
        interval = analysis["lcx_crown_interval"]
        start, end = interval["start_index"], interval["end_index"]
        scale = max(float(analysis["lcx"]["total_length_mm"]), EPS)
        axis.plot(local["plane_x_mm"] / scale, local["plane_y_mm"] / scale, color=BRANCH_COLORS["lcx"], alpha=0.25, linewidth=2)
        axis.plot(local["plane_x_mm"][start:end + 1] / scale, local["plane_y_mm"][start:end + 1] / scale, color=BRANCH_COLORS["lcx"], linewidth=3.4)
        axis.scatter(local["plane_x_mm"][0] / scale, local["plane_y_mm"][0] / scale, color="#2ca02c", s=35)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{patient_id}\nscore {analysis['population_crown_support_score']:.2f} | interval {interval['arc_fraction']:.0%}\nsweep {interval['net_angular_sweep_deg']:.1f}°")
        axis.set_xlabel("RCA-plane u / LCX length"); axis.set_ylabel("RCA-plane v / LCX length")
    figure.suptitle(title, fontsize=18, weight="bold")
    figure.text(0.5, 0.035, "Unchanged LCX projections; highlighted portions are continuous source-index intervals. No ellipse used.", ha="center")
    save_figure(figure, path, dpi)


def plot_distribution(path: Path, values: np.ndarray, xlabel: str, title: str, dpi: int, thresholds: list[tuple[float, str]] | None = None) -> None:
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.hist(values, bins=max(10, int(round(math.sqrt(len(values))))), color="#4c91bf", alpha=0.82, edgecolor="white")
    axis.axvline(np.median(values), color="#111111", linewidth=2, label=f"median {np.median(values):.3g}")
    if thresholds:
        for value, label in thresholds:
            axis.axvline(value, linestyle="--", linewidth=1.8, label=label)
    axis.set_xlabel(xlabel); axis.set_ylabel("Accepted-case count"); axis.set_title(title, fontsize=17, weight="bold"); axis.legend()
    save_figure(figure, path, dpi)


def plot_role_scatter(path: Path, rows: list[dict[str, Any]], dpi: int) -> None:
    available = [row for row in rows if row["reference_plane_status"] == "available"]
    x = np.asarray([row["lad_downward_apical_dominance"] - row["lcx_downward_apical_dominance"] for row in available])
    y = np.asarray([row["lcx_full_in_plane_path_fraction"] - row["lad_in_plane_path_fraction"] for row in available])
    questionable = np.asarray([bool(row["assignment_questionable"]) for row in available])
    figure, axis = plt.subplots(figsize=(11, 8))
    axis.scatter(x[~questionable], y[~questionable], color="#2ca02c", alpha=0.72, label="assignment consistent")
    axis.scatter(x[questionable], y[questionable], color="#d62728", marker="x", s=65, label="assignment questionable")
    axis.axvline(0, color="#555555", linewidth=1); axis.axhline(0, color="#555555", linewidth=1)
    axis.set_xlabel("LAD minus LCX downward/apical dominance")
    axis.set_ylabel("LCX minus LAD crown-plane path fraction")
    axis.set_title("LAD-vs-LCX Raw Role Separation", fontsize=17, weight="bold")
    axis.legend()
    save_figure(figure, path, dpi)


def render_population(root: Path, rows: list[dict[str, Any]], analyses: dict[str, dict[str, Any]], policy: dict[str, Any], dpi: int) -> dict[str, list[str]]:
    root.mkdir(parents=True, exist_ok=True)
    representatives = {
        "supported": representative_ids(rows, "supported"),
        "ambiguous": representative_ids(rows, "ambiguous"),
        "not_supported": representative_ids(rows, "not_supported"),
    }
    plot_population_representatives(root / POPULATION_IMAGES[0], analyses, representatives["supported"], "Representative LCX Crown-Supported Cases", dpi)
    plot_population_representatives(root / POPULATION_IMAGES[1], analyses, representatives["ambiguous"], "Representative LCX Crown-Ambiguous Cases", dpi)
    plot_population_representatives(root / POPULATION_IMAGES[2], analyses, representatives["not_supported"], "Representative LCX Crown-Not-Supported Cases", dpi)
    available = [row for row in rows if row["reference_plane_status"] == "available"]
    plot_distribution(root / POPULATION_IMAGES[3], np.asarray([row["lcx_plane_rmse_mm"] for row in available]), "LCX RMSE to RCA-only plane (mm)", "Accepted-Cohort LCX Plane RMSE", dpi)
    plot_distribution(root / POPULATION_IMAGES[4], np.asarray([row["lcx_mean_in_plane_tangent_fraction"] for row in available]), "Mean LCX in-plane tangent fraction", "Accepted-Cohort LCX In-Plane Tangent Dominance", dpi)
    plot_distribution(root / POPULATION_IMAGES[5], np.asarray([row["crown_interval_angular_sweep_deg"] for row in available]), "Detected-interval net angular sweep (degrees)", "Accepted-Cohort LCX Angular Sweep", dpi)
    plot_distribution(root / POPULATION_IMAGES[6], np.asarray([row["crown_interval_arc_fraction"] for row in available]), "Detected continuous interval / total LCX length", "Accepted-Cohort LCX Crown-Interval Fraction", dpi)
    plot_role_scatter(root / POPULATION_IMAGES[7], rows, dpi)
    return representatives


def update_analysis_from_population(analysis: dict[str, Any], row: dict[str, Any], policy: dict[str, Any]) -> None:
    analysis["population_crown_support_score"] = row.get("population_crown_support_score")
    analysis["lcx_crown_support"] = row["lcx_crown_support"]
    analysis["classification_basis"] = "population-derived QC thresholds (not clinical thresholds)"
    analysis["population_qc_policy"] = policy


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=repository / "outputs/lca_ssm")
    parser.add_argument("--validation-root", type=Path, default=repository / "outputs/lca_ssm/lcx_crown_validation")
    parser.add_argument("--pilot-case", default="118.label")
    parser.add_argument("--tangent-window", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pilot-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tangent_window < 1:
        raise SystemExit("--tangent-window must be at least 1")
    configure_plotting()
    config = ValidationConfig(tangent_window=args.tangent_window)
    pilot_case = load_case(args.output_root, args.pilot_case)
    pilot = case_analysis(pilot_case, config)
    if pilot.get("reference_plane_status") != "available":
        raise SystemExit(f"{args.pilot_case}: {pilot.get('reference_plane_reason')}")
    pilot["lcx_crown_interval"]["selection_method"]["tangent_window"] = args.tangent_window
    pilot_root = args.validation_root / args.pilot_case
    render_pilot(pilot_root, pilot, args.dpi)
    pilot["source_integrity"] = validate_source_after(pilot, args.output_root)
    pilot_validation = {
        "implementation_operational": True,
        "reference_plane_independent_of_lcx": True,
        "reference_plane_source": "inferred_RCA_candidate_only",
        "ellipse_used_for_measurement_or_selection": False,
        "selected_interval_continuous": pilot["lcx_crown_interval"]["selected_indices_are_continuous"],
        "source_integrity_pass": pilot["source_integrity"]["pass"],
        "local_measurements_finite": bool(all(np.all(np.isfinite(pilot["_lcx_local"][key])) for key in (
            "s_mm", "s_normalized", "tangents", "plane_signed_distance_mm",
            "in_plane_fraction", "out_of_plane_fraction", "tangent_plane_angle_deg",
            "curvature_per_mm", "angular_coordinate_deg",
        ))),
    }
    pilot_validation["pass"] = bool(
        pilot_validation["implementation_operational"]
        and pilot_validation["reference_plane_independent_of_lcx"]
        and not pilot_validation["ellipse_used_for_measurement_or_selection"]
        and pilot_validation["selected_interval_continuous"]
        and pilot_validation["source_integrity_pass"]
        and pilot_validation["local_measurements_finite"]
    )
    pilot["pilot_validation"] = pilot_validation
    write_json(pilot_root / "118.label_lcx_crown_validation.json", compact_analysis(pilot))
    if args.pilot_only:
        print(json.dumps(jsonable({
            "mode": "pilot_only", "pilot_root": pilot_root,
            "pilot_validation": pilot_validation,
            "lcx_crown_support": pilot["lcx_crown_support"],
            "assignment_consistent": pilot["assignment_consistent"],
            "lcx": pilot["lcx"],
        }), indent=2))
        return
    if not pilot_validation["pass"]:
        raise SystemExit("pilot implementation validation failed; population analysis was not started")

    accepted = json.loads((args.output_root / "quality_control/accepted_patient_index.json").read_text(encoding="utf-8"))["patient_ids"]
    analyses: dict[str, dict[str, Any]] = {args.pilot_case: pilot}
    failures: dict[str, str] = {}
    for index, patient_id in enumerate(accepted, start=1):
        if patient_id == args.pilot_case:
            continue
        try:
            analyses[patient_id] = case_analysis(load_case(args.output_root, patient_id), config)
        except Exception as exc:
            failures[patient_id] = f"{type(exc).__name__}: {exc}"
        if index % 20 == 0:
            print(f"analysed {index}/{len(accepted)} accepted cases; failures={len(failures)}", flush=True)
    rows = [population_row(analyses[patient_id]) for patient_id in accepted if patient_id in analyses]
    policy = classify_population(rows)
    row_by_id = {row["case"]: row for row in rows}
    for patient_id, analysis in analyses.items():
        update_analysis_from_population(analysis, row_by_id[patient_id], policy)
    population_root = args.validation_root / "population"
    representatives = render_population(population_root, rows, analyses, policy, args.dpi)

    integrity_results = {}
    for patient_id, analysis in analyses.items():
        integrity = validate_source_after(analysis, args.output_root)
        analysis["source_integrity"] = integrity
        integrity_results[patient_id] = integrity
        row_by_id[patient_id]["source_hash_unchanged"] = integrity.get("source_file_sha256_unchanged", False)
    write_population_csv(population_root / "population_lcx_crown_validation.csv", rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["lcx_crown_support"]] = counts.get(row["lcx_crown_support"], 0) + 1
    questionable = [row["case"] for row in rows if row.get("assignment_questionable")]
    summary = {
        "scope": "accepted cohort only; unchanged saved Stage-1 centerlines",
        "accepted_case_count": len(accepted),
        "successfully_analysed_case_count": len(rows),
        "analysis_failures": failures,
        "classification_counts": counts,
        "classification_interpretation": (
            "supported/ambiguous/not_supported are relative population-derived QC classes; "
            "they are not clinical diagnoses or proof of verified AV-groove anatomy"
        ),
        "population_qc_policy": policy,
        "representative_cases": representatives,
        "assignment_questionable_count": len(questionable),
        "assignment_questionable_cases": questionable,
        "source_integrity": {
            "all_source_file_hashes_unchanged": all(item.get("source_file_sha256_unchanged", False) for item in integrity_results.values()),
            "all_available_case_integrity_checks_pass": all(item.get("pass", True) for item in integrity_results.values()),
            "maximum_rigid_segment_length_error_mm": max(
                (item.get("maximum_rigid_segment_length_error_mm", 0.0) for item in integrity_results.values()), default=0.0
            ),
        },
        "method_guards": {
            "reference_plane_uses_lcx": False,
            "ellipse_used": False,
            "isolated_point_removal_used": False,
            "source_geometry_modified": False,
            "generative_or_pca_geometry_used": False,
            "clinical_thresholds_used": False,
        },
        "limitations": [
            RCA_LIMITATION,
            "The RCA-only SVD plane is an engineering reference, not verified AV-groove ground truth.",
            "Population QC thresholds describe this accepted cohort and are not clinical thresholds.",
            "A population-supported label means stronger relative geometric evidence within this cohort, not independently verified coronary anatomy.",
            "Angular sweep depends on the RCA-only centroid and plane basis, although both are LCX-independent.",
            "Current LAD/LCX identities remain inferred rather than definitive dataset labels.",
        ],
    }
    write_json(population_root / "population_lcx_crown_summary.json", summary)
    pilot = analyses[args.pilot_case]
    write_json(pilot_root / "118.label_lcx_crown_validation.json", compact_analysis(pilot))
    print(json.dumps(jsonable({
        "pilot_root": pilot_root,
        "population_root": population_root,
        "pilot_lcx_crown_support": pilot["lcx_crown_support"],
        "pilot_assignment_consistent": pilot["assignment_consistent"],
        "classification_counts": counts,
        "source_integrity": summary["source_integrity"],
        "analysis_failures": failures,
    }), indent=2))


if __name__ == "__main__":
    main()
