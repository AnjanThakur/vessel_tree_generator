"""Build the final pre-generative, dataset-derived Stage-1 LCA model.

This module never writes source centerlines.  It reconstructs neutral daughter
branches from saved assignment provenance, evaluates both LAD/LCX role
assignments against unchanged RAS geometry, validates the resolved LCX against
an RCA-only centroid-SVD plane, and builds descriptive two-plane/ellipse
references only for a conservative clean cohort.

No PCA, generated geometry, disease, radius, tube, mesh, or motion code is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from create_canonical_heart_evidence import (
    acute_plane_angle,
    construct_canonical_frame,
    exact_arc_landmarks,
    jsonable,
    load_case,
    normalize,
    segment_lengths,
    sha256,
    tangent,
    write_json,
)
from lca_ssm_curve_fitting import fit_constrained_ellipse_arc
from lca_ssm_planes import fit_plane_svd
from validate_lcx_crown_trajectory import (
    ValidationConfig,
    array_sha256,
    branch_role_metrics,
    compact_plane,
    local_geometry,
    select_crown_interval,
)


EPS = 1.0e-12
COLORS = {
    "lmca": "#232323",
    "lad": "#D62828",
    "lcx": "#1679B8",
    "rca": "#77589A",
    "daughter_a": "#E68613",
    "daughter_b": "#4C78A8",
    "coronary_plane": "#6EC5E9",
    "lad_plane": "#F3A447",
    "coronary_ellipse": "#0072B2",
    "lad_ellipse": "#D55E00",
    "landmark": "#FFBE0B",
    "bifurcation": "#2CA02C",
    "ok": "#2E8B57",
    "warn": "#E09F3E",
    "bad": "#C73E1D",
}

FIGURES = [
    "01_final_two_plane_lca_anatomical_model",
    "02_ppt_like_two_plane_schematic",
    "03_source_geometry_overlay",
    "04_branch_role_resolution_proof",
    "05_svd_plane_technical_proof",
    "06_canonical_front_view",
    "07_canonical_superior_view",
    "08_canonical_lad_plane_view",
]


@dataclass(frozen=True)
class Stage1Config:
    tangent_window: int = 3
    assignment_min_votes: int = 5
    assignment_min_margin: float = 0.16
    assignment_min_winner_score: float = 0.08
    crown_min_arc_fraction: float = 0.25
    crown_max_start_fraction: float = 0.20
    crown_min_sweep_deg: float = 15.0
    crown_min_monotonicity: float = 0.75
    crown_min_in_plane_path_fraction: float = 0.65
    crown_min_mean_in_plane_tangent: float = 0.65
    crown_max_normalized_rmse: float = 0.18
    crown_absolute_rmse_floor_mm: float = 8.0
    crown_reference_rmse_multiplier: float = 2.5
    rigid_tolerance_mm: float = 1.0e-9


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def contrast(high_role: float, low_role: float) -> float:
    """Signed within-patient contrast; positive supports the first role."""
    return float((high_role - low_role) / (abs(high_role) + abs(low_role) + EPS))


def curve_length(points: np.ndarray) -> float:
    return float(np.sum(segment_lengths(points)))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(jsonable(value), sort_keys=True) if isinstance(value, (dict, list, tuple, np.ndarray)) else value for key, value in row.items()})


def recover_neutral_daughters(case: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Recover branch_a/branch_b without changing either source sequence."""
    raw = case["raw"]
    saved = case["metadata"].get("assignment", {})
    saved_lad = saved.get("selected_lad_source")
    saved_lcx = saved.get("selected_lcx_source")
    if {saved_lad, saved_lcx} != {"branch_a", "branch_b"}:
        raise ValueError(f"{case['patient_id']}: saved daughter provenance is incomplete")
    neutral = {
        saved_lad: raw["lad"],
        saved_lcx: raw["lcx"],
    }
    adapter = case["metadata"].get("extraction", {}).get("daughter_classification", {})
    provenance = {
        "previous_saved_assignment": {
            "lad_source": saved_lad,
            "lcx_source": saved_lcx,
            "method": saved.get("method"),
            "score": saved.get("score"),
            "margin": saved.get("margin"),
        },
        "previous_adapter_assignment": {
            "lad_source": adapter.get("selected_lad_source"),
            "lcx_source": adapter.get("selected_lcx_source"),
            "status": adapter.get("status"),
            "method": adapter.get("rule"),
        },
        "neutral_reconstruction": {
            "branch_a_source_array": "lad" if saved_lad == "branch_a" else "lcx",
            "branch_b_source_array": "lad" if saved_lad == "branch_b" else "lcx",
            "source_sequences_reordered_or_modified": False,
        },
    }
    return neutral, provenance


def source_snapshot(case: dict[str, Any], neutral: dict[str, np.ndarray]) -> dict[str, Any]:
    arrays = {
        "lmca": case["raw"]["lmca"],
        "daughter_a": neutral["branch_a"],
        "daughter_b": neutral["branch_b"],
    }
    if "rca" in case["raw"]:
        arrays["rca"] = case["raw"]["rca"]
    return {
        "source_file": str(case["paths"]["raw"]),
        "source_file_sha256": sha256(case["paths"]["raw"]),
        "arrays": {
            name: {
                "point_count": int(len(points)),
                "coordinate_sha256": array_sha256(points),
                "segment_lengths_sha256": array_sha256(segment_lengths(points)),
            }
            for name, points in arrays.items()
        },
        "arrays_loaded_read_only": bool(all(not points.flags.writeable for points in case["raw"].values())),
    }


def plane_record(points: np.ndarray, name: str, source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    plane = fit_plane_svd(points, name=name)
    return plane, compact_plane(plane, source)


def lad_role_evidence(points: np.ndarray, bifurcation: np.ndarray) -> dict[str, Any]:
    points = np.asarray(points, dtype=float)
    length = curve_length(points)
    displacement = points[-1] - bifurcation
    inferior_depth = bifurcation[2] - points[:, 2]
    most_inferior_index = int(np.argmax(inferior_depth))
    dz = np.diff(points[:, 2])
    segment_len = segment_lengths(points)
    downward_travel = float(np.sum(np.maximum(-dz, 0.0)))
    upward_travel = float(np.sum(np.maximum(dz, 0.0)))
    plane, plane_compact = plane_record(points, "raw daughter LAD-role plane", "all unchanged daughter points")
    return {
        "point_count": int(len(points)),
        "total_centerline_length_mm": length,
        "inferior_apical_reach_ras_z_mm": float(max(0.0, np.max(inferior_depth))),
        "terminal_inferior_displacement_ras_z_mm": float(max(0.0, -displacement[2])),
        "signed_terminal_inferior_displacement_ras_z_mm": float(-displacement[2]),
        "terminal_displacement_ras_mm": displacement,
        "terminal_displacement_magnitude_mm": float(np.linalg.norm(displacement)),
        "downward_apical_dominance": float(downward_travel / max(downward_travel + upward_travel, EPS)),
        "downward_path_fraction": float(np.sum(segment_len[dz < 0.0]) / max(length, EPS)),
        "most_inferior_source_index": most_inferior_index,
        "most_inferior_index_fraction": float(most_inferior_index / max(len(points) - 1, 1)),
        "progression_toward_most_inferior_point": float(most_inferior_index / max(len(points) - 1, 1)),
        "best_fit_plane": plane_compact,
        "best_fit_plane_normalized_rmse": float(plane["rms_residual_mm"] / max(length, EPS)),
    }


def lcx_role_evidence(
    points: np.ndarray,
    rca_plane: dict[str, Any],
    config: Stage1Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics, local = branch_role_metrics(points, rca_plane, config.tangent_window)
    metrics["initial_direction_ras"] = local["tangents"][0]
    metrics["initial_tangent_to_rca_plane_angle_deg"] = float(local["tangent_plane_angle_deg"][0])
    interval = select_crown_interval(
        local,
        ValidationConfig(
            tangent_window=config.tangent_window,
            candidate_start_max_fraction=config.crown_max_start_fraction,
            candidate_min_arc_fraction=config.crown_min_arc_fraction,
        ),
    )
    return {
        "full_branch": metrics,
        "best_continuous_crown_interval": interval,
        "ellipse_used_for_measurement_or_selection": False,
        "reference_plane_uses_lcx": False,
    }, local


def crown_coherence_gate(interval: dict[str, Any], rca_plane: dict[str, Any], branch_length: float, config: Stage1Config) -> dict[str, Any]:
    absolute_limit = max(
        config.crown_absolute_rmse_floor_mm,
        config.crown_reference_rmse_multiplier * finite(rca_plane["rms_residual_mm"]),
    )
    votes = {
        "absolute_plane_rmse": finite(interval["plane_rmse_mm"]) <= absolute_limit,
        "normalized_plane_rmse": finite(interval["plane_rmse_mm"]) / max(branch_length, EPS) <= config.crown_max_normalized_rmse,
        "in_plane_path_fraction": finite(interval["in_plane_path_fraction"]) >= config.crown_min_in_plane_path_fraction,
        "mean_in_plane_tangent_fraction": finite(interval["mean_in_plane_fraction"]) >= config.crown_min_mean_in_plane_tangent,
        "net_angular_sweep": finite(interval["net_angular_sweep_deg"]) >= config.crown_min_sweep_deg,
        "angular_monotonicity": finite(interval["angular_monotonic_fraction"]) >= config.crown_min_monotonicity,
        "continuous_coverage": finite(interval["arc_fraction"]) >= config.crown_min_arc_fraction,
    }
    coherent_votes = sum(bool(votes[key]) for key in (
        "in_plane_path_fraction", "mean_in_plane_tangent_fraction",
        "net_angular_sweep", "angular_monotonicity", "continuous_coverage",
    ))
    has_plane_support = bool(votes["absolute_plane_rmse"] and votes["normalized_plane_rmse"])
    coherent = bool(has_plane_support and coherent_votes >= 3)
    return {
        "coherent": coherent,
        "votes": votes,
        "vote_count": int(sum(votes.values())),
        "absolute_plane_rmse_limit_mm": absolute_limit,
        "criterion_type": "transparent geometric QC criterion; not a clinical threshold",
    }


def assignment_scores(
    daughter_metrics: dict[str, dict[str, Any]],
    rca_plane: dict[str, Any],
    config: Stage1Config,
) -> dict[str, Any]:
    a, b = daughter_metrics["branch_a"], daughter_metrics["branch_b"]
    lad_a, lad_b = a["lad"], b["lad"]
    lcx_a, lcx_b = a["lcx"], b["lcx"]

    # Each group is averaged before combination so correlated metrics cannot
    # dominate by simple repetition.
    lad_contrasts_a = {
        "inferior_apical_reach": contrast(lad_a["inferior_apical_reach_ras_z_mm"], lad_b["inferior_apical_reach_ras_z_mm"]),
        "terminal_inferior_displacement": contrast(lad_a["terminal_inferior_displacement_ras_z_mm"], lad_b["terminal_inferior_displacement_ras_z_mm"]),
        "downward_apical_dominance": contrast(lad_a["downward_apical_dominance"], lad_b["downward_apical_dominance"]),
        "inferior_progression": contrast(lad_a["progression_toward_most_inferior_point"], lad_b["progression_toward_most_inferior_point"]),
    }
    ia, ib = lcx_a["best_continuous_crown_interval"], lcx_b["best_continuous_crown_interval"]
    la, lb = a["lcx_full_length_mm"], b["lcx_full_length_mm"]
    lcx_contrasts_a = {
        "lower_normalized_rca_plane_rmse": contrast(finite(ib["plane_rmse_mm"]) / max(lb, EPS), finite(ia["plane_rmse_mm"]) / max(la, EPS)),
        "in_plane_path_fraction": contrast(ia["in_plane_path_fraction"], ib["in_plane_path_fraction"]),
        "mean_in_plane_tangent_fraction": contrast(ia["mean_in_plane_fraction"], ib["mean_in_plane_fraction"]),
        "coherent_angular_progression": contrast(
            (ia["net_angular_sweep_deg"] / 60.0) * ia["angular_monotonic_fraction"],
            (ib["net_angular_sweep_deg"] / 60.0) * ib["angular_monotonic_fraction"],
        ),
    }
    lad_group_a = float(np.mean(list(lad_contrasts_a.values())))
    lcx_group_a = float(np.mean(list(lcx_contrasts_a.values())))
    # Assignment 1 wants A to be LAD and B to be LCX, so LCX evidence reverses.
    score_1 = 0.5 * (lad_group_a - lcx_group_a)
    score_2 = -score_1

    votes_assignment_1 = {
        **{f"LAD_{key}": value > 0.0 for key, value in lad_contrasts_a.items()},
        **{f"LCX_{key}": value < 0.0 for key, value in lcx_contrasts_a.items()},
    }
    votes_1 = int(sum(votes_assignment_1.values()))
    votes_2 = len(votes_assignment_1) - votes_1
    winner = 1 if score_1 >= score_2 else 2
    winner_votes = votes_1 if winner == 1 else votes_2
    lad_votes_1 = int(sum(value > 0.0 for value in lad_contrasts_a.values()))
    lcx_votes_1 = int(sum(value < 0.0 for value in lcx_contrasts_a.values()))
    winning_lad_votes = lad_votes_1 if winner == 1 else len(lad_contrasts_a) - lad_votes_1
    winning_lcx_votes = lcx_votes_1 if winner == 1 else len(lcx_contrasts_a) - lcx_votes_1
    winning_lcx = b if winner == 1 else a
    coherence = crown_coherence_gate(
        winning_lcx["lcx"]["best_continuous_crown_interval"],
        rca_plane,
        winning_lcx["lcx_full_length_mm"],
        config,
    )
    margin = abs(score_1 - score_2)
    winner_score = max(score_1, score_2)
    resolved = bool(
        winner_votes >= config.assignment_min_votes
        and winning_lad_votes >= 3
        and winning_lcx_votes >= 3
        and margin >= config.assignment_min_margin
        and winner_score >= config.assignment_min_winner_score
        and coherence["coherent"]
    )
    reason_parts = []
    if winner_votes < config.assignment_min_votes:
        reason_parts.append(f"only {winner_votes}/{len(votes_assignment_1)} independent votes support the winner")
    if winning_lad_votes < 3:
        reason_parts.append(f"only {winning_lad_votes}/4 LAD-role measurements support the proposed LAD")
    if winning_lcx_votes < 3:
        reason_parts.append(f"only {winning_lcx_votes}/4 LCX-role measurements support the proposed LCX")
    if margin < config.assignment_min_margin:
        reason_parts.append(f"score margin {margin:.3f} is below {config.assignment_min_margin:.3f}")
    if winner_score < config.assignment_min_winner_score:
        reason_parts.append(f"winner score {winner_score:.3f} is below {config.assignment_min_winner_score:.3f}")
    if not coherence["coherent"]:
        reason_parts.append("proposed LCX lacks minimum independent coronary-plane coherence")
    return {
        "assignment_1": {"lad_source": "branch_a", "lcx_source": "branch_b", "score": score_1, "evidence_votes": votes_1},
        "assignment_2": {"lad_source": "branch_b", "lcx_source": "branch_a", "score": score_2, "evidence_votes": votes_2},
        "score_margin": margin,
        "winning_candidate": winner,
        "winning_score": winner_score,
        "winning_votes": winner_votes,
        "winning_LAD_group_votes": winning_lad_votes,
        "winning_LCX_group_votes": winning_lcx_votes,
        "conflicting_evidence_count": int(len(votes_assignment_1) - winner_votes),
        "evidence_vote_count": len(votes_assignment_1),
        "assignment_1_votes": votes_assignment_1,
        "lad_group_contrasts_assignment_1": lad_contrasts_a,
        "lcx_group_contrasts_assignment_1": {key: -value for key, value in lcx_contrasts_a.items()},
        "confidence": float(0.5 * (winner_votes / len(votes_assignment_1)) + 0.5 * min(1.0, margin)),
        "resolved": resolved,
        "proposed_lcx_coherence": coherence,
        "rule": {
            "pairwise_within_patient": True,
            "cohort_thresholds_used": False,
            "minimum_votes": config.assignment_min_votes,
            "minimum_LAD_group_votes": 3,
            "minimum_LCX_group_votes": 3,
            "minimum_score_margin": config.assignment_min_margin,
            "minimum_winner_score": config.assignment_min_winner_score,
            "proposed_lcx_must_show_coronary_plane_coherence": True,
        },
        "reason": "clear pairwise majority and coherent proposed LCX" if resolved else "; ".join(reason_parts),
    }


def analyze_case(output_root: Path, patient_id: str, config: Stage1Config) -> dict[str, Any]:
    case = load_case(output_root, patient_id)
    neutral, provenance = recover_neutral_daughters(case)
    snapshot = source_snapshot(case, neutral)
    rca_meta = case["metadata"].get("rca", {})
    if "rca" not in case["raw"] or len(case["raw"].get("rca", [])) < 3 or not rca_meta.get("available", False):
        return {
            "case": patient_id,
            "source_qc_status": case["qc"].get("status"),
            **provenance,
            "resolution_status": "insufficient_reference_geometry",
            "reason": "inferred RCA candidate unavailable or contains fewer than three source points",
            "source_snapshot": snapshot,
            "_case": case,
            "_neutral": neutral,
        }

    rca_plane, rca_plane_record = plane_record(case["raw"]["rca"], f"{patient_id} RCA-only reference", "all unchanged inferred-RCA candidate points only")
    daughters: dict[str, dict[str, Any]] = {}
    for name, points in neutral.items():
        lad = lad_role_evidence(points, case["raw"]["lmca"][-1])
        lcx, local = lcx_role_evidence(points, rca_plane, config)
        daughters[name] = {
            "lad": lad,
            "lcx": lcx,
            "lcx_full_length_mm": curve_length(points),
            "_lcx_local": local,
        }
    scores = assignment_scores(daughters, rca_plane, config)
    if scores["resolved"]:
        winning = scores[f"assignment_{scores['winning_candidate']}"]
        lad_source, lcx_source = winning["lad_source"], winning["lcx_source"]
        prior = provenance["previous_saved_assignment"]
        status = "resolved_existing_assignment" if lad_source == prior["lad_source"] else "resolved_swapped_assignment"
        changed = lad_source != prior["lad_source"]
    else:
        lad_source = lcx_source = None
        status = "unresolved_manual_review"
        changed = False
    return {
        "case": patient_id,
        "source_qc_status": case["qc"].get("status"),
        **provenance,
        "rca_reference_plane": rca_plane_record,
        "rca_reference_limitation": "The RCA path is an inferred disconnected component and is not annotated RCA ground truth.",
        "daughter_metrics": daughters,
        "assignment_scoring": scores,
        "new_resolved_assignment": {"lad_source": lad_source, "lcx_source": lcx_source},
        "assignment_changed": changed,
        "changed_or_unchanged": "changed" if changed else "unchanged" if scores["resolved"] else "unresolved",
        "resolution_status": status,
        "reason": scores["reason"],
        "source_snapshot": snapshot,
        "method_guards": {
            "raw_geometry_modified": False,
            "ellipse_used_for_role_resolution": False,
            "lcx_used_in_independent_reference_plane": False,
            "cohort_thresholds_used_for_assignment": False,
            "single_feature_assignment": False,
        },
        "_case": case,
        "_neutral": neutral,
        "_rca_plane": rca_plane,
    }


def classify_crown(analysis: dict[str, Any], config: Stage1Config) -> dict[str, Any]:
    if analysis["resolution_status"] not in {"resolved_existing_assignment", "resolved_swapped_assignment"}:
        return {"status": "not_evaluated_unresolved_assignment", "reason": "branch roles unresolved"}
    source = analysis["new_resolved_assignment"]["lcx_source"]
    daughter = analysis["daughter_metrics"][source]
    interval = daughter["lcx"]["best_continuous_crown_interval"]
    gate = crown_coherence_gate(interval, analysis["_rca_plane"], daughter["lcx_full_length_mm"], config)
    votes = gate["votes"]
    critical_plane = bool(votes["absolute_plane_rmse"] and votes["normalized_plane_rmse"])
    shape_votes = sum(bool(votes[key]) for key in (
        "in_plane_path_fraction", "mean_in_plane_tangent_fraction",
        "net_angular_sweep", "angular_monotonicity", "continuous_coverage",
    ))
    full_branch_poor = bool(
        interval["arc_fraction"] >= 0.90
        and (not critical_plane or shape_votes < 4)
    )
    if critical_plane and shape_votes >= 4 and not full_branch_poor:
        status = "crown_supported"
        reason = "absolute and normalized plane proximity pass with at least four of five trajectory-shape criteria"
    elif gate["vote_count"] >= 4 and not full_branch_poor:
        status = "crown_ambiguous"
        reason = "mixed geometric evidence; support criteria are not all satisfied"
    else:
        status = "crown_not_supported"
        reason = "insufficient independent plane/trajectory evidence"
        if full_branch_poor:
            reason += "; near-full-branch selection does not rescue poor plane behaviour"
    return {
        "status": status,
        "reason": reason,
        "criterion_type": "explicit multi-signal geometric QC criterion; not clinical",
        "criteria": gate,
        "full_branch_interval_with_poor_support": full_branch_poor,
        "interval": interval,
        "reference_plane_source": "inferred RCA candidate only",
        "ellipse_used": False,
    }


def verify_source_integrity(analysis: dict[str, Any], output_root: Path) -> dict[str, Any]:
    patient_id = analysis["case"]
    before = analysis["source_snapshot"]
    reloaded = load_case(output_root, patient_id)
    neutral_after, _ = recover_neutral_daughters(reloaded)
    arrays_after = {
        "lmca": reloaded["raw"]["lmca"],
        "daughter_a": neutral_after["branch_a"],
        "daughter_b": neutral_after["branch_b"],
    }
    if "rca" in reloaded["raw"]:
        arrays_after["rca"] = reloaded["raw"]["rca"]
    arrays_before = {
        "lmca": analysis["_case"]["raw"]["lmca"],
        "daughter_a": analysis["_neutral"]["branch_a"],
        "daughter_b": analysis["_neutral"]["branch_b"],
    }
    if "rca" in analysis["_case"]["raw"]:
        arrays_before["rca"] = analysis["_case"]["raw"]["rca"]
    branches = {}
    for name, points in arrays_before.items():
        after = arrays_after[name]
        coordinate_difference = float(np.max(np.abs(points - after))) if points.shape == after.shape else math.inf
        segment_difference = float(np.max(np.abs(segment_lengths(points) - segment_lengths(after)))) if points.shape == after.shape else math.inf
        branches[name] = {
            "point_count_before": int(len(points)),
            "point_count_after": int(len(after)),
            "point_count_unchanged": bool(len(points) == len(after)),
            "coordinate_hash_before": before["arrays"][name]["coordinate_sha256"],
            "coordinate_hash_after": array_sha256(after),
            "coordinate_hash_unchanged": bool(before["arrays"][name]["coordinate_sha256"] == array_sha256(after)),
            "maximum_coordinate_difference_mm": coordinate_difference,
            "maximum_source_segment_length_difference_mm": segment_difference,
        }
    result = {
        "source_file_sha256_before": before["source_file_sha256"],
        "source_file_sha256_after": sha256(reloaded["paths"]["raw"]),
        "source_file_sha256_unchanged": before["source_file_sha256"] == sha256(reloaded["paths"]["raw"]),
        "branches": branches,
        "maximum_coordinate_difference_mm": max(item["maximum_coordinate_difference_mm"] for item in branches.values()),
        "maximum_source_segment_length_difference_mm": max(item["maximum_source_segment_length_difference_mm"] for item in branches.values()),
        "source_arrays_overwritten": False,
    }
    result["pass"] = bool(
        result["source_file_sha256_unchanged"]
        and all(item["point_count_unchanged"] and item["coordinate_hash_unchanged"] for item in branches.values())
        and result["maximum_coordinate_difference_mm"] == 0.0
        and result["maximum_source_segment_length_difference_mm"] == 0.0
    )
    return result


def project_2d(points: np.ndarray, plane: dict[str, Any]) -> np.ndarray:
    centered = points - np.asarray(plane["centroid"])
    return np.column_stack((centered @ np.asarray(plane["basis_u"]), centered @ np.asarray(plane["basis_v"])))


def sparse_full_ellipse(points_2d: np.ndarray, label: str) -> dict[str, Any]:
    """Robust five-parameter ellipse fitted only to exact source landmarks."""
    points = np.asarray(points_2d, dtype=float)
    if len(points) < 5:
        raise ValueError(f"{label}: at least five exact landmarks are required")
    center_mean = points.mean(axis=0)
    covariance = np.cov(points - center_mean, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    theta0 = float(math.atan2(vectors[1, 0], vectors[0, 0]))
    span = max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])), 1.0)
    local0 = (points - center_mean) @ vectors
    axes0 = np.maximum(0.6 * np.ptp(local0, axis=0), 0.05 * span)

    def residuals(parameters: np.ndarray) -> np.ndarray:
        cx, cy, log_a, log_b, theta = parameters
        a, b = np.exp([log_a, log_b])
        c, s = math.cos(theta), math.sin(theta)
        centered = points - np.array([cx, cy])
        x = c * centered[:, 0] + s * centered[:, 1]
        y = -s * centered[:, 0] + c * centered[:, 1]
        radial = np.sqrt((x / a) ** 2 + (y / b) ** 2)
        return (radial - 1.0) * math.sqrt(float(a * b))

    lower = np.array([points[:, 0].min() - 2 * span, points[:, 1].min() - 2 * span, math.log(0.5), math.log(0.5), -4 * math.pi])
    upper = np.array([points[:, 0].max() + 2 * span, points[:, 1].max() + 2 * span, math.log(3 * span), math.log(3 * span), 4 * math.pi])
    starts = []
    for center in (center_mean, 0.5 * (points.min(axis=0) + points.max(axis=0))):
        starts.append(np.array([*center, math.log(axes0[0]), math.log(axes0[1]), theta0]))
        starts.append(np.array([*center, math.log(axes0[1]), math.log(axes0[0]), theta0 + math.pi / 2]))
    solutions = [least_squares(residuals, np.clip(start, lower + 1e-8, upper - 1e-8), bounds=(lower, upper), loss="soft_l1", f_scale=max(0.5, 0.02 * span), max_nfev=5000) for start in starts]
    best = min(solutions, key=lambda result: float(np.mean(residuals(result.x) ** 2)))
    cx, cy, log_a, log_b, theta = best.x
    a, b = float(np.exp(log_a)), float(np.exp(log_b))
    if b > a:
        a, b = b, a
        theta += math.pi / 2
    theta = float((theta + math.pi) % math.pi)
    t = np.linspace(0.0, 2 * math.pi, 1440, endpoint=False)
    ellipse_local = np.column_stack((a * np.cos(t), b * np.sin(t)))
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    outline = ellipse_local @ rotation.T + np.array([cx, cy])
    distances, _ = cKDTree(outline).query(points)
    centered = (points - np.array([cx, cy])) @ rotation
    angles = np.arctan2(centered[:, 1] / b, centered[:, 0] / a)
    return {
        "label": label,
        "method": "robust_nonlinear_ellipse_from_sparse_exact_source_landmarks",
        "center_2d_mm": np.array([cx, cy]),
        "semi_major_axis_mm": a,
        "semi_minor_axis_mm": b,
        "eccentricity": float(math.sqrt(max(0.0, 1.0 - (b * b) / (a * a)))),
        "axis_ratio": float(a / max(b, EPS)),
        "orientation_angle_deg": math.degrees(theta),
        "orientation_rad": theta,
        "outline_2d_mm": outline,
        "outline_angles_rad": t,
        "landmark_angular_positions_rad": angles,
        "landmark_residuals_mm": distances,
        "landmark_rmse_mm": float(np.sqrt(np.mean(distances**2))),
        "maximum_landmark_residual_mm": float(np.max(distances)),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
    }


def affine_arc_full_ellipse(fit: dict[str, Any], points_2d: np.ndarray) -> dict[str, Any]:
    affine = np.asarray(fit["affine_matrix"])
    center = np.asarray(fit["center_2d"])
    t = np.linspace(0.0, 2 * math.pi, 1440, endpoint=False)
    outline = center + np.column_stack((np.cos(t), np.sin(t))) @ affine.T
    distances, _ = cKDTree(outline).query(points_2d)
    return {
        "label": "LAD interventricular ellipse",
        "method": fit["method"] + "_from_sparse_exact_source_landmarks",
        "center_2d_mm": center,
        "semi_major_axis_mm": fit["semi_major_axis_mm"],
        "semi_minor_axis_mm": fit["semi_minor_axis_mm"],
        "eccentricity": float(math.sqrt(max(0.0, 1.0 - (fit["semi_minor_axis_mm"] / fit["semi_major_axis_mm"]) ** 2))),
        "axis_ratio": fit["axis_ratio"],
        "orientation_angle_deg": fit["orientation_deg"],
        "outline_2d_mm": outline,
        "outline_angles_rad": t,
        "landmark_angular_positions_rad": np.asarray(fit["angular_positions_rad"]),
        "observed_angular_interval_rad": [fit["arc_start_rad"], fit["arc_end_rad"]],
        "landmark_residuals_mm": distances,
        "landmark_rmse_mm": float(np.sqrt(np.mean(distances**2))),
        "maximum_landmark_residual_mm": float(np.max(distances)),
        "initial_tangent_error_deg": fit["initial_tangent_error_deg"],
        "endpoint_error_mm": max(fit["start_endpoint_error_mm"], fit["terminal_endpoint_error_mm"]),
        "optimizer_success": fit["optimizer_success"],
    }


def circular_support_mask(outline_angles: np.ndarray, groups: list[np.ndarray]) -> tuple[np.ndarray, list[list[float]]]:
    mask = np.zeros(len(outline_angles), dtype=bool)
    intervals: list[list[float]] = []
    for raw in groups:
        values = np.unwrap(np.asarray(raw, dtype=float))
        if not len(values):
            continue
        start, end = float(min(values[0], values[-1])), float(max(values[0], values[-1]))
        intervals.append([start, end])
        for shift in (-2 * math.pi, 0.0, 2 * math.pi):
            mask |= (outline_angles + shift >= start) & (outline_angles + shift <= end)
    return mask, intervals


def embed_outline(ellipse: dict[str, Any], plane: dict[str, Any], rotation: np.ndarray, bifurcation: np.ndarray) -> np.ndarray:
    points_ras = (
        np.asarray(plane["centroid"])
        + ellipse["outline_2d_mm"][:, 0, None] * np.asarray(plane["basis_u"])
        + ellipse["outline_2d_mm"][:, 1, None] * np.asarray(plane["basis_v"])
    )
    return (rotation @ (points_ras - bifurcation).T).T


def full_source_ellipse_rmse(points: np.ndarray, plane: dict[str, Any], ellipse: dict[str, Any]) -> tuple[float, float]:
    distances_2d, _ = cKDTree(ellipse["outline_2d_mm"]).query(project_2d(points, plane))
    plane_distance = np.abs((points - np.asarray(plane["centroid"])) @ np.asarray(plane["normal"]))
    distance_3d = np.sqrt(distances_2d**2 + plane_distance**2)
    return float(np.sqrt(np.mean(distance_3d**2))), float(np.max(distance_3d))


def build_clean_model(analysis: dict[str, Any], config: Stage1Config) -> dict[str, Any]:
    case = analysis["_case"]
    neutral = analysis["_neutral"]
    lad_source = analysis["new_resolved_assignment"]["lad_source"]
    lcx_source = analysis["new_resolved_assignment"]["lcx_source"]
    lad, lcx = neutral[lad_source], neutral[lcx_source]
    rca = case["raw"]["rca"]
    interval = analysis["lcx_crown_validation"]["interval"]
    start, end = int(interval["start_index"]), int(interval["end_index"])
    lcx_crown = lcx[start:end + 1]
    coronary_sources = np.vstack((rca, lcx_crown))
    coronary_plane, coronary_record = plane_record(coronary_sources, f"{analysis['case']} final coronary plane", "unchanged inferred RCA + validated continuous resolved-LCX crown interval")
    lad_plane, lad_record = plane_record(lad, f"{analysis['case']} final LAD plane", "all unchanged resolved-LAD points")
    bifurcation = np.asarray(case["raw"]["lmca"][-1])
    frame = construct_canonical_frame(coronary_plane["normal"], lad_plane["normal"], lad[-1] - bifurcation, lcx[-1] - bifurcation)
    rotation = np.asarray(frame["rotation_ras_to_canonical"])
    source_ras = {"lmca": case["raw"]["lmca"], "lad": lad, "lcx": lcx, "rca": rca}
    canonical = {name: (rotation @ (points - bifurcation).T).T for name, points in source_ras.items()}
    rigid_errors = {name: float(np.max(np.abs(segment_lengths(points) - segment_lengths(canonical[name])))) for name, points in source_ras.items()}

    rca_local = local_geometry(rca, analysis["_rca_plane"], config.tangent_window)
    rca_interval = select_crown_interval(
        rca_local,
        ValidationConfig(
            tangent_window=config.tangent_window,
            candidate_start_max_fraction=config.crown_max_start_fraction,
            candidate_min_arc_fraction=config.crown_min_arc_fraction,
        ),
    )
    rca_start, rca_end = int(rca_interval["start_index"]), int(rca_interval["end_index"])
    rca_crown = rca[rca_start:rca_end + 1]
    rca_local_idx, rca_landmarks = exact_arc_landmarks(rca_crown, min(7, len(rca_crown)))
    rca_idx = rca_local_idx + rca_start
    lcx_local_idx, lcx_landmarks = exact_arc_landmarks(lcx_crown, min(7, len(lcx_crown)))
    lcx_idx = lcx_local_idx + start
    coronary_landmarks = np.vstack((rca_landmarks, lcx_landmarks))
    coronary_ellipse = sparse_full_ellipse(project_2d(coronary_landmarks, coronary_plane), "coronary crown ellipse")
    cor_angles = []
    for group in (rca_landmarks, lcx_landmarks):
        projected = project_2d(group, coronary_plane)
        center = np.asarray(coronary_ellipse["center_2d_mm"])
        theta = math.radians(coronary_ellipse["orientation_angle_deg"])
        rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        local = (projected - center) @ rot
        cor_angles.append(np.arctan2(local[:, 1] / coronary_ellipse["semi_minor_axis_mm"], local[:, 0] / coronary_ellipse["semi_major_axis_mm"]))
    coronary_ellipse["observed_support_mask"], coronary_ellipse["observed_angular_intervals_rad"] = circular_support_mask(coronary_ellipse["outline_angles_rad"], cor_angles)

    lad_idx, lad_landmarks = exact_arc_landmarks(lad, min(7, len(lad)))
    lad_landmarks_2d = project_2d(lad_landmarks, lad_plane)
    lad_tangent_3d = tangent(lad, start=True, window=min(7, len(lad) - 1))
    lad_tangent_2d = normalize(np.array([np.dot(lad_tangent_3d, lad_plane["basis_u"]), np.dot(lad_tangent_3d, lad_plane["basis_v"])]))
    lad_fit = fit_constrained_ellipse_arc(lad_landmarks_2d, lad_tangent_2d, name="LAD sparse landmark ellipse", maximum_axis_ratio=30.0)
    lad_ellipse = affine_arc_full_ellipse(lad_fit, lad_landmarks_2d)
    lad_ellipse["observed_support_mask"], intervals = circular_support_mask(lad_ellipse["outline_angles_rad"], [lad_ellipse["landmark_angular_positions_rad"]])
    lad_ellipse["observed_angular_intervals_rad"] = intervals

    coronary_ellipse["full_source_descriptive_rmse_mm"], coronary_ellipse["full_source_descriptive_maximum_residual_mm"] = full_source_ellipse_rmse(coronary_sources, coronary_plane, coronary_ellipse)
    lad_ellipse["full_source_descriptive_rmse_mm"], lad_ellipse["full_source_descriptive_maximum_residual_mm"] = full_source_ellipse_rmse(lad, lad_plane, lad_ellipse)
    coronary_ellipse["outline_canonical_mm"] = embed_outline(coronary_ellipse, coronary_plane, rotation, bifurcation)
    lad_ellipse["outline_canonical_mm"] = embed_outline(lad_ellipse, lad_plane, rotation, bifurcation)
    landmark_ras = {"rca": rca_landmarks, "lcx": lcx_landmarks, "lad": lad_landmarks}
    landmark_canonical = {name: (rotation @ (points - bifurcation).T).T for name, points in landmark_ras.items()}
    plane_angle = acute_plane_angle(coronary_plane["normal"], lad_plane["normal"])
    return {
        "case": analysis["case"],
        "assignment": analysis["new_resolved_assignment"],
        "assignment_confidence": analysis["assignment_scoring"]["confidence"],
        "crown_validation": analysis["lcx_crown_validation"],
        "source_ras": source_ras,
        "canonical": canonical,
        "bifurcation_ras": bifurcation,
        "frame": frame,
        "rotation": rotation,
        "coronary_plane": coronary_plane,
        "coronary_plane_record": coronary_record,
        "lad_plane": lad_plane,
        "lad_plane_record": lad_record,
        "plane_separation_angle_deg": plane_angle,
        "lcx_crown_source_indices": [start, end],
        "rca_compatible_crown_source_indices": [rca_start, rca_end],
        "coronary_ellipse": coronary_ellipse,
        "lad_ellipse": lad_ellipse,
        "landmark_source_indices": {"rca": rca_idx, "lcx": lcx_idx, "lad": lad_idx},
        "landmark_ras": landmark_ras,
        "landmark_canonical": landmark_canonical,
        "maximum_exact_landmark_coordinate_error_mm": 0.0,
        "rigid_segment_length_errors_mm": rigid_errors,
        "maximum_rigid_segment_length_error_mm": max(rigid_errors.values()),
        "source_geometry_modified": False,
        "measurement_planes_pass_through_centroids": True,
        "display_planes_parallel_and_translated_to_bifurcation_only": True,
    }


def compact_ellipse(ellipse: dict[str, Any], indices: Any, coordinates: Any) -> dict[str, Any]:
    excluded = {"outline_2d_mm", "outline_angles_rad", "outline_canonical_mm", "observed_support_mask", "landmark_residuals_mm", "landmark_angular_positions_rad"}
    result = {key: value for key, value in ellipse.items() if key not in excluded}
    result["landmark_indices"] = indices
    result["landmark_coordinates_ras_mm"] = coordinates
    result["ellipse_is_diagnostic_reference_only"] = True
    return result


def model_measurement_row(model: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    lad = model["source_ras"]["lad"]
    lcx = model["source_ras"]["lcx"]
    crown = analysis["lcx_crown_validation"]["interval"]
    return {
        "case": model["case"],
        "resolved_LAD_source": model["assignment"]["lad_source"],
        "resolved_LCX_source": model["assignment"]["lcx_source"],
        "coronary_plane_rmse_mm": model["coronary_plane_record"]["rmse_mm"],
        "LAD_plane_rmse_mm": model["lad_plane_record"]["rmse_mm"],
        "plane_separation_angle_deg": model["plane_separation_angle_deg"],
        "LAD_length_mm": curve_length(lad),
        "LCX_length_mm": curve_length(lcx),
        "LAD_inferior_reach_mm": analysis["daughter_metrics"][model["assignment"]["lad_source"]]["lad"]["inferior_apical_reach_ras_z_mm"],
        "LCX_crown_interval_fraction": crown["arc_fraction"],
        "LCX_crown_interval_RMSE_mm": crown["plane_rmse_mm"],
        "LCX_in_plane_tangent_fraction": crown["mean_in_plane_fraction"],
        "LCX_angular_sweep_deg": crown["net_angular_sweep_deg"],
        "source_integrity_status": analysis["source_integrity"]["pass"],
        "maximum_rigid_segment_length_error_mm": model["maximum_rigid_segment_length_error_mm"],
        "coronary_ellipse_landmark_rmse_mm": model["coronary_ellipse"]["landmark_rmse_mm"],
        "LAD_ellipse_landmark_rmse_mm": model["lad_ellipse"]["landmark_rmse_mm"],
    }


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    scale = 1.4826 * np.median(np.abs(values - median))
    if scale <= EPS:
        scale = max(np.std(values), EPS)
    return (values - median) / scale


def rank_models(models: list[dict[str, Any]], analyses: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not models:
        return []
    angle = np.array([m["plane_separation_angle_deg"] for m in models])
    typical_angle = np.median(angle)
    features = {
        "assignment": np.array([m["assignment_confidence"] for m in models]),
        "crown_votes": np.array([analyses[m["case"]]["lcx_crown_validation"]["criteria"]["vote_count"] for m in models]),
        "cor_rmse": np.array([m["coronary_plane_record"]["rmse_mm"] for m in models]),
        "lad_rmse": np.array([m["lad_plane_record"]["rmse_mm"] for m in models]),
        "coverage": np.array([analyses[m["case"]]["lcx_crown_validation"]["interval"]["arc_fraction"] for m in models]),
        "apical": np.array([analyses[m["case"]]["daughter_metrics"][m["assignment"]["lad_source"]]["lad"]["inferior_apical_reach_ras_z_mm"] for m in models]),
        "angle_typical": -np.abs(angle - typical_angle),
    }
    score = (
        0.22 * robust_z(features["assignment"])
        + 0.18 * robust_z(features["crown_votes"])
        - 0.16 * robust_z(features["cor_rmse"])
        - 0.14 * robust_z(features["lad_rmse"])
        + 0.12 * robust_z(features["coverage"])
        + 0.10 * robust_z(features["apical"])
        + 0.08 * robust_z(features["angle_typical"])
    )
    ranking = []
    for index, (model, value) in enumerate(zip(models, score)):
        ranking.append({"case": model["case"], "ranking_score": float(value), "rank_features": {key: float(values[index]) for key, values in features.items()}})
    return sorted(ranking, key=lambda item: item["ranking_score"], reverse=True)


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.15,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


class FigureWriter:
    def __init__(self, root: Path, dpi: int):
        self.root = root
        self.dpi = dpi
        self.manifest: list[dict[str, Any]] = []

    def save(self, figure: plt.Figure, relative_stem: str, description: str, case: str | None = None) -> None:
        png = self.root / f"{relative_stem}.png"
        svg = self.root / f"{relative_stem}.svg"
        png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png, dpi=self.dpi, bbox_inches="tight")
        figure.savefig(svg, bbox_inches="tight")
        plt.close(figure)
        self.manifest.append({"png": str(png.relative_to(self.root)), "svg": str(svg.relative_to(self.root)), "description": description, "case": case, "dpi": self.dpi})


def equal_3d(axis: Any, sets: Iterable[np.ndarray], padding: float = 0.06) -> None:
    arrays = [np.asarray(item) for item in sets if item is not None and len(item)]
    points = np.vstack(arrays)
    low, high = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (low + high)
    radius = max(float(np.max(high - low)) * (0.5 + padding), 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def plane_mesh(model: dict[str, Any], name: str, anchored: bool, extent: float) -> np.ndarray:
    plane = model[name]
    normal_c = model["rotation"] @ np.asarray(plane["normal"])
    origin = np.zeros(3) if anchored else model["rotation"] @ (np.asarray(plane["centroid"]) - model["bifurcation_ras"])
    u = normalize(model["rotation"] @ np.asarray(plane["basis_u"]))
    v = normalize(np.cross(normal_c, u))
    grid = np.linspace(-extent, extent, 7)
    x, y = np.meshgrid(grid, grid)
    return origin + x[..., None] * u + y[..., None] * v


def independent_plane_mesh(model: dict[str, Any], plane: dict[str, Any], extent: float) -> np.ndarray:
    origin = model["rotation"] @ (np.asarray(plane["centroid"]) - model["bifurcation_ras"])
    u = normalize(model["rotation"] @ np.asarray(plane["basis_u"]))
    v = normalize(model["rotation"] @ np.asarray(plane["basis_v"]))
    grid = np.linspace(-extent, extent, 7)
    x, y = np.meshgrid(grid, grid)
    return origin + x[..., None] * u + y[..., None] * v


def draw_source(axis: Any, model: dict[str, Any], alpha: float = 1.0, landmarks: bool = True) -> list[np.ndarray]:
    sets = []
    for name in ("lmca", "lad", "lcx", "rca"):
        points = model["canonical"][name]
        sets.append(points)
        label = {"lmca": "unchanged LMCA", "lad": "unchanged resolved LAD", "lcx": "unchanged resolved LCX", "rca": "inferred RCA candidate"}[name]
        style = "--" if name == "rca" else "-"
        axis.plot(*points.T, color=COLORS[name], lw=3.0 if name != "rca" else 2.0, ls=style, alpha=alpha, label=label)
    axis.scatter(0, 0, 0, s=70, color=COLORS["bifurcation"], edgecolor="black", zorder=20, label="shared LMCA bifurcation")
    if landmarks:
        for name, points in model["landmark_canonical"].items():
            axis.scatter(*points.T, s=30, color=COLORS["landmark"], edgecolor=COLORS[name], linewidth=0.8, zorder=15)
    return sets


def draw_planes(axis: Any, model: dict[str, Any], anchored: bool, alpha: float = 0.18) -> list[np.ndarray]:
    extent = 0.55 * max(max(np.ptp(points, axis=0)) for points in model["canonical"].values())
    meshes = []
    for key, color, label in (("coronary_plane", COLORS["coronary_plane"], "coronary / AV-groove plane"), ("lad_plane", COLORS["lad_plane"], "LAD interventricular plane")):
        mesh = plane_mesh(model, key, anchored, extent)
        meshes.append(mesh.reshape(-1, 3))
        axis.plot_surface(mesh[..., 0], mesh[..., 1], mesh[..., 2], color=color, alpha=alpha, linewidth=0, label=label)
    return meshes


def draw_masked_ellipse_3d(axis: Any, ellipse: dict[str, Any], color: str, lw: float = 3.0) -> np.ndarray:
    points = ellipse["outline_canonical_mm"]
    mask = np.asarray(ellipse["observed_support_mask"], dtype=bool)
    axis.plot(*points.T, color=color, lw=1.7, ls="--", alpha=0.45)
    solid = points.copy()
    solid[~mask] = np.nan
    axis.plot(*solid.T, color=color, lw=lw)
    return points


def draw_ellipses(axis: Any, model: dict[str, Any]) -> list[np.ndarray]:
    return [
        draw_masked_ellipse_3d(axis, model["coronary_ellipse"], COLORS["coronary_ellipse"]),
        draw_masked_ellipse_3d(axis, model["lad_ellipse"], COLORS["lad_ellipse"]),
    ]


def decorate_3d(axis: Any, title: str | None = None) -> None:
    axis.set_xlabel("Canonical X: crown")
    axis.set_ylabel("Canonical Y")
    axis.set_zlabel("Canonical Z: superior (+)")
    if title:
        axis.set_title(title)
    axis.view_init(elev=19, azim=-63)


def figure_main(writer: FigureWriter, model: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    planes = draw_planes(ax, model, anchored=False, alpha=0.16)
    ellipses = draw_ellipses(ax, model)
    sources = draw_source(ax, model)
    equal_3d(ax, planes + ellipses + sources)
    decorate_3d(ax)
    crown = model["crown_validation"]
    fig.suptitle("Dataset-Derived Two-Plane LCA Anatomical Model", fontsize=20, weight="bold")
    ax.set_title(
        "Coronary / AV-Groove Plane + LAD Interventricular Plane\n"
        f"Source Centerlines Unchanged | {model['case']} | plane angle {model['plane_separation_angle_deg']:.1f}° | "
        f"assignment confidence {model['assignment_confidence']:.2f} | {crown['status']}"
    )
    ax.legend(loc="upper left", fontsize=8)
    writer.save(fig, "representative_case/" + FIGURES[0], "Final clean-case two-plane anatomical model with unchanged sources and diagnostic ellipses.", model["case"])


def figure_schematic(writer: FigureWriter, model: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_subplot(111, projection="3d")
    planes = draw_planes(ax, model, anchored=True, alpha=0.35)
    ellipses = draw_ellipses(ax, model)
    scale = 0.45 * max(np.max(np.ptp(points, axis=0)) for points in model["canonical"].values())
    ax.scatter(0, 0, 0, s=100, color=COLORS["bifurcation"], edgecolor="black")
    ax.quiver(0, 0, 0, scale, 0, 0, color=COLORS["coronary_ellipse"], linewidth=3, arrow_length_ratio=0.1)
    ax.text(scale, 0, 0, "  +X crown direction", color=COLORS["coronary_ellipse"], weight="bold")
    ax.quiver(0, 0, 0, 0, 0, -scale, color=COLORS["lad_ellipse"], linewidth=3, arrow_length_ratio=0.1)
    ax.text(0, 0, -scale, "  −Z apex direction", color=COLORS["lad_ellipse"], weight="bold")
    equal_3d(ax, planes + ellipses)
    decorate_3d(ax)
    fig.suptitle("Dataset-Derived Two-Plane Heart Schematic", fontsize=20, weight="bold")
    ax.set_title("Display planes are parallel copies translated to the bifurcation; ellipse parameters remain measured")
    writer.save(fig, "representative_case/" + FIGURES[1], "PPT-like schematic without fabricated patient geometry.", model["case"])


def figure_overlay(writer: FigureWriter, model: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    planes = draw_planes(ax, model, anchored=False, alpha=0.13)
    ellipses = draw_ellipses(ax, model)
    sources = draw_source(ax, model, landmarks=True)
    equal_3d(ax, planes + ellipses + sources)
    decorate_3d(ax)
    fig.suptitle("Source Geometry Over the Dataset-Derived Model", fontsize=20, weight="bold")
    ax.set_title("Source coordinates unchanged. Ellipses are diagnostic references only.")
    ax.legend(loc="upper left", fontsize=8)
    writer.save(fig, "representative_case/" + FIGURES[2], "Immutable source geometry over measured planes and diagnostic references.", model["case"])


def figure_assignment(writer: FigureWriter, model: dict[str, Any], analysis: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(17, 8.5))
    neutral = analysis["_neutral"]
    bif = analysis["_case"]["raw"]["lmca"][-1]
    rotation = model["rotation"]
    canonical = {key: (rotation @ (points - bif).T).T for key, points in neutral.items()}
    scores = analysis["assignment_scoring"]
    candidates = [
        ("A", "branch_a", "branch_b", scores["assignment_1"]),
        ("B", "branch_b", "branch_a", scores["assignment_2"]),
    ]
    sets = list(canonical.values()) + [model["canonical"]["lmca"]]
    for idx, (name, lad_source, lcx_source, candidate) in enumerate(candidates, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        ax.plot(*model["canonical"]["lmca"].T, color=COLORS["lmca"], lw=3)
        ax.plot(*canonical[lad_source].T, color=COLORS["lad"], lw=3, label=f"{lad_source} → LAD")
        ax.plot(*canonical[lcx_source].T, color=COLORS["lcx"], lw=3, label=f"{lcx_source} → LCX")
        ax.scatter(0, 0, 0, s=65, color=COLORS["bifurcation"], edgecolor="black")
        equal_3d(ax, sets)
        decorate_3d(ax, f"Candidate {name}: score {candidate['score']:+.3f}; votes {candidate['evidence_votes']}/{scores['evidence_vote_count']}")
        ax.legend(loc="upper left")
        winner = scores["winning_candidate"] == idx
        ax.text2D(0.03, 0.04, "SELECTED" if winner and scores["resolved"] else "NOT SELECTED", transform=ax.transAxes, color=COLORS["ok"] if winner and scores["resolved"] else COLORS["bad"], weight="bold", fontsize=14)
    previous = analysis["previous_saved_assignment"]
    resolved = analysis["new_resolved_assignment"]
    fig.suptitle(
        "Branch Role Resolution Proof\n"
        f"Previous LAD={previous['lad_source']} / LCX={previous['lcx_source']}  →  "
        f"Resolved LAD={resolved['lad_source']} / LCX={resolved['lcx_source']} | margin={scores['score_margin']:.3f}",
        fontsize=18, weight="bold",
    )
    writer.save(fig, "representative_case/" + FIGURES[3], "Both neutral daughter assignments with raw pairwise scores and votes.", model["case"])


def figure_svd(writer: FigureWriter, model: dict[str, Any], analysis: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(121, projection="3d")
    independent = analysis["_rca_plane"]
    rca = model["canonical"]["rca"]
    ax.plot(*rca.T, color=COLORS["rca"], lw=2.5, label="RCA-only independent validation source")
    extent = 0.45 * max(max(np.ptp(points, axis=0)) for points in model["canonical"].values())
    independent_mesh = independent_plane_mesh(model, independent, extent)
    ax.plot_surface(
        independent_mesh[..., 0], independent_mesh[..., 1], independent_mesh[..., 2],
        color=COLORS["rca"], alpha=0.12, linewidth=0,
    )
    planes = draw_planes(ax, model, anchored=False, alpha=0.18)
    sources = draw_source(ax, model, alpha=0.7, landmarks=False)
    equal_3d(ax, [independent_mesh.reshape(-1, 3)] + planes + sources)
    decorate_3d(ax, "Centroid-SVD measurement planes")
    ax.legend(loc="upper left", fontsize=7)
    tx = fig.add_subplot(122)
    tx.axis("off")
    cor = model["coronary_plane_record"]
    lad = model["lad_plane_record"]
    rca_rec = analysis["rca_reference_plane"]
    body = (
        "C = mean(pᵢ)\nQᵢ = pᵢ − C\nU, S, Vᵀ = SVD(Q)\nn = Vᵀ[-1]\nn · (x − C) = 0\n\n"
        "INDEPENDENT VALIDATION PLANE\n"
        f"RCA-only S = {np.array(rca_rec['singular_values']).round(3).tolist()}\n"
        f"RMSE / median / P95 / max = {rca_rec['rmse_mm']:.2f} / {rca_rec['median_residual_mm']:.2f} / {rca_rec['p95_residual_mm']:.2f} / {rca_rec['max_residual_mm']:.2f} mm\n"
        f"S3/S1={rca_rec['s3_over_s1']:.3f}; S3/S2={rca_rec['s3_over_s2']:.3f}\n\n"
        "FINAL CORONARY PLANE\n"
        f"source = RCA + validated continuous LCX interval\nS = {np.array(cor['singular_values']).round(3).tolist()}\nRMSE={cor['rmse_mm']:.2f} mm\n\n"
        "FINAL LAD PLANE\n"
        f"source = all resolved LAD points\nS = {np.array(lad['singular_values']).round(3).tolist()}\nRMSE={lad['rmse_mm']:.2f} mm\n\n"
        f"Normal separation = {model['plane_separation_angle_deg']:.2f}°\n\n"
        "Minimum three non-collinear points define a plane; all available relevant source points are used with least-squares SVD for robustness.\n\n"
        "Measurement planes pass through centroids. Bifurcation-translated planes are display references only."
    )
    tx.text(0.02, 0.98, body, va="top", family="monospace", fontsize=11, linespacing=1.35)
    fig.suptitle("SVD Plane Technical Proof", fontsize=20, weight="bold")
    writer.save(fig, "representative_case/" + FIGURES[4], "RCA-only validation plane and final centroid-SVD plane mathematics.", model["case"])


def plot_orthographic(writer: FigureWriter, model: dict[str, Any], stem: str, title: str, x: int, y: int, labels: tuple[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    for name in ("lmca", "lad", "lcx", "rca"):
        points = model["canonical"][name]
        ax.plot(points[:, x], points[:, y], color=COLORS[name], lw=3 if name != "rca" else 2, ls="--" if name == "rca" else "-", label=name.upper())
    for key, color in (("coronary_ellipse", COLORS["coronary_ellipse"]), ("lad_ellipse", COLORS["lad_ellipse"])):
        ellipse = model[key]
        points, mask = ellipse["outline_canonical_mm"], ellipse["observed_support_mask"]
        ax.plot(points[:, x], points[:, y], color=color, ls="--", alpha=0.4, lw=1.5)
        solid = points.copy(); solid[~mask] = np.nan
        ax.plot(solid[:, x], solid[:, y], color=color, lw=3)
    ax.scatter(0, 0, s=75, color=COLORS["bifurcation"], edgecolor="black", zorder=10)
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title + "\nFixed canonical axes; no per-view or per-branch rotation")
    ax.legend(ncol=3, fontsize=8)
    writer.save(fig, "representative_case/" + stem, title, model["case"])


def figure_montage(writer: FigureWriter, selected: list[dict[str, Any]]) -> None:
    count = len(selected)
    cols = 3
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5.2 * rows), squeeze=False)
    all_points = np.vstack([points for model in selected for points in model["canonical"].values()])
    scale = float(np.percentile(np.abs(all_points), 98))
    for ax, model in zip(axes.flat, selected):
        for name in ("lmca", "lad", "lcx", "rca"):
            points = model["canonical"][name]
            ax.plot(points[:, 0], points[:, 2], color=COLORS[name], lw=2.2 if name != "rca" else 1.5, ls="--" if name == "rca" else "-")
        ax.scatter(0, 0, s=30, color=COLORS["bifurcation"], edgecolor="black")
        ax.set_xlim(-scale, scale); ax.set_ylim(-scale, scale); ax.set_aspect("equal")
        ax.set_title(f"{model['case']} | plane angle {model['plane_separation_angle_deg']:.1f}°")
        ax.set_xlabel("Canonical X"); ax.set_ylabel("Canonical Z")
    for ax in axes.flat[count:]:
        ax.axis("off")
    fig.suptitle("Clean Stage‑1 Cohort — Fixed Canonical Front View", fontsize=19, weight="bold")
    fig.text(0.5, 0.01, "Same canonical definition, axes, view, and robust global scale; no arbitrary panel rotation.", ha="center")
    writer.save(fig, "representative_population/clean_case_montage", "Six clean cases near the robust population center in one fixed canonical view.")


def figure_qc_flow(writer: FigureWriter, counts: dict[str, int]) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    boxes = {
        "accepted": (0.5, 0.88, f"Accepted Stage‑1 source cases\n{counts['accepted']}", COLORS["coronary_plane"]),
        "resolved": (0.34, 0.66, f"Assignment resolved\n{counts['resolved']}", COLORS["ok"]),
        "unresolved": (0.72, 0.66, f"Unresolved / insufficient\n{counts['unresolved'] + counts['insufficient']}", COLORS["bad"]),
        "supported": (0.18, 0.42, f"Crown supported\n{counts['supported']}", COLORS["ok"]),
        "ambiguous": (0.42, 0.42, f"Crown ambiguous\n{counts['ambiguous']}", COLORS["warn"]),
        "not": (0.66, 0.42, f"Crown not supported\n{counts['not_supported']}", COLORS["bad"]),
        "clean": (0.18, 0.17, f"Clean Stage‑1 anatomical cohort\n{counts['clean']}", COLORS["coronary_plane"]),
    }
    for _, (x, y, text_value, color) in boxes.items():
        ax.text(x, y, text_value, ha="center", va="center", fontsize=14, weight="bold", bbox=dict(boxstyle="round,pad=0.7", fc=color, ec="#333333", alpha=0.25, lw=2))
    arrows = [("accepted", "resolved"), ("accepted", "unresolved"), ("resolved", "supported"), ("resolved", "ambiguous"), ("resolved", "not"), ("supported", "clean")]
    for source, target in arrows:
        x1, y1, *_ = boxes[source]; x2, y2, *_ = boxes[target]
        ax.add_patch(FancyArrowPatch((x1, y1 - 0.055), (x2, y2 + 0.055), arrowstyle="-|>", mutation_scale=18, lw=2, color="#555555"))
    ax.set_title("Stage‑1 Anatomical QC Flow\nCases are quarantined, never geometrically forced", fontsize=21, weight="bold")
    writer.save(fig, "population/stage1_qc_flow", "Accepted-source to clean-cohort QC flow with actual counts.")


def compact_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    excluded = {key for key in analysis if key.startswith("_")}
    result = {key: value for key, value in analysis.items() if key not in excluded}
    if "daughter_metrics" in result:
        result["daughter_metrics"] = {
            name: {key: value for key, value in metrics.items() if not key.startswith("_")}
            for name, metrics in result["daughter_metrics"].items()
        }
    return result


def assignment_row(analysis: dict[str, Any]) -> dict[str, Any]:
    scores = analysis.get("assignment_scoring", {})
    previous = analysis["previous_saved_assignment"]
    resolved = analysis.get("new_resolved_assignment", {})
    return {
        "case": analysis["case"],
        "daughter_A_previous_role": "LAD" if previous.get("lad_source") == "branch_a" else "LCX",
        "daughter_B_previous_role": "LAD" if previous.get("lad_source") == "branch_b" else "LCX",
        "resolved_LAD_source": resolved.get("lad_source"),
        "resolved_LCX_source": resolved.get("lcx_source"),
        "assignment_changed": analysis.get("assignment_changed", False),
        "assignment_score_1": scores.get("assignment_1", {}).get("score"),
        "assignment_score_2": scores.get("assignment_2", {}).get("score"),
        "margin": scores.get("score_margin"),
        "evidence_votes": f"{scores.get('winning_votes', 0)}/{scores.get('evidence_vote_count', 0)}",
        "confidence": scores.get("confidence"),
        "resolution_status": analysis["resolution_status"],
        "reason": analysis["reason"],
    }


def crown_row(analysis: dict[str, Any]) -> dict[str, Any]:
    result = analysis.get("lcx_crown_validation", {})
    interval = result.get("interval", {})
    return {
        "case": analysis["case"],
        "resolved_LCX_source": analysis.get("new_resolved_assignment", {}).get("lcx_source"),
        "status": result.get("status", "not_evaluated"),
        "interval_start_index": interval.get("start_index"),
        "interval_end_index": interval.get("end_index"),
        "interval_fraction": interval.get("arc_fraction"),
        "plane_RMSE_mm": interval.get("plane_rmse_mm"),
        "normalized_plane_RMSE": interval.get("normalized_plane_rmse"),
        "in_plane_path_fraction": interval.get("in_plane_path_fraction"),
        "mean_in_plane_tangent_fraction": interval.get("mean_in_plane_fraction"),
        "angular_sweep_deg": interval.get("net_angular_sweep_deg"),
        "angular_monotonicity": interval.get("angular_monotonic_fraction"),
        "angular_reversal_amount_deg": interval.get("total_reverse_angular_travel_deg"),
        "reason": result.get("reason"),
    }


def build_report(summary: dict[str, Any], selection: dict[str, Any], config: Stage1Config) -> str:
    counts = summary["counts"]
    primary = selection.get("primary_case")
    secondary = selection.get("secondary_cases", [])
    diagnostic = summary.get("case_118", {})
    return f"""# Stage‑1 Final Anatomical Model Report

Generated: {utc_now()}

## Outcome

- Accepted source cases: **{counts['accepted']}**
- Existing assignments retained: **{counts['kept']}**
- Daughter roles resolved by swapping: **{counts['swapped']}**
- Assignment unresolved/manual review: **{counts['unresolved']}**
- Insufficient RCA reference geometry: **{counts['insufficient']}**
- LCX crown supported: **{counts['supported']}**
- LCX crown ambiguous: **{counts['ambiguous']}**
- LCX crown not supported: **{counts['not_supported']}**
- Crown not evaluated because assignment/reference remained unresolved: **{counts['crown_not_evaluated']}**
- Final clean anatomical-model cohort: **{counts['clean']}**
- Primary representative: **{primary}**
- Secondary representatives: **{', '.join(secondary)}**

## `118.label` diagnostic

- Previous saved assignment: **{diagnostic.get('previous_assignment')}**
- New result: **{diagnostic.get('resolution_status')}**
- Assignment 1 / 2 scores: **{diagnostic.get('score_1')} / {diagnostic.get('score_2')}**
- Score margin: **{diagnostic.get('margin')}**
- Winning evidence votes: **{diagnostic.get('votes')}**
- LCX crown result: **{diagnostic.get('crown_status', 'not evaluated because assignment remained unresolved')}**

## Scientific interpretation

The model is dataset-derived. Source centrelines were not anatomically corrected. Cases inconsistent with the intended anatomical abstraction were quarantined rather than geometrically forced to conform. The inferred RCA component is not annotated RCA ground truth. This is a pre-generative anatomical/statistical preparation stage, not a clinical coronary model.

## Branch-role resolution

Saved LAD/LCX arrays were reconstructed as neutral `branch_a` and `branch_b` from their saved provenance. Both assignments were scored with pairwise, within-patient normalized contrasts. Four LAD measurements (inferior reach, terminal inferior displacement, downward dominance, and progression to the inferior point) form one averaged evidence group. Four LCX measurements (RCA-plane residual, in-plane travel, in-plane tangent behaviour, and angular coherence) form a second averaged group. Repeated measurements are averaged within their role group before the two groups are combined. Automatic resolution requires at least {config.assignment_min_votes}/8 independent votes, score margin ≥ {config.assignment_min_margin:.2f}, winner score ≥ {config.assignment_min_winner_score:.2f}, and a minimally coherent proposed LCX. Otherwise the case remains unresolved.

## Independent LCX crown validation

The independent plane is fitted from all unchanged inferred-RCA candidate points only. LCX never contributes to this validation plane. Candidate intervals are continuous source-index ranges beginning within the proximal {config.crown_max_start_fraction:.0%} and spanning at least {config.crown_min_arc_fraction:.0%}. No ellipse, smoothing, isolated-point rejection, or fitted curve participates. Crown classes use explicit multi-signal geometric QC criteria—not cohort medians and not clinical thresholds—including absolute and normalized plane residuals, in-plane path/tangent fractions, angular sweep, monotonicity, and coverage. A near-full branch with poor plane behaviour cannot be promoted to supported.

## Plane fitting mathematics

For relevant unchanged points `P`, `C = mean(P)`, `Q = P - C`, and `U, S, Vᵀ = SVD(Q)`. The plane normal is `n = Vᵀ[-1]`, giving `n · (x - C) = 0`. The final coronary measurement plane uses inferred RCA plus the independently validated continuous LCX crown interval. The LAD plane uses every unchanged resolved-LAD point. Plane separation is `acos(|n_cor · n_lad|)`. Display planes are parallel copies translated to the LMCA bifurcation and are never used for residual measurement.

## Dataset-derived ellipse references

All ellipse landmarks are exact source points selected at normalized arc-length positions. The coronary ellipse uses sparse landmarks from the inferred RCA and validated LCX interval. The LAD affine ellipse uses sparse ordered LAD landmarks and exact endpoints. Solid arcs indicate source-supported parameter intervals; dashed arcs are extrapolated full references. Landmark and full-source descriptive residuals are reported, but ellipse residuals neither validate LCX nor modify vessels.

## Canonical frame

One right-handed rigid frame is constructed per case from the final measured plane normals. Canonical X follows the plane intersection/crown direction, X sign follows resolved LCX progression, and Z is the signed coronary-plane normal with LAD descent toward negative Z. Every branch receives the same `p_canonical = R @ (p_original - B)` transform. No scaling or branch-specific transform is used. Determinants, orthonormality, and segment-length preservation are validated.

## Source-integrity proof

Every source file and array is reloaded after analysis. Point counts, coordinate hashes, coordinates, and original segment lengths must match exactly. Display-transform segment lengths must agree within floating-point tolerance. The final validation records the maxima across the cohort.

## Limitations

- RAS inferior direction and inferred RCA geometry provide engineering evidence, not annotated coronary labels.
- The RCA component is inferred from disconnected source topology and may not represent true RCA anatomy.
- Explicit geometric QC limits are transparent engineering choices, not clinically validated thresholds.
- Ellipses are simplified descriptive references and do not reproduce every noisy centerline point.
- Only clean cases support this two-plane abstraction; excluded cases remain available in QC tables.
- No PCA or generative geometry is used or trained in this increment.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "lca_ssm")
    parser.add_argument("--stage1-root", type=Path, default=None)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--pilot-case", default="118.label")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--reset-output", action="store_true", help="replace only the versioned Stage-1 final output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_plotting()
    config = Stage1Config()
    output_root = args.output_root.resolve()
    stage1_root = (args.stage1_root or output_root / "stage1_final_anatomical_model").resolve()
    if args.reset_output and stage1_root.exists():
        shutil.rmtree(stage1_root)
    stage1_root.mkdir(parents=True, exist_ok=True)

    accepted_index = json.loads((output_root / "quality_control" / "accepted_patient_index.json").read_text(encoding="utf-8"))
    accepted = list(accepted_index["patient_ids"])

    if args.pilot_only:
        analysis = analyze_case(output_root, args.pilot_case, config)
        analysis["source_integrity"] = verify_source_integrity(analysis, output_root)
        if analysis["resolution_status"] in {"resolved_existing_assignment", "resolved_swapped_assignment"}:
            analysis["lcx_crown_validation"] = classify_crown(analysis, config)
        pilot_path = stage1_root / "branch_resolution" / f"pilot_{args.pilot_case}.json"
        write_json(pilot_path, compact_analysis(analysis))
        print(json.dumps(jsonable({
            "case": args.pilot_case,
            "resolution_status": analysis["resolution_status"],
            "assignment": analysis.get("new_resolved_assignment"),
            "scores": analysis.get("assignment_scoring"),
            "crown": analysis.get("lcx_crown_validation"),
            "source_integrity": analysis["source_integrity"],
            "output": str(pilot_path),
        }), indent=2))
        return

    analyses: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for index, patient_id in enumerate(accepted, start=1):
        try:
            analysis = analyze_case(output_root, patient_id, config)
            analysis["source_integrity"] = verify_source_integrity(analysis, output_root)
            analysis["lcx_crown_validation"] = classify_crown(analysis, config)
            analyses[patient_id] = analysis
        except Exception as exc:
            failures[patient_id] = f"{type(exc).__name__}: {exc}"
        if index % 20 == 0 or index == len(accepted):
            print(f"analysed {index}/{len(accepted)} accepted cases; failures={len(failures)}", flush=True)

    clean_ids = [
        case for case, analysis in analyses.items()
        if analysis["resolution_status"] in {"resolved_existing_assignment", "resolved_swapped_assignment"}
        and analysis["lcx_crown_validation"].get("status") == "crown_supported"
        and analysis["source_integrity"]["pass"]
        and analysis["assignment_scoring"]["winning_score"] > 0.0
    ]

    models: list[dict[str, Any]] = []
    model_failures: dict[str, str] = {}
    for patient_id in clean_ids:
        try:
            models.append(build_clean_model(analyses[patient_id], config))
        except Exception as exc:
            model_failures[patient_id] = f"{type(exc).__name__}: {exc}"
    # A clean anatomical model requires both dataset-derived ellipse references.
    clean_ids = [model["case"] for model in models]

    ranking = rank_models(models, analyses)
    if not ranking:
        raise SystemExit("no case satisfied the conservative clean Stage-1 model and ellipse requirements")
    model_by_case = {model["case"]: model for model in models}
    primary_id = ranking[0]["case"]
    primary = model_by_case[primary_id]
    # Montage: prefer high-ranked cases, then spread around median plane angle.
    candidate_ids = [item["case"] for item in ranking[: min(20, len(ranking))]]
    median_angle = float(np.median([model_by_case[case]["plane_separation_angle_deg"] for case in candidate_ids]))
    secondary_ids = sorted(candidate_ids, key=lambda case: abs(model_by_case[case]["plane_separation_angle_deg"] - median_angle))
    secondary_ids = [case for case in secondary_ids if case != primary_id][:8]
    montage_ids = [primary_id] + secondary_ids[:8]

    assignment_rows = [assignment_row(analyses[case]) for case in accepted if case in analyses]
    crown_rows = [crown_row(analyses[case]) for case in accepted if case in analyses]
    measurement_rows = [model_measurement_row(model, analyses[model["case"]]) for model in models]
    unresolved_rows = [row for row in assignment_rows if row["resolution_status"] in {"unresolved_manual_review", "insufficient_reference_geometry"}]
    excluded_rows = []
    for case in accepted:
        if case in clean_ids:
            continue
        if case in failures:
            excluded_rows.append({"case": case, "group": "analysis_failure", "reason": failures[case]})
            continue
        analysis = analyses[case]
        if case in model_failures:
            group, reason = "model_reference_failure", model_failures[case]
        elif analysis["resolution_status"] == "unresolved_manual_review":
            group, reason = "assignment_unresolved_cases", analysis["reason"]
        elif analysis["resolution_status"] == "insufficient_reference_geometry":
            group, reason = "reference_plane_unavailable_cases", analysis["reason"]
        else:
            crown_status = analysis["lcx_crown_validation"]["status"]
            group = "lcx_crown_ambiguous_cases" if crown_status == "crown_ambiguous" else "lcx_crown_not_supported_cases"
            reason = analysis["lcx_crown_validation"]["reason"]
        excluded_rows.append({"case": case, "group": group, "reason": reason})

    counts = {
        "accepted": len(accepted),
        "kept": sum(row["resolution_status"] == "resolved_existing_assignment" for row in assignment_rows),
        "swapped": sum(row["resolution_status"] == "resolved_swapped_assignment" for row in assignment_rows),
        "resolved": sum(row["resolution_status"] in {"resolved_existing_assignment", "resolved_swapped_assignment"} for row in assignment_rows),
        "unresolved": sum(row["resolution_status"] == "unresolved_manual_review" for row in assignment_rows),
        "insufficient": sum(row["resolution_status"] == "insufficient_reference_geometry" for row in assignment_rows),
        "supported": sum(row["status"] == "crown_supported" for row in crown_rows),
        "ambiguous": sum(row["status"] == "crown_ambiguous" for row in crown_rows),
        "not_supported": sum(row["status"] == "crown_not_supported" for row in crown_rows),
        "crown_not_evaluated": sum(row["status"].startswith("not_evaluated") for row in crown_rows),
        "clean": len(clean_ids),
    }

    # Persist authoritative derived maps and per-case validation without touching sources.
    write_json(stage1_root / "branch_resolution" / "resolved_branch_assignments.json", {
        "generated_at": utc_now(),
        "authoritative_scope": "derived Stage-1 branch roles only; raw extraction remains unchanged",
        "assignment_rule": analyses[next(iter(analyses))].get("assignment_scoring", {}).get("rule"),
        "cases": {case: compact_analysis(analysis) for case, analysis in analyses.items()},
        "failures": failures,
    })
    write_csv(stage1_root / "branch_resolution" / "resolved_branch_assignments.csv", assignment_rows)
    write_csv(stage1_root / "branch_resolution" / "unresolved_cases.csv", unresolved_rows)
    write_csv(stage1_root / "lcx_validation" / "lcx_crown_validation_final.csv", crown_rows)
    for case, analysis in analyses.items():
        write_json(stage1_root / "lcx_validation" / "per_case" / f"{case}.json", compact_analysis(analysis))
    write_csv(stage1_root / "clean_cohort" / "clean_stage1_cases.csv", [{"case": case} for case in clean_ids], ["case"])
    write_csv(stage1_root / "clean_cohort" / "excluded_stage1_cases.csv", excluded_rows)
    write_csv(stage1_root / "technical_tables" / "two_plane_measurements_final.csv", measurement_rows)

    integrity = {
        "all_analysed_source_integrity_pass": bool(all(analysis["source_integrity"]["pass"] for analysis in analyses.values())),
        "all_source_file_hashes_unchanged": bool(all(analysis["source_integrity"]["source_file_sha256_unchanged"] for analysis in analyses.values())),
        "maximum_coordinate_difference_mm": max(analysis["source_integrity"]["maximum_coordinate_difference_mm"] for analysis in analyses.values()),
        "maximum_source_segment_length_difference_mm": max(analysis["source_integrity"]["maximum_source_segment_length_difference_mm"] for analysis in analyses.values()),
        "maximum_rigid_display_segment_length_error_mm": max((model["maximum_rigid_segment_length_error_mm"] for model in models), default=0.0),
        "rigid_tolerance_mm": config.rigid_tolerance_mm,
        "source_geometry_modified": False,
        "per_case": {case: analysis["source_integrity"] for case, analysis in analyses.items()},
    }
    write_json(stage1_root / "technical_tables" / "source_integrity_validation.json", integrity)

    selection = {
        "primary_case": primary_id,
        "secondary_cases": secondary_ids,
        "montage_cases": montage_ids,
        "ranking": ranking,
        "logic": "weighted robust-z ranking of assignment confidence, explicit crown evidence, plane residuals, crown coverage, LAD apical reach, and typical plane angle; failures/warnings/integrity failures excluded",
    }
    write_json(stage1_root / "representative_population" / "selected_clean_cases.json", selection)
    write_json(stage1_root / "representative_case" / "primary_case_model.json", {
        "case": primary_id,
        "assignment": primary["assignment"],
        "coronary_plane": primary["coronary_plane_record"],
        "lad_plane": primary["lad_plane_record"],
        "plane_separation_angle_deg": primary["plane_separation_angle_deg"],
        "coronary_ellipse": compact_ellipse(primary["coronary_ellipse"], primary["landmark_source_indices"]["rca"].tolist() + primary["landmark_source_indices"]["lcx"].tolist(), np.vstack((primary["landmark_ras"]["rca"], primary["landmark_ras"]["lcx"]))),
        "lad_ellipse": compact_ellipse(primary["lad_ellipse"], primary["landmark_source_indices"]["lad"], primary["landmark_ras"]["lad"]),
        "frame": primary["frame"],
        "source_geometry_modified": False,
    })

    writer = FigureWriter(stage1_root, args.dpi)
    figure_main(writer, primary)
    figure_schematic(writer, primary)
    figure_overlay(writer, primary)
    figure_assignment(writer, primary, analyses[primary_id])
    figure_svd(writer, primary, analyses[primary_id])
    plot_orthographic(writer, primary, FIGURES[5], "Canonical Front View", 0, 2, ("Canonical X: crown", "Canonical Z: apex descent"))
    plot_orthographic(writer, primary, FIGURES[6], "Canonical Superior View", 0, 1, ("Canonical X: crown", "Canonical Y"))
    plot_orthographic(writer, primary, FIGURES[7], "Canonical LAD-Plane / Lateral View", 0, 2, ("Canonical X", "Canonical Z: apex descent"))
    figure_montage(writer, [model_by_case[case] for case in montage_ids])
    figure_qc_flow(writer, counts)
    write_json(stage1_root / "image_manifest.json", {"images": writer.manifest, "png_dpi": args.dpi, "matching_svg_generated": True})

    diagnostic_118 = analyses.get("118.label", {})
    diagnostic_scores = diagnostic_118.get("assignment_scoring", {})
    summary = {
        "counts": counts,
        "failures": failures,
        "model_reference_failures": model_failures,
        "case_118": {
            "previous_assignment": (
                f"LAD={diagnostic_118.get('previous_saved_assignment', {}).get('lad_source')}, "
                f"LCX={diagnostic_118.get('previous_saved_assignment', {}).get('lcx_source')}"
            ),
            "resolution_status": diagnostic_118.get("resolution_status"),
            "score_1": diagnostic_scores.get("assignment_1", {}).get("score"),
            "score_2": diagnostic_scores.get("assignment_2", {}).get("score"),
            "margin": diagnostic_scores.get("score_margin"),
            "votes": f"{diagnostic_scores.get('winning_votes')}/{diagnostic_scores.get('evidence_vote_count')}",
            "crown_status": diagnostic_118.get("lcx_crown_validation", {}).get("status"),
        },
    }
    report = build_report(summary, selection, config)
    (stage1_root / "STAGE1_FINAL_ANATOMICAL_MODEL_REPORT.md").write_text(report, encoding="utf-8")

    required_files = [
        "branch_resolution/resolved_branch_assignments.json",
        "branch_resolution/resolved_branch_assignments.csv",
        "branch_resolution/unresolved_cases.csv",
        "lcx_validation/lcx_crown_validation_final.csv",
        "clean_cohort/clean_stage1_cases.csv",
        "clean_cohort/excluded_stage1_cases.csv",
        "technical_tables/two_plane_measurements_final.csv",
        "technical_tables/source_integrity_validation.json",
        "representative_population/selected_clean_cases.json",
        "representative_case/primary_case_model.json",
        "population/stage1_qc_flow.png",
        "STAGE1_FINAL_ANATOMICAL_MODEL_REPORT.md",
        "image_manifest.json",
    ]
    required_files += [f"representative_case/{stem}.{suffix}" for stem in FIGURES for suffix in ("png", "svg")]
    required_files += [f"representative_population/clean_case_montage.{suffix}" for suffix in ("png", "svg")]
    required_files += [f"population/stage1_qc_flow.{suffix}" for suffix in ("png", "svg")]
    missing = [path for path in required_files if not (stage1_root / path).is_file()]
    validation = {
        "generated_at": utc_now(),
        "pass": bool(
            not missing
            and not failures
            and integrity["all_analysed_source_integrity_pass"]
            and integrity["maximum_coordinate_difference_mm"] == 0.0
            and integrity["maximum_source_segment_length_difference_mm"] == 0.0
            and integrity["maximum_rigid_display_segment_length_error_mm"] <= config.rigid_tolerance_mm
            and counts["accepted"] == len(assignment_rows)
            and counts["resolved"] + counts["unresolved"] + counts["insufficient"] == counts["accepted"]
            and counts["clean"] > 0
        ),
        "counts": counts,
        "primary_case": primary_id,
        "secondary_cases": secondary_ids,
        "source_integrity": {key: value for key, value in integrity.items() if key != "per_case"},
        "missing_required_files": missing,
        "analysis_failures": failures,
        "model_reference_failures": model_failures,
        "method_guards": {
            "PCA_or_generated_geometry_used": False,
            "source_geometry_modified": False,
            "cohort_median_used_for_assignment": False,
            "cohort_median_used_for_crown_classification": False,
            "ellipse_used_for_LCX_validation": False,
            "LCX_used_for_independent_validation_plane": False,
            "measurement_planes_use_centroid_SVD": True,
            "one_global_rigid_transform_per_case": True,
        },
    }
    write_json(stage1_root / "validation.json", validation)
    print(json.dumps(jsonable({"stage1_root": str(stage1_root), **validation}), indent=2))
    if not validation["pass"]:
        raise SystemExit("Stage-1 final validation failed; inspect validation.json")


if __name__ == "__main__":
    main()
