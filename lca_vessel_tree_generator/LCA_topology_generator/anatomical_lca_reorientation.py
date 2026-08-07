import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bspline import interpolate_lca_tree
from .heart_landmark_model import (
    evaluate_heart_guided_lca,
    generate_heart_guided_lca_tree,
    lad_interventricular_groove_guide,
    lcx_atrioventricular_groove_guide,
    lmca_short_trunk_guide,
    serialize_landmarks,
)
from .lca_validation import validate_lca_tree
from .paths import lca_generated_control_points_dir, output_path
from .tortuosity import calculate_lca_tortuosity, calculate_path_length, calculate_tortuosity
from .visualize import set_axes_equal


BRANCH_SLICES = {
    "LMCA": slice(0, 5),
    "LAD": slice(5, 17),
    "LCX": slice(17, 27),
}

BRANCH_COLORS = {
    "LMCA": "black",
    "LAD": "red",
    "LCX": "blue",
}

BRANCH_IDS = {
    "LMCA": 1,
    "LAD": 2,
    "LCX": 3,
}


ANATOMICAL_VALIDATION_THRESHOLDS = {
    "lad_downward_score_min": 0.78,
    "lad_upward_step_fraction_max": 0.05,
    "lcx_lateral_score_min": 0.74,
    "lcx_vertical_score_max": 0.35,
    "lmca_to_child_length_ratio_max": 0.45,
    "min_lad_lcx_separation_mm": 4.0,
    "min_lad_lcx_angle_deg": 45.0,
    "lad_tortuosity_range": [1.0, 1.30],
    "lcx_tortuosity_range": [1.02, 1.55],
    "branch_length_preservation_ratio_range": [0.90, 1.10],
    "lad_guide_distance_mean_ratio_max": 0.035,
    "lcx_guide_distance_mean_ratio_max": 0.045,
    "lad_apex_axis_alignment_min": 0.88,
    "lcx_coronary_plane_alignment_min": 0.94,
    "lcx_coronary_plane_offset_mean_ratio_max": 0.08,
}


def _write_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)


def _branch_lengths(control_tree: np.ndarray) -> dict:
    return {
        branch_name: float(calculate_path_length(control_tree[branch_slice]))
        for branch_name, branch_slice in BRANCH_SLICES.items()
    }


def _unique_point_count(points: np.ndarray, decimals: int = 6) -> int:
    rounded = np.round(np.asarray(points, dtype=float), decimals=decimals)
    return int(len(np.unique(rounded, axis=0)))


def _pairwise_min_distance(points_a: np.ndarray, points_b: np.ndarray) -> float:
    if len(points_a) == 0 or len(points_b) == 0:
        return 0.0
    distances = np.linalg.norm(points_a[:, None, :] - points_b[None, :, :], axis=2)
    return float(np.min(distances))


def _score_between(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return float(value >= upper)
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


def _score_below(value: float, good: float, bad: float) -> float:
    if bad <= good:
        return float(value <= good)
    return float(np.clip((bad - value) / (bad - good), 0.0, 1.0))


def _anatomical_lengths(input_lengths: dict) -> tuple:
    lengths = dict(input_lengths)
    fallback_used = {}
    child_lengths = [
        length for length in [lengths["LAD"], lengths["LCX"]]
        if length > 1e-9
    ]
    child_scale = min(child_lengths) if child_lengths else 50.0

    if lengths["LMCA"] <= 1e-9:
        lengths["LMCA"] = float(np.clip(0.16 * child_scale, 6.0, 22.0))
        fallback_used["LMCA"] = "input LMCA was zero-length; used short trunk length from child branch scale"
    else:
        max_lmca_length = float(np.clip(0.36 * child_scale, 6.0, 28.0))
        if lengths["LMCA"] > max_lmca_length:
            lengths["LMCA"] = max_lmca_length
            fallback_used["LMCA"] = (
                "input LMCA was longer than anatomical short-trunk rule; "
                "capped relative to child branch scale"
            )

    return lengths, fallback_used


def _rescale_path_to_length(points: np.ndarray, target_length: float, anchor_index: int = 0) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    current_length = calculate_path_length(points)
    if current_length <= 1e-9:
        return points.copy()
    anchor = points[anchor_index].copy()
    return anchor + (points - anchor) * (target_length / current_length)


def _lmca_template(length_mm: float, num_points: int = 5) -> np.ndarray:
    return lmca_short_trunk_guide(length_mm, num_points)


def _lad_template(bifurcation: np.ndarray, length_mm: float, num_points: int = 12) -> np.ndarray:
    return lad_interventricular_groove_guide(bifurcation, length_mm, num_points)


def _lcx_template(bifurcation: np.ndarray, length_mm: float, num_points: int = 10) -> np.ndarray:
    return lcx_atrioventricular_groove_guide(bifurcation, length_mm, num_points)


def reorient_lca_control_tree(control_tree: np.ndarray) -> tuple:
    """
    Converts one 27x3 LCA control tree into a rule-based anatomical layout.

    The output preserves the strict topology:
    LMCA 0:5, LAD 5:17, LCX 17:27, with LAD/LCX starts duplicated from
    the LMCA bifurcation point.
    """
    control_tree = np.asarray(control_tree, dtype=float)
    if control_tree.shape != (27, 3):
        raise ValueError("Expected LCA control tree shape (27, 3).")

    input_lengths = _branch_lengths(control_tree)
    lengths, fallback_used = _anatomical_lengths(input_lengths)
    anatomical, landmarks = generate_heart_guided_lca_tree(lengths)
    metadata = {
        "input_branch_lengths_mm": input_lengths,
        "anatomical_branch_lengths_mm": lengths,
        "fallback_lengths": fallback_used,
        "output_branch_lengths_mm": _branch_lengths(anatomical),
        "heart_landmark_model": serialize_landmarks(landmarks),
        "rules": {
            "LMCA": "ostium to short trunk to bifurcation",
            "LAD": "bifurcation to anterior interventricular groove toward apex",
            "LCX": "bifurcation to left atrioventricular/coronary groove crown arc",
        },
        "axis_convention": {
            "X": "LMCA/local anterior-left template axis",
            "Y": "lateral circumflex/crown direction in coronary plane",
            "Z": "base-to-apex axis; negative Z is apex direction",
        },
    }
    return anatomical, metadata


def validate_anatomical_lca_tree(control_tree: np.ndarray, reference_lengths: dict = None) -> dict:
    control_tree = np.asarray(control_tree, dtype=float)
    errors = []
    warnings = []

    if control_tree.shape != (27, 3):
        errors.append(f"Expected shape (27, 3), got {list(control_tree.shape)}")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
        }

    if not np.all(np.isfinite(control_tree)):
        errors.append("Control tree contains NaN or Inf values")

    lmca = control_tree[BRANCH_SLICES["LMCA"]]
    lad = control_tree[BRANCH_SLICES["LAD"]]
    lcx = control_tree[BRANCH_SLICES["LCX"]]
    lengths = _branch_lengths(control_tree)
    thresholds = ANATOMICAL_VALIDATION_THRESHOLDS

    for branch_name, length in lengths.items():
        if length <= 1e-9:
            errors.append(f"{branch_name} has zero-length control path")

    unique_points = _unique_point_count(control_tree)
    expected_unique_points = 25
    # LAD[0] and LCX[0] intentionally duplicate LMCA[-1] at the bifurcation.
    no_unexpected_duplicates = unique_points >= expected_unique_points
    if not no_unexpected_duplicates:
        errors.append(
            f"Control tree has {unique_points} unique points; expected at least {expected_unique_points}"
        )

    lad_start_gap = float(np.linalg.norm(lad[0] - lmca[-1]))
    lcx_start_gap = float(np.linalg.norm(lcx[0] - lmca[-1]))
    if lad_start_gap > 1e-6:
        errors.append(f"LAD start does not match LMCA bifurcation: gap {lad_start_gap:.6f} mm")
    if lcx_start_gap > 1e-6:
        errors.append(f"LCX start does not match LMCA bifurcation: gap {lcx_start_gap:.6f} mm")

    lad_vector = lad[-1] - lad[0]
    lcx_vector = lcx[-1] - lcx[0]
    lad_abs = np.abs(lad_vector)
    lcx_abs = np.abs(lcx_vector)
    lad_displacement_norm = max(float(np.linalg.norm(lad_vector)), 1e-9)
    lcx_displacement_norm = max(float(np.linalg.norm(lcx_vector)), 1e-9)
    lad_downward_score = float(max(-lad_vector[2], 0.0) / lad_displacement_norm)
    lcx_lateral_score = float(np.linalg.norm(lcx_vector[:2]) / lcx_displacement_norm)
    lcx_vertical_score = float(abs(lcx_vector[2]) / lcx_displacement_norm)
    lmca_child_ratio = float(lengths["LMCA"] / max(min(lengths["LAD"], lengths["LCX"]), 1e-9))

    lad_z_steps = np.diff(lad[:, 2])
    lad_upward_step_fraction = float(np.count_nonzero(lad_z_steps > 1e-6) / max(len(lad_z_steps), 1))
    lcx_total_vertical_range_mm = float(np.max(lcx[:, 2]) - np.min(lcx[:, 2]))
    lcx_vertical_range_ratio = float(lcx_total_vertical_range_mm / max(lengths["LCX"], 1e-9))
    min_lad_lcx_separation_mm = _pairwise_min_distance(lad[1:], lcx[1:])
    heart_guided_metrics = evaluate_heart_guided_lca(control_tree)
    lad_tortuosity = calculate_tortuosity(lad)["tortuosity"]
    lcx_tortuosity = calculate_tortuosity(lcx)["tortuosity"]
    lad_tortuosity_min, lad_tortuosity_max = thresholds["lad_tortuosity_range"]
    lcx_tortuosity_min, lcx_tortuosity_max = thresholds["lcx_tortuosity_range"]
    lad_tortuosity_ok = bool(
        lad_tortuosity is not None
        and lad_tortuosity_min <= lad_tortuosity <= lad_tortuosity_max
    )
    lcx_tortuosity_ok = bool(
        lcx_tortuosity is not None
        and lcx_tortuosity_min <= lcx_tortuosity <= lcx_tortuosity_max
    )

    reference_lengths = reference_lengths or lengths
    preservation_min, preservation_max = thresholds["branch_length_preservation_ratio_range"]
    branch_length_preservation_ratios = {}
    branch_length_preservation_ok = True
    for branch_name in ["LMCA", "LAD", "LCX"]:
        ratio = lengths[branch_name] / max(float(reference_lengths.get(branch_name, lengths[branch_name])), 1e-9)
        branch_length_preservation_ratios[branch_name] = float(ratio)
        branch_length_preservation_ok = branch_length_preservation_ok and preservation_min <= ratio <= preservation_max

    lad_dominant_downward = bool(
        lad_vector[2] < 0
        and lad_abs[2] > lad_abs[0]
        and lad_abs[2] > lad_abs[1]
        and lad_downward_score >= thresholds["lad_downward_score_min"]
        and lad_upward_step_fraction <= thresholds["lad_upward_step_fraction_max"]
    )
    lcx_dominant_lateral = bool(
        lcx_lateral_score >= thresholds["lcx_lateral_score_min"]
        and lcx_abs[1] > lcx_abs[2]
        and lcx_abs[1] >= 0.65 * lcx_abs[0]
    )
    lcx_limited_descent = bool(
        lcx_vertical_score <= thresholds["lcx_vertical_score_max"]
        and lcx_vertical_range_ratio <= 0.35
    )
    lcx_crown_plane_score = float(_score_below(lcx_vertical_range_ratio, 0.18, 0.35))
    lmca_short_trunk = bool(
        lengths["LMCA"] < lengths["LAD"]
        and lengths["LMCA"] < lengths["LCX"]
        and lmca_child_ratio <= thresholds["lmca_to_child_length_ratio_max"]
    )
    lad_lcx_separation_ok = bool(min_lad_lcx_separation_mm >= thresholds["min_lad_lcx_separation_mm"])
    lad_not_circumflex = bool(lad_downward_score > lcx_vertical_score + 0.55)
    lad_follows_apex_groove = bool(
        heart_guided_metrics["lad_guide_distance_mean_ratio"]
        <= thresholds["lad_guide_distance_mean_ratio_max"]
        and heart_guided_metrics["lad_apex_axis_alignment"]
        >= thresholds["lad_apex_axis_alignment_min"]
    )
    lcx_follows_coronary_groove = bool(
        heart_guided_metrics["lcx_guide_distance_mean_ratio"]
        <= thresholds["lcx_guide_distance_mean_ratio_max"]
        and heart_guided_metrics["lcx_coronary_plane_alignment"]
        >= thresholds["lcx_coronary_plane_alignment_min"]
        and heart_guided_metrics["lcx_coronary_plane_offset_mean_ratio"]
        <= thresholds["lcx_coronary_plane_offset_mean_ratio_max"]
    )

    if not lad_dominant_downward:
        errors.append("LAD does not have dominant downward negative-Z direction")
    if not lcx_dominant_lateral:
        errors.append("LCX does not have dominant lateral/circumflex direction")
    if not lcx_limited_descent:
        errors.append("LCX descends too much for a crown/circumflex branch")
    if not lad_tortuosity_ok:
        errors.append("LAD tortuosity is outside anatomical MVP range")
    if not lcx_tortuosity_ok:
        errors.append("LCX tortuosity is outside anatomical MVP range")
    if not branch_length_preservation_ok:
        errors.append("Branch length preservation ratio is outside configured range")
    if not lmca_short_trunk:
        errors.append("LMCA is not a short trunk relative to LAD and LCX")
    if not lad_lcx_separation_ok:
        errors.append(
            f"LAD/LCX non-bifurcation separation is too small: {min_lad_lcx_separation_mm:.3f} mm"
        )
    if not lad_follows_apex_groove:
        errors.append("LAD does not follow the heart-guided apex/interventricular groove")
    if not lcx_follows_coronary_groove:
        errors.append("LCX does not follow the heart-guided coronary/AV groove")

    lad_lcx_angle = None
    lad_lcx_separation_angle_ok = False
    lad_norm = np.linalg.norm(lad_vector)
    lcx_norm = np.linalg.norm(lcx_vector)
    if lad_norm > 1e-9 and lcx_norm > 1e-9:
        cosine = np.clip(np.dot(lad_vector, lcx_vector) / (lad_norm * lcx_norm), -1.0, 1.0)
        lad_lcx_angle = float(math.degrees(math.acos(cosine)))
        lad_lcx_separation_angle_ok = bool(lad_lcx_angle >= thresholds["min_lad_lcx_angle_deg"])
        if not lad_lcx_separation_angle_ok:
            warnings.append(f"LAD/LCX separation angle is small: {lad_lcx_angle:.2f} degrees")

    branch_topology_ok = bool(lad_start_gap <= 1e-6 and lcx_start_gap <= 1e-6)
    score_terms = {
        "lad_downward_score": _score_between(lad_downward_score, 0.55, thresholds["lad_downward_score_min"]),
        "lad_no_upward_loop_score": _score_below(
            lad_upward_step_fraction,
            thresholds["lad_upward_step_fraction_max"],
            0.25,
        ),
        "lcx_lateral_score": _score_between(lcx_lateral_score, 0.55, thresholds["lcx_lateral_score_min"]),
        "lcx_limited_vertical_score": _score_below(
            lcx_vertical_score,
            thresholds["lcx_vertical_score_max"],
            0.65,
        ),
        "lcx_crown_plane_score": lcx_crown_plane_score,
        "lad_tortuosity_score": float(lad_tortuosity_ok),
        "lcx_tortuosity_score": float(lcx_tortuosity_ok),
        "branch_length_preservation_score": float(branch_length_preservation_ok),
        "lad_apex_groove_score": _score_below(
            heart_guided_metrics["lad_guide_distance_mean_ratio"],
            thresholds["lad_guide_distance_mean_ratio_max"],
            thresholds["lad_guide_distance_mean_ratio_max"] * 3.0,
        ),
        "lad_apex_axis_alignment_score": _score_between(
            heart_guided_metrics["lad_apex_axis_alignment"],
            0.72,
            thresholds["lad_apex_axis_alignment_min"],
        ),
        "lcx_coronary_groove_score": _score_below(
            heart_guided_metrics["lcx_guide_distance_mean_ratio"],
            thresholds["lcx_guide_distance_mean_ratio_max"],
            thresholds["lcx_guide_distance_mean_ratio_max"] * 3.0,
        ),
        "lcx_coronary_plane_alignment_score": _score_between(
            heart_guided_metrics["lcx_coronary_plane_alignment"],
            0.78,
            thresholds["lcx_coronary_plane_alignment_min"],
        ),
        "lcx_coronary_plane_offset_score": _score_below(
            heart_guided_metrics["lcx_coronary_plane_offset_mean_ratio"],
            thresholds["lcx_coronary_plane_offset_mean_ratio_max"],
            thresholds["lcx_coronary_plane_offset_mean_ratio_max"] * 2.5,
        ),
        "lmca_short_trunk_score": _score_below(
            lmca_child_ratio,
            thresholds["lmca_to_child_length_ratio_max"],
            0.85,
        ),
        "bifurcation_consistency_score": float(branch_topology_ok),
        "lad_lcx_separation_score": _score_between(
            min_lad_lcx_separation_mm,
            0.0,
            thresholds["min_lad_lcx_separation_mm"],
        ),
        "no_unexpected_duplicate_score": float(no_unexpected_duplicates),
    }
    anatomical_score = float(np.mean(list(score_terms.values())))

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "validation_thresholds": thresholds,
        "shape": list(control_tree.shape),
        "is_finite": bool(np.all(np.isfinite(control_tree))),
        "unique_points": unique_points,
        "expected_unique_points": expected_unique_points,
        "branch_lengths_mm": lengths,
        "branch_length_ratios": {
            "LMCA_to_min_child": lmca_child_ratio,
            "LAD_to_LCX": float(lengths["LAD"] / max(lengths["LCX"], 1e-9)),
        },
        "branch_length_preservation_ratios": branch_length_preservation_ratios,
        "bifurcation_gaps_mm": {
            "LAD_to_LMCA": lad_start_gap,
            "LCX_to_LMCA": lcx_start_gap,
        },
        "direction_vectors": {
            "LAD": lad_vector.tolist(),
            "LCX": lcx_vector.tolist(),
        },
        "anatomical_metrics": {
            "lad_downward_score": lad_downward_score,
            "lad_upward_step_fraction": lad_upward_step_fraction,
            "lcx_lateral_score": lcx_lateral_score,
            "lcx_vertical_score": lcx_vertical_score,
            "lcx_vertical_range_mm": lcx_total_vertical_range_mm,
            "lcx_vertical_range_ratio": lcx_vertical_range_ratio,
            "lcx_crown_plane_score": lcx_crown_plane_score,
            "lad_tortuosity": lad_tortuosity,
            "lcx_tortuosity": lcx_tortuosity,
            "heart_guided": heart_guided_metrics,
            "min_lad_lcx_separation_mm": min_lad_lcx_separation_mm,
            "anatomical_score": anatomical_score,
            "score_terms": score_terms,
        },
        "anatomical_checks": {
            "lad_dominant_downward": lad_dominant_downward,
            "lad_distal_lower_than_start": bool(lad[-1, 2] < lad[0, 2]),
            "lad_no_upward_loop": bool(lad_upward_step_fraction <= thresholds["lad_upward_step_fraction_max"]),
            "lad_not_circumflex": lad_not_circumflex,
            "lad_curvature_tortuosity_ok": lad_tortuosity_ok,
            "lad_follows_apex_groove": lad_follows_apex_groove,
            "lcx_dominant_lateral": lcx_dominant_lateral,
            "lcx_limited_descent": lcx_limited_descent,
            "lcx_crown_plane_behavior": bool(lcx_crown_plane_score >= 0.99),
            "lcx_curvature_tortuosity_ok": lcx_tortuosity_ok,
            "lcx_follows_coronary_groove": lcx_follows_coronary_groove,
            "lcx_not_second_lad": bool(lcx_lateral_score > lad_downward_score * 0.65),
            "lmca_shorter_than_lad_lcx": lmca_short_trunk,
            "branch_topology_lmca_to_lad_lcx": branch_topology_ok,
            "lad_lcx_separation_ok": lad_lcx_separation_ok,
            "lad_lcx_separation_angle_ok": lad_lcx_separation_angle_ok,
            "no_unexpected_duplicate_points": no_unexpected_duplicates,
            "branch_length_preservation_ok": branch_length_preservation_ok,
        },
        "lad_lcx_angle_deg": lad_lcx_angle,
    }


def _plot_tree(ax, control_tree: np.ndarray, title: str, linestyle: str = "-", alpha: float = 0.9):
    for branch_name, branch_slice in BRANCH_SLICES.items():
        points = control_tree[branch_slice]
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=BRANCH_COLORS[branch_name],
            linewidth=2.0,
            linestyle=linestyle,
            alpha=alpha,
            label=branch_name,
        )
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=BRANCH_COLORS[branch_name],
            s=22,
            alpha=alpha,
            depthshade=False,
        )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y lateral")
    ax.set_zlabel("Z vertical")
    set_axes_equal(ax)


def save_anatomical_tree_plot(path: Path, control_tree: np.ndarray, title: str):
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(projection="3d")
    _plot_tree(ax, control_tree, title)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_before_after_plot(path: Path, original: np.ndarray, anatomical: np.ndarray, patient_id: str):
    fig = plt.figure(figsize=(12.0, 5.4))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_tree(ax1, original, f"{patient_id} original", alpha=0.72)
    _plot_tree(ax2, anatomical, f"{patient_id} anatomical", alpha=0.92)
    for ax in [ax1, ax2]:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_centerline_preview(path: Path, control_tree: np.ndarray, patient_id: str):
    centerlines = interpolate_lca_tree(control_tree)
    fig = plt.figure(figsize=(7.2, 5.8))
    ax = fig.add_subplot(projection="3d")
    for branch_name in ["LMCA", "LAD", "LCX"]:
        points = centerlines[branch_name]
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=BRANCH_COLORS[branch_name],
            linewidth=2.4,
            label=f"{branch_name} spline",
        )
    ax.set_title(f"{patient_id} anatomical B-spline centerlines")
    ax.set_xlabel("X")
    ax.set_ylabel("Y lateral")
    ax.set_zlabel("Z vertical")
    set_axes_equal(ax)
    ax.legend(fontsize=8, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_branch_polyline_vtk(path: Path, branches: dict, title: str):
    """
    Writes a dependency-free legacy VTK PolyData file for interactive viewing.

    The file contains branch polylines plus point data:
    branch_id: LMCA=1, LAD=2, LCX=3
    point_index: index within each branch
    """
    ordered_branch_names = ["LMCA", "LAD", "LCX"]
    points = []
    lines = []
    branch_ids = []
    point_indices = []

    for branch_name in ordered_branch_names:
        branch_points = np.asarray(branches[branch_name], dtype=float)
        start_index = len(points)
        points.extend(branch_points.tolist())
        lines.append([start_index + idx for idx in range(len(branch_points))])
        branch_ids.extend([BRANCH_IDS[branch_name]] * len(branch_points))
        point_indices.extend(list(range(len(branch_points))))

    with open(path, "w", encoding="utf-8") as vtk_file:
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write(f"{title}\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET POLYDATA\n")
        vtk_file.write(f"POINTS {len(points)} float\n")
        for point in points:
            vtk_file.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f}\n")

        line_size = sum(len(line) + 1 for line in lines)
        vtk_file.write(f"LINES {len(lines)} {line_size}\n")
        for line in lines:
            vtk_file.write(f"{len(line)} {' '.join(str(idx) for idx in line)}\n")

        vtk_file.write(f"POINT_DATA {len(points)}\n")
        vtk_file.write("SCALARS branch_id int 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for branch_id in branch_ids:
            vtk_file.write(f"{branch_id}\n")

        vtk_file.write("SCALARS point_index int 1\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for point_index in point_indices:
            vtk_file.write(f"{point_index}\n")


def save_anatomical_vtk_files(patient_dir: Path, anatomical: np.ndarray, patient_id: str):
    control_branches = {
        branch_name: anatomical[branch_slice]
        for branch_name, branch_slice in BRANCH_SLICES.items()
    }
    save_branch_polyline_vtk(
        patient_dir / "anatomical_control_points.vtk",
        control_branches,
        f"{patient_id} anatomical LCA control points",
    )

    centerlines = interpolate_lca_tree(anatomical)
    save_branch_polyline_vtk(
        patient_dir / "anatomical_centerlines.vtk",
        centerlines,
        f"{patient_id} anatomical LCA centerlines",
    )


def _load_control_trees(dataset_dir: Path) -> np.ndarray:
    path = dataset_dir / "LCA_tree_ctrl_points.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing LCA control tree dataset: {path}")
    trees = np.load(path)
    if trees.ndim != 3 or trees.shape[1:] != (27, 3):
        raise ValueError(f"Expected LCA_tree_ctrl_points.npy shape (N, 27, 3), got {trees.shape}")
    return trees


def main():
    default_dataset_dir = lca_generated_control_points_dir()
    default_output_dir = output_path("dataset_lca_anatomical")

    parser = argparse.ArgumentParser(description="Generate rule-based anatomical LCA 27x3 control-point layouts.")
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir)
    parser.add_argument("--output", type=Path, default=default_output_dir)
    parser.add_argument("--num-trees", type=int, default=None)
    parser.add_argument("--no-clean", action="store_true", help="Do not clear previous anatomical output before writing.")
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Also reorient source trees that fail original dataset validation.",
    )
    args = parser.parse_args()

    output_dir = args.output
    tree_dir = output_dir / "trees"
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    tree_dir.mkdir(parents=True, exist_ok=True)

    original_trees = _load_control_trees(args.dataset_dir)
    if args.num_trees is not None:
        original_trees = original_trees[:args.num_trees]

    anatomical_trees = []
    tree_reports = []
    skipped_reports = []
    valid_count = 0

    for tree_index, original in enumerate(original_trees):
        patient_id = f"patient_{tree_index:04d}"
        original_centerlines = interpolate_lca_tree(original)
        original_metrics = calculate_lca_tortuosity(original_centerlines)
        source_validation = validate_lca_tree(original, original_centerlines, original_metrics)

        if not source_validation["is_valid"] and not args.include_invalid:
            skipped_reports.append({
                "tree_index": tree_index,
                "patient_id": patient_id,
                "reason": "source LCA tree failed original dataset validation",
                "source_validation": source_validation,
            })
            tree_reports.append({
                "tree_index": tree_index,
                "patient_id": patient_id,
                "source_validation_status": "invalid",
                "anatomical_validation_status": "skipped",
                "source_validation": source_validation,
            })
            continue

        patient_dir = tree_dir / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

        anatomical, metadata = reorient_lca_control_tree(original)
        validation = validate_anatomical_lca_tree(anatomical)
        anatomical_trees.append(anatomical)
        valid_count += int(validation["is_valid"])

        np.save(patient_dir / "control_points_27x3_original.npy", original)
        np.save(patient_dir / "control_points_27x3_anatomical.npy", anatomical)
        _write_json(patient_dir / "anatomical_validation.json", validation)
        _write_json(
            patient_dir / "anatomical_summary.json",
            {
                "patient_id": patient_id,
                "input_shape": list(original.shape),
                "output_shape": list(anatomical.shape),
                "source_validation": source_validation,
                "metadata": metadata,
                "validation": validation,
            },
        )
        save_anatomical_tree_plot(
            patient_dir / "anatomical_control_points_3d.png",
            anatomical,
            f"{patient_id} anatomical LCA control points",
        )
        save_before_after_plot(
            patient_dir / "before_vs_after_control_points.png",
            original,
            anatomical,
            patient_id,
        )
        save_centerline_preview(
            patient_dir / "anatomical_centerlines_preview.png",
            anatomical,
            patient_id,
        )
        save_anatomical_vtk_files(patient_dir, anatomical, patient_id)

        tree_reports.append({
            "tree_index": tree_index,
            "patient_id": patient_id,
            "source_validation_status": "valid" if source_validation["is_valid"] else "invalid",
            "anatomical_validation_status": "valid" if validation["is_valid"] else "invalid",
            "input_branch_lengths_mm": metadata["input_branch_lengths_mm"],
            "output_branch_lengths_mm": metadata["output_branch_lengths_mm"],
            "source_validation": source_validation,
            "validation": validation,
        })

    anatomical_trees = np.asarray(anatomical_trees, dtype=float)
    np.save(output_dir / "LCA_tree_ctrl_points_anatomical.npy", anatomical_trees)

    report = {
        "mode": "rule_based_anatomical_lca_reorientation_mvp",
        "description": "Converts existing dataset-derived LCA control trees into a heart-like 27x3 anatomical layout.",
        "input_dataset": str(args.dataset_dir / "LCA_tree_ctrl_points.npy"),
        "output_dataset": str(output_dir / "LCA_tree_ctrl_points_anatomical.npy"),
        "total_source_trees": int(len(original_trees)),
        "total_anatomical_trees": int(len(anatomical_trees)),
        "include_invalid": bool(args.include_invalid),
        "skipped_invalid_source_trees": skipped_reports,
        "valid_trees": int(valid_count),
        "invalid_trees": int(len(anatomical_trees) - valid_count),
        "control_point_indexing": {
            "LMCA": "0:5",
            "LAD": "5:17",
            "LCX": "17:27",
            "shared_bifurcation_index": 4,
        },
        "interactive_vtk_exports": {
            "control_points": "trees/patient_XXXX/anatomical_control_points.vtk",
            "centerlines": "trees/patient_XXXX/anatomical_centerlines.vtk",
            "branch_id_mapping": BRANCH_IDS,
            "point_data": ["branch_id", "point_index"],
        },
        "rules": {
            "LMCA": "short slightly curved trunk",
            "LAD": "descends in negative Z toward apex",
            "LCX": "curves laterally in a crown/circumflex arc",
        },
        "limitations": "Rule-based anatomical approximation, not patient-specific clinical reconstruction.",
        "trees": tree_reports,
    }
    _write_json(output_dir / "anatomical_validation.json", report)
    _write_json(output_dir / "anatomical_summary.json", report)

    print(f"Generated {len(anatomical_trees)} anatomical LCA control trees in: {output_dir}")
    print(f"Valid anatomical trees: {valid_count}")
    print(f"Invalid anatomical trees: {len(anatomical_trees) - valid_count}")
    print(f"Anatomical dataset: {output_dir / 'LCA_tree_ctrl_points_anatomical.npy'}")


if __name__ == "__main__":
    main()
