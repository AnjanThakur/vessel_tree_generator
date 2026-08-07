"""Build a dataset-derived LCA landmark SSM from unchanged source centerlines."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

from lca_ssm_curve_fitting import discrete_curve_metrics, fit_constrained_ellipse_arc
from lca_ssm_label_adapter import (
    LOCAL_RADIUS_WINDOW_MM,
    coordinates_to_world,
    extract_lca_from_label_case,
    extract_path_coordinates,
    load_binary_nifti,
    load_graph,
    node_image_coordinates,
    read_nifti_header,
)
from lca_ssm_planes import angle_degrees, fit_plane_svd, lift_plane_points, project_points_to_plane


VERSION = "3.0.0"
EPS = 1.0e-12
BRANCHES = ("lmca", "lad", "lcx")
LANDMARK_COUNTS = {"lmca": 5, "lad": 12, "lcx": 10}
LANDMARK_ORDER = ("lmca", "lad", "lcx")
SHARED_POINT_INDICES = (4, 5, 17)
ASSIGNMENT_WARNING_MARGIN = 0.20
ASSIGNMENT_MANUAL_MARGIN = 0.08
ROOT_WARNING_MARGIN_MM = 0.05
REFERENCE_WARNING_AXIS_RATIO = 30.0
RCA_MINIMUM_LENGTH_MM = 30.0
RCA_MINIMUM_POINTS = 20
PILOT_IDS = (
    "1.label", "23.label", "45.label", "67.label", "89.label",
    "111.label", "133.label", "155.label", "177.label", "199.label",
)
MANAGED_DIRECTORIES = (
    "raw_cases", "aligned_cases", "per_patient_parameters", "planes",
    "ellipse_references", "landmarks", "quality_control",
    "population_statistics", "pca_ssm", "generated_trees",
    "generated_validation", "visualizations",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize(vector: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    length = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(length) or length <= EPS:
        raise ValueError(f"{name} must be a finite non-zero 3-D vector")
    return vector / length


def branch_tangent(points: np.ndarray, *, at_start: bool, window: int = 5) -> np.ndarray:
    segments = np.diff(np.asarray(points, dtype=float), axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    unit = segments[lengths > EPS] / lengths[lengths > EPS, None]
    if not len(unit):
        raise ValueError("branch contains no non-zero segments")
    chosen = unit[:window] if at_start else unit[-window:]
    return normalize(chosen.mean(axis=0), name="branch tangent")


def cumulative_positions(points: np.ndarray) -> tuple[np.ndarray, float]:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    if total <= EPS:
        raise ValueError("zero-length branch")
    return cumulative / total, total


def build_local_frame(lmca_tangent: np.ndarray) -> dict[str, Any]:
    axis_x = normalize(lmca_tangent, name="LMCA tangent")
    inferior = np.array([0.0, 0.0, -1.0])
    axis_z = inferior - np.dot(inferior, axis_x) * axis_x
    if np.linalg.norm(axis_z) <= EPS:
        fallback = np.array([0.0, 1.0, 0.0])
        axis_z = fallback - np.dot(fallback, axis_x) * axis_x
    axis_z = normalize(axis_z, name="inferior frame axis")
    axis_y = normalize(np.cross(axis_z, axis_x), name="frame cross axis")
    axis_z = normalize(np.cross(axis_x, axis_y), name="inferior frame axis")
    if np.dot(axis_z, inferior) < 0.0:
        axis_y = -axis_y
        axis_z = -axis_z
    rotation = np.column_stack((axis_x, axis_y, axis_z))
    return {
        "origin_definition": "LMCA bifurcation",
        "axis_x_lmca_tangent_ras": axis_x,
        "axis_y_cross_product_ras": axis_y,
        "axis_z_inferior_reference_ras": axis_z,
        "ras_to_local_row_rotation": rotation,
        "local_to_ras_row_rotation": rotation.T,
        "handedness_determinant": float(np.linalg.det(rotation)),
        "scale": 1.0,
    }


def transform_points(points: np.ndarray, origin: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return (np.asarray(points, dtype=float) - origin) @ rotation


def describe(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def discover_cases(input_dir: Path) -> list[Path]:
    cases = [
        path for path in input_dir.iterdir()
        if path.is_dir()
        and (path / "summary.csv").is_file()
        and (path / "branches.npz").is_file()
    ]
    return sorted(cases, key=lambda path: int(path.name.split(".", 1)[0]))


def raw_candidates_from_adapter(
    labelled: dict[str, np.ndarray], metadata: dict[str, Any]
) -> dict[str, np.ndarray]:
    classification = metadata["daughter_classification"]
    raw = {
        classification["selected_lad_source"]: labelled["lad"],
        classification["selected_lcx_source"]: labelled["lcx"],
    }
    if set(raw) != {"branch_a", "branch_b"}:
        raise ValueError("adapter did not expose both raw daughter candidates")
    return raw


def daughter_features(points: np.ndarray) -> dict[str, Any]:
    metrics = discrete_curve_metrics(points)
    displacement = points[-1] - points[0]
    initial = branch_tangent(points, at_start=True)
    overall = normalize(displacement, name="daughter displacement")
    plane = fit_plane_svd(points, name="daughter assignment plane")
    horizontal_path = float(np.linalg.norm(np.diff(points, axis=0)[:, :2], axis=1).sum())
    return {
        "length_mm": metrics["length_mm"],
        "displacement_ras_mm": displacement,
        "inferior_endpoint_mm": float(max(0.0, -displacement[2])),
        "inferior_reach_mm": float(max(0.0, points[0, 2] - np.min(points[:, 2]))),
        "lateral_displacement_xy_mm": float(np.linalg.norm(displacement[:2])),
        "horizontal_path_fraction": float(horizontal_path / metrics["length_mm"]),
        "initial_direction_ras": initial,
        "overall_direction_ras": overall,
        "initial_inferior_score": float(max(0.0, -initial[2])),
        "initial_crown_score": float(1.0 - abs(initial[2])),
        "overall_inferior_score": float(max(0.0, -overall[2])),
        "crown_plane_score": float(abs(plane["normal"][2])),
    }


def pair_share(value: float, other: float) -> float:
    total = value + other
    return 0.5 if total <= EPS else float(value / total)


def score_assignment(
    lad_source: str,
    raw: dict[str, np.ndarray],
    features: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    lcx_source = "branch_b" if lad_source == "branch_a" else "branch_a"
    lad = features[lad_source]
    lcx = features[lcx_source]
    inferior_share = pair_share(lad["inferior_reach_mm"], lcx["inferior_reach_mm"])
    lad_role = (
        0.25 * pair_share(lad["inferior_endpoint_mm"], lcx["inferior_endpoint_mm"])
        + 0.20 * inferior_share
        + 0.20 * pair_share(lad["length_mm"], lcx["length_mm"])
        + 0.20 * lad["initial_inferior_score"]
        + 0.15 * lad["overall_inferior_score"]
    )
    lcx_role = (
        0.35 * inferior_share
        + 0.20 * lcx["initial_crown_score"]
        + 0.15 * lcx["horizontal_path_fraction"]
        + 0.15 * lcx["crown_plane_score"]
        + 0.15 * pair_share(
            lcx["lateral_displacement_xy_mm"], lad["lateral_displacement_xy_mm"]
        )
    )
    relationship = (
        0.35 * pair_share(lad["inferior_endpoint_mm"], lcx["inferior_endpoint_mm"])
        + 0.35 * inferior_share
        + 0.30 * pair_share(lad["length_mm"], lcx["length_mm"])
    )
    penalty = 0.0
    lcx_main_apex = (
        lcx["inferior_reach_mm"] > lad["inferior_reach_mm"]
        and lcx["length_mm"] > lad["length_mm"]
    )
    if lcx["inferior_reach_mm"] > lad["inferior_reach_mm"]:
        penalty += 0.12
    if lcx_main_apex:
        penalty += 0.08
    score = float(np.clip(0.45 * lad_role + 0.30 * lcx_role + 0.25 * relationship - penalty, 0, 1))
    return {
        "lad_source": lad_source,
        "lcx_source": lcx_source,
        "score": score,
        "lad_role_score": lad_role,
        "lcx_role_score": lcx_role,
        "relationship_score": relationship,
        "penalty": penalty,
        "lcx_behaves_as_main_apex_branch": lcx_main_apex,
    }


def classify_daughters(
    raw: dict[str, np.ndarray], adapter_metadata: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    features = {name: daughter_features(points) for name, points in raw.items()}
    candidates = {
        "branch_a_as_lad": score_assignment("branch_a", raw, features),
        "branch_b_as_lad": score_assignment("branch_b", raw, features),
    }
    ranked = sorted(candidates.values(), key=lambda item: item["score"], reverse=True)
    selected, runner_up = ranked
    margin = float(selected["score"] - runner_up["score"])
    branches = {"lad": raw[selected["lad_source"]], "lcx": raw[selected["lcx_source"]]}
    adapter_selected = adapter_metadata["daughter_classification"]["selected_lad_source"]
    return branches, {
        "method": "multi_signal_ras_anatomy_score",
        "root_radius_used_for_assignment": False,
        "single_direction_vector_can_select_assignment": False,
        "candidate_features": features,
        "candidate_scores": candidates,
        "selected_lad_source": selected["lad_source"],
        "selected_lcx_source": selected["lcx_source"],
        "score": selected["score"],
        "margin": margin,
        "adapter_lad_source": adapter_selected,
        "agrees_with_adapter": bool(adapter_selected == selected["lad_source"]),
        "coordinate_system": "NIfTI physical RAS",
    }


def endpoint_radius(
    node: int,
    coordinates: dict[int, np.ndarray],
    mask: np.ndarray,
    spacing: np.ndarray,
) -> float:
    center = coordinates[node]
    window = np.ceil(LOCAL_RADIUS_WINDOW_MM / spacing).astype(int)
    lower = np.maximum(center - window, 0)
    upper = np.minimum(center + window + 1, mask.shape)
    crop = mask[lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]]
    return float(distance_transform_edt(crop, sampling=spacing)[tuple(center - lower)])


def infer_rca_candidate(
    patient_dir: Path,
    nifti_path: Path,
    lca_root_node: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    header = read_nifti_header(nifti_path)
    graph, rows, archive = load_graph(patient_dir)
    mask = load_binary_nifti(nifti_path, header)
    try:
        components = list(nx.connected_components(graph))
        lca_component = next(component for component in components if lca_root_node in component)
        node_coordinates = node_image_coordinates(rows)
        candidates = []
        for component in components:
            if component is lca_component or lca_root_node in component or len(component) < 2:
                continue
            subgraph = graph.subgraph(component).copy()
            endpoints = sorted(node for node, degree in subgraph.degree if degree == 1)
            if len(endpoints) < 2:
                continue
            radii = sorted(
                [(endpoint_radius(node, node_coordinates, mask, header["spacing_mm"]), node)
                 for node in endpoints],
                reverse=True,
            )
            root = radii[0][1]
            distances = nx.single_source_dijkstra_path_length(subgraph, root, weight="weight")
            terminal = max(distances, key=distances.get)
            path_nodes = nx.shortest_path(subgraph, root, terminal, weight="weight")
            image_points = extract_path_coordinates(subgraph, path_nodes, archive)
            world_points = coordinates_to_world(image_points, header)
            length_mm = float(np.linalg.norm(np.diff(world_points, axis=0), axis=1).sum())
            candidates.append({
                "points": world_points,
                "component_node_count": len(component),
                "endpoint_count": len(endpoints),
                "root_node": root,
                "terminal_node": terminal,
                "root_radius_mm": radii[0][0],
                "second_endpoint_radius_mm": radii[1][0],
                "path_length_mm": length_mm,
                "path_point_count": len(world_points),
            })
    finally:
        archive.close()
        del mask
    structurally_usable = [
        candidate for candidate in candidates
        if candidate["path_length_mm"] >= RCA_MINIMUM_LENGTH_MM
        and candidate["path_point_count"] >= RCA_MINIMUM_POINTS
    ]
    if not structurally_usable:
        return None, {
            "available": False,
            "status": "no_structurally_usable_disconnected_rca_candidate",
            "component_count": len(components),
            "candidate_summaries": [
                {key: value for key, value in candidate.items() if key != "points"}
                for candidate in candidates
            ],
        }
    selected = max(structurally_usable, key=lambda item: item["path_length_mm"])
    metadata = {key: value for key, value in selected.items() if key != "points"}
    metadata.update({
        "available": True,
        "status": "inferred_disconnected_rca_candidate_not_ground_truth",
        "selection_rule": "longest structurally usable non-LCA connected-component main path",
        "component_count": len(components),
    })
    return selected["points"], metadata


def select_landmarks(
    points_original: np.ndarray,
    points_aligned: np.ndarray,
    branch: str,
) -> dict[str, Any]:
    count = LANDMARK_COUNTS[branch]
    positions, _ = cumulative_positions(points_original)
    targets = np.linspace(0.0, 1.0, count)
    indices: list[int] = []
    for target in targets:
        index = int(np.argmin(np.abs(positions - target)))
        if indices and index <= indices[-1]:
            index = indices[-1] + 1
        if index >= len(points_original):
            raise ValueError(f"{branch} cannot supply {count} unique ordered source landmarks")
        indices.append(index)
    records = []
    for index, target in zip(indices, targets):
        lower = max(0, index - 2)
        upper = min(len(points_original) - 1, index + 2)
        tangent_ras = normalize(
            points_original[upper] - points_original[lower],
            name=f"{branch} landmark tangent",
        )
        tangent_aligned = normalize(
            points_aligned[upper] - points_aligned[lower],
            name=f"{branch} aligned landmark tangent",
        )
        if 0 < index < len(points_original) - 1:
            before = normalize(points_original[index] - points_original[index - 1], name="before")
            after = normalize(points_original[index + 1] - points_original[index], name="after")
            turning_deg = angle_degrees(before, after)
            local_scale = 0.5 * (
                np.linalg.norm(points_original[index] - points_original[index - 1])
                + np.linalg.norm(points_original[index + 1] - points_original[index])
            )
            curvature = math.radians(turning_deg) / max(local_scale, EPS)
        else:
            turning_deg = 0.0
            curvature = 0.0
        records.append({
            "branch": branch,
            "original_point_index": index,
            "target_normalized_arc_position": float(target),
            "actual_normalized_arc_position": float(positions[index]),
            "original_coordinate_ras_mm": points_original[index],
            "aligned_coordinate_mm": points_aligned[index],
            "local_tangent_ras": tangent_ras,
            "local_tangent_aligned": tangent_aligned,
            "turning_angle_deg": turning_deg,
            "curvature_per_mm": curvature,
            "selection_method": "nearest_existing_source_sample_by_normalized_arc_length",
        })
    return {
        "branch": branch,
        "count": count,
        "source_indices": np.asarray(indices, dtype=int),
        "original_coordinates": points_original[indices],
        "aligned_coordinates": points_aligned[indices],
        "records": records,
        "all_coordinates_are_exact_source_samples": bool(all(
            np.array_equal(points_original[index], points_original[indices][position])
            for position, index in enumerate(indices)
        )),
        "curvature_replacement_enabled": False,
    }


def plane_payload(plane: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "plane_source": source,
        "centroid": plane["centroid"],
        "normal": plane["normal"],
        "basis_u": plane["basis_u"],
        "basis_v": plane["basis_v"],
        "singular_values": plane["singular_values"],
        "rank": plane["rank"],
        "point_distances_mm": plane["point_residuals_mm"],
        "rmse_mm": plane["rms_residual_mm"],
        "maximum_distance_mm": plane["max_residual_mm"],
    }


def nearest_curve_distances(points: np.ndarray, curve: np.ndarray) -> np.ndarray:
    differences = points[:, None, :] - curve[None, :, :]
    return np.sqrt(np.min(np.sum(differences * differences, axis=2), axis=1))


def ellipse_reference(
    branch: str,
    raw_points: np.ndarray,
    landmark_points: np.ndarray,
    plane: dict[str, Any],
) -> dict[str, Any]:
    landmark_2d, _ = project_points_to_plane(landmark_points, plane)
    tangent = branch_tangent(raw_points, at_start=True)
    tangent_2d = np.array([
        np.dot(tangent, plane["basis_u"]),
        np.dot(tangent, plane["basis_v"]),
    ])
    ellipse = fit_constrained_ellipse_arc(
        landmark_2d,
        tangent_2d,
        name=f"{branch.upper()} reference",
        maximum_axis_ratio=REFERENCE_WARNING_AXIS_RATIO,
        allow_axis_ratio_guard_exceedance=True,
    )
    fitted_landmarks = lift_plane_points(ellipse["fitted_points_2d"], plane)
    arc_3d = lift_plane_points(ellipse["sampled_arc_2d"], plane)
    landmark_vectors = landmark_points - fitted_landmarks
    full_distances = nearest_curve_distances(raw_points, arc_3d)
    stored = {
        key: value for key, value in ellipse.items()
        if key not in {"fitted_points_2d", "sampled_arc_2d", "point_residuals_mm"}
    }
    return {
        "interpretation": "diagnostic_statistical_reference_not_source_geometry",
        "used_for_patient_acceptance": False,
        "ellipse": stored,
        "arc_3d_ras_mm": arc_3d,
        "fitted_landmarks_3d_ras_mm": fitted_landmarks,
        "landmark_residual_vectors_ras_mm": landmark_vectors,
        "landmark_residual_magnitudes_mm": np.linalg.norm(landmark_vectors, axis=1),
        "full_centerline_residuals_mm": full_distances,
        "full_centerline_residual_label": "descriptive_only",
    }


def curvature_statistics(points: np.ndarray) -> dict[str, Any]:
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    unit = segments / np.maximum(lengths[:, None], EPS)
    if len(unit) < 2:
        values = np.zeros(1)
    else:
        turns = np.arccos(np.clip(np.sum(unit[:-1] * unit[1:], axis=1), -1, 1))
        scales = 0.5 * (lengths[:-1] + lengths[1:])
        values = turns / np.maximum(scales, EPS)
    return {"per_point_curvature_per_mm": values, **describe(values)}


def raw_measurements(
    branches: dict[str, np.ndarray],
    planes: dict[str, dict[str, Any]],
    origin: np.ndarray,
    assignment: dict[str, Any],
) -> dict[str, Any]:
    metrics = {name: discrete_curve_metrics(points) for name, points in branches.items()}
    lmca_tangent = branch_tangent(branches["lmca"], at_start=False)
    lad_tangent = branch_tangent(branches["lad"], at_start=True)
    lcx_tangent = branch_tangent(branches["lcx"], at_start=True)
    plane_angle = angle_degrees(planes["lad"]["normal"], planes["coronary"]["normal"])
    plane_angle = min(plane_angle, 180.0 - plane_angle)
    inferior = np.array([0.0, 0.0, -1.0])
    lcx_displacement = branches["lcx"][-1] - origin
    lcx_lateral = float(np.linalg.norm(lcx_displacement[:2]))
    lcx_inferior = float(max(0.0, -lcx_displacement[2]))
    return {
        "angles_deg": {
            "lad_lcx_bifurcation": angle_degrees(lad_tangent, lcx_tangent),
            "lmca_to_lad": angle_degrees(lmca_tangent, lad_tangent),
            "lmca_to_lcx": angle_degrees(lmca_tangent, lcx_tangent),
            "lad_initial_to_inferior_ras_z": angle_degrees(lad_tangent, inferior),
            "lad_terminal_direction_to_inferior_ras_z": angle_degrees(
                branches["lad"][-1] - origin, inferior
            ),
            "lad_coronary_plane": plane_angle,
        },
        "lengths_mm": {name: metrics[name]["length_mm"] for name in BRANCHES},
        "length_ratios": {
            "lad_to_lcx": metrics["lad"]["length_mm"] / metrics["lcx"]["length_mm"],
            "lad_to_lmca": metrics["lad"]["length_mm"] / metrics["lmca"]["length_mm"],
            "lcx_to_lmca": metrics["lcx"]["length_mm"] / metrics["lmca"]["length_mm"],
        },
        "tortuosity": {name: metrics[name]["tortuosity"] for name in BRANCHES},
        "curvature": {name: curvature_statistics(branches[name]) for name in BRANCHES},
        "plane_quality": {
            "lad_rmse_mm": planes["lad"]["rms_residual_mm"],
            "lad_maximum_mm": planes["lad"]["max_residual_mm"],
            "coronary_rmse_mm": planes["coronary"]["rms_residual_mm"],
            "coronary_maximum_mm": planes["coronary"]["max_residual_mm"],
        },
        "lcx_crown": {
            "lateral_displacement_xy_mm": lcx_lateral,
            "inferior_displacement_mm": lcx_inferior,
            "lateral_to_inferior_ratio": lcx_lateral / max(lcx_inferior, EPS),
        },
        "terminal_separation_mm": float(
            np.linalg.norm(branches["lad"][-1] - branches["lcx"][-1])
        ),
        "assignment_confidence": {
            "score": assignment["score"],
            "margin": assignment["margin"],
            "selected_lad_source": assignment["selected_lad_source"],
            "selected_lcx_source": assignment["selected_lcx_source"],
        },
    }


def classify_quality(
    branches: dict[str, np.ndarray],
    extraction: dict[str, Any],
    assignment: dict[str, Any],
    landmarks: dict[str, dict[str, Any]],
    rca_metadata: dict[str, Any],
    references: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    warnings = []
    for name, points in branches.items():
        if not np.all(np.isfinite(points)):
            failures.append(f"{name}: non-finite coordinates")
        if len(points) < LANDMARK_COUNTS.get(name, 2):
            failures.append(f"{name}: insufficient samples for fixed correspondence")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= EPS):
            failures.append(f"{name}: repeated consecutive coordinates")
    if extraction["maximum_junction_offset_mm"] > 1.0:
        failures.append("broken shared bifurcation")
    if not all(item["all_coordinates_are_exact_source_samples"] for item in landmarks.values()):
        failures.append("landmark correspondence moved away from source samples")
    margin = assignment["margin"]
    if margin < ASSIGNMENT_MANUAL_MARGIN:
        warnings.append(f"very low daughter-assignment margin ({margin:.4f})")
        status = "manual_review"
    elif margin < ASSIGNMENT_WARNING_MARGIN:
        warnings.append(f"low daughter-assignment margin ({margin:.4f})")
        status = "accepted_with_warnings"
    else:
        status = "accepted"
    root_margin = extraction["root_selection"]["radius_margin_mm"]
    if root_margin < ROOT_WARNING_MARGIN_MM:
        warnings.append(f"low root-radius margin ({root_margin:.4f} mm)")
        if status == "accepted":
            status = "accepted_with_warnings"
    if not rca_metadata["available"]:
        warnings.append("coronary plane uses LCX-only approximation because no usable RCA candidate was found")
        if status == "accepted":
            status = "accepted_with_warnings"
    for branch, reference in references.items():
        axis_ratio = float(reference["ellipse"]["axis_ratio"])
        if axis_ratio > REFERENCE_WARNING_AXIS_RATIO:
            warnings.append(
                f"{branch.upper()} diagnostic ellipse is slender "
                f"(axis ratio {axis_ratio:.3f}); raw anatomy remains usable"
            )
            if status == "accepted":
                status = "accepted_with_warnings"
    if failures:
        status = "rejected"
    return {
        "status": status,
        "raw_anatomy_validation": {
            "source_geometry_modified": False,
            "junction_error_mm": extraction["maximum_junction_offset_mm"],
            "failure_reasons": failures,
        },
        "branch_assignment_confidence": {
            "score": assignment["score"], "margin": margin,
            "manual_review_threshold": ASSIGNMENT_MANUAL_MARGIN,
            "warning_threshold": ASSIGNMENT_WARNING_MARGIN,
        },
        "plane_fit_quality": "usable" if not failures else "unusable",
        "ellipse_reference_quality": "diagnostic_only_not_rejection_basis",
        "warnings": warnings,
        "rejection_reasons": failures,
    }


def process_case(patient_dir: Path, nifti_dir: Path) -> dict[str, Any]:
    patient_id = patient_dir.name
    nifti_path = nifti_dir / f"{patient_id}.nii.gz"
    labelled, extraction = extract_lca_from_label_case(patient_dir, nifti_path)
    raw = raw_candidates_from_adapter(labelled, extraction)
    daughters, assignment = classify_daughters(raw, extraction)
    branches = {"lmca": labelled["lmca"], **daughters}
    origin = np.asarray(extraction["junction_coordinate_mm"], dtype=float)
    frame = build_local_frame(branch_tangent(branches["lmca"], at_start=False))
    aligned = {
        name: transform_points(points, origin, frame["ras_to_local_row_rotation"])
        for name, points in branches.items()
    }
    preservation = {}
    for name in BRANCHES:
        raw_lengths = np.linalg.norm(np.diff(branches[name], axis=0), axis=1)
        aligned_lengths = np.linalg.norm(np.diff(aligned[name], axis=0), axis=1)
        preservation[name] = float(np.max(np.abs(raw_lengths - aligned_lengths)))

    landmark_data = {
        name: select_landmarks(branches[name], aligned[name], name) for name in BRANCHES
    }
    rca, rca_metadata = infer_rca_candidate(
        patient_dir, nifti_path, extraction["root_selection"]["node_id"]
    )
    lad_plane = fit_plane_svd(branches["lad"], name="unchanged LAD")
    if rca is not None:
        coronary_points = np.vstack((rca, branches["lcx"]))
        coronary_source = "inferred_rca_plus_lcx"
    else:
        coronary_points = branches["lcx"]
        coronary_source = "lcx_only_approximation"
    coronary_plane = fit_plane_svd(coronary_points, name="coronary plane points")
    planes = {"lad": lad_plane, "coronary": coronary_plane}
    references = {
        "lad": ellipse_reference(
            "lad", branches["lad"], landmark_data["lad"]["original_coordinates"], lad_plane
        ),
        "lcx": ellipse_reference(
            "lcx", branches["lcx"], landmark_data["lcx"]["original_coordinates"], coronary_plane
        ),
    }
    measurements = raw_measurements(branches, planes, origin, assignment)
    quality = classify_quality(
        branches, extraction, assignment, landmark_data, rca_metadata, references
    )
    return {
        "patient_id": patient_id,
        "status": quality["status"],
        "branches_original": branches,
        "rca_original": rca,
        "branches_initial_aligned": aligned,
        "origin_ras_mm": origin,
        "frame": frame,
        "segment_length_preservation_error_mm": preservation,
        "landmarks": landmark_data,
        "planes": {
            "lad": plane_payload(lad_plane, "unchanged_lad_points"),
            "coronary": plane_payload(coronary_plane, coronary_source),
        },
        "ellipse_references": references,
        "measurements": measurements,
        "quality": quality,
        "extraction_metadata": extraction,
        "assignment": assignment,
        "rca_metadata": rca_metadata,
    }


def landmark_matrix(case: dict[str, Any], key: str = "aligned_coordinates") -> np.ndarray:
    return np.vstack([case["landmarks"][name][key] for name in LANDMARK_ORDER])


def kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(source.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def generalized_procrustes(
    shapes: np.ndarray,
    *,
    scale_normalized: bool,
    maximum_iterations: int = 100,
    tolerance: float = 1.0e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    working = shapes.copy()
    scales = np.ones(len(working))
    if scale_normalized:
        for index, shape in enumerate(working):
            scale = float(np.linalg.norm(shape))
            if scale <= EPS:
                raise ValueError("cannot scale-normalize a zero landmark configuration")
            working[index] /= scale
            scales[index] = scale
    mean = working.mean(axis=0)
    rotations = np.repeat(np.eye(3)[None, :, :], len(working), axis=0)
    history = []
    for _ in range(maximum_iterations):
        aligned = np.empty_like(working)
        for index, shape in enumerate(working):
            incremental_rotation = kabsch_rotation(shape, mean)
            rotations[index] = rotations[index] @ incremental_rotation
            aligned[index] = shape @ incremental_rotation
        updated = aligned.mean(axis=0)
        updated[list(SHARED_POINT_INDICES)] = 0.0
        delta = float(np.linalg.norm(updated - mean))
        history.append(delta)
        mean = updated
        working = aligned
        if delta < tolerance:
            break
    return working, mean, rotations, scales, history


def pca_model(shape_matrix: np.ndarray) -> dict[str, Any]:
    mean_vector = shape_matrix.mean(axis=0)
    centered = shape_matrix - mean_vector
    u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    eigenvalues = singular_values ** 2 / max(len(shape_matrix) - 1, 1)
    tolerance = max(float(eigenvalues[0]) if len(eigenvalues) else 0.0, EPS) * 1.0e-12
    retained = eigenvalues > tolerance
    eigenvalues = eigenvalues[retained]
    components = vt[retained]
    coefficients = centered @ components.T
    explained = eigenvalues / max(float(eigenvalues.sum()), EPS)
    return {
        "mean_vector": mean_vector,
        "components": components,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained,
        "cumulative_explained_variance": np.cumsum(explained),
        "coefficients": coefficients,
        "singular_values": singular_values[retained],
    }


def scalar_parameter_row(case: dict[str, Any]) -> dict[str, float]:
    m = case["measurements"]
    lad_e = case["ellipse_references"]["lad"]["ellipse"]
    lcx_e = case["ellipse_references"]["lcx"]["ellipse"]
    return {
        "lmca_length_mm": m["lengths_mm"]["lmca"],
        "lad_length_mm": m["lengths_mm"]["lad"],
        "lcx_length_mm": m["lengths_mm"]["lcx"],
        "lad_lcx_length_ratio": m["length_ratios"]["lad_to_lcx"],
        "lad_lcx_bifurcation_deg": m["angles_deg"]["lad_lcx_bifurcation"],
        "lmca_to_lad_deg": m["angles_deg"]["lmca_to_lad"],
        "lmca_to_lcx_deg": m["angles_deg"]["lmca_to_lcx"],
        "plane_angle_deg": m["angles_deg"]["lad_coronary_plane"],
        "lad_tortuosity": m["tortuosity"]["lad"],
        "lcx_tortuosity": m["tortuosity"]["lcx"],
        "lad_plane_rmse_mm": m["plane_quality"]["lad_rmse_mm"],
        "coronary_plane_rmse_mm": m["plane_quality"]["coronary_rmse_mm"],
        "lad_ellipse_a_mm": lad_e["semi_major_axis_mm"],
        "lad_ellipse_b_mm": lad_e["semi_minor_axis_mm"],
        "lad_ellipse_tilt_deg": lad_e["orientation_deg"],
        "lad_ellipse_extent_deg": lad_e["signed_angular_extent_deg"],
        "lad_ellipse_arc_length_mm": lad_e["arc_length_mm"],
        "lcx_ellipse_a_mm": lcx_e["semi_major_axis_mm"],
        "lcx_ellipse_b_mm": lcx_e["semi_minor_axis_mm"],
        "lcx_ellipse_tilt_deg": lcx_e["orientation_deg"],
        "lcx_ellipse_extent_deg": lcx_e["signed_angular_extent_deg"],
        "lcx_ellipse_arc_length_mm": lcx_e["arc_length_mm"],
        "assignment_margin": case["assignment"]["margin"],
    }


def parameter_statistics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [scalar_parameter_row(case) for case in cases]
    names = list(rows[0])
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=float)
    return {
        "parameter_names": names,
        "matrix": matrix,
        "covariance": np.cov(matrix, rowvar=False),
        "correlation": np.corrcoef(matrix, rowvar=False),
        "statistics": {name: describe(matrix[:, index]) for index, name in enumerate(names)},
    }


def save_case_artifacts(case: dict[str, Any], output_dir: Path) -> None:
    patient = case["patient_id"]
    raw_dir = output_dir / "raw_cases" / patient
    aligned_dir = output_dir / "aligned_cases" / patient
    landmark_dir = output_dir / "landmarks" / patient
    plane_dir = output_dir / "planes" / patient
    ellipse_dir = output_dir / "ellipse_references" / patient
    parameter_dir = output_dir / "per_patient_parameters" / patient
    for directory in (raw_dir, aligned_dir, landmark_dir, plane_dir, ellipse_dir, parameter_dir):
        directory.mkdir(parents=True, exist_ok=True)
    raw_arrays = {name: case["branches_original"][name] for name in BRANCHES}
    if case["rca_original"] is not None:
        raw_arrays["rca"] = case["rca_original"]
    np.savez_compressed(raw_dir / "original_centerlines.npz", **raw_arrays)
    write_json(raw_dir / "source_metadata.json", {
        "patient_id": patient,
        "geometry_modified": False,
        "extraction": case["extraction_metadata"],
        "assignment": case["assignment"],
        "rca": case["rca_metadata"],
    })
    np.savez_compressed(
        aligned_dir / "initial_rigid_aligned_centerlines.npz",
        **case["branches_initial_aligned"],
    )
    write_json(aligned_dir / "initial_rigid_transform.json", {
        "origin_ras_mm": case["origin_ras_mm"],
        "frame": case["frame"],
        "segment_length_preservation_error_mm": case["segment_length_preservation_error_mm"],
    })
    landmark_npz = {}
    landmark_json = {"schema": LANDMARK_COUNTS, "branches": {}}
    for name in BRANCHES:
        data = case["landmarks"][name]
        landmark_npz[f"{name}_indices"] = data["source_indices"]
        landmark_npz[f"{name}_original"] = data["original_coordinates"]
        landmark_npz[f"{name}_initial_aligned"] = data["aligned_coordinates"]
        landmark_json["branches"][name] = {
            "count": data["count"],
            "records": data["records"],
            "all_exact_source_samples": data["all_coordinates_are_exact_source_samples"],
        }
    np.savez_compressed(landmark_dir / "source_landmarks.npz", **landmark_npz)
    write_json(landmark_dir / "source_landmarks.json", landmark_json)
    write_json(plane_dir / "planes.json", case["planes"])
    ellipse_arrays = {}
    ellipse_json = {}
    for name in ("lad", "lcx"):
        reference = case["ellipse_references"][name]
        ellipse_arrays[f"{name}_arc_ras"] = reference["arc_3d_ras_mm"]
        ellipse_arrays[f"{name}_residual_vectors_ras"] = reference[
            "landmark_residual_vectors_ras_mm"
        ]
        ellipse_json[name] = {
            key: value for key, value in reference.items()
            if key not in {"arc_3d_ras_mm", "fitted_landmarks_3d_ras_mm",
                           "landmark_residual_vectors_ras_mm", "full_centerline_residuals_mm"}
        }
        ellipse_json[name]["full_centerline_residuals_mm"] = reference[
            "full_centerline_residuals_mm"
        ]
    np.savez_compressed(ellipse_dir / "reference_arrays.npz", **ellipse_arrays)
    write_json(ellipse_dir / "reference_parameters.json", ellipse_json)
    write_json(parameter_dir / "parameters.json", {
        "patient_id": patient,
        "status": case["status"],
        "measurements": case["measurements"],
        "assignment": case["assignment"],
        "plane_sources": {
            "lad": case["planes"]["lad"]["plane_source"],
            "coronary": case["planes"]["coronary"]["plane_source"],
        },
    })
    write_json(output_dir / "quality_control" / f"{patient}.json", case["quality"])


def save_case_plot(case: dict[str, Any], path: Path) -> None:
    origin = case["origin_ras_mm"]
    figure, axis = plt.subplots(figsize=(8.5, 7.5))
    colors = {"lmca": "#222222", "lad": "#d62728", "lcx": "#1f77b4"}
    for name in BRANCHES:
        points = case["branches_original"][name] - origin
        axis.plot(points[:, 0], points[:, 2], color=colors[name], lw=2.1,
                  label=f"raw {name.upper()}")
        landmarks = case["landmarks"][name]["original_coordinates"] - origin
        axis.scatter(landmarks[:, 0], landmarks[:, 2], s=22, color="#ffbf00",
                     edgecolor="#333333", linewidth=0.4)
    for name in ("lad", "lcx"):
        arc = case["ellipse_references"][name]["arc_3d_ras_mm"] - origin
        axis.plot(arc[:, 0], arc[:, 2], color=colors[name], lw=1.5,
                  linestyle="--", alpha=0.45, label=f"{name.upper()} reference")
    axis.scatter(0, 0, s=65, color="#2ca02c", label="bifurcation")
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("RAS X from bifurcation (mm)")
    axis.set_ylabel("RAS Z from bifurcation (mm; inferior negative)")
    axis.set_title(f"{case['patient_id']}: raw geometry and diagnostic references")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=7)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=155)
    plt.close(figure)


def save_population_overlay(shapes: np.ndarray, mean: np.ndarray, path: Path, title: str) -> None:
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    slices = {"lmca": slice(0, 5), "lad": slice(5, 17), "lcx": slice(17, 27)}
    colors = {"lmca": "#444444", "lad": "#d62728", "lcx": "#1f77b4"}
    for shape in shapes:
        for name, branch_slice in slices.items():
            axis.plot(*shape[branch_slice].T, color=colors[name], alpha=0.10, lw=0.7)
    for name, branch_slice in slices.items():
        axis.plot(*mean[branch_slice].T, color=colors[name], lw=3.0, label=f"mean {name.upper()}")
    axis.set_xlabel("aligned X (mm)")
    axis.set_ylabel("aligned Y (mm)")
    axis.set_zlabel("aligned Z (mm)")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_pca_plots(model: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mean = model["mean_vector"].reshape(27, 3)
    slices = {"lmca": slice(0, 5), "lad": slice(5, 17), "lcx": slice(17, 27)}
    colors = {"lmca": "#333333", "lad": "#d62728", "lcx": "#1f77b4"}
    for mode_index in range(min(2, len(model["eigenvalues"]))):
        variation = 2.0 * math.sqrt(model["eigenvalues"][mode_index])
        variants = {
            "minus_2sd": mean - variation * model["components"][mode_index].reshape(27, 3),
            "mean": mean,
            "plus_2sd": mean + variation * model["components"][mode_index].reshape(27, 3),
        }
        figure = plt.figure(figsize=(9, 7))
        axis = figure.add_subplot(111, projection="3d")
        styles = {"minus_2sd": "--", "mean": "-", "plus_2sd": ":"}
        for variant_name, shape in variants.items():
            for branch_name, branch_slice in slices.items():
                axis.plot(*shape[branch_slice].T, color=colors[branch_name],
                          linestyle=styles[variant_name],
                          lw=2.5 if variant_name == "mean" else 1.5,
                          alpha=1.0 if variant_name == "mean" else 0.65,
                          label=(f"{variant_name} {branch_name.upper()}"
                                 if branch_name == "lmca" else None))
        axis.set_title(f"PCA mode {mode_index + 1}: mean and ±2 SD")
        axis.set_xlabel("X (mm)"); axis.set_ylabel("Y (mm)"); axis.set_zlabel("Z (mm)")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output_dir / f"pca_mode_{mode_index + 1}_plus_minus_2sd.png", dpi=170)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 5))
    x = np.arange(1, len(model["cumulative_explained_variance"]) + 1)
    axis.plot(x, model["cumulative_explained_variance"] * 100.0, marker="o", ms=3)
    axis.axhline(95.0, color="#d62728", linestyle="--", label="95%")
    axis.set_xlabel("principal component count")
    axis.set_ylabel("cumulative explained variance (%)")
    axis.set_ylim(0, 101)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "cumulative_explained_variance.png", dpi=170)
    plt.close(figure)


def write_index_files(cases: list[dict[str, Any]], rejected: list[dict[str, Any]], output: Path) -> None:
    groups = {
        "accepted": [case for case in cases if case["status"] == "accepted"],
        "warning": [case for case in cases if case["status"] == "accepted_with_warnings"],
        "manual_review": [case for case in cases if case["status"] == "manual_review"],
    }
    write_json(output / "quality_control" / "accepted_patient_index.json", {
        "count": len(groups["accepted"]), "patient_ids": [case["patient_id"] for case in groups["accepted"]]
    })
    write_json(output / "quality_control" / "warning_patient_index.json", {
        "count": len(groups["warning"]),
        "patients": [{"patient_id": case["patient_id"], "warnings": case["quality"]["warnings"]}
                     for case in groups["warning"]],
    })
    write_json(output / "quality_control" / "manual_review_patient_index.json", {
        "count": len(groups["manual_review"]),
        "patients": [{"patient_id": case["patient_id"], "warnings": case["quality"]["warnings"]}
                     for case in groups["manual_review"]],
    })
    write_json(output / "quality_control" / "rejected_patient_index.json", {
        "count": len(rejected), "patients": rejected
    })


def pilot_run(case_dirs: list[Path], nifti_dir: Path, output: Path, workers: int) -> int:
    mapping = {case.name: case for case in case_dirs}
    selected_dirs = [mapping[patient] for patient in PILOT_IDS if patient in mapping]
    cases, rejected = process_cases(selected_dirs, nifti_dir, workers)
    for case in cases:
        save_case_plot(case, output / "visualizations" / "pilot" / f"{case['patient_id']}_front.png")
    usable = [case for case in cases if case["status"] in {"accepted", "accepted_with_warnings"}]
    correspondence_pass = all(
        all(case["landmarks"][name]["count"] == LANDMARK_COUNTS[name] for name in BRANCHES)
        and all(case["landmarks"][name]["all_coordinates_are_exact_source_samples"] for name in BRANCHES)
        for case in usable
    )
    assignment_pass = all(case["status"] != "manual_review" for case in cases)
    pilot_pass = (
        len(selected_dirs) == 10 and len(rejected) == 0 and len(usable) == 10
        and correspondence_pass and assignment_pass
    )
    if usable:
        shapes = np.asarray([landmark_matrix(case) for case in usable])
        aligned, mean, _, _, history = generalized_procrustes(shapes, scale_normalized=False)
        save_population_overlay(
            aligned, mean,
            output / "visualizations" / "pilot" / "pilot_aligned_overlay.png",
            "10-case pilot: rigid size-preserving landmark overlay",
        )
    else:
        history = []
    report = {
        "pilot_ids": list(PILOT_IDS),
        "processed_count": len(selected_dirs),
        "usable_count": len(usable),
        "rejected": rejected,
        "statuses": {case["patient_id"]: case["status"] for case in cases},
        "fixed_landmark_schema": LANDMARK_COUNTS,
        "correspondence_pass": correspondence_pass,
        "assignment_consistency_pass": assignment_pass,
        "gpa_convergence_history": history,
        "pilot_pass": pilot_pass,
        "next_action": "run population mode" if pilot_pass else "stop for manual review",
    }
    write_json(output / "quality_control" / "pilot_report.json", report)
    print(json.dumps(jsonable(report), indent=2))
    return 0 if pilot_pass else 2


def process_cases(
    case_dirs: list[Path], nifti_dir: Path, workers: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cases: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_case, case, nifti_dir): case for case in case_dirs}
        for future in as_completed(futures):
            case_dir = futures[future]
            try:
                case = future.result()
                cases.append(case)
                print(f"{case['status'].upper()} {case['patient_id']}", flush=True)
            except Exception as exc:
                rejected.append({
                    "patient_id": case_dir.name,
                    "status": "rejected",
                    "reason": str(exc),
                })
                print(f"REJECTED {case_dir.name}: {exc}", flush=True)
    cases.sort(key=lambda case: int(case["patient_id"].split(".", 1)[0]))
    rejected.sort(key=lambda case: int(case["patient_id"].split(".", 1)[0]))
    return cases, rejected


def population_run(
    case_dirs: list[Path],
    nifti_dir: Path,
    output: Path,
    workers: int,
    scale_normalized: bool,
    skip_patient_plots: bool,
) -> int:
    pilot_path = output / "quality_control" / "pilot_report.json"
    if not pilot_path.is_file() or not json.loads(pilot_path.read_text(encoding="utf-8"))["pilot_pass"]:
        raise SystemExit("population run requires a passing quality_control/pilot_report.json")
    cases, rejected = process_cases(case_dirs, nifti_dir, workers)
    for case in cases:
        save_case_artifacts(case, output)
        if not skip_patient_plots:
            save_case_plot(
                case, output / "visualizations" / "per_patient" / f"{case['patient_id']}_front.png"
            )
    write_index_files(cases, rejected, output)
    model_cases = [
        case for case in cases if case["status"] in {"accepted", "accepted_with_warnings"}
    ]
    if len(model_cases) < 3:
        raise SystemExit("fewer than three accepted/warning cases remain for PCA")
    initial_shapes = np.asarray([landmark_matrix(case) for case in model_cases])
    aligned_shapes, mean_shape, rotations, scales, history = generalized_procrustes(
        initial_shapes, scale_normalized=scale_normalized
    )
    for case, shape, rotation, scale in zip(model_cases, aligned_shapes, rotations, scales):
        patient_dir = output / "aligned_cases" / case["patient_id"]
        combined_rotation = case["frame"]["ras_to_local_row_rotation"] @ rotation
        final_centerlines = {
            name: transform_points(case["branches_original"][name], case["origin_ras_mm"], combined_rotation)
            / scale
            for name in BRANCHES
        }
        np.savez_compressed(patient_dir / "gpa_aligned_centerlines.npz", **final_centerlines)
        np.save(patient_dir / "gpa_aligned_landmarks.npy", shape)
        write_json(patient_dir / "gpa_transform.json", {
            "origin_ras_mm": case["origin_ras_mm"],
            "ras_to_initial_local_rotation": case["frame"]["ras_to_local_row_rotation"],
            "gpa_rotation": rotation,
            "combined_ras_to_gpa_rotation": combined_rotation,
            "inverse_gpa_to_ras_rotation": combined_rotation.T,
            "scale": scale,
            "scale_normalized": scale_normalized,
        })
    shape_matrix = aligned_shapes.reshape(len(aligned_shapes), -1)
    model = pca_model(shape_matrix)
    pca_dir = output / "pca_ssm"
    pca_dir.mkdir(parents=True, exist_ok=True)
    np.save(pca_dir / "shape_matrix.npy", shape_matrix)
    np.save(pca_dir / "mean_shape.npy", mean_shape)
    np.save(pca_dir / "pca_components.npy", model["components"])
    np.save(pca_dir / "pca_eigenvalues.npy", model["eigenvalues"])
    np.savez_compressed(
        pca_dir / "ssm_model.npz",
        shape_matrix=shape_matrix,
        mean_shape=mean_shape,
        mean_vector=model["mean_vector"],
        components=model["components"],
        eigenvalues=model["eigenvalues"],
        explained_variance_ratio=model["explained_variance_ratio"],
        cumulative_explained_variance=model["cumulative_explained_variance"],
        patient_ids=np.asarray([case["patient_id"] for case in model_cases]),
        landmark_counts=np.asarray([LANDMARK_COUNTS[name] for name in LANDMARK_ORDER]),
        scale_normalized=np.asarray(scale_normalized),
    )
    explained = {
        "component_count": int(len(model["eigenvalues"])),
        "eigenvalues": model["eigenvalues"],
        "explained_variance_ratio": model["explained_variance_ratio"],
        "cumulative_explained_variance": model["cumulative_explained_variance"],
        "components_for_90_percent": int(
            np.searchsorted(model["cumulative_explained_variance"], 0.90) + 1
        ),
        "components_for_95_percent": int(
            np.searchsorted(model["cumulative_explained_variance"], 0.95) + 1
        ),
        "components_for_99_percent": int(
            np.searchsorted(model["cumulative_explained_variance"], 0.99) + 1
        ),
    }
    write_json(pca_dir / "explained_variance.json", explained)
    with (pca_dir / "patient_shape_coefficients.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["patient_id"] + [f"pc_{index + 1}" for index in range(len(model["eigenvalues"]))]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case, coefficients in zip(model_cases, model["coefficients"]):
            writer.writerow({"patient_id": case["patient_id"], **{
                f"pc_{index + 1}": value for index, value in enumerate(coefficients)
            }})
    pre_landmark_distances = np.linalg.norm(
        initial_shapes - initial_shapes.mean(axis=0), axis=2
    )
    post_landmark_distances = np.linalg.norm(
        aligned_shapes - mean_shape, axis=2
    )
    per_patient_alignment = [
        {
            "patient_id": case["patient_id"],
            "pre_alignment_mean_distance_mm": float(pre.mean()),
            "pre_alignment_rms_distance_mm": float(np.sqrt(np.mean(pre ** 2))),
            "post_alignment_mean_distance_mm": float(post.mean()),
            "post_alignment_rms_distance_mm": float(np.sqrt(np.mean(post ** 2))),
        }
        for case, pre, post in zip(model_cases, pre_landmark_distances, post_landmark_distances)
    ]
    alignment_report = {
        "method": "rigid generalized Procrustes after LMCA/inferior local-frame initialization",
        "physical_size_retained": not scale_normalized,
        "scale_normalized": scale_normalized,
        "iteration_count": len(history),
        "convergence_history": history,
        "pre_alignment_mean_distance_to_mean_mm": float(pre_landmark_distances.mean()),
        "post_alignment_mean_distance_to_mean_mm": float(post_landmark_distances.mean()),
        "per_patient": per_patient_alignment,
    }
    write_json(output / "aligned_cases" / "alignment_report.json", alignment_report)
    parameters = parameter_statistics(model_cases)
    stats_dir = output / "population_statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    np.save(stats_dir / "parameter_matrix.npy", parameters["matrix"])
    np.save(stats_dir / "parameter_covariance.npy", parameters["covariance"])
    write_json(stats_dir / "population_statistics.json", {
        "case_count": len(model_cases),
        "parameter_names": parameters["parameter_names"],
        "statistics": parameters["statistics"],
        "covariance": parameters["covariance"],
        "correlation": parameters["correlation"],
        "warning": "ellipse tilt is summarized linearly; production circular statistics remain a limitation",
    })
    save_population_overlay(
        aligned_shapes, mean_shape,
        output / "visualizations" / "aligned_population_overlay.png",
        f"Aligned LCA landmark population (n={len(model_cases)})",
    )
    save_population_overlay(
        mean_shape[None, :, :], mean_shape,
        output / "visualizations" / "mean_shape.png",
        "Mean 27-landmark LCA shape",
    )
    save_pca_plots(model, output / "visualizations")
    rca_count = sum(case["rca_metadata"]["available"] for case in cases)
    lcx_only_count = sum(
        case["planes"]["coronary"]["plane_source"] == "lcx_only_approximation"
        for case in cases
    )
    status_counts = {
        status: sum(case["status"] == status for case in cases)
        for status in ("accepted", "accepted_with_warnings", "manual_review")
    }
    status_counts["rejected"] = len(rejected)
    manifest = {
        "version": VERSION,
        "total_source_cases": len(case_dirs),
        "successfully_extracted_cases": len(cases),
        "status_counts": status_counts,
        "model_training_case_count": len(model_cases),
        "rca_inferred_available_count": rca_count,
        "lcx_only_plane_approximation_count": lcx_only_count,
        "fixed_landmark_schema": LANDMARK_COUNTS,
        "alignment": alignment_report,
        "pca": explained,
        "protected_modules_used": False,
    }
    write_json(output / "run_manifest_v3.json", manifest)
    write_reports(output, manifest, parameters["statistics"])
    print(json.dumps(jsonable(manifest), indent=2))
    return 0


def write_reports(output: Path, manifest: dict[str, Any], stats: dict[str, Any]) -> None:
    reports = {
        "LCA_SSM_IMPLEMENTATION_REPORT.md": f"""# LCA SSM implementation report

This is a dataset-derived statistical/anatomical LCA shape model, not a clinical-grade model.

- Source cases: {manifest['total_source_cases']}
- Successfully extracted: {manifest['successfully_extracted_cases']}
- Training cases: {manifest['model_training_case_count']}
- Fixed landmarks: LMCA 5, LAD 12, LCX 10 (27 total)
- Alignment: {manifest['alignment']['method']}
- Physical size retained: {manifest['alignment']['physical_size_retained']}
- Source centerline geometry is never moved non-rigidly; planes and ellipses are diagnostic/statistical references.
- PCA components: {manifest['pca']['component_count']}
- Components for 95% variance: {manifest['pca']['components_for_95_percent']}
""",
        "LCA_SSM_DATA_QUALITY_REPORT.md": f"""# LCA SSM data quality report

- Status counts: `{json.dumps(manifest['status_counts'], sort_keys=True)}`
- Inferred RCA candidates: {manifest['rca_inferred_available_count']}
- LCX-only coronary-plane approximations: {manifest['lcx_only_plane_approximation_count']}
- RCA paths are inferred from disconnected components and are not annotated ground truth.
- Manual-review cases are excluded from PCA training.
""",
        "LCA_SSM_POPULATION_STATISTICS.md": "# LCA SSM population statistics\n\n" + "\n".join(
            f"- {name}: mean {values['mean']:.4g}, SD {values['std']:.4g}, "
            f"range [{values['min']:.4g}, {values['max']:.4g}]"
            for name, values in stats.items()
        ) + "\n",
        "LCA_SSM_PCA_REPORT.md": f"""# LCA SSM PCA report

- Components retained: {manifest['pca']['component_count']}
- 90% variance: {manifest['pca']['components_for_90_percent']} components
- 95% variance: {manifest['pca']['components_for_95_percent']} components
- 99% variance: {manifest['pca']['components_for_99_percent']} components
- PCA is trained on fixed corresponding, rigidly aligned 27-landmark shapes.
""",
    }
    for filename, text in reports.items():
        (output / filename).write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    repo_root = project_dir.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "population"), default="pilot")
    parser.add_argument("--input-dir", type=Path, default=project_dir / "centerlines")
    parser.add_argument("--nifti-dir", type=Path, default=project_dir / "nii files")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "lca_ssm")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--scale-normalized", action="store_true",
                        help="Build a shape-only SSM; default retains physical size")
    parser.add_argument("--skip-patient-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    nifti_dir = args.nifti_dir.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for directory in MANAGED_DIRECTORIES:
        (output / directory).mkdir(parents=True, exist_ok=True)
    cases = discover_cases(input_dir)
    if args.mode == "pilot":
        return pilot_run(cases, nifti_dir, output, args.workers)
    return population_run(
        cases, nifti_dir, output, args.workers,
        args.scale_normalized, args.skip_patient_plots,
    )


if __name__ == "__main__":
    raise SystemExit(main())
