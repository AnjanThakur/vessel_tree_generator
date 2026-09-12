"""Build the post-landmark LCA statistical shape parameter dataset.

This stage is intentionally separate from vessel radius, disease, tube, mesh,
hub, and anatomical-generation code. It reads the PCA label volumes and their
unlabelled centerline graphs, extracts an LCA candidate, and writes derived SSM
artifacts under outputs/lca_ssm.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lca_ssm_curve_fitting import (
    cumulative_parameter,
    discrete_curve_metrics,
    fit_lcx_ellipse_arc,
    fit_quadratic_guide,
)
from lca_ssm_label_adapter import extract_lca_from_label_case
from lca_ssm_planes import (
    angle_degrees,
    as_points,
    fit_plane_svd,
    project_points_to_plane,
)


VERSION = "2.0.0"
BRANCHES = ("lmca", "lad", "lcx")
CONTROL_POINT_TARGETS = {"lmca": 5, "lad": 12, "lcx": 10}
CLASSIFICATION_REVIEW_MARGIN = 0.20
ROOT_RADIUS_REVIEW_MARGIN_MM = 0.05


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plane_payload(plane: dict) -> dict:
    keys = (
        "centroid", "normal", "basis_u", "basis_v", "singular_values", "rank",
        "point_residuals_mm", "rms_residual_mm", "mean_residual_mm", "max_residual_mm",
    )
    return {key: plane[key] for key in keys}


def curve_payload(curve: dict) -> dict:
    return {key: value for key, value in curve.items() if key != "fitted_points_2d"}


def process_patient(
    patient_dir: Path, nifti_dir: Path, maximum_junction_offset_mm: float
) -> tuple[dict, dict]:
    nifti_path = nifti_dir / f"{patient_dir.name}.nii.gz"
    branches, source_metadata = extract_lca_from_label_case(patient_dir, nifti_path)
    for branch, minimum in (("lmca", 2), ("lad", 5), ("lcx", 5)):
        as_points(branches[branch], name=branch.upper(), minimum=minimum)
    if source_metadata["maximum_junction_offset_mm"] > maximum_junction_offset_mm:
        raise ValueError(
            f"branch endpoints disagree by {source_metadata['maximum_junction_offset_mm']:.3f} mm "
            f"(limit {maximum_junction_offset_mm:.3f} mm)"
        )

    metrics = {name: discrete_curve_metrics(points) for name, points in branches.items()}
    lcx_plane = fit_plane_svd(branches["lcx"], name="LCX")
    lad_plane = fit_plane_svd(branches["lad"], name="LAD")
    lcx_2d, lcx_projected_3d = project_points_to_plane(branches["lcx"], lcx_plane)
    lad_2d, lad_projected_3d = project_points_to_plane(branches["lad"], lad_plane)
    lcx_arc = fit_lcx_ellipse_arc(lcx_2d)
    lad_guide = fit_quadratic_guide(lad_2d)
    lcx_path_positions, _ = cumulative_parameter(lcx_2d)
    lcx_control_positions = np.linspace(0.0, 1.0, CONTROL_POINT_TARGETS["lcx"])
    lcx_control_angles = np.interp(
        lcx_control_positions, lcx_path_positions, lcx_arc["angular_positions_rad"]
    )
    lad_control_positions = np.linspace(0.0, 1.0, CONTROL_POINT_TARGETS["lad"])

    plane_angle = angle_degrees(lcx_plane["normal"], lad_plane["normal"])
    plane_angle = min(plane_angle, 180.0 - plane_angle)
    bifurcation_angle = angle_degrees(
        metrics["lad"]["initial_tangent"], metrics["lcx"]["initial_tangent"]
    )
    lmca_forward = metrics["lmca"]["distal_tangent"]
    lmca_to_lad = angle_degrees(lmca_forward, metrics["lad"]["initial_tangent"])
    lmca_to_lcx = angle_degrees(lmca_forward, metrics["lcx"]["initial_tangent"])
    lad_reference_axis = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)
    lcx_reference_axis = np.array([-1.0, 0.0, 0.0])
    lad_descent_score = float(np.dot(metrics["lad"]["initial_tangent"], lad_reference_axis))
    lcx_circumflex_score = float(np.dot(metrics["lcx"]["initial_tangent"], lcx_reference_axis))
    assignment_margin = source_metadata["daughter_classification"]["assignment_margin"]
    root_radius_margin = source_metadata["root_selection"]["radius_margin_mm"]
    warnings = []
    if assignment_margin < CLASSIFICATION_REVIEW_MARGIN:
        warnings.append(
            f"low LAD/LCX assignment margin ({assignment_margin:.3f} < {CLASSIFICATION_REVIEW_MARGIN:.3f})"
        )
    if root_radius_margin < ROOT_RADIUS_REVIEW_MARGIN_MM:
        warnings.append(
            f"small endpoint-radius root margin ({root_radius_margin:.3f} < {ROOT_RADIUS_REVIEW_MARGIN_MM:.3f} mm)"
        )

    parameters = {
        "schema_version": VERSION,
        "patient_id": patient_dir.name,
        "status": "accepted",
        "source": {
            "dataset": "pca_ssm_vessel_tree_generator 200-case PCA label cohort",
            "directory": patient_dir.name,
            "coordinate_units": "mm",
            "coordinate_system": "NIfTI sform RAS",
            "branch_labels": "LCA topology extracted from graph; LAD/LCX identities anatomically inferred",
            "extraction_metadata": source_metadata,
        },
        "quality": {
            "point_counts": {name: len(points) for name, points in branches.items()},
            "junction": {
                "junction_coordinate_mm": source_metadata["junction_coordinate_mm"],
                "branch_junction_offsets_mm": source_metadata["branch_junction_offsets_mm"],
                "maximum_junction_offset_mm": source_metadata["maximum_junction_offset_mm"],
            },
            "daughter_assignment_margin": assignment_margin,
            "root_radius_margin_mm": root_radius_margin,
            "review_recommended": bool(warnings),
            "warnings": warnings,
            "validation_checks": [
                "finite_coordinates", "nonzero_branch_lengths", "shared_junction",
                "radius_selected_root", "stage_2_bifurcation_thresholds",
                "ras_based_daughter_classification", "noncollinear_lad_plane",
                "noncollinear_lcx_plane", "finite_curve_fits",
            ],
        },
        "landmarks": {
            "ostium_mm": branches["lmca"][0],
            "bifurcation_mm": source_metadata["junction_coordinate_mm"],
            "lad_terminal_mm": branches["lad"][-1],
            "lcx_terminal_mm": branches["lcx"][-1],
            "lcx_curve_angular_positions_rad": lcx_arc["angular_positions_rad"],
            "lad_curve_parameter_positions": lad_guide["parameter_positions"],
            "generator_control_positions": {
                "lcx_path_fractions": lcx_control_positions,
                "lcx_angular_positions_rad": lcx_control_angles,
                "lad_guide_parameters": lad_control_positions,
            },
        },
        "branches": metrics,
        "ratios": {
            "lad_to_lmca_length": metrics["lad"]["length_mm"] / metrics["lmca"]["length_mm"],
            "lcx_to_lmca_length": metrics["lcx"]["length_mm"] / metrics["lmca"]["length_mm"],
            "lad_to_lcx_length": metrics["lad"]["length_mm"] / metrics["lcx"]["length_mm"],
        },
        "angles_deg": {
            "lad_lcx_bifurcation": bifurcation_angle,
            "lmca_to_lad_continuity": lmca_to_lad,
            "lmca_to_lcx_continuity": lmca_to_lcx,
            "lad_lcx_plane": plane_angle,
        },
        "anatomy_scores": {
            "lad_initial_descent_score": lad_descent_score,
            "lcx_initial_circumflex_score": lcx_circumflex_score,
            "lad_reference_axis_ras": lad_reference_axis,
            "lcx_reference_axis_ras": lcx_reference_axis,
            "score_definition": (
                "cosine alignment in the common NIfTI RAS frame: LAD with the anterior-inferior "
                "axis and LCX with the patient-left axis"
            ),
        },
        "planes": {"lcx_coronary_av": plane_payload(lcx_plane), "lad_interventricular": plane_payload(lad_plane)},
        "plane_summary": {
            "coronary_plane_centroid": lcx_plane["centroid"],
            "coronary_plane_normal": lcx_plane["normal"],
            "coronary_plane_error": lcx_plane["rms_residual_mm"],
            "interventricular_plane_centroid": lad_plane["centroid"],
            "interventricular_plane_normal": lad_plane["normal"],
            "interventricular_plane_error": lad_plane["rms_residual_mm"],
            "error_units": "mm RMS point-to-plane distance",
        },
        "lcx_arc": curve_payload(lcx_arc),
        "lad_guide": curve_payload(lad_guide),
    }
    if not all(np.isfinite(value) for value in (
        bifurcation_angle, plane_angle, lcx_arc["rms_residual_mm"], lad_guide["rms_residual_mm"]
    )):
        raise ValueError("one or more derived parameters are not finite")
    plot_data = {
        "branches": branches,
        "lcx_plane": lcx_plane,
        "lad_plane": lad_plane,
        "lcx_2d": lcx_2d,
        "lad_2d": lad_2d,
        "lcx_projected_3d": lcx_projected_3d,
        "lad_projected_3d": lad_projected_3d,
        "lcx_fitted_2d": lcx_arc["fitted_points_2d"],
        "lad_fitted_2d": lad_guide["fitted_points_2d"],
    }
    return parameters, plot_data


def set_axes_equal(axis: Any, points: np.ndarray) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    radius = max(float((maximum - minimum).max()) / 2.0, 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def add_plane_patch(axis: Any, plane: dict, extent: float, color: str) -> None:
    grid = np.linspace(-extent, extent, 9)
    uu, vv = np.meshgrid(grid, grid)
    points = (
        plane["centroid"][None, None, :]
        + uu[:, :, None] * plane["basis_u"][None, None, :]
        + vv[:, :, None] * plane["basis_v"][None, None, :]
    )
    axis.plot_surface(points[:, :, 0], points[:, :, 1], points[:, :, 2], color=color, alpha=0.12)


def save_patient_plots(patient_id: str, data: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"lmca": "#333333", "lad": "#d62728", "lcx": "#1f77b4"}
    all_points = np.vstack(list(data["branches"].values()))
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    for name, points in data["branches"].items():
        axis.plot(*points.T, color=colors[name], linewidth=2.0, label=name.upper())
    extent = max(float(np.ptp(all_points, axis=0).max()) * 0.35, 2.0)
    add_plane_patch(axis, data["lcx_plane"], extent, colors["lcx"])
    add_plane_patch(axis, data["lad_plane"], extent, colors["lad"])
    axis.set_title(f"{patient_id}: fitted LCX and LAD planes")
    axis.set_xlabel("X (mm)"); axis.set_ylabel("Y (mm)"); axis.set_zlabel("Z (mm)")
    axis.legend(loc="best")
    set_axes_equal(axis, all_points)
    figure.tight_layout()
    figure.savefig(output_dir / f"{patient_id}_planes_3d.png", dpi=160)
    plt.close(figure)

    for branch, raw, fitted, color, title in (
        ("lcx", data["lcx_2d"], data["lcx_fitted_2d"], colors["lcx"], "LCX ellipse/arc fit"),
        ("lad", data["lad_2d"], data["lad_fitted_2d"], colors["lad"], "LAD guide-curve fit"),
    ):
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.plot(raw[:, 0], raw[:, 1], "o", ms=2.5, alpha=0.65, color=color, label="projected centerline")
        axis.plot(fitted[:, 0], fitted[:, 1], color="#111111", lw=2.0, label="fit")
        axis.scatter(raw[0, 0], raw[0, 1], marker="s", color="#2ca02c", label="bifurcation")
        axis.scatter(raw[-1, 0], raw[-1, 1], marker="x", color="#ff7f0e", label="terminal")
        axis.set_aspect("equal", adjustable="datalim")
        axis.set_xlabel("plane u (mm)"); axis.set_ylabel("plane v (mm)")
        axis.set_title(f"{patient_id}: {title}"); axis.grid(alpha=0.2); axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / f"{patient_id}_{branch}_fit.png", dpi=160)
        plt.close(figure)


SCALAR_PARAMETERS = {
    "lmca_length_mm": ("branches", "lmca", "length_mm"),
    "lad_length_mm": ("branches", "lad", "length_mm"),
    "lcx_length_mm": ("branches", "lcx", "length_mm"),
    "lmca_tortuosity": ("branches", "lmca", "tortuosity"),
    "lad_tortuosity": ("branches", "lad", "tortuosity"),
    "lcx_tortuosity": ("branches", "lcx", "tortuosity"),
    "lad_mean_curvature_per_mm": ("branches", "lad", "mean_curvature_per_mm"),
    "lcx_mean_curvature_per_mm": ("branches", "lcx", "mean_curvature_per_mm"),
    "lad_to_lmca_length_ratio": ("ratios", "lad_to_lmca_length"),
    "lcx_to_lmca_length_ratio": ("ratios", "lcx_to_lmca_length"),
    "lad_to_lcx_length_ratio": ("ratios", "lad_to_lcx_length"),
    "bifurcation_angle_deg": ("angles_deg", "lad_lcx_bifurcation"),
    "lmca_to_lad_continuity_deg": ("angles_deg", "lmca_to_lad_continuity"),
    "lmca_to_lcx_continuity_deg": ("angles_deg", "lmca_to_lcx_continuity"),
    "plane_angle_deg": ("angles_deg", "lad_lcx_plane"),
    "lad_initial_descent_score": ("anatomy_scores", "lad_initial_descent_score"),
    "lcx_initial_circumflex_score": ("anatomy_scores", "lcx_initial_circumflex_score"),
    "lcx_plane_rms_mm": ("planes", "lcx_coronary_av", "rms_residual_mm"),
    "lad_plane_rms_mm": ("planes", "lad_interventricular", "rms_residual_mm"),
    "lcx_semi_major_axis_mm": ("lcx_arc", "semi_major_axis_mm"),
    "lcx_semi_minor_axis_mm": ("lcx_arc", "semi_minor_axis_mm"),
    "lcx_axis_ratio": ("lcx_arc", "axis_ratio"),
    "lcx_arc_extent_deg": ("lcx_arc", "angular_extent_deg"),
    "lcx_arc_start_rad": ("lcx_arc", "arc_start_rad"),
    "lcx_arc_end_rad": ("lcx_arc", "arc_end_rad"),
    "lcx_ellipse_orientation_deg": ("lcx_arc", "orientation_deg"),
    "lcx_arc_length_mm": ("lcx_arc", "arc_length_mm"),
    "lcx_arc_rms_mm": ("lcx_arc", "rms_residual_mm"),
    "lad_guide_length_mm": ("lad_guide", "guide_length_mm"),
    "lad_guide_rms_mm": ("lad_guide", "rms_residual_mm"),
}

VECTOR_PARAMETERS = {
    "lad_overall_direction": ("branches", "lad", "overall_direction"),
    "lcx_overall_direction": ("branches", "lcx", "overall_direction"),
    "lad_initial_tangent": ("branches", "lad", "initial_tangent"),
    "lcx_initial_tangent": ("branches", "lcx", "initial_tangent"),
    "lad_plane_normal": ("planes", "lad_interventricular", "normal"),
    "lcx_plane_normal": ("planes", "lcx_coronary_av", "normal"),
}

PROFILE_PARAMETERS = {
    "lcx_control_angular_positions_rad": (
        "landmarks", "generator_control_positions", "lcx_angular_positions_rad"
    ),
    "lad_control_guide_parameters": (
        "landmarks", "generator_control_positions", "lad_guide_parameters"
    ),
}

POOLED_NOISE_PARAMETERS = {
    "lad_curve_deviation_mm": ("lad_guide", "point_residuals_mm"),
    "lcx_curve_deviation_mm": ("lcx_arc", "point_residuals_mm"),
    "lad_plane_deviation_mm": ("planes", "lad_interventricular", "point_residuals_mm"),
    "lcx_plane_deviation_mm": ("planes", "lcx_coronary_av", "point_residuals_mm"),
}


def get_nested(payload: dict, path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        value = value[key]
    return value


def describe(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        return {
            "count": int(len(values)), "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)), "p05": float(np.percentile(values, 5)),
            "p25": float(np.percentile(values, 25)), "median": float(np.median(values)),
            "p75": float(np.percentile(values, 75)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return {
        "count": int(len(values)), "components": [describe(values[:, index]) for index in range(values.shape[1])]
    }


def build_statistics(patients: list[dict]) -> dict:
    assignment_margins = np.array([
        patient["quality"]["daughter_assignment_margin"] for patient in patients
    ])
    root_margins = np.array([
        patient["quality"]["root_radius_margin_mm"] for patient in patients
    ])
    return {
        "schema_version": VERSION,
        "accepted_patient_count": len(patients),
        "scalar_parameters": {
            name: describe(np.array([get_nested(patient, path) for patient in patients]))
            for name, path in SCALAR_PARAMETERS.items()
        },
        "vector_parameters": {
            name: describe(np.array([get_nested(patient, path) for patient in patients]))
            for name, path in VECTOR_PARAMETERS.items()
        },
        "control_position_profiles": {
            name: describe(np.array([get_nested(patient, path) for patient in patients]))
            for name, path in PROFILE_PARAMETERS.items()
        },
        "pooled_deviation_noise": {
            name: describe(np.concatenate([
                np.asarray(get_nested(patient, path), dtype=float) for patient in patients
            ]))
            for name, path in POOLED_NOISE_PARAMETERS.items()
        },
        "source_quality": {
            "daughter_assignment_margin": describe(assignment_margins),
            "root_radius_margin_mm": describe(root_margins),
            "classification_review_threshold": CLASSIFICATION_REVIEW_MARGIN,
            "root_review_threshold_mm": ROOT_RADIUS_REVIEW_MARGIN_MM,
            "review_recommended_count": int(sum(patient["quality"]["review_recommended"] for patient in patients)),
            "review_recommended_patient_ids": [
                patient["patient_id"] for patient in patients if patient["quality"]["review_recommended"]
            ],
            "lad_source_counts": {
                source: sum(
                    patient["source"]["extraction_metadata"]["daughter_classification"]["selected_lad_source"] == source
                    for patient in patients
                )
                for source in ("branch_a", "branch_b")
            },
        },
    }


def write_statistics_csv(path: Path, statistics: dict) -> None:
    fields = ("parameter", "component", "count", "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, stats in statistics["scalar_parameters"].items():
            writer.writerow({"parameter": name, "component": "scalar", **stats})
        labels = ("x", "y", "z")
        for name, vector in statistics["vector_parameters"].items():
            for label, stats in zip(labels, vector["components"]):
                writer.writerow({"parameter": name, "component": label, **stats})
        for name, profile in statistics["control_position_profiles"].items():
            for index, stats in enumerate(profile["components"]):
                writer.writerow({"parameter": name, "component": str(index), **stats})
        for name, stats in statistics["pooled_deviation_noise"].items():
            writer.writerow({"parameter": name, "component": "pooled_points", **stats})
        for name in ("daughter_assignment_margin", "root_radius_margin_mm"):
            writer.writerow({"parameter": name, "component": "source_quality", **statistics["source_quality"][name]})


def build_generator_schema(statistics: dict) -> dict:
    distributions = {}
    for name, stats in statistics["scalar_parameters"].items():
        distributions[name] = {
            "distribution": "empirical_or_truncated_normal",
            "mean": stats["mean"], "standard_deviation": stats["std"],
            "observed_range": [stats["min"], stats["max"]],
            "recommended_sampling_range": [stats["p05"], stats["p95"]],
        }
    for name, stats in statistics["vector_parameters"].items():
        distributions[name] = {
            "distribution": "componentwise_empirical_then_unit_normalize",
            "components": stats["components"],
        }
    for name, stats in statistics["control_position_profiles"].items():
        distributions[name] = {
            "distribution": "componentwise_empirical_control_positions",
            "components": stats["components"],
        }
    for name, stats in statistics["pooled_deviation_noise"].items():
        distributions[name] = {
            "distribution": "zero_centered_empirical_magnitude",
            "statistics": stats,
            "recommended_sampling_range_mm": [0.0, stats["p95"]],
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "LCA geometry generator parameter schema",
        "schema_version": VERSION,
        "coordinate_units": "millimetres",
        "topology": "LMCA bifurcating into LAD and LCX",
        "control_point_targets": CONTROL_POINT_TARGETS,
        "training_population_size": statistics["accepted_patient_count"],
        "training_data_quality": statistics["source_quality"],
        "sampling_warning": (
            "LAD/LCX daughter identities are anatomically inferred from the common NIfTI RAS frame, "
            "not manually annotated ground truth. Preserve cross-parameter covariance in a future PCA "
            "model; independent sampling can create implausible anatomy. Unit vectors must be renormalized."
        ),
        "parameter_distributions": distributions,
        "future_extension_points": [
            "resampled control-point PCA after rigid/scale alignment",
            "covariance-aware latent sampling",
            "manual or metadata-based LAD/LCX validation for low-margin cases",
            "explicit patient heart axes beyond the shared scan RAS frame",
        ],
    }


def save_summary_plot(patients: list[dict], output_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = (
        ("LAD length (mm)", ("branches", "lad", "length_mm")),
        ("LCX length (mm)", ("branches", "lcx", "length_mm")),
        ("LAD-LCX bifurcation angle (deg)", ("angles_deg", "lad_lcx_bifurcation")),
        ("Plane RMS residual (mm)", None),
    )
    for axis, (title, path) in zip(axes.flat, panels):
        if path is not None:
            values = [get_nested(patient, path) for patient in patients]
            axis.hist(values, bins=min(20, max(7, int(np.sqrt(len(values))))), color="#4c78a8", edgecolor="white")
        else:
            lcx = [patient["planes"]["lcx_coronary_av"]["rms_residual_mm"] for patient in patients]
            lad = [patient["planes"]["lad_interventricular"]["rms_residual_mm"] for patient in patients]
            bins = min(20, max(7, int(np.sqrt(len(patients)))))
            axis.hist(lad, bins=bins, alpha=0.65, label="LAD", color="#d62728")
            axis.hist(lcx, bins=bins, alpha=0.65, label="LCX", color="#1f77b4")
            axis.legend()
        axis.set_title(title); axis.set_ylabel("patient count"); axis.grid(axis="y", alpha=0.2)
    figure.suptitle(f"LCA SSM PCA-label cohort, inferred daughters (n={len(patients)})")
    figure.tight_layout()
    figure.savefig(output_dir / "population_parameter_summary.png", dpi=170)
    plt.close(figure)


def discover_patients(input_dir: Path) -> list[Path]:
    patients = [
        directory for directory in input_dir.iterdir()
        if directory.is_dir()
        and (directory / "summary.csv").is_file()
        and (directory / "branches.npz").is_file()
    ]
    return sorted(
        patients,
        key=lambda directory: (
            int(directory.name.split(".", 1)[0])
            if directory.name.split(".", 1)[0].isdigit()
            else float("inf"),
            directory.name,
        ),
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=project_dir / "centerlines")
    parser.add_argument("--nifti-dir", type=Path, default=project_dir / "nii files")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "outputs" / "lca_ssm")
    parser.add_argument("--maximum-junction-offset-mm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--max-patients", type=int, help="Process only the first N cases (for smoke tests)")
    parser.add_argument("--skip-visualizations", action="store_true", help="Skip plots during a smoke test")
    parser.add_argument("--clean", action="store_true", help="Remove only the selected lca_ssm output directory before rebuilding")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    nifti_dir = args.nifti_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"input directory does not exist: {input_dir}")
    if not nifti_dir.is_dir():
        raise SystemExit(f"NIfTI directory does not exist: {nifti_dir}")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    patient_output = output_dir / "patients"
    visualization_output = output_dir / "visualizations"
    patient_output.mkdir(parents=True, exist_ok=True)
    visualization_output.mkdir(parents=True, exist_ok=True)

    patient_dirs = discover_patients(input_dir)
    if args.max_patients is not None:
        patient_dirs = patient_dirs[:args.max_patients]
    accepted: list[dict] = []
    rejected: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_patient, patient_dir, nifti_dir, args.maximum_junction_offset_mm
            ): patient_dir
            for patient_dir in patient_dirs
        }
        completed = as_completed(futures)
        for future in completed:
            patient_dir = futures[future]
            try:
                parameters, plot_data = future.result()
                accepted.append(parameters)
                write_json(patient_output / patient_dir.name / "ssm_parameters.json", parameters)
                if not args.skip_visualizations:
                    save_patient_plots(patient_dir.name, plot_data, visualization_output)
                print(f"ACCEPT {patient_dir.name}", flush=True)
            except Exception as exc:
                rejection = {"patient_id": patient_dir.name, "status": "rejected", "reason": str(exc)}
                rejected.append(rejection)
                write_json(patient_output / patient_dir.name / "rejection.json", rejection)
                print(f"REJECT {patient_dir.name}: {exc}", flush=True)

    if not accepted:
        raise SystemExit("no valid patients remained after validation")
    accepted.sort(key=lambda patient: patient["patient_id"])
    rejected.sort(key=lambda patient: patient["patient_id"])
    statistics = build_statistics(accepted)
    write_json(output_dir / "ssm_population_statistics.json", statistics)
    write_statistics_csv(output_dir / "ssm_population_statistics.csv", statistics)
    write_json(output_dir / "ssm_generator_schema.json", build_generator_schema(statistics))
    valid_index = {"count": len(accepted), "patient_ids": [p["patient_id"] for p in accepted]}
    rejected_index = {"count": len(rejected), "patients": rejected}
    write_json(output_dir / "valid_patient_index.json", valid_index)
    write_json(output_dir / "rejected_patient_index.json", rejected_index)
    if not args.skip_visualizations:
        save_summary_plot(accepted, visualization_output)
    write_json(output_dir / "run_manifest.json", {
        "schema_version": VERSION, "input_directory": input_dir, "nifti_directory": nifti_dir,
        "output_directory": output_dir, "discovered_case_count": len(discover_patients(input_dir)),
        "processed_case_count": len(patient_dirs),
        "accepted_count": len(accepted), "rejected_count": len(rejected),
        "accepted_patient_ids": [patient["patient_id"] for patient in accepted],
        "rejected_patients": rejected,
    })
    print(f"Completed: {len(accepted)} accepted, {len(rejected)} rejected -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
