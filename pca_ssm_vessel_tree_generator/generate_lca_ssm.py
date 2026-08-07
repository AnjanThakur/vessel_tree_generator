"""Generate and validate LCA trees from the dataset-derived landmark SSM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


EPS = 1.0e-12
COUNTS = {"lmca": 5, "lad": 12, "lcx": 10}
ORDER = ("lmca", "lad", "lcx")
SLICES = {
    "lmca": slice(0, 5),
    "lad": slice(5, 17),
    "lcx": slice(17, 27),
}
SHARED = (4, 5, 17)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize(vector: np.ndarray, name: str = "vector") -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPS or not math.isfinite(norm):
        raise ValueError(f"cannot normalize {name}")
    return np.asarray(vector, dtype=float) / norm


def angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(normalize(first), normalize(second)), -1.0, 1.0))))


def curve_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def split_controls(shape: np.ndarray) -> dict[str, np.ndarray]:
    points = np.asarray(shape, dtype=float).reshape(27, 3)
    return {name: points[section].copy() for name, section in SLICES.items()}


def combine_controls(branches: dict[str, np.ndarray]) -> np.ndarray:
    return np.vstack([branches[name] for name in ORDER])


def exact_shared_bifurcation(branches: dict[str, np.ndarray]) -> np.ndarray:
    shared = np.mean(np.vstack((branches["lmca"][-1], branches["lad"][0], branches["lcx"][0])), axis=0)
    branches["lmca"][-1] = shared
    branches["lad"][0] = shared
    branches["lcx"][0] = shared
    return shared


def rotation_matrix_xyz(degrees_xyz: tuple[float, float, float]) -> np.ndarray:
    x, y, z = np.radians(degrees_xyz)
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def apply_physical_transform(
    branches: dict[str, np.ndarray],
    scale: float,
    rotation_degrees: tuple[float, float, float],
    translation: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    if scale <= 0.0 or not math.isfinite(scale):
        raise ValueError("physical scale must be positive and finite")
    rotation = rotation_matrix_xyz(rotation_degrees)
    offset = np.asarray(translation, dtype=float)
    return {name: (points * scale) @ rotation.T + offset for name, points in branches.items()}


def interpolate_branch(points: np.ndarray, sample_count: int) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(segment_lengths <= EPS):
        raise ValueError("generated controls contain duplicate consecutive points")
    parameter = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    parameter /= parameter[-1]
    spline = make_interp_spline(parameter, points, k=min(3, len(points) - 1), axis=0)
    sampled = spline(np.linspace(0.0, 1.0, sample_count))
    sampled[0] = points[0]
    sampled[-1] = points[-1]
    return sampled


def interpolate_tree(branches: dict[str, np.ndarray], sample_count: int) -> dict[str, np.ndarray]:
    curves = {name: interpolate_branch(branches[name], sample_count) for name in ORDER}
    shared = branches["lmca"][-1]
    curves["lmca"][-1] = shared
    curves["lad"][0] = shared
    curves["lcx"][0] = shared
    return curves


def lift_plane_point(point_2d: np.ndarray, plane: dict[str, Any]) -> np.ndarray:
    return (
        np.asarray(plane["centroid"], dtype=float)
        + float(point_2d[0]) * np.asarray(plane["basis_u"], dtype=float)
        + float(point_2d[1]) * np.asarray(plane["basis_v"], dtype=float)
    )


def lift_plane_vector(vector_2d: np.ndarray, plane: dict[str, Any]) -> np.ndarray:
    return (
        float(vector_2d[0]) * np.asarray(plane["basis_u"], dtype=float)
        + float(vector_2d[1]) * np.asarray(plane["basis_v"], dtype=float)
    )


def load_ellipse_library(output: Path, model: Any) -> dict[str, Any]:
    cache_path = output / "population_statistics" / "parametric_ellipse_library.npz"
    if cache_path.is_file():
        with np.load(cache_path) as cached:
            return {
                "patient_ids": [str(item) for item in cached["patient_ids"]],
                "shapes": np.array(cached["shapes"]),
                "centers": {
                    "lad": np.array(cached["lad_centers"]),
                    "lcx": np.array(cached["lcx_centers"]),
                },
                "e1_vectors": {
                    "lad": np.array(cached["lad_e1"]),
                    "lcx": np.array(cached["lcx_e1"]),
                },
                "e2_vectors": {
                    "lad": np.array(cached["lad_e2"]),
                    "lcx": np.array(cached["lcx_e2"]),
                },
                "angles": {
                    "lad": np.array(cached["lad_angles"]),
                    "lcx": np.array(cached["lcx_angles"]),
                },
                "residuals": {
                    "lad": np.array(cached["lad_residuals"]),
                    "lcx": np.array(cached["lcx_residuals"]),
                },
                "scalar_parameters": {"lad": [], "lcx": []},
            }
    patient_ids = [str(item) for item in model["patient_ids"]]
    shapes = model["shape_matrix"].reshape(-1, 27, 3)
    centers = {"lad": [], "lcx": []}
    e1_vectors = {"lad": [], "lcx": []}
    e2_vectors = {"lad": [], "lcx": []}
    angles = {"lad": [], "lcx": []}
    residuals = {"lad": [], "lcx": []}
    scalar_parameters = {"lad": [], "lcx": []}
    for patient_id in patient_ids:
        planes = json.loads((output / "planes" / patient_id / "planes.json").read_text(encoding="utf-8"))
        references = json.loads(
            (output / "ellipse_references" / patient_id / "reference_parameters.json").read_text(encoding="utf-8")
        )
        arrays = np.load(output / "ellipse_references" / patient_id / "reference_arrays.npz")
        transform = json.loads(
            (output / "aligned_cases" / patient_id / "gpa_transform.json").read_text(encoding="utf-8")
        )
        origin = np.asarray(transform["origin_ras_mm"], dtype=float)
        rotation = np.asarray(transform["combined_ras_to_gpa_rotation"], dtype=float)
        scale = float(transform["scale"])
        for branch, plane_name in (("lad", "lad"), ("lcx", "coronary")):
            plane = planes[plane_name]
            ellipse = references[branch]["ellipse"]
            affine = np.asarray(ellipse["affine_matrix"], dtype=float)
            center_ras = lift_plane_point(np.asarray(ellipse["center_2d"], dtype=float), plane)
            e1_ras = lift_plane_vector(affine[:, 0], plane)
            e2_ras = lift_plane_vector(affine[:, 1], plane)
            centers[branch].append(((center_ras - origin) @ rotation) / scale)
            e1_vectors[branch].append((e1_ras @ rotation) / scale)
            e2_vectors[branch].append((e2_ras @ rotation) / scale)
            angles[branch].append(np.asarray(ellipse["angular_positions_rad"], dtype=float))
            residuals[branch].append(
                (np.asarray(arrays[f"{branch}_residual_vectors_ras"], dtype=float) @ rotation) / scale
            )
            scalar_parameters[branch].append({
                "semi_major_axis_mm": float(ellipse["semi_major_axis_mm"]) / scale,
                "semi_minor_axis_mm": float(ellipse["semi_minor_axis_mm"]) / scale,
                "signed_angular_extent_deg": float(ellipse["signed_angular_extent_deg"]),
                "arc_length_mm": float(ellipse["arc_length_mm"]) / scale,
            })
    library = {
        "patient_ids": patient_ids,
        "shapes": shapes,
        "centers": {key: np.asarray(value) for key, value in centers.items()},
        "e1_vectors": {key: np.asarray(value) for key, value in e1_vectors.items()},
        "e2_vectors": {key: np.asarray(value) for key, value in e2_vectors.items()},
        "angles": {key: np.asarray(value) for key, value in angles.items()},
        "residuals": {key: np.asarray(value) for key, value in residuals.items()},
        "scalar_parameters": scalar_parameters,
    }
    np.savez_compressed(
        cache_path,
        patient_ids=np.asarray(patient_ids),
        shapes=shapes,
        lad_centers=library["centers"]["lad"],
        lcx_centers=library["centers"]["lcx"],
        lad_e1=library["e1_vectors"]["lad"],
        lcx_e1=library["e1_vectors"]["lcx"],
        lad_e2=library["e2_vectors"]["lad"],
        lcx_e2=library["e2_vectors"]["lcx"],
        lad_angles=library["angles"]["lad"],
        lcx_angles=library["angles"]["lcx"],
        lad_residuals=library["residuals"]["lad"],
        lcx_residuals=library["residuals"]["lcx"],
    )
    return library


def pca_sample(model: Any, rng: np.random.Generator, sigma_scale: float, coefficient_limit: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    components = model["components"]
    coefficients_observed = (model["shape_matrix"] - model["mean_vector"]) @ components.T
    cumulative = model["cumulative_explained_variance"]
    retained = min(int(np.searchsorted(cumulative, 0.95) + 1), len(components))
    observed = coefficients_observed[:, :retained]
    means = observed.mean(axis=0)
    standard_deviations = observed.std(axis=0, ddof=1)
    coefficients = rng.normal(means, standard_deviations * sigma_scale)
    lower = np.maximum(observed.min(axis=0), means - coefficient_limit * standard_deviations)
    upper = np.minimum(observed.max(axis=0), means + coefficient_limit * standard_deviations)
    coefficients = np.clip(coefficients, lower, upper)
    full_coefficients = np.zeros(len(components))
    full_coefficients[:retained] = coefficients
    shape = model["mean_vector"] + full_coefficients @ components
    branches = split_controls(shape)
    shared_before = np.vstack((branches["lmca"][-1], branches["lad"][0], branches["lcx"][0]))
    exact_shared_bifurcation(branches)
    return branches, {
        "retained_component_count": retained,
        "target_cumulative_variance": 0.95,
        "coefficient_sigma_scale": sigma_scale,
        "coefficient_limit_standard_deviations": coefficient_limit,
        "coefficients": full_coefficients,
        "shared_bifurcation_pre_enforcement_max_error_mm": float(
            np.max(np.linalg.norm(shared_before - shared_before.mean(axis=0), axis=1))
        ),
    }


def aligned_parametric_template(library: dict[str, Any], branch: str, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        library["centers"][branch][index].copy(),
        library["e1_vectors"][branch][index].copy(),
        library["e2_vectors"][branch][index].copy(),
        library["angles"][branch][index].copy(),
    )


def parametric_sample(library: dict[str, Any], rng: np.random.Generator, maximum_residual_scale: float) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    count = len(library["patient_ids"])
    first, second, donor = rng.integers(0, count, size=3)
    blend = float(rng.uniform(0.15, 0.85))
    residual_scale = float(rng.uniform(0.0, maximum_residual_scale))
    branches = {
        "lmca": (1.0 - blend) * library["shapes"][first, SLICES["lmca"]] + blend * library["shapes"][second, SLICES["lmca"]]
    }
    sampled_ellipse_parameters = {}
    for branch in ("lad", "lcx"):
        center_a, e1_a, e2_a, angles_a = aligned_parametric_template(library, branch, first)
        center_b, e1_b, e2_b, angles_b = aligned_parametric_template(library, branch, second)
        if np.sign(angles_a[-1]) != np.sign(angles_b[-1]):
            angles_b *= -1.0
            e2_b *= -1.0
        center = (1.0 - blend) * center_a + blend * center_b
        e1 = (1.0 - blend) * e1_a + blend * e1_b
        e2 = (1.0 - blend) * e2_a + blend * e2_b
        branch_angles = (1.0 - blend) * angles_a + blend * angles_b
        reference = center + np.cos(branch_angles)[:, None] * e1 + np.sin(branch_angles)[:, None] * e2
        # The paired residual restores the measured off-plane/source deviation
        # associated with the jointly blended ellipse templates.  A separate,
        # mean-centered donor adds controlled population-derived variability.
        paired_residual = (
            (1.0 - blend) * library["residuals"][branch][first]
            + blend * library["residuals"][branch][second]
        )
        donor_noise = (
            library["residuals"][branch][donor]
            - library["residuals"][branch].mean(axis=0)
        )
        branches[branch] = reference + paired_residual + residual_scale * donor_noise
        axes = np.linalg.svd(np.column_stack((e1, e2)), compute_uv=False)
        sampled_ellipse_parameters[branch] = {
            "center_model_mm": center,
            "ellipse_vector_1_model_mm": e1,
            "ellipse_vector_2_model_mm": e2,
            "semi_major_axis_mm": float(axes[0]),
            "semi_minor_axis_mm": float(axes[1]),
            "axis_ratio": float(axes[0] / max(axes[1], EPS)),
            "landmark_angular_positions_rad": branch_angles,
            "signed_angular_extent_deg": float(np.degrees(branch_angles[-1] - branch_angles[0])),
            "residual_scale": residual_scale,
        }
    shared_before = np.vstack((branches["lmca"][-1], branches["lad"][0], branches["lcx"][0]))
    exact_shared_bifurcation(branches)
    return branches, {
        "sampling_method": "joint_empirical_pair_interpolation_with_population_residual_donor",
        "template_patient_a": library["patient_ids"][first],
        "template_patient_b": library["patient_ids"][second],
        "residual_donor_patient": library["patient_ids"][donor],
        "template_blend_weight_b": blend,
        "maximum_residual_scale": maximum_residual_scale,
        "sampled_residual_scale": residual_scale,
        "ellipse_parameters": sampled_ellipse_parameters,
        "shared_bifurcation_pre_enforcement_max_error_mm": float(
            np.max(np.linalg.norm(shared_before - shared_before.mean(axis=0), axis=1))
        ),
    }


def validation_axes(mean_shape: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    branches = split_controls(mean_shape)
    descent = normalize(branches["lad"][-1] - branches["lad"][0], "population LAD descent axis")
    lateral = branches["lcx"][-1] - branches["lcx"][0]
    lateral -= np.dot(lateral, descent) * descent
    lateral = normalize(lateral, "population LCX lateral axis")
    return descent, lateral


def shape_metrics(shape: np.ndarray, descent: np.ndarray, lateral: np.ndarray) -> dict[str, float]:
    branches = split_controls(shape)
    lmca_tangent = branches["lmca"][-1] - branches["lmca"][-2]
    lad_tangent = branches["lad"][1] - branches["lad"][0]
    lcx_tangent = branches["lcx"][1] - branches["lcx"][0]
    lad_terminal = branches["lad"][-1] - branches["lad"][0]
    lcx_terminal = branches["lcx"][-1] - branches["lcx"][0]
    lad_length = curve_length(branches["lad"])
    lcx_length = curve_length(branches["lcx"])
    return {
        "lmca_length_mm": curve_length(branches["lmca"]),
        "lad_length_mm": lad_length,
        "lcx_length_mm": lcx_length,
        "lad_lcx_length_ratio": lad_length / max(lcx_length, EPS),
        "lad_lcx_bifurcation_deg": angle_degrees(lad_tangent, lcx_tangent),
        "lmca_to_lad_deg": angle_degrees(lmca_tangent, lad_tangent),
        "lmca_to_lcx_deg": angle_degrees(lmca_tangent, lcx_tangent),
        "lad_initial_descent_cos": float(np.dot(normalize(lad_tangent), descent)),
        "lcx_initial_lateral_cos": float(np.dot(normalize(lcx_tangent), lateral)),
        "lad_terminal_descent_mm": float(np.dot(lad_terminal, descent)),
        "lcx_terminal_descent_mm": float(np.dot(lcx_terminal, descent)),
        "lcx_terminal_lateral_mm": float(np.dot(lcx_terminal, lateral)),
        "lad_minus_lcx_descent_mm": float(np.dot(lad_terminal - lcx_terminal, descent)),
    }


def empirical_validation_context(model: Any) -> dict[str, Any]:
    descent, lateral = validation_axes(model["mean_shape"])
    metric_rows = [shape_metrics(shape, descent, lateral) for shape in model["shape_matrix"].reshape(-1, 27, 3)]
    names = list(metric_rows[0])
    matrix = np.asarray([[row[name] for name in names] for row in metric_rows])
    thresholds = {
        name: {
            "minimum": float(matrix[:, index].min()),
            "p01": float(np.percentile(matrix[:, index], 1)),
            "p99": float(np.percentile(matrix[:, index], 99)),
            "maximum": float(matrix[:, index].max()),
        }
        for index, name in enumerate(names)
    }
    return {"descent_axis": descent, "lateral_axis": lateral, "thresholds": thresholds}


def has_self_near_intersection(points: np.ndarray, tolerance: float) -> bool:
    differences = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    row, column = np.indices(distances.shape)
    mask = np.abs(row - column) > 4
    return bool(np.any(distances[mask] < tolerance))


def validate_generated(
    branches: dict[str, np.ndarray],
    curves: dict[str, np.ndarray],
    context: dict[str, Any],
    mode: str,
    sampled: dict[str, Any],
    intersection_tolerance: float,
) -> dict[str, Any]:
    controls = combine_controls(branches)
    metrics = shape_metrics(controls, context["descent_axis"], context["lateral_axis"])
    failures = []
    warnings = []
    if not np.all(np.isfinite(controls)):
        failures.append("non-finite generated control points")
    shared = np.vstack((branches["lmca"][-1], branches["lad"][0], branches["lcx"][0]))
    shared_error = float(np.max(np.linalg.norm(shared - shared[0], axis=1)))
    if shared_error > 1.0e-9:
        failures.append("disconnected shared bifurcation")
    for name in ORDER:
        if np.any(np.linalg.norm(np.diff(branches[name], axis=0), axis=1) <= EPS):
            failures.append(f"{name}: duplicate generated control points")
        if has_self_near_intersection(curves[name], intersection_tolerance):
            failures.append(f"{name}: B-spline self-intersection/loop proximity")
    for name, value in metrics.items():
        limits = context["thresholds"][name]
        if value < limits["minimum"] - 1.0e-8 or value > limits["maximum"] + 1.0e-8:
            failures.append(f"{name}={value:.4g} outside empirical training range")
        elif value < limits["p01"] or value > limits["p99"]:
            warnings.append(f"{name}={value:.4g} outside empirical 1st-99th percentile band")
    if metrics["lad_initial_descent_cos"] <= 0.0:
        failures.append("LAD initial tangent does not point along the population descent direction")
    if metrics["lcx_initial_lateral_cos"] <= 0.0:
        failures.append("LCX initial tangent does not point along the population lateral direction")
    if metrics["lad_terminal_descent_mm"] <= 0.0:
        failures.append("LAD terminal does not reach the descending/apical half-space")
    if metrics["lcx_terminal_lateral_mm"] <= 0.0:
        failures.append("LCX terminal does not reach the lateral/circumflex half-space")
    if metrics["lad_minus_lcx_descent_mm"] <= 0.0:
        failures.append("LCX behaves as the dominant descending branch")
    coefficient_check = None
    if mode == "pca":
        coefficients = np.asarray(sampled["coefficients"], dtype=float)
        coefficient_check = bool(np.all(np.isfinite(coefficients)))
        if not coefficient_check:
            failures.append("non-finite PCA coefficients")
    return {
        "validation_pass": not failures,
        "classification": "passed_with_warnings" if not failures and warnings else ("passed" if not failures else "failed"),
        "threshold_source": "min/max and 1st-99th percentiles of the 187-case aligned training distribution",
        "intersection_tolerance_mm": intersection_tolerance,
        "shared_bifurcation_error_mm": shared_error,
        "metrics": metrics,
        "empirical_thresholds": context["thresholds"],
        "pca_coefficients_finite": coefficient_check,
        "failures": failures,
        "warnings": warnings,
    }


def save_tree_plot(branches: dict[str, np.ndarray], curves: dict[str, np.ndarray], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    colors = {"lmca": "#333333", "lad": "#d62728", "lcx": "#1f77b4"}
    for name in ORDER:
        curve = curves[name]
        control = branches[name]
        axis.plot(*curve.T, color=colors[name], linewidth=2.2, label=f"{name.upper()} B-spline")
        axis.scatter(*control.T, color=colors[name], s=22, edgecolor="white", linewidth=0.5)
    shared = branches["lmca"][-1]
    axis.scatter(*shared, color="#2ca02c", s=70, label="shared bifurcation")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.set_title(title)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    model_path = output / "pca_ssm" / "ssm_model.npz"
    if not model_path.is_file():
        raise SystemExit(f"missing population model: {model_path}")
    model = np.load(model_path)
    if bool(model["scale_normalized"]):
        raise SystemExit("generator currently requires the default physical-size-preserving SSM")
    library = load_ellipse_library(output, model)
    context = empirical_validation_context(model)
    rng = np.random.default_rng(args.seed)
    modes = ("pca", "parametric") if args.mode == "both" else (args.mode,)
    results = []
    for mode in modes:
        for index in range(args.count):
            tree_id = f"{mode}_{index + 1:03d}"
            rejected_attempts = []
            for attempt in range(1, args.max_sampling_attempts + 1):
                if mode == "pca":
                    branches, sampled = pca_sample(
                        model, rng, args.pca_sigma_scale, args.coefficient_limit
                    )
                else:
                    branches, sampled = parametric_sample(library, rng, args.residual_scale)
                exact_shared_bifurcation(branches)
                try:
                    model_space_curves = interpolate_tree(branches, args.spline_samples)
                    validation = validate_generated(
                        branches,
                        model_space_curves,
                        context,
                        mode,
                        sampled,
                        args.intersection_tolerance_mm,
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    validation = {
                        "validation_pass": False,
                        "classification": "failed",
                        "failures": [f"generation/interpolation error: {exc}"],
                        "warnings": [],
                    }
                if validation["validation_pass"]:
                    break
                rejected_attempts.append({
                    "attempt": attempt,
                    "failures": validation["failures"],
                })
            sampled["sampling_attempt_count"] = attempt
            sampled["rejected_sampling_attempts"] = rejected_attempts
            sampled["maximum_sampling_attempts"] = args.max_sampling_attempts
            validation["validation_coordinate_space"] = (
                "size-preserving aligned SSM space before requested rigid/scale output transform"
            )
            branches = apply_physical_transform(
                branches, args.scale, tuple(args.rotation_deg), tuple(args.translation)
            )
            exact_shared_bifurcation(branches)
            curves = interpolate_tree(branches, args.spline_samples)
            tree_dir = output / "generated_trees" / tree_id
            tree_dir.mkdir(parents=True, exist_ok=True)
            np.save(tree_dir / "control_points_27x3.npy", combine_controls(branches))
            np.savez_compressed(tree_dir / "centerlines_bspline.npz", **curves)
            sampled.update({
                "tree_id": tree_id,
                "mode": mode,
                "seed": args.seed,
                "physical_scale": args.scale,
                "rotation_degrees_xyz": args.rotation_deg,
                "translation_mm": args.translation,
                "spline_samples_per_branch": args.spline_samples,
            })
            write_json(tree_dir / "sampled_parameters.json", sampled)
            validation_path = output / "generated_validation" / f"{tree_id}_validation.json"
            write_json(validation_path, validation)
            save_tree_plot(
                branches,
                curves,
                tree_dir / "tree_3d.png",
                f"{tree_id}: dataset-derived LCA ({validation['classification']})",
            )
            results.append({
                "tree_id": tree_id,
                "mode": mode,
                "validation_pass": validation["validation_pass"],
                "classification": validation["classification"],
                "sampling_attempt_count": attempt,
                "rejected_sampling_attempt_count": len(rejected_attempts),
                "tree_directory": tree_dir,
                "validation_file": validation_path,
            })
    passed = sum(item["validation_pass"] for item in results)
    summary = {
        "description": "dataset-derived statistical/anatomical LCA generator; not clinical grade",
        "seed": args.seed,
        "requested_mode": args.mode,
        "generated_tree_count": len(results),
        "validation_pass_count": passed,
        "validation_fail_count": len(results) - passed,
        "rejected_sampling_attempt_count": sum(
            item["rejected_sampling_attempt_count"] for item in results
        ),
        "mode_counts": {mode: sum(item["mode"] == mode for item in results) for mode in modes},
        "fixed_landmark_schema": COUNTS,
        "b_spline_applied_after_control_generation": True,
        "shared_bifurcation_exactly_enforced": True,
        "clinical_grade": False,
        "results": results,
    }
    write_json(output / "generated_validation" / "generated_validation_index.json", summary)
    report = f"""# LCA SSM generation report

This is a dataset-derived statistical/anatomical LCA generator, not a clinical-grade model.

- Seed: {args.seed}
- Modes: {', '.join(modes)}
- Generated trees: {len(results)}
- Validation passed: {passed}
- Validation failed: {len(results) - passed}
- Invalid sampled combinations rejected before saving: {summary['rejected_sampling_attempt_count']}
- Fixed controls: LMCA 5, LAD 12, LCX 10
- PCA sampling: observed coefficient distributions, clipped to observed/configured limits
- Parametric sampling: joint empirical ellipse/angle templates with population-derived residuals
- Topology: one exact LMCA/LAD/LCX shared bifurcation
- Interpolation: cubic B-splines applied only after control-point generation
- Validation thresholds: aligned training-population empirical ranges; numerical intersection tolerance is configurable

Disease, radius, tube, mesh, hub, and old anatomical-correction modules were not called.
"""
    (output / "LCA_SSM_GENERATION_REPORT.md").write_text(report, encoding="utf-8")
    manifest_path = output / "run_manifest_v3.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["generation"] = {key: value for key, value in summary.items() if key != "results"}
        write_json(manifest_path, manifest)
    print(json.dumps(jsonable(summary), indent=2))
    return 0 if passed == len(results) else 2


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pca", "parametric", "both"), default="both")
    parser.add_argument("--count", type=int, default=3, help="trees per selected mode")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output-dir", type=Path, default=project_dir.parent / "outputs" / "lca_ssm")
    parser.add_argument("--pca-sigma-scale", type=float, default=0.30)
    parser.add_argument("--coefficient-limit", type=float, default=2.5)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rotation-deg", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--translation", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--spline-samples", type=int, default=256)
    parser.add_argument("--intersection-tolerance-mm", type=float, default=1.0e-3)
    parser.add_argument("--max-sampling-attempts", type=int, default=50)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be at least one")
    if args.spline_samples < 16:
        parser.error("--spline-samples must be at least 16")
    if args.max_sampling_attempts < 1:
        parser.error("--max-sampling-attempts must be at least one")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
