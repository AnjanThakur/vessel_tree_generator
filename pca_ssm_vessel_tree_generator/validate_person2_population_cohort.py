#!/usr/bin/env python
"""Compare a generated coronary cohort with the real Person 1 reference population."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import circmean, ks_2samp

from generation.deviation_sampler import DeviationSampler
from generation.landmark_sampler import LandmarkSampler
from generation.parameter_sampler import EllipsoidParameters, ParameterSampler
from generation.trajectory_sampler import TrajectorySampler
from generation.tree_assembler import TreeAssembler
from generation.validator import bifurcation_angle_deg, path_length, path_progression_metrics, tortuosity
from generation.validator import TreeValidator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERSON1 = REPO_ROOT / "outputs/lca_ssm/person1_week1"
DEFAULT_COHORT = REPO_ROOT / "outputs/lca_ssm/person2_population_cohort"
FIXED_COUNTS = {"LMCA": 5, "LAD": 12, "LCX": 10, "RCA": 15}
BRANCHES = ("LMCA", "LAD", "LCX", "RCA")
LANDMARKS = ("lca_ostium", "bifurcation", "lad_endpoint", "lcx_endpoint", "rca_ostium", "rca_endpoint")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append(store: dict[str, list[float]], name: str, value: float) -> None:
    value = float(value)
    if np.isfinite(value):
        store.setdefault(name, []).append(value)


def numeric_case_key(path: Path) -> tuple[int, str]:
    token = path.name.split("_", 1)[-1]
    return (int(token) if token.isdigit() else 10**9, path.name)


def load_real_reference(person1: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    distributions: dict[str, list[float]] = {}
    real_rows: dict[str, dict[str, Any]] = {}
    ellipsoid_path = person1 / "population_ellipsoid_parameters.csv"
    with ellipsoid_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["is_valid"].strip().lower() != "true":
                continue
            case = real_rows.setdefault(row["case_id"], {"case_id": row["case_id"]})
            for axis in "abc":
                value = float(row[axis])
                append(distributions, f"ellipsoid_{axis}_mm", value)
                case[f"ellipsoid_{axis}_mm"] = value

    # Compare the generated smooth paths with the matching smooth validation
    # scaffolds of the resolved/anatomy-gated cases.  Raw metrics from all 191
    # source archives cannot be pooled here because most daughter assignments
    # are explicitly unresolved and raw sampling noise is not generator shape.
    scaffold_path = person1 / "validation_scaffold_case_metrics.csv"
    with scaffold_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            case = real_rows.setdefault(row["case_id"], {"case_id": row["case_id"]})
            for name, text in row.items():
                if name == "case_id" or text in {None, ""}:
                    continue
                value = float(text)
                append(distributions, name, value)
                case[name] = value

    landmark_path = person1 / "population_landmarks.csv"
    with landmark_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["statistics_eligible"].strip().lower() != "true":
                continue
            landmark = row["landmark_name"]
            if landmark not in LANDMARKS:
                continue
            case = real_rows.setdefault(row["case_id"], {"case_id": row["case_id"]})
            for coordinate in ("u", "v", "offset"):
                suffix = "rad" if coordinate in {"u", "v"} else "mm"
                name = f"landmark_{landmark}_{coordinate}_{suffix}"
                value = float(row[coordinate])
                append(distributions, name, value)
                case[name] = value

    fixed_path = person1 / "fixed_branch_surface_coordinates.npz"
    with np.load(fixed_path, allow_pickle=False) as data:
        for branch in BRANCHES:
            values = np.asarray(data[f"{branch}_uvo"], dtype=float)
            unwrapped = np.unwrap(values[:, :, 0], axis=1)
            obliquity = unwrapped[:, -1] - unwrapped[:, 0]
            distributions[f"{branch.lower()}_obliquity_rad"] = obliquity.tolist()
            distributions[f"{branch.lower()}_surface_offset_mm"] = values[:, :, 2].reshape(-1).tolist()

    pca_path = person1 / "surface_deviation_pca.npz"
    with np.load(pca_path, allow_pickle=False) as data:
        standardized = np.asarray(data["standardized_training_scores"], dtype=float)
        for mode in range(standardized.shape[1]):
            distributions[f"pca_mode_{mode + 1:02d}_standard_score"] = standardized[:, mode].tolist()
    return {name: np.asarray(values, dtype=float) for name, values in distributions.items()}, list(real_rows.values())


def resample_offset(points: np.ndarray, offsets: np.ndarray, count: int) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    normalized = cumulative / max(float(cumulative[-1]), 1.0e-12)
    return np.interp(np.linspace(0.0, 1.0, count), normalized, offsets)


def load_generated_reference(cohort: Path, pca_path: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    distributions: dict[str, list[float]] = {}
    rows: list[dict[str, Any]] = []
    with np.load(pca_path, allow_pickle=False) as pca:
        eigen_scales = np.sqrt(np.maximum(np.asarray(pca["eigenvalues"][: len(pca["components"])]), 0.0))
    tree_directories = sorted((path for path in cohort.glob("tree_*" ) if path.is_dir()), key=numeric_case_key)
    if not tree_directories:
        raise ValueError(f"no generated tree directories found in {cohort}")
    for tree in tree_directories:
        parameters = json.loads((tree / "parameters.json").read_text(encoding="utf-8"))
        validation = json.loads((tree / "validation.json").read_text(encoding="utf-8"))
        if not validation["accepted"]:
            raise ValueError(f"generated tree is not accepted: {tree}")
        metrics = validation["metrics"]
        row: dict[str, Any] = {"tree_id": tree.name, "source_case_id": parameters["ellipsoid"]["source_case_id"]}
        for axis in "abc":
            name = f"ellipsoid_{axis}_mm"
            value = float(parameters["ellipsoid"][axis])
            append(distributions, name, value)
            row[name] = value
        append(distributions, "bifurcation_angle_deg", metrics["bifurcation_angle_deg"])
        row["bifurcation_angle_deg"] = metrics["bifurcation_angle_deg"]
        for branch in metrics["branch_lengths_mm"]:
            key = branch.lower()
            branch_values = {
                "length_mm": metrics["branch_lengths_mm"][branch],
                "tortuosity": metrics["branch_tortuosity"][branch],
                "obliquity_rad": metrics["branch_obliquity_rad"][branch],
                **metrics["branch_progression"][branch],
            }
            for metric, value in branch_values.items():
                name = f"{key}_{metric}"
                append(distributions, name, value)
                row[name] = value
        for landmark, values in parameters["landmarks"].items():
            for coordinate in ("u", "v", "offset"):
                suffix = "rad" if coordinate in {"u", "v"} else "mm"
                name = f"landmark_{landmark}_{coordinate}_{suffix}"
                append(distributions, name, values[coordinate])
                row[name] = values[coordinate]
        with np.load(tree / "surface_coordinates.npz", allow_pickle=False) as surface:
            for branch in BRANCHES:
                if f"{branch}_uvo" not in surface.files:
                    continue
                uvo = np.asarray(surface[f"{branch}_uvo"], dtype=float)
                points = np.load(tree / f"{branch}.npy", allow_pickle=False)
                offsets = resample_offset(points, uvo[:, 2], FIXED_COUNTS[branch])
                distributions.setdefault(f"{branch.lower()}_surface_offset_mm", []).extend(offsets.tolist())
        coefficients = np.asarray(parameters["generation"]["deviation_sample"]["coefficients"], dtype=float)
        standardized = np.divide(
            coefficients,
            eigen_scales,
            out=np.zeros_like(coefficients),
            where=eigen_scales > 1.0e-12,
        )
        for mode, value in enumerate(standardized, start=1):
            append(distributions, f"pca_mode_{mode:02d}_standard_score", value)
        rows.append(row)
    return {name: np.asarray(values, dtype=float) for name, values in distributions.items()}, rows


def audit_generator_baselines(person1: Path) -> list[dict[str, Any]]:
    """Evaluate every frozen baseline both with and without optional RCA."""
    stats = person1 / "generator_statistics"
    parameter_sampler = ParameterSampler.from_person1_output(stats)
    landmark_sampler = LandmarkSampler.from_json(stats / "landmark_stats.json")
    trajectory_sampler = TrajectorySampler.from_npz(stats / "fixed_branch_surface_coordinates.npz")
    deviation_sampler = DeviationSampler.from_npz(
        stats / "surface_deviation_pca.npz", include_mean=True, variation_scale=0.0
    )
    validator = TreeValidator.from_person1_output(stats / "population_validation_thresholds.json")
    rows: list[dict[str, Any]] = []
    for source in parameter_sampler.empirical_rows:
        case_id = str(source["case_id"])
        ellipsoid = EllipsoidParameters(
            float(source["a"]), float(source["b"]), float(source["c"]),
            "zero_innovation_baseline_audit", case_id,
        )
        reports = {}
        for include_rca in (False, True):
            landmarks = landmark_sampler.sample(
                np.random.default_rng(1), include_rca=include_rca, source_case_id=case_id
            )
            tree = TreeAssembler(
                ellipsoid,
                np.random.default_rng(1),
                deviation_sampler=deviation_sampler,
                trajectory_sampler=trajectory_sampler,
            ).assemble(landmarks, include_rca=include_rca)
            reports[include_rca] = validator.validate(tree)
        report = reports[False]
        rca_report = reports[True]
        rows.append({
            "case_id": case_id,
            "accepted_lca_only": report["accepted"],
            "accepted_with_rca": rca_report["accepted"],
            "error_count": len(report["errors"]),
            "rca_error_count": len(rca_report["errors"]),
            "warning_count": len(report["warnings"]),
            "errors": "; ".join(report["errors"]),
            "rca_errors": "; ".join(rca_report["errors"]),
            "warnings": "; ".join(report["warnings"]),
            **{f"{name.lower()}_length_mm": value for name, value in report["metrics"]["branch_lengths_mm"].items()},
            **{f"{name.lower()}_tortuosity": value for name, value in report["metrics"]["branch_tortuosity"].items()},
        })
    return rows


def align_circular(real: np.ndarray, generated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = float(circmean(real, low=-math.pi, high=math.pi))
    wrap = lambda values: (values - center + math.pi) % (2.0 * math.pi) - math.pi
    return wrap(real), wrap(generated)


def summarize_comparison(name: str, real: np.ndarray, generated: np.ndarray) -> dict[str, Any]:
    real_values, generated_values = real, generated
    circular = name.startswith("landmark_") and name.endswith("_u_rad")
    if circular:
        real_values, generated_values = align_circular(real, generated)
    real_mean = float(np.mean(real_values))
    generated_mean = float(np.mean(generated_values))
    real_std = float(np.std(real_values, ddof=1)) if len(real_values) > 1 else 0.0
    generated_std = float(np.std(generated_values, ddof=1)) if len(generated_values) > 1 else 0.0
    lower, upper = np.percentile(real_values, [2.5, 97.5])
    smd = (generated_mean - real_mean) / real_std if real_std > 1.0e-12 else 0.0
    std_ratio = generated_std / real_std if real_std > 1.0e-12 else 1.0
    coverage = float(np.mean((generated_values >= lower) & (generated_values <= upper)))
    ks = ks_2samp(real_values, generated_values)
    mean_pass = abs(smd) <= 0.75
    coverage_pass = coverage >= 0.75
    spread_pass = 0.20 <= std_ratio <= 3.0 if real_std > 1.0e-12 else True
    return {
        "metric": name,
        "circularly_aligned": circular,
        "real_count": len(real_values),
        "generated_count": len(generated_values),
        "real_mean": real_mean,
        "generated_mean": generated_mean,
        "real_std": real_std,
        "generated_std": generated_std,
        "real_median": float(np.median(real_values)),
        "generated_median": float(np.median(generated_values)),
        "real_p2_5": float(lower),
        "real_p97_5": float(upper),
        "generated_p2_5": float(np.percentile(generated_values, 2.5)),
        "generated_p97_5": float(np.percentile(generated_values, 97.5)),
        "standardized_mean_difference": float(smd),
        "generated_to_real_std_ratio": float(std_ratio),
        "generated_within_real_95_fraction": coverage,
        "ks_statistic": float(ks.statistic),
        "ks_pvalue_descriptive_only": float(ks.pvalue),
        "mean_pass": mean_pass,
        "coverage_pass": coverage_pass,
        "spread_pass": spread_pass,
        "comparison_pass": mean_pass and coverage_pass and spread_pass,
    }


def distribution_plot(path: Path, comparisons: list[dict[str, Any]], real: dict[str, np.ndarray], generated: dict[str, np.ndarray]) -> None:
    preferred = [
        "ellipsoid_a_mm", "ellipsoid_b_mm", "ellipsoid_c_mm", "bifurcation_angle_deg",
        "lmca_length_mm", "lad_length_mm", "lcx_length_mm", "rca_length_mm",
        "lmca_tortuosity", "lad_tortuosity", "lcx_tortuosity", "rca_tortuosity",
        "lmca_obliquity_rad", "lad_obliquity_rad", "lcx_obliquity_rad", "rca_obliquity_rad",
    ]
    available = [name for name in preferred if name in real and name in generated]
    columns = 4
    rows = int(math.ceil(len(available) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(17, 3.25 * rows))
    axes_array = np.atleast_1d(axes).reshape(rows, columns)
    for axis in axes_array.flat:
        axis.set_visible(False)
    for axis, name in zip(axes_array.flat, available):
        axis.set_visible(True)
        axis.hist(real[name], bins=20, density=True, alpha=0.55, label="Real", color="#4c78a8")
        axis.hist(generated[name], bins=min(12, max(5, len(generated[name]) // 2)), density=True, alpha=0.55, label="Generated", color="#f58518")
        axis.set_title(name.replace("_", " "), fontsize=9)
        axis.grid(alpha=0.2)
    axes_array.flat[0].legend()
    figure.suptitle("Real versus generated coronary distributions", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def landmark_plot(path: Path, real: dict[str, np.ndarray], generated: dict[str, np.ndarray]) -> None:
    available_landmarks = [
        landmark for landmark in LANDMARKS
        if all(
            f"landmark_{landmark}_{coordinate}_{suffix}" in real
            and f"landmark_{landmark}_{coordinate}_{suffix}" in generated
            for coordinate, suffix in (("u", "rad"), ("v", "rad"), ("offset", "mm"))
        )
    ]
    figure, axes = plt.subplots(3, len(available_landmarks), figsize=(3.2 * len(available_landmarks), 9))
    axes = np.asarray(axes).reshape(3, len(available_landmarks))
    for column, landmark in enumerate(available_landmarks):
        for row, (coordinate, suffix) in enumerate((("u", "rad"), ("v", "rad"), ("offset", "mm"))):
            name = f"landmark_{landmark}_{coordinate}_{suffix}"
            axis = axes[row, column]
            axis.boxplot([real[name], generated[name]], tick_labels=["Real", "Gen"], showfliers=False)
            axis.set_title(f"{landmark}\n{coordinate}", fontsize=9)
            axis.grid(alpha=0.2)
    figure.suptitle("Surface-relative landmark comparison", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def pca_plot(path: Path, real: dict[str, np.ndarray], generated: dict[str, np.ndarray]) -> None:
    names = sorted(name for name in real if name.startswith("pca_mode_"))
    real_std = [np.std(real[name], ddof=1) for name in names]
    generated_std = [np.std(generated[name], ddof=1) for name in names]
    generated_mean = [np.mean(generated[name]) for name in names]
    x = np.arange(1, len(names) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(x, real_std, marker="o", label="Real score SD")
    axes[0].plot(x, generated_std, marker="o", label="Generated score SD")
    axes[0].axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_ylabel("Standard deviation")
    axes[0].legend()
    axes[1].bar(x, generated_mean, color="#f58518")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Generated mean score")
    axes[1].set_xlabel("PCA mode")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.suptitle("PCA score distribution comparison", fontsize=16)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def surface_offset_plot(path: Path, real: dict[str, np.ndarray], generated: dict[str, np.ndarray]) -> None:
    available = [
        branch for branch in BRANCHES
        if f"{branch.lower()}_surface_offset_mm" in real
        and f"{branch.lower()}_surface_offset_mm" in generated
    ]
    figure, axes = plt.subplots(1, len(available), figsize=(3.8 * len(available), 4.5))
    axes = np.atleast_1d(axes)
    for axis, branch in zip(axes, available):
        name = f"{branch.lower()}_surface_offset_mm"
        axis.boxplot([real[name], generated[name]], tick_labels=["Real", "Gen"], showfliers=False)
        axis.set_title(branch)
        axis.set_ylabel("Surface offset (mm)")
        axis.grid(alpha=0.2)
    figure.suptitle("Fixed-point surface-offset comparison", fontsize=15)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    person1 = args.person1_dir.resolve()
    cohort = args.cohort_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else cohort / "population_validation"
    if output.exists():
        if not args.clean:
            raise FileExistsError(f"refusing to overwrite {output}; pass --clean")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    real, real_rows = load_real_reference(person1)
    generated, generated_rows = load_generated_reference(cohort, person1 / "surface_deviation_pca.npz")
    baseline_audit = audit_generator_baselines(person1)
    common = sorted(set(real) & set(generated))
    comparisons = [summarize_comparison(name, real[name], generated[name]) for name in common]
    write_csv(output / "real_vs_generated_metrics.csv", comparisons)
    write_csv(output / "real_reference_case_metrics.csv", real_rows)
    write_csv(output / "generated_case_metrics.csv", generated_rows)
    write_csv(output / "generator_baseline_eligibility.csv", baseline_audit)
    distribution_plot(output / "real_vs_generated_distributions.png", comparisons, real, generated)
    landmark_plot(output / "landmark_comparison.png", real, generated)
    pca_plot(output / "pca_score_comparison.png", real, generated)
    surface_offset_plot(output / "surface_offset_comparison.png", real, generated)

    non_pca = [row for row in comparisons if not row["metric"].startswith("pca_mode_")]
    pca = [row for row in comparisons if row["metric"].startswith("pca_mode_")]
    warnings = [row["metric"] for row in comparisons if not row["comparison_pass"]]
    manifest = json.loads((cohort / "cohort_manifest.json").read_text(encoding="utf-8"))
    assignment_gate = json.loads((person1 / "branch_assignment_gate.json").read_text(encoding="utf-8"))
    structural_pass = bool(
        manifest["all_trees_accepted"] and manifest["exact_lca_topology_for_all_trees"]
    )
    eligible_baselines = {
        row["case_id"] for row in baseline_audit if row["accepted_lca_only"]
    }
    rca_eligible_baselines = {
        row["case_id"] for row in baseline_audit if row["accepted_with_rca"]
    }
    represented_baselines = {str(row["source_case_id"]) for row in generated_rows}
    result = {
        "schema_version": 2,
        "status": "PASS" if structural_pass else "FAIL",
        "structural_and_anatomical_acceptance_pass": structural_pass,
        "generated_tree_count": len(generated_rows),
        "source_population_case_count_audited": int(assignment_gate["input_case_count"]),
        "resolved_assignment_count": int(assignment_gate["resolved_assignment_count"]),
        "real_anatomy_eligible_reference_case_count": len(real_rows),
        "unresolved_assignments_used": False,
        "compared_metric_count": len(comparisons),
        "non_pca_metric_count": len(non_pca),
        "pca_mode_count": len(pca),
        "rca_decision": "excluded_from_primary_cohort",
        "rca_decision_basis": (
            "The available RCA arrays are documented inferred disconnected candidates rather than resolved RCA "
            "ground truth, and their preview courses are not consistently stable. RCA remains optional in code "
            "and is audited separately, but is not published in the primary anatomy-accepted cohort."
        ),
        "eligible_zero_innovation_lca_baseline_count": len(eligible_baselines),
        "eligible_zero_innovation_lca_baselines": sorted(eligible_baselines),
        "eligible_zero_innovation_with_optional_rca_count": len(rca_eligible_baselines),
        "represented_eligible_baseline_count": len(represented_baselines & eligible_baselines),
        "represented_eligible_baseline_fraction": (
            len(represented_baselines & eligible_baselines) / max(len(eligible_baselines), 1)
        ),
        "descriptive_distribution_pass_count": sum(row["comparison_pass"] for row in comparisons),
        "descriptive_distribution_warning_count": len(warnings),
        "descriptive_distribution_warnings": warnings,
        "interpretation": (
            "Every generated LCA tree passed individual structural, anatomical, collision, progression, and "
            "scaffold-reference checks. Distribution comparisons use only resolved, anatomy-gated real references; "
            "all unresolved daughter assignments are excluded rather than used to make generated shapes appear valid."
        ),
        "artifacts": {
            "metrics_csv": str((output / "real_vs_generated_metrics.csv").resolve()),
            "real_case_metrics_csv": str((output / "real_reference_case_metrics.csv").resolve()),
            "generated_case_metrics_csv": str((output / "generated_case_metrics.csv").resolve()),
            "generator_baseline_eligibility_csv": str((output / "generator_baseline_eligibility.csv").resolve()),
            "distribution_plot": str((output / "real_vs_generated_distributions.png").resolve()),
            "landmark_plot": str((output / "landmark_comparison.png").resolve()),
            "pca_plot": str((output / "pca_score_comparison.png").resolve()),
            "surface_offset_plot": str((output / "surface_offset_comparison.png").resolve()),
            "published_artifact_manifest": str((cohort / "published_artifact_manifest.json").resolve()),
        },
    }
    write_json(output / "real_vs_generated_validation.json", result)
    warning_lines = "\n".join(f"- `{name}`" for name in warnings) or "- None"
    (output / "README.md").write_text(
        "# Real-versus-generated cohort validation\n\n"
        "This directory compares the accepted synthetic cohort with resolved, anatomy-gated immutable real-patient references. "
        "KS p-values are descriptive only because the generated cohort is intentionally smaller and individual "
        "samples are produced by empirical bootstrap plus low-scale PCA innovation.\n\n"
        f"Status: `{result['status']}`. Compared metrics: {len(comparisons)}; descriptive passes: "
        f"{result['descriptive_distribution_pass_count']}; warnings: {len(warnings)}.\n\n"
        "RCA decision: excluded from the primary cohort because the available RCA paths are inferred disconnected "
        "candidates rather than resolved ground truth; optional RCA support remains in the code and baseline audit.\n\n"
        "## Descriptive warnings\n\n"
        f"{warning_lines}\n",
        encoding="utf-8",
    )
    manifest["population_validation"] = {
        "status": result["status"],
        "report": str((output / "real_vs_generated_validation.json").resolve()),
        "descriptive_warning_count": len(warnings),
    }
    write_json(cohort / "cohort_manifest.json", manifest)
    hash_path = cohort / "published_artifact_manifest.json"
    published_files = sorted(
        path for path in cohort.rglob("*")
        if path.is_file() and path != hash_path
    )
    write_json(hash_path, {
        "schema_version": 1,
        "root": ".",
        "file_count": len(published_files),
        "total_bytes": sum(path.stat().st_size for path in published_files),
        "files": {
            path.relative_to(cohort).as_posix(): file_sha256(path)
            for path in published_files
        },
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person1-dir", type=Path, default=DEFAULT_PERSON1)
    parser.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
