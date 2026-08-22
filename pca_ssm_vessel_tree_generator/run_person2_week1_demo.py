#!/usr/bin/env python
"""Run a controlled or population-derived synthetic LCA-tree realization."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from generation.deviation_sampler import DeviationSampler, ZeroDeviationSampler
from generation.landmark_sampler import LandmarkSampler
from generation.parameter_sampler import ParameterSampler
from generation.tree_assembler import TreeAssembler
from generation.trajectory_sampler import TrajectorySampler
from generation.validator import TreeValidator
from generation.vtk_export import export_tree_vtk


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "outputs/lca_ssm/lca_population_demo/tree_0001"
COLORS = {"LMCA": "#222222", "LAD": "#d62728", "LCX": "#1f77b4", "RCA": "#9467bd"}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _plot_projection(ax, tree, horizontal: int, vertical: int, labels: tuple[str, str]) -> None:
    axis_values = (tree.ellipsoid.a, tree.ellipsoid.b, tree.ellipsoid.c)
    theta = np.linspace(0.0, 2.0 * np.pi, 300)
    ax.plot(
        axis_values[horizontal] * np.cos(theta),
        axis_values[vertical] * np.sin(theta),
        color="#b8c4cf",
        linewidth=1.2,
        alpha=0.65,
        label="support ellipsoid",
    )
    # Scale the anatomy view from the vessels, not from a potentially long-
    # tailed fitted support axis.  The support outline may be clipped; this is
    # intentional so a tall ellipsoid cannot make the coronary branches too
    # small to inspect.
    bounds: list[float] = []
    for name in ("LMCA", "LAD", "LCX", "RCA"):
        if name not in tree.branches:
            continue
        points = tree.branches[name]
        ax.plot(
            points[:, horizontal], points[:, vertical],
            color=COLORS[name], linewidth=3.0, label=name,
        )
        ax.scatter(
            points[-1, horizontal], points[-1, vertical],
            color=COLORS[name], s=24, zorder=4,
        )
        bounds.extend((float(np.max(np.abs(points[:, horizontal]))), float(np.max(np.abs(points[:, vertical])))))
    bifurcation = tree.branches["LMCA"][-1]
    ax.scatter(
        bifurcation[horizontal], bifurcation[vertical],
        color="#000000", s=52, zorder=5, label="shared bifurcation",
    )
    radius = max(1.15 * max(bounds), 1.0)
    ax.set_xlim(-radius, radius)
    ax.set_ylim(-radius, radius)
    ax.set_aspect("equal", adjustable="box")
    ax.axhline(0.0, color="#d9d9d9", linewidth=0.7, zorder=0)
    ax.axvline(0.0, color="#d9d9d9", linewidth=0.7, zorder=0)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.grid(alpha=0.18)


def preview(tree, output: Path, *, population_derived: bool) -> None:
    """Write the fixed cardiac X-Z front view used for anatomy acceptance."""
    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    _plot_projection(ax, tree, 0, 2, ("Cardiac X (mm)", "Cardiac Z (mm; apex is negative)"))
    title = (
        "Population-derived coronary tree — fixed cardiac front view"
        if population_derived
        else "Controlled coronary tree — fixed cardiac front view"
    )
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def preview_multiview(tree, output: Path, *, population_derived: bool) -> None:
    """Write fixed front, lateral and crown views without camera ambiguity."""
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.8))
    specifications = (
        (0, 2, ("Cardiac X (mm)", "Cardiac Z (mm)"), "Front: LAD descent"),
        (1, 2, ("Cardiac Y (mm)", "Cardiac Z (mm)"), "Lateral: apex course"),
        (0, 1, ("Cardiac X (mm)", "Cardiac Y (mm)"), "Crown: LCX circumference"),
    )
    for ax, (horizontal, vertical, labels, title) in zip(axes, specifications):
        _plot_projection(ax, tree, horizontal, vertical, labels)
        ax.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    kind = "population-derived" if population_derived else "controlled"
    fig.suptitle(f"{kind.capitalize()} coronary tree — fixed common-frame anatomy views")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.95))
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def build_inputs(args, rng: np.random.Generator):
    if args.stats_dir is None:
        parameter_sampler = ParameterSampler.controlled()
        landmark_sampler = LandmarkSampler.controlled()
        deviation_sampler = ZeroDeviationSampler()
        trajectory_sampler = None
        validator = TreeValidator()
        mode = "controlled_week1_demo_not_population_derived"
    else:
        stats_dir = args.stats_dir.resolve()
        landmark_path = args.landmark_stats or (stats_dir / "landmark_stats.json")
        pca_path = args.pca or (stats_dir / "surface_deviation_pca.npz")
        thresholds_path = args.thresholds or (stats_dir / "population_validation_thresholds.json")
        trajectory_path = stats_dir / "fixed_branch_surface_coordinates.npz"
        assignment_gate_path = stats_dir / "branch_assignment_gate.json"
        missing = [
            path for path in (
                landmark_path, pca_path, thresholds_path, trajectory_path, assignment_gate_path
            ) if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Real-statistics mode requires a complete frozen LCA package; missing: "
                + ", ".join(str(path) for path in missing)
            )
        assignment_gate = json.loads(assignment_gate_path.read_text(encoding="utf-8"))
        if assignment_gate.get("unresolved_cases_used_for_statistics"):
            raise ValueError("LCA statistics package contains unresolved daughter assignments")
        if int(assignment_gate.get("statistics_eligible_count", 0)) < 2:
            raise ValueError("LCA statistics package has fewer than two anatomy-eligible resolved cases")
        parameter_sampler = ParameterSampler.from_person1_output(stats_dir)
        landmark_sampler = LandmarkSampler.from_json(landmark_path)
        deviation_sampler = DeviationSampler.from_npz(
            pca_path,
            include_mean=True,
            variation_scale=args.pca_scale,
        )
        trajectory_sampler = TrajectorySampler.from_npz(trajectory_path)
        if not trajectory_sampler.exact_local_deviations_available:
            raise ValueError("LCA statistics package lacks exact local trajectory coefficients")
        if not trajectory_sampler.exact_cardiac_points_available:
            raise ValueError("LCA statistics package lacks exact matched cardiac trajectory controls")
        validator = TreeValidator.from_person1_output(thresholds_path)
        mode = "frozen_lca_statistics"
    return parameter_sampler, landmark_sampler, deviation_sampler, trajectory_sampler, validator, mode


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    if output.exists():
        if not args.clean:
            raise FileExistsError(f"refusing to overwrite existing demo: {output}; pass --clean explicitly")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    parameter_sampler, landmark_sampler, deviation_sampler, trajectory_sampler, validator, mode = build_inputs(args, rng)
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least one")
    attempt_records = []
    tree = validation = ellipsoid = landmarks = None
    for attempt in range(1, args.max_attempts + 1):
        ellipsoid = parameter_sampler.sample(
            rng, source_case_id=getattr(args, "source_case_id", None)
        )
        landmarks = landmark_sampler.sample(
            rng, include_rca=args.include_rca, source_case_id=ellipsoid.source_case_id
        )
        candidate = TreeAssembler(
            ellipsoid, rng, deviation_sampler=deviation_sampler, trajectory_sampler=trajectory_sampler
        ).assemble(landmarks, include_rca=args.include_rca)
        candidate_validation = validator.validate(candidate)
        attempt_records.append({
            "attempt": attempt,
            "accepted": candidate_validation["accepted"],
            "ellipsoid_source_case_id": ellipsoid.source_case_id,
            "trajectory_source_case_id": candidate.generation_metadata["empirical_trajectory_source_case_id"],
            "errors": candidate_validation["errors"],
            "metrics": candidate_validation["metrics"],
        })
        if candidate_validation["accepted"]:
            tree, validation = candidate, candidate_validation
            break
    write_json(output / "sampling_attempts.json", {
        "seed": args.seed,
        "maximum_attempts": args.max_attempts,
        "attempt_count": len(attempt_records),
        "accepted_attempt": None if validation is None else len(attempt_records),
        "attempts": attempt_records,
    })
    if tree is None or validation is None or ellipsoid is None or landmarks is None:
        raise RuntimeError(
            f"No valid synthetic tree found in {args.max_attempts} attempts; inspect {output / 'sampling_attempts.json'}"
        )

    for name, points in tree.branches.items():
        np.save(output / f"{name}.npy", points, allow_pickle=False)
    surface_arrays: dict[str, np.ndarray] = {}
    for name, path in tree.surface_paths.items():
        surface_arrays[f"{name}_uvo"] = np.column_stack((path.u, path.v, path.normal_offset))
        surface_arrays[f"{name}_local_deviation"] = path.local_deviation
    np.savez_compressed(output / "surface_coordinates.npz", **surface_arrays)
    vtk_paths = export_tree_vtk(tree, output / "vtk")
    population_derived = mode == "frozen_lca_statistics"
    preview(tree, output / "preview.png", population_derived=population_derived)
    preview_multiview(tree, output / "preview_multiview.png", population_derived=population_derived)
    parameters = {
        "seed": args.seed,
        "generation_mode": mode,
        "population_derived": population_derived,
        "ellipsoid": ellipsoid.to_dict(),
        "landmarks": {name: landmark.to_dict() for name, landmark in landmarks.items()},
        "generation": tree.generation_metadata,
        "accepted_generation_attempt": len(attempt_records),
    }
    write_json(output / "parameters.json", parameters)
    write_json(output / "validation.json", validation)
    expected = [
        output / "parameters.json", output / "validation.json", output / "sampling_attempts.json",
        output / "preview.png", output / "preview_multiview.png",
        output / "surface_coordinates.npz",
        output / "LMCA.npy", output / "LAD.npy", output / "LCX.npy",
        output / "vtk/synthetic_tree.vtm", output / "vtk/synthetic_ellipsoid.vtp",
        output / "vtk/LMCA.vtp", output / "vtk/LAD.vtp", output / "vtk/LCX.vtp",
        output / "vtk/landmarks.vtp",
    ]
    if args.include_rca:
        expected.extend((output / "RCA.npy", output / "vtk/RCA.vtp"))
    missing = [str(path.relative_to(output)) for path in expected if not path.is_file()]
    topology_exact = bool(
        np.array_equal(tree.branches["LMCA"][-1], tree.branches["LAD"][0])
        and np.array_equal(tree.branches["LMCA"][-1], tree.branches["LCX"][0])
    )
    artifact_validation = {
        "expected_files_present": not missing,
        "missing_files": missing,
        "topology_exact_by_array_equality": topology_exact,
        "vtk_readback_completed": True,
        "source_statistics_consumed": mode == "frozen_lca_statistics",
        "pca_applied": tree.generation_metadata["deviation_sample"]["pca_applied"],
    }
    overall_pass = validation["accepted"] and not missing and topology_exact
    manifest = {
        "implementation": (
            "Population-derived LCA statistical generator"
            if population_derived
            else "Controlled LCA generator architecture demonstration"
        ),
        "status": "PASS" if overall_pass else "FAIL",
        "generation_mode": mode,
        "tree_directory": output.name,
        "sampling_attempt_count": len(attempt_records),
        "branch_point_counts": {name: len(points) for name, points in tree.branches.items()},
        "vtk_files": {
            name: str(Path(path).relative_to(output).as_posix())
            for name, path in vtk_paths.items()
        },
        "artifact_validation": artifact_validation,
        "scientific_limitations": [
            (
                "This is one accepted realization from the frozen LCA statistics package; "
                "it is not population-level validation."
                if population_derived
                else "The controlled case does not use learned population distributions."
            ),
            (
                "Centered PCA variation is applied at the requested scale over a matched empirical trajectory."
                if population_derived
                else "No PCA variation is added unless a complete frozen statistics package is supplied."
            ),
            "No independent point-by-point random noise is used.",
            "Engineering acceptance does not establish clinical validity.",
        ],
    }
    write_json(output / "week1_demo_manifest.json", manifest)
    mode_description = (
        "This tree was sampled from the frozen LCA ellipsoid, landmark, empirical trajectory, "
        "PCA, and validation package."
        if population_derived
        else "This controlled architecture test is not a population-derived coronary model."
    )
    readme = f"""# LCA synthetic-tree realization

Open `vtk/synthetic_tree.vtm` in ParaView.

Generation mode: `{mode}`. {mode_description} The generator consumes ellipsoid surface functions, generates smooth B-spline paths, enforces `LMCA[-1] == LAD[0] == LCX[0]`, validates the result, and exports modular VTK.

Validation status: `{manifest['status']}`. Population statistics and PCA are used only when a complete frozen LCA package is explicitly supplied.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    if not overall_pass:
        raise RuntimeError(f"LCA realization validation failed; inspect {output / 'validation.json'}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--include-rca", action="store_true")
    parser.add_argument("--stats-dir", type=Path)
    parser.add_argument("--landmark-stats", type=Path)
    parser.add_argument("--pca", type=Path)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument(
        "--source-case-id",
        help="Force one anatomy-eligible empirical baseline for deterministic auditing.",
    )
    parser.add_argument(
        "--pca-scale", type=float, default=0.04,
        help="Validated centered PCA innovation scale around an exact empirical baseline (default: 0.04).",
    )
    args = parser.parse_args()
    manifest = run(args)
    print(json.dumps({"status": manifest["status"], "output": manifest["tree_directory"]}, indent=2))


if __name__ == "__main__":
    main()
