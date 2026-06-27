# LCA_topology_generator/generate_lca.py

import argparse
from pathlib import Path

import numpy as np

from .parameters import (
    DEFAULT_K_EXPORT,
    DEFAULT_CANDIDATES,
    USE_PATIENT_PRIOR,
    DEFAULT_LMCA_CENTERLINE_POINTS,
    DEFAULT_LAD_CENTERLINE_POINTS,
    DEFAULT_LCX_CENTERLINE_POINTS,
)
from .control_points import generate_one_lca_candidate
from .validation import validate_lca_tree
from .visualize import plot_lca_tree
from .export import export_lca_population, export_side_branch_parametric_positions
from .population_statistics import export_population_statistics
from .angle_distribution import export_angle_distribution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate topology-driven LCA control points."
    )

    parser.add_argument("--k", type=int, default=DEFAULT_K_EXPORT)
    parser.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="LCA_branch_control_points/generated",
    )
    parser.add_argument(
        "--export-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export mean/std population statistics.",
    )
    parser.add_argument(
        "--export-angle-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export LAD-LCX angle distribution JSON and plot.",
    )
    parser.add_argument(
        "--export-side-branch-config",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export future side-branch parametric-position config.",
    )
    parser.add_argument(
        "--use-prior",
        action=argparse.BooleanOptionalAction,
        default=USE_PATIENT_PRIOR,
        help="Use patient-derived SSM prior model instead of procedural equations.",
    )
    parser.add_argument(
        "--lmca-points",
        type=int,
        default=DEFAULT_LMCA_CENTERLINE_POINTS,
        help="Number of centerline points for LMCA.",
    )
    parser.add_argument(
        "--lad-points",
        type=int,
        default=DEFAULT_LAD_CENTERLINE_POINTS,
        help="Number of centerline points for LAD.",
    )
    parser.add_argument(
        "--lcx-points",
        type=int,
        default=DEFAULT_LCX_CENTERLINE_POINTS,
        help="Number of centerline points for LCX.",
    )

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    selected_branches = []
    selected_metadata = []
    validation_reports = []

    attempted = 0

    for candidate_id in range(args.candidates):
        attempted += 1

        branches, metadata = generate_one_lca_candidate(
            rng=rng,
            tree_id=candidate_id,
            use_patient_prior=args.use_prior,
        )

        passed, report = validate_lca_tree(branches, metadata)

        if passed:
            selected_branches.append(branches)
            selected_metadata.append(metadata)
            validation_reports.append(report)

        if len(selected_branches) >= args.k:
            break

    if len(selected_branches) < args.k:
        raise RuntimeError(
            f"Only generated {len(selected_branches)} valid trees "
            f"out of {attempted}. Increase --candidates."
        )

    output_dir = Path(args.output)

    export_lca_population(
        selected_branches=selected_branches,
        selected_metadata=selected_metadata,
        validation_reports=validation_reports,
        output_dir=str(output_dir),
        lmca_points=args.lmca_points,
        lad_points=args.lad_points,
        lcx_points=args.lcx_points,
    )

    if args.export_stats:
        export_population_statistics(output_dir)

    angle_report = None
    if args.export_angle_report:
        angle_report = export_angle_distribution(output_dir)

    if args.export_side_branch_config:
        export_side_branch_parametric_positions(str(output_dir))

    preview_path = output_dir / "preview_plots" / "lca_template_0.png"
    plot_lca_tree(
        selected_branches[0],
        output_path=str(preview_path),
        title="Generated LCA Template 0: LMCA -> LAD + LCX",
    )

    print("LCA control-point generation completed.")
    print(f"Valid templates exported: {len(selected_branches)}")
    if angle_report is not None:
        print(
            "Actual LAD-LCX angle mean/std: "
            f"{angle_report['actual_mean_deg']:.2f} / "
            f"{angle_report['actual_std_deg']:.2f} deg"
        )
    print(f"Output folder: {output_dir}")
    print(f"Preview plot: {preview_path}")


if __name__ == "__main__":
    main()
