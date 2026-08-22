"""Canonical command-line entry point for the LCA statistical generator."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pca_ssm_vessel_tree_generator"))

from pca_ssm_vessel_tree_generator.run_person1_week1_pipeline import run_pipeline as build_statistics
from pca_ssm_vessel_tree_generator.run_person2_population_cohort import run as generate_cohort
from pca_ssm_vessel_tree_generator.validate_person2_population_cohort import run as validate_cohort
from run_batch6_cardiac_motion import run_batch6
from run_batch7_output_formatting import run_batch7


LCA_OUTPUT = ROOT / "outputs" / "lca_ssm"
DEFAULT_POPULATION = LCA_OUTPUT / "lca_population_model"
DEFAULT_STATS = DEFAULT_POPULATION / "generator_statistics"
DEFAULT_COHORT = LCA_OUTPUT / "lca_population_cohort"
DEFAULT_VALIDATION = DEFAULT_COHORT / "population_validation"
DEFAULT_MOTION = LCA_OUTPUT / "lca_population_motion"
DEFAULT_EXPORT = LCA_OUTPUT / "lca_population_export"


def _add_generation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--base-seed", type=int, default=20260822)
    parser.add_argument("--seed-stride", type=int, default=1009)
    parser.add_argument("--max-attempts", type=int, default=250)
    parser.add_argument("--pca-scale", type=float, default=0.08)
    parser.add_argument("--clean", action="store_true")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LCA statistical vessel-tree pipeline (real cases to 81-D PCA to validated trees)"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    stats = commands.add_parser("compute-stats", help="Audit real cases and build the LCA-only model")
    stats.add_argument("--output-dir", type=Path, default=DEFAULT_POPULATION)
    stats.add_argument("--smoke-test", action="store_true")
    stats.add_argument("--clean", action="store_true")

    generate = commands.add_parser("generate", help="Generate an accepted static LCA cohort")
    _add_generation_arguments(generate)

    validate = commands.add_parser("validate", help="Compare the accepted cohort with eligible real references")
    validate.add_argument("--population-dir", type=Path, default=DEFAULT_POPULATION)
    validate.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT)
    validate.add_argument("--output-dir", type=Path, default=DEFAULT_VALIDATION)
    validate.add_argument("--clean", action="store_true")

    motion = commands.add_parser("motion", help="Apply optional design-default cardiac motion")
    motion.add_argument("--input-dir", type=Path, default=DEFAULT_COHORT)
    motion.add_argument("--output-dir", type=Path, default=DEFAULT_MOTION)
    motion.add_argument("--num-phases", type=int, default=10)
    motion.add_argument("--radial-amplitude", type=float, default=0.15)
    motion.add_argument("--longitudinal-amplitude", type=float, default=0.10)
    motion.add_argument("--torsion-amplitude-deg", type=float, default=10.0)

    export = commands.add_parser("export", help="Package static/cine XYZ-radius arrays")
    export.add_argument("--input-dir", type=Path, default=DEFAULT_MOTION)
    export.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT)
    export.add_argument("--num-points", type=int, default=50)

    run_all = commands.add_parser(
        "run-all", help="Run statistics, generation, validation, motion and export"
    )
    run_all.add_argument("--population-dir", type=Path, default=DEFAULT_POPULATION)
    run_all.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT)
    run_all.add_argument("--motion-dir", type=Path, default=DEFAULT_MOTION)
    run_all.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    run_all.add_argument("--count", type=int, default=25)
    run_all.add_argument("--num-phases", type=int, default=10)
    run_all.add_argument("--num-points", type=int, default=50)
    run_all.add_argument("--base-seed", type=int, default=20260822)
    run_all.add_argument("--seed-stride", type=int, default=1009)
    run_all.add_argument("--max-attempts", type=int, default=250)
    run_all.add_argument("--pca-scale", type=float, default=0.08)
    run_all.add_argument("--clean", action="store_true")
    return parser.parse_args(argv)


def _run_generation(args: argparse.Namespace) -> None:
    generate_cohort(Namespace(
        stats_dir=args.stats_dir,
        output_dir=args.output_dir,
        count=args.count,
        base_seed=args.base_seed,
        seed_stride=args.seed_stride,
        max_attempts=args.max_attempts,
        pca_scale=args.pca_scale,
        include_rca=False,
        clean=args.clean,
    ))


def _run_motion(args: argparse.Namespace) -> None:
    status = run_batch6(Namespace(
        batch5_dir=args.input_dir,
        output_dir=args.output_dir,
        num_phases=args.num_phases,
        radial_amplitude=args.radial_amplitude,
        longitudinal_amplitude=args.longitudinal_amplitude,
        torsion_amplitude_deg=args.torsion_amplitude_deg,
    ))
    if status:
        raise RuntimeError("motion validation failed")


def _run_validation(args: argparse.Namespace) -> None:
    result = validate_cohort(Namespace(
        population_dir=args.population_dir,
        cohort_dir=args.cohort_dir,
        output_dir=args.output_dir,
        clean=args.clean,
    ))
    if result["status"] != "PASS":
        raise RuntimeError("real-versus-generated cohort validation failed")


def _run_export(args: argparse.Namespace) -> None:
    status = run_batch7(Namespace(
        batch6_dir=args.input_dir,
        output_dir=args.output_dir,
        num_points=args.num_points,
    ))
    if status:
        raise RuntimeError("export validation failed")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "compute-stats":
        build_statistics(
            smoke_test=args.smoke_test,
            output=args.output_dir.resolve(),
            clean=args.clean,
        )
    elif args.command == "generate":
        _run_generation(args)
    elif args.command == "validate":
        _run_validation(args)
    elif args.command == "motion":
        _run_motion(args)
    elif args.command == "export":
        _run_export(args)
    elif args.command == "run-all":
        build_statistics(
            smoke_test=False,
            output=args.population_dir.resolve(),
            clean=args.clean,
        )
        generation_args = Namespace(
            stats_dir=args.population_dir / "generator_statistics",
            output_dir=args.cohort_dir,
            count=args.count,
            base_seed=args.base_seed,
            seed_stride=args.seed_stride,
            max_attempts=args.max_attempts,
            pca_scale=args.pca_scale,
            clean=args.clean,
        )
        _run_generation(generation_args)
        _run_validation(Namespace(
            population_dir=args.population_dir,
            cohort_dir=args.cohort_dir,
            output_dir=args.cohort_dir / "population_validation",
            clean=args.clean,
        ))
        _run_motion(Namespace(
            input_dir=args.cohort_dir,
            output_dir=args.motion_dir,
            num_phases=args.num_phases,
            radial_amplitude=0.15,
            longitudinal_amplitude=0.10,
            torsion_amplitude_deg=10.0,
        ))
        _run_export(Namespace(
            input_dir=args.motion_dir,
            output_dir=args.export_dir,
            num_points=args.num_points,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
