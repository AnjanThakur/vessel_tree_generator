"""Command-line interface for the submission-ready 4D LCA generator."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from .api import (
    DEFAULT_STATISTICS_DIR,
    CoronaryTreeGenerator,
    GenerationConfig,
    MotionConfig,
    PulsatilityConfig,
)
from .disease import healthy_config, stenosis_config


# Selected by robust multivariate centrality among the warning-free eligible
# population cases (lengths, distance-metric tortuosity and bifurcation angle).
# This chooses a real population member; it does not constrain or hand-edit the
# generated centerlines.
CURATED_DEMO_SOURCE_CASE_ID = "63.label"


def _disease_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.config is not None:
        return json.loads(args.config.read_text(encoding="utf-8"))
    if args.preset == "healthy":
        return healthy_config(args.case_id or "healthy")
    return stenosis_config(
        args.branch,
        args.position,
        args.length,
        args.severity,
        args.preset,
        case_id=args.case_id,
        tandem_positions=args.tandem_positions,
    )


def _generation(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        seed=args.seed,
        source_case_id=args.source_case_id,
        pca_scale=args.pca_scale,
        maximum_attempts=args.maximum_attempts,
        heart_rate_bpm=args.heart_rate,
    )


def _motion(args: argparse.Namespace) -> MotionConfig:
    return MotionConfig(
        number_of_phases=args.phases,
        radial_amplitude=args.radial_motion,
        longitudinal_amplitude=args.longitudinal_motion,
        torsion_amplitude_deg=args.torsion,
        peak_phase=args.peak_phase,
    )


def _pulsatility(args: argparse.Namespace) -> PulsatilityConfig:
    return PulsatilityConfig(
        amplitude=args.pulsatility,
        stenosis_compliance_factor=args.stenosis_compliance,
        peak_phase=args.pulse_peak_phase,
    )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--source-case-id")
    parser.add_argument("--pca-scale", type=float, default=0.04)
    parser.add_argument("--maximum-attempts", type=int, default=250)
    parser.add_argument("--phases", type=int, default=10)
    parser.add_argument("--phase-values", type=float, nargs="+")
    parser.add_argument("--heart-rate", type=float, default=60.0)
    parser.add_argument("--radial-motion", type=float, default=0.14)
    parser.add_argument("--longitudinal-motion", type=float, default=0.10)
    parser.add_argument("--torsion", type=float, default=10.0)
    parser.add_argument("--peak-phase", type=float, default=0.35)
    parser.add_argument("--pulsatility", type=float, default=0.03)
    parser.add_argument("--stenosis-compliance", type=float, default=0.35)
    parser.add_argument("--pulse-peak-phase", type=float, default=0.60)
    parser.add_argument("--points-per-branch", type=int, default=50)
    parser.add_argument("--clean", action="store_true", help="Explicitly replace the selected output directory.")


def _run_generate(args: argparse.Namespace) -> dict[str, Any]:
    from .export import export_case

    generator = CoronaryTreeGenerator(args.statistics_dir)
    case = generator.generate_case(
        _disease_from_args(args),
        generation=_generation(args),
        motion=_motion(args),
        pulsatility=_pulsatility(args),
        phase_values=args.phase_values,
    )
    return export_case(
        case,
        args.output_dir,
        points_per_branch=args.points_per_branch,
        clean=args.clean,
    )


def _run_demo(args: argparse.Namespace) -> dict[str, Any]:
    from .export import export_case

    output = args.output_dir.resolve()
    if output.exists():
        if not args.clean:
            raise FileExistsError(f"refusing to overwrite {output}; pass --clean explicitly")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    generator = CoronaryTreeGenerator(args.statistics_dir)
    generation = _generation(args)
    if generation.source_case_id is None:
        generation = replace(generation, source_case_id=CURATED_DEMO_SOURCE_CASE_ID)
    reference = generator.sample_reference(generation)
    presets = {
        "healthy": healthy_config("healthy_reference"),
        "focal_lad": stenosis_config("LAD", 0.45, 0.12, 0.65, "focal", case_id="focal_lad"),
        "diffuse_lcx": stenosis_config("LCX", 0.55, 0.45, 0.45, "diffuse", case_id="diffuse_lcx"),
        "tandem_lad": stenosis_config(
            "LAD", 0.45, 0.08, 0.55, "tandem",
            case_id="tandem_lad", tandem_positions=(0.28, 0.66),
        ),
    }
    manifests = {}
    for name, disease in presets.items():
        case = generator.generate_case(
            disease,
            generation=generation,
            motion=_motion(args),
            pulsatility=_pulsatility(args),
            phase_values=args.phase_values,
            reference=reference,
        )
        manifests[name] = export_case(
            case,
            output / name,
            points_per_branch=args.points_per_branch,
            clean=False,
        )
    summary = {
        "status": "PASS" if all(item["status"] == "PASS" for item in manifests.values()) else "FAIL",
        "shared_anatomy_seed": args.seed,
        "purpose": "controlled comparison of healthy, focal, diffuse, and tandem disease on one anatomy",
        "cases": manifests,
    }
    (output / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Submission demonstration cases\n\n"
        "These four cases share one seeded anatomy and differ only by the explicit disease configuration. "
        "Open each case's `vtk/cine.pvd` in ParaView. See `demo_summary.json` for validation status.\n",
        encoding="utf-8",
    )
    return summary


def _run_verify(args: argparse.Namespace) -> dict[str, Any]:
    from .export import verify_export

    return verify_export(args.input_dir)


def _run_visualize(args: argparse.Namespace) -> dict[str, Any]:
    from .visualization import create_case_visualizations

    return create_case_visualizations(
        args.input_dir,
        args.output_dir,
        clean=args.clean,
        standalone_html=args.standalone_html,
        create_gif=not args.no_gif,
    )


def _run_compare(args: argparse.Namespace) -> dict[str, Any]:
    from .visualization import create_disease_comparison

    return create_disease_comparison(args.input_dir, args.output_file)


def _run_audit(args: argparse.Namespace) -> dict[str, Any]:
    from .audit import audit_demo_release, audit_exported_case

    result = (
        audit_demo_release(args.input_dir, write_outputs=True)
        if args.release
        else audit_exported_case(args.input_dir, write_outputs=True)
    )
    # Release audits already expose ``status``. A single-case audit uses the
    # more explicit ``overall_status`` field; normalize it for the CLI result
    # envelope without discarding the detailed field.
    if "status" not in result:
        result = {"status": result["overall_status"], **result}
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coronary4d",
        description="Generate validated, disease-aware 4D LCA research models.",
    )
    parser.add_argument(
        "--statistics-dir",
        type=Path,
        default=DEFAULT_STATISTICS_DIR,
        help="Frozen validated LCA statistics package.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate and export one case.")
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--config", type=Path, help="Disease JSON; overrides preset parameters.")
    generate.add_argument("--preset", choices=("healthy", "focal", "diffuse", "tandem"), default="healthy")
    generate.add_argument("--case-id")
    generate.add_argument("--branch", choices=("LMCA", "LAD", "LCX"), default="LAD")
    generate.add_argument("--position", type=float, default=0.45)
    generate.add_argument("--length", type=float, default=0.12)
    generate.add_argument("--severity", type=float, default=0.65)
    generate.add_argument("--tandem-positions", type=float, nargs="+")
    _add_common(generate)

    demo = commands.add_parser("demo", help="Export healthy/focal/diffuse/tandem cases on one anatomy.")
    demo.add_argument("--output-dir", type=Path, required=True)
    _add_common(demo)
    verify = commands.add_parser("verify", help="Verify checksums, arrays, topology, and VTK readback.")
    verify.add_argument("--input-dir", type=Path, required=True)
    visualize = commands.add_parser(
        "visualize", help="Create dashboards, tortuosity plots, GIF, and interactive 3D HTML."
    )
    visualize.add_argument("--input-dir", type=Path, required=True)
    visualize.add_argument("--output-dir", type=Path)
    visualize.add_argument("--standalone-html", action="store_true", help="Embed Plotly for offline HTML use.")
    visualize.add_argument("--no-gif", action="store_true")
    visualize.add_argument("--clean", action="store_true")
    compare = commands.add_parser(
        "compare", help="Compare healthy, focal, diffuse, and tandem cases on shared anatomy."
    )
    compare.add_argument("--input-dir", type=Path, required=True)
    compare.add_argument("--output-file", type=Path)
    audit = commands.add_parser(
        "audit", help="Write quantitative anatomy, disease, motion and pulsatility validation."
    )
    audit.add_argument("--input-dir", type=Path, required=True)
    audit.add_argument("--release", action="store_true", help="Audit all four curated demo cases.")
    present = commands.add_parser(
        "present", help="Start the local Coronary4D presentation and verification center."
    )
    present.add_argument("--host", default="127.0.0.1")
    present.add_argument("--port", type=int, default=8765)
    present.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        result = _run_generate(args)
        location = args.output_dir
    elif args.command == "demo":
        result = _run_demo(args)
        location = args.output_dir
    elif args.command == "verify":
        result = _run_verify(args)
        location = args.input_dir
    elif args.command == "visualize":
        result = _run_visualize(args)
        location = args.output_dir or (args.input_dir / "visualizations")
    elif args.command == "compare":
        result = _run_compare(args)
        location = args.output_file or (args.input_dir / "disease_mode_comparison.png")
    elif args.command == "audit":
        result = _run_audit(args)
        location = args.input_dir
    else:
        from submission_release.presentation_center.app import serve

        return serve(args.host, args.port, not args.no_browser)
    print(json.dumps({
        "status": result["status"],
        "command": args.command,
        "output": str(location.resolve()),
    }, indent=2))
    return 0 if str(result["status"]).startswith("PASS") else 1
