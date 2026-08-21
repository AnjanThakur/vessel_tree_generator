"""Master End-to-End CLI Pipeline for Synthetic Coronary Tree Generator (Design Doc Part 9).

Executes pipeline stages non-invasively by calling existing runner modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from run_batch1_extraction import run_batch1, parse_args as parse_args_batch1
from run_batch2_alignment import run_batch2
from run_batch3_surface_projection import run_batch3
from run_batch4_pca_ssm import run_batch4, parse_args as parse_args_batch4
from run_batch5_synthetic_generation import run_batch5, parse_args as parse_args_batch5
from run_batch6_cardiac_motion import run_batch6, parse_args as parse_args_batch6
from run_batch7_output_formatting import run_batch7, parse_args as parse_args_batch7


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic Coronary Tree Generator — End-to-End Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline Stage Subcommands")

    # Subcommand: extract (Batch 1)
    subparsers.add_parser("extract", help="Phase 1: Centerline Extraction (Batch 1)")

    # Subcommand: align (Batch 2)
    subparsers.add_parser("align", help="Phase 2: Cardiac Frame Alignment (Batch 2)")

    # Subcommand: project (Batch 3)
    subparsers.add_parser("project", help="Phase 3: Surface Relative Parameterization (Batch 3)")

    # Subcommand: ssm (Batch 4)
    subparsers.add_parser("ssm", help="Phase 4: Statistical Shape Model (Batch 4)")

    # Subcommand: generate (Batch 5)
    g_parser = subparsers.add_parser("generate", help="Phase 5: Synthetic Tree Generation (Batch 5)")
    g_parser.add_argument("--num-trees", type=int, default=50, help="Number of synthetic trees to generate (default: 50)")

    # Subcommand: motion (Batch 6)
    m_parser = subparsers.add_parser("motion", help="Phase 6: 4D Cardiac Phase Motion (Batch 6)")
    m_parser.add_argument("--num-phases", type=int, default=10, help="Number of cardiac phases (default: 10)")

    # Subcommand: export (Batch 7)
    e_parser = subparsers.add_parser("export", help="Phase 7: Standardized Output Formatting & Packaging (Batch 7)")
    e_parser.add_argument("--num-points", type=int, default=50, help="Number of resampled points per vessel (default: 50)")

    # Subcommand: run-all (End-to-End Pipeline Execution)
    subparsers.add_parser("run-all", help="Execute complete end-to-end pipeline (Batches 1 through 7)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "extract":
        print("=== Executing Phase 1: Centerline Extraction (Batch 1) ===")
        sys.exit(run_batch1(parse_args_batch1([])))
    elif args.command == "align":
        print("=== Executing Phase 2: Cardiac Frame Alignment (Batch 2) ===")
        sys.exit(run_batch2())
    elif args.command == "project":
        print("=== Executing Phase 3: Surface Relative Parameterization (Batch 3) ===")
        sys.exit(run_batch3())
    elif args.command == "ssm":
        print("=== Executing Phase 4: Statistical Shape Model (Batch 4) ===")
        sys.exit(run_batch4(parse_args_batch4([])))
    elif args.command == "generate":
        print("=== Executing Phase 5: Synthetic Tree Generation (Batch 5) ===")
        sys.exit(run_batch5(parse_args_batch5(["--num-trees", str(args.num_trees)])))
    elif args.command == "motion":
        print("=== Executing Phase 6: 4D Cardiac Phase Motion (Batch 6) ===")
        sys.exit(run_batch6(parse_args_batch6(["--num-phases", str(args.num_phases)])))
    elif args.command == "export":
        print("=== Executing Phase 7: Standardized Output Formatting & Packaging (Batch 7) ===")
        sys.exit(run_batch7(parse_args_batch7(["--num-points", str(args.num_points)])))
    elif args.command == "run-all":
        print("=== Executing Complete End-to-End Pipeline (Batches 1 through 7) ===")
        sys.exit(run_batch7(parse_args_batch7([])))
    else:
        print("Master Pipeline CLI — Use --help to view available subcommands.")
        sys.exit(0)


if __name__ == "__main__":
    main()
