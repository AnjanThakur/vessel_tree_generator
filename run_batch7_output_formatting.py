"""Runner script for Batch 7 Standardized Output Formatting and Packaging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from output.batch7_pipeline import process_batch7_output


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for clean JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    return obj


def parse_args():
    parser = argparse.ArgumentParser(description="Batch 7 — Standardized Output Formatting and Packaging")
    parser.add_argument(
        "--batch6-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch6_cardiac_motion",
        help="Input Batch 6 4D cardiac motion directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch7_exported_trees",
        help="Output Batch 7 exported trees directory",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=50,
        help="Number of resampled points per vessel branch (default: 50)",
    )
    return parser.parse_args()


def run_batch7(args) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting Batch 7 Output Formatting and Packaging ===")
    print(f" Source Batch 6 Directory: {args.batch6_dir.resolve()}")
    print(f" Output Directory:         {output_dir}")
    print(f" Resampled Points/Branch:  {args.num_points}\n")

    res = process_batch7_output(
        batch6_dir=args.batch6_dir,
        output_dir=output_dir,
        num_points=args.num_points,
    )

    summary_report = {
        "batch": res["batch"],
        "output_summary": res["output_summary"],
        "verification": res["verification"],
        "pipeline_report": res["pipeline_report"],
    }

    with open(output_dir / "exported_trees_summary.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(summary_report), f, indent=2)

    o_sum = res["output_summary"]
    verif = res["verification"]

    print("=" * 80)
    print(" BATCH 7 OUTPUT FORMATTING & PACKAGING SUMMARY REPORT")
    print("=" * 80)
    print(f" Total Trees Exported:               {o_sum['num_trees_exported']}")
    print(f" Points per Vessel Branch:           {o_sum['num_points_per_vessel']}")
    print(f" Static Array Shape (geometry_static): {o_sum['static_array_shape']}")
    print(f" Cine Array Shape (geometry_cine):   {o_sum['cine_array_shape']}")
    print(f" Branch Order:                       {o_sum['branch_order']}")
    print(f" Total Execution Time:               {o_sum['total_execution_time_seconds']:.2f} s")
    print(f" Static == Cine[0] Invariant Check:  {'PASSED' if verif['geometry_static_equals_geometry_cine0'] else 'FAILED'}")
    print(f" Max Invariant Error:                {verif['max_identity_error_mm']:.2e} mm")
    print(f" Overall Verification Status:        {'PASSED' if verif['overall_pass'] else 'FAILED'}")
    print("=" * 80 + "\n")

    return 0 if verif["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(run_batch7(parse_args()))
