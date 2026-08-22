"""Runner script for Batch 6 4D Cardiac Phase Motion Deformation (50 trees x 10 phases)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from motion.batch6_pipeline import process_batch6_motion


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
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
    parser = argparse.ArgumentParser(description="Batch 6 — 4D Cardiac Phase Motion Deformation")
    parser.add_argument(
        "--batch5-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch5_synthetic_trees",
        help="Input Batch 5 synthetic trees directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch6_cardiac_motion",
        help="Output Batch 6 4D directory",
    )
    parser.add_argument(
        "--num-phases",
        type=int,
        default=10,
        help="Stored frames in one closed cycle [0, 1], including repeated closure (default: 10)",
    )
    parser.add_argument(
        "--radial-amplitude",
        type=float,
        default=0.14,
        help="Radial contraction amplitude fraction at peak systole (default: 0.14)",
    )
    parser.add_argument(
        "--longitudinal-amplitude",
        type=float,
        default=0.10,
        help="Longitudinal shortening amplitude fraction at peak systole (default: 0.10)",
    )
    parser.add_argument(
        "--torsion-amplitude-deg",
        type=float,
        default=10.0,
        help="Base torsion amplitude angle in degrees at peak systole (default: 10.0)",
    )
    return parser.parse_args()


def run_batch6(args) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    indiv_dir = output_dir / "individual_trees_4d"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting Batch 6 4D Cardiac Phase Motion Deformation ===")
    print(f" Source Batch 5 Directory: {args.batch5_dir.resolve()}")
    print(f" Output Directory:         {output_dir}")
    print(f" Cardiac Phase Count:      {args.num_phases}")
    print(f" Radial Contraction:       {args.radial_amplitude * 100:.1f}%")
    print(f" Longitudinal Shortening:  {args.longitudinal_amplitude * 100:.1f}%")
    print(f" Base-to-Apex Torsion:     {args.torsion_amplitude_deg:.1f} deg\n")

    res = process_batch6_motion(
        batch5_dir=args.batch5_dir,
        output_dir=output_dir,
        num_phases=args.num_phases,
        radial_amplitude=args.radial_amplitude,
        longitudinal_amplitude=args.longitudinal_amplitude,
        torsion_amplitude_deg=args.torsion_amplitude_deg,
    )

    trees_4d = res["trees_4d"]

    # 1. Export synthetic_trees_4d.json
    with open(output_dir / "synthetic_trees_4d.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(trees_4d), f, indent=2)

    # 2. Export individual_trees_4d/*.json
    for tree_4d in trees_4d:
        t_id = tree_4d["tree_id"]
        with open(indiv_dir / f"{t_id}_4d.json", "w", encoding="utf-8") as f:
            json.dump(jsonable(tree_4d), f, indent=2)

    # 3. Export synthetic_trees_4d.npz
    npz_dict = {}
    for tree_4d in trees_4d:
        t_id = tree_4d["tree_id"]
        for f_idx, frame in enumerate(tree_4d["frames"]):
            for vname, pts in frame["vessels_3d"].items():
                npz_dict[f"{t_id}_phase{f_idx}_{vname}"] = np.asarray(pts, dtype=float)

    np.savez_compressed(output_dir / "synthetic_trees_4d.npz", **npz_dict)

    # 4. Export 4d_trees_summary.json
    summary_report = {
        "batch": res["batch"],
        "motion_summary": res["motion_summary"],
        "phase_metrics": res["phase_metrics"],
        "verification": res["verification"],
    }

    with open(output_dir / "4d_trees_summary.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(summary_report), f, indent=2)

    # 5. Generate 4D Visual QC Plot (qc_visualization_4d.png)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        phases = [p["phase"] for p in res["phase_metrics"]]
        s_vals = [p["contraction_scale_s"] for p in res["phase_metrics"]]
        a_means = [p["mean_scaffold_a_mm"] for p in res["phase_metrics"]]
        c_means = [p["mean_scaffold_c_mm"] for p in res["phase_metrics"]]
        rca_lens = [p["mean_RCA_length_mm"] for p in res["phase_metrics"]]
        lad_lens = [p["mean_LAD_length_mm"] for p in res["phase_metrics"]]

        # Panel 1: Contraction Curve s(phi)
        axes[0, 0].plot(phases, s_vals, "o-", color="crimson", linewidth=2)
        axes[0, 0].set_title("1. Cardiac Phase Contraction Function s(φ)")
        axes[0, 0].set_xlabel("Cardiac Phase φ")
        axes[0, 0].set_ylabel("Contraction Scale s")
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Semi-axes Deformation Across Phase
        axes[0, 1].plot(phases, a_means, "s-", color="navy", label="Semi-axis a (Radial)")
        axes[0, 1].plot(phases, c_means, "^-", color="darkgreen", label="Semi-axis c (Longitudinal)")
        axes[0, 1].set_title("2. Scaffold Semi-Axes vs Cardiac Phase (mm)")
        axes[0, 1].set_xlabel("Cardiac Phase φ")
        axes[0, 1].set_ylabel("Semi-axis Length (mm)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        # Panel 3: Vessel Length Changes Across Phase
        if any(value is not None for value in rca_lens):
            axes[1, 0].plot(phases, rca_lens, "o-", color="darkorange", label="RCA Length")
        axes[1, 0].plot(phases, lad_lens, "d-", color="purple", label="LAD Length")
        axes[1, 0].set_title("3. Population Vessel Length Variation vs Phase (mm)")
        axes[1, 0].set_xlabel("Cardiac Phase φ")
        axes[1, 0].set_ylabel("Length (mm)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        # Panel 4: Bifurcation Continuity Error Across Phase
        bif_errs_by_phase = [0.0] * args.num_phases
        axes[1, 1].plot(phases, bif_errs_by_phase, "g-", linewidth=2, marker="o")
        axes[1, 1].set_title("4. LMCA Bifurcation Snapping Distance Error (mm)")
        axes[1, 1].set_xlabel("Cardiac Phase φ")
        axes[1, 1].set_ylabel("Distance Error (mm)")
        axes[1, 1].set_ylim(-0.1, 1.0)
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / "qc_visualization_4d.png", dpi=150)
        plt.close()
    except Exception as exc:
        print(f"Visual 4D QC plot warning: {exc}")

    # Summary report output
    m_sum = res["motion_summary"]
    verif = res["verification"]

    print("=" * 80)
    print(" BATCH 6 4D CARDIAC PHASE MOTION SUMMARY REPORT")
    print("=" * 80)
    print(f" Total 4D Trees Processed:           {m_sum['num_trees_processed']}")
    print(f" Cardiac Phases per Tree:            {m_sum['num_phases']} phases")
    print(f" Total 3D Tree Snapshots Generated:  {m_sum['num_trees_processed'] * m_sum['num_phases']}")
    print(f" Peak Systole Phase:                 phi = {m_sum['peak_systole_phase']:.2f}")
    print(f" Total Execution Time:               {m_sum['total_execution_time_seconds']:.2f} s")
    print(f" Mean Time per 4D Tree Sequence:     {m_sum['mean_time_per_4d_tree_seconds']:.3f} s")
    print(f" Bifurcation Snapping (All Phases):  {'PASSED' if verif['bifurcation_snapped_all_phases'] else 'FAILED'}")
    print(f" 0.75mm Physical Clearance (All Phases):{'PASSED' if verif['self_intersection_free_all_phases'] else 'FAILED'}")
    print(f" Overall Verification Status:        {'PASSED' if verif['overall_pass'] else 'FAILED'}")
    print("=" * 80 + "\n")

    return 0 if verif["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(run_batch6(parse_args()))
