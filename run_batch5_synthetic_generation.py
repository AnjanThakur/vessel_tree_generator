"""Runner script for Batch 5 Synthetic Coronary Tree Generation (M=50 trees)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from generation.batch5_pipeline import process_batch5_generation


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
    parser = argparse.ArgumentParser(description="Batch 5 — Synthetic Coronary Tree Generation")
    parser.add_argument(
        "--batch4-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch4_pca_ssm",
        help="Input Batch 4 SSM directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch5_synthetic_trees",
        help="Output Batch 5 directory",
    )
    parser.add_argument(
        "--num-trees",
        type=int,
        default=50,
        help="Number of accepted synthetic 3D trees to generate (default: 50)",
    )
    parser.add_argument(
        "--max-attempts-per-tree",
        type=int,
        default=100,
        help="Maximum rejection sampling attempts per tree (default: 100)",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def run_batch5(args) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    indiv_dir = output_dir / "individual_trees"
    indiv_dir.mkdir(parents=True, exist_ok=True)

    print("=== Starting Batch 5 Synthetic Coronary Tree Generation ===")
    print(f" Source Batch 4 Directory: {args.batch4_dir.resolve()}")
    print(f" Output Directory:         {output_dir}")
    print(f" Target Trees Count (M):   {args.num_trees}")
    print(f" Random Seed:              {args.rng_seed}\n")

    res = process_batch5_generation(
        batch4_dir=args.batch4_dir,
        output_dir=output_dir,
        num_trees=args.num_trees,
        max_attempts_per_tree=args.max_attempts_per_tree,
        rng_seed=args.rng_seed,
    )

    accepted_trees = res["accepted_trees"]

    # 1. Export synthetic_trees.json
    with open(output_dir / "synthetic_trees.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(accepted_trees), f, indent=2)

    # 2. Export individual_trees/*.json
    for tree in accepted_trees:
        t_id = tree["tree_id"]
        with open(indiv_dir / f"{t_id}.json", "w", encoding="utf-8") as f:
            json.dump(jsonable(tree), f, indent=2)

    # 3. Export synthetic_trees.npz
    npz_dict = {}
    for tree in accepted_trees:
        t_id = tree["tree_id"]
        for vname, pts in tree["vessels_3d"].items():
            npz_dict[f"{t_id}_{vname}"] = np.asarray(pts, dtype=float)

    np.savez_compressed(output_dir / "synthetic_trees.npz", **npz_dict)

    # 4. Export synthetic_trees_summary.json
    summary_report = {
        "batch": res["batch"],
        "generation_summary": res["generation_summary"],
        "accepted_tree_statistics": res["accepted_tree_statistics"],
        "verification": res["verification"],
    }

    with open(output_dir / "synthetic_trees_summary.json", "w", encoding="utf-8") as f:
        json.dump(jsonable(summary_report), f, indent=2)

    # 5. Generate Visual QC Plot (qc_visualization.png)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Panel 1: Semi-axes scatter
        a_vals = [t["ellipsoid_params"]["a_mm"] for t in accepted_trees]
        b_vals = [t["ellipsoid_params"]["b_mm"] for t in accepted_trees]
        c_vals = [t["ellipsoid_params"]["c_mm"] for t in accepted_trees]

        axes[0, 0].scatter(a_vals, b_vals, c="crimson", alpha=0.7, label="a vs b")
        axes[0, 0].scatter(a_vals, c_vals, c="navy", alpha=0.7, label="a vs c")
        axes[0, 0].set_title("1. Scaffold Ellipsoid Semi-Axes (mm)")
        axes[0, 0].set_xlabel("Semi-axis a (mm)")
        axes[0, 0].set_ylabel("Semi-axes b, c (mm)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Branch lengths distribution
        vnames = ("RCA", "LMCA", "LAD", "LCX")
        lengths_per_v = {v: [float(np.sum(np.linalg.norm(np.diff(t["vessels_3d"][v], axis=0), axis=1))) for t in accepted_trees] for v in vnames}
        axes[0, 1].boxplot([lengths_per_v[v] for v in vnames], labels=vnames)
        axes[0, 1].set_title("2. Synthetic Branch Lengths Distribution (mm)")
        axes[0, 1].set_ylabel("Length (mm)")
        axes[0, 1].grid(True, linestyle="--", alpha=0.5)

        # Panel 3: Bifurcation angles histogram
        bif_angles = []
        for t in accepted_trees:
            v_lad = t["vessels_3d"]["LAD"][1] - t["vessels_3d"]["LAD"][0]
            v_lcx = t["vessels_3d"]["LCX"][1] - t["vessels_3d"]["LCX"][0]
            n1, n2 = np.linalg.norm(v_lad), np.linalg.norm(v_lcx)
            if n1 > 1e-12 and n2 > 1e-12:
                bif_angles.append(float(np.degrees(np.arccos(np.clip(np.dot(v_lad, v_lcx) / (n1 * n2), -1.0, 1.0)))))

        axes[1, 0].hist(bif_angles, bins=12, color="teal", edgecolor="black", alpha=0.7)
        axes[1, 0].set_title("3. LMCA Bifurcation Angles (deg)")
        axes[1, 0].set_xlabel("Bifurcation Angle (deg)")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].grid(True, linestyle="--", alpha=0.5)

        # Panel 4: Attempts per tree distribution
        attempts_list = [t["attempts_required"] for t in accepted_trees]
        axes[1, 1].hist(attempts_list, bins=max(10, max(attempts_list)), color="darkorange", edgecolor="black", alpha=0.7)
        axes[1, 1].set_title("4. Rejection Sampling Attempts per Accepted Tree")
        axes[1, 1].set_xlabel("Attempts Required")
        axes[1, 1].set_ylabel("Tree Count")
        axes[1, 1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / "qc_visualization.png", dpi=150)
        plt.close()
    except Exception as exc:
        print(f"Visual QC plot warning: {exc}")

    # Summary report output
    g_sum = res["generation_summary"]
    verif = res["verification"]

    print("=" * 80)
    print(" BATCH 5 SYNTHETIC CORONARY TREE GENERATION SUMMARY REPORT")
    print("=" * 80)
    print(f" Target Accepted Trees Count (M):   {g_sum['num_trees_requested']}")
    print(f" Total Accepted Trees Generated:    {g_sum['num_trees_accepted']}")
    print(f" Total Sampling Attempts:           {g_sum['total_attempts']}")
    print(f" Total Rejection Count:             {g_sum['rejection_count']}")
    print(f" Sampling Acceptance Rate:          {g_sum['acceptance_rate_percent']:.2f}%")
    print(f" Total Generation Time:             {g_sum['total_generation_time_seconds']:.2f} s")
    print(f" Mean Time per Accepted Tree:       {g_sum['mean_generation_time_per_accepted_tree_seconds']:.3f} s")
    print(f" LMCA Bifurcation Snapping (100%):  {'PASSED' if verif['bifurcation_snapped_all'] else 'FAILED'}")
    print(f" Level 6 Population Bounds (100%):  {'PASSED' if verif['level6_passed_all'] else 'FAILED'}")
    print(f" 1.0mm Self-Intersection Test (100%): {'PASSED' if verif['self_intersection_free_all'] else 'FAILED'}")
    print(f" 126-D SSM Vector Ordering (100%):   {'PASSED' if verif['vector_ordering_preserved_all'] else 'FAILED'}")
    print(f" Overall Verification Status:       {'PASSED' if verif['all_accepted_trees_valid'] else 'FAILED'}")
    print("=" * 80 + "\n")

    return 0 if verif["all_accepted_trees_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(run_batch5(parse_args()))
