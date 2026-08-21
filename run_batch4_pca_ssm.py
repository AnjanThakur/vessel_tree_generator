"""Batch 4 PCA / Statistical Shape Model Runner Script.

Executes 126-D joint deviation PCA and population statistics fitting across all 174 Batch-3 accepted patient records.
Conforms strictly to Part 5 of coronary_tree_generator_design (1).md.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from pca_ssm_vessel_tree_generator.ssm.batch4_pipeline import (
    process_batch4_population,
)


def jsonable(val: Any) -> Any:
    if isinstance(val, dict):
        return {str(k): jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [jsonable(v) for v in val]
    if isinstance(val, np.ndarray):
        return jsonable(val.tolist())
    if isinstance(val, (np.floating, np.integer, np.bool_)):
        return jsonable(val.item())
    if isinstance(val, float) and not math.isfinite(val):
        return None
    if isinstance(val, Path):
        return str(val)
    return val


def generate_batch4_qc_plot(
    model_summary: dict[str, Any],
    reconstruction_rmses: list[float],
    output_path: Path,
) -> None:
    """Generate 4-panel Batch 4 PCA QC visual diagnostic plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Batch 4 Statistical Shape Model (SSM) QC Diagnostic — N = {model_summary['n_samples']} Patients",
        fontsize=14,
        fontweight="bold",
    )

    # 1. Cumulative Explained Variance Curve
    cum_var = np.array(model_summary["cumulative_explained_variance"]) * 100.0
    k_retained = model_summary["k_retained"]
    cutoff = model_summary["variance_cutoff"] * 100.0

    ax1 = axes[0, 0]
    ax1.plot(range(1, len(cum_var) + 1), cum_var, color="#1f77b4", linewidth=2.5, label="Cumulative Variance")
    ax1.axhline(cutoff, color="#d62728", linestyle="--", label=f"Threshold ({cutoff:.0f}%)")
    ax1.axvline(k_retained, color="#2ca02c", linestyle=":", linewidth=2, label=f"k = {k_retained} Retained Modes")
    ax1.scatter([k_retained], [cum_var[k_retained - 1]], color="#2ca02c", s=80, zorder=5)
    ax1.set_title("1. Cumulative Explained Variance Curve", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Principal Component Index")
    ax1.set_ylabel("Cumulative Variance (%)")
    ax1.set_xlim(1, min(40, len(cum_var)))
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    # 2. Eigenvalue Spectrum
    eigenvalues = np.array(model_summary["eigenvalues"])
    ax2 = axes[0, 1]
    ax2.semilogy(range(1, len(eigenvalues) + 1), eigenvalues, color="#ff7f0e", marker="o", markersize=4, label="Eigenvalue λ_j")
    ax2.set_title("2. Eigenvalue Spectrum (Log Scale)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Principal Component Index")
    ax2.set_ylabel("Eigenvalue λ_j")
    ax2.set_xlim(1, min(40, len(eigenvalues)))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # 3. First 4 PCA Mode Variances
    mode_evs = np.array(model_summary["retained_mode_eigenvalues"])[:min(8, k_retained)]
    mode_ratios = np.array(model_summary["explained_variance_ratio"])[:len(mode_evs)] * 100.0
    ax3 = axes[1, 0]
    bars = ax3.bar(range(1, len(mode_ratios) + 1), mode_ratios, color="#9467bd", alpha=0.85, edgecolor="black")
    for bar, r in zip(bars, mode_ratios):
        ax3.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.5, f"{r:.1f}%", ha="center", va="bottom", fontsize=8)
    ax3.set_title("3. Individual Mode Variance Contribution (%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Retained PCA Mode Index")
    ax3.set_ylabel("Explained Variance (%)")
    ax3.set_ylim(0, max(mode_ratios) * 1.15 if len(mode_ratios) > 0 else 10)
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. Population Reconstruction RMSE Histogram
    ax4 = axes[1, 1]
    rmses = np.array(reconstruction_rmses)
    ax4.hist(rmses, bins=20, color="#2ca02c", alpha=0.75, edgecolor="black")
    mean_rmse = float(np.mean(rmses))
    std_rmse = float(np.std(rmses))
    ax4.axvline(mean_rmse, color="#d62728", linestyle="-", linewidth=2, label=f"Mean = {mean_rmse:.2f}mm")
    ax4.set_title("4. Patient Reconstruction RMSE Distribution", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Reconstruction RMSE (mm)")
    ax4.set_ylabel("Patient Count")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_batch4(args: argparse.Namespace) -> int:
    batch3_dir = args.batch3_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Batch 4 PCA / Statistical Shape Model Fitting ===")
    print(f" Source Batch 3 Directory: {batch3_dir}")
    print(f" Output Directory:         {output_dir}")
    print(f" Variance Cutoff:          {args.variance_cutoff * 100:.1f}%\n")

    res = process_batch4_population(batch3_dir=batch3_dir, variance_cutoff=args.variance_cutoff)

    model: StatisticalShapeModel = res["shape_model"]

    # 1. Export Statistical Shape Model JSON & NPZ
    model.save(output_dir / "statistical_shape_model.json", output_dir / "statistical_shape_model.npz")

    # 2. Export Population Statistics JSON
    (output_dir / "population_statistics.json").write_text(
        json.dumps(jsonable(res["population_statistics"]), indent=2) + "\n", encoding="utf-8"
    )

    # 3. Export Validation Thresholds JSON
    (output_dir / "validation_thresholds.json").write_text(
        json.dumps(jsonable(res["validation_thresholds"]), indent=2) + "\n", encoding="utf-8"
    )

    # 4. Export Patient Reconstructions JSON
    (output_dir / "patient_reconstructions.json").write_text(
        json.dumps(jsonable(res["patient_reconstructions"]), indent=2) + "\n", encoding="utf-8"
    )

    # 5. Export Invariant Validation JSON
    (output_dir / "validation.json").write_text(
        json.dumps(jsonable(res["validation"]), indent=2) + "\n", encoding="utf-8"
    )

    # 6. Generate 4-Panel QC Visual Diagnostic Plot
    reconstruction_rmses = [p["reconstruction_rmse"] for p in res["patient_reconstructions"]]
    qc_plot_path = output_dir / "qc_visualization.png"
    generate_batch4_qc_plot(
        model_summary=model.to_dict(),
        reconstruction_rmses=reconstruction_rmses,
        output_path=qc_plot_path,
    )

    # 7. Export Batch 4 Summary Report JSON
    summary_report = {
        "batch": "Batch 4 — PCA / Statistical Shape Model (SSM)",
        "population": {
            "total_accepted_batch3_patients": res["n_patients"],
            "population_shape_matrix_dimensions": res["population_shape_matrix_shape"],
            "shape_vector_length": 126,
            "vessel_ordering": ["RCA (0:45)", "LMCA (45:60)", "LAD (60:96)", "LCX (96:126)"],
        },
        "pca_summary": {
            "variance_cutoff": args.variance_cutoff,
            "k_retained_modes": model.k_retained,
            "retained_cumulative_variance_percent": float(model.cumulative_explained_variance[model.k_retained - 1] * 100.0),
            "retained_mode_eigenvalues": model.eigenvalues_retained.tolist(),
            "retained_mode_std": model.mode_stds.tolist(),
        },
        "reconstruction_performance": res["population_statistics"]["vessel_deviations_level_3"]["reconstruction_rmse"],
        "scaffold_statistics": res["population_statistics"]["scaffold_level_1"],
        "validation": res["validation"],
    }

    (output_dir / "batch4_summary.json").write_text(
        json.dumps(jsonable(summary_report), indent=2) + "\n", encoding="utf-8"
    )

    print("=" * 80)
    print(" BATCH 4 PCA / STATISTICAL SHAPE MODEL SUMMARY REPORT")
    print("=" * 80)
    print(f" Total Population (Batch 3 Accepted): {res['n_patients']}")
    print(f" Population Matrix Shape:              {res['population_shape_matrix_shape']}")
    print(f" Target Cumulative Variance Cutoff:   {args.variance_cutoff * 100:.1f}%")
    print(f" Retained PCA Modes (k):               {model.k_retained} modes")
    print(f" Actual Cumulative Variance Explained: {model.cumulative_explained_variance[model.k_retained - 1] * 100:.2f}%")
    print(f" Mean Reconstruction RMSE:             {res['population_statistics']['vessel_deviations_level_3']['reconstruction_rmse']['mean']:.4f} mm")
    print(f" Max Reconstruction RMSE:              {res['population_statistics']['vessel_deviations_level_3']['reconstruction_rmse']['max']:.4f} mm")
    print(f" SVD Basis Orthogonality Error:        {res['validation']['orthogonality_error']:.2e}")
    print(f" Full-Rank Reconstruction RMSE Error:  {res['validation']['max_full_rank_reconstruction_rmse']:.2e}")
    print(f" Overall Validation Status:            {'PASSED' if res['validation']['overall_pass'] else 'FAILED'}")
    print("=" * 80 + "\n")

    return 0 if res["validation"]["overall_pass"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch 4 PCA / Statistical Shape Model Runner")
    parser.add_argument(
        "--batch3-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch3_surface_projection",
        help="Path to Batch 3 surface projection output directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch4_pca_ssm",
        help="Output directory for Batch 4 PCA / SSM artifacts",
    )
    parser.add_argument(
        "--variance-cutoff",
        type=float,
        default=0.95,
        help="Cumulative explained variance threshold for retaining PCA modes (default 0.95)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_batch4(parse_args()))
