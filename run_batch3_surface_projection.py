"""Runner script for Batch 3 — Axis-Aligned Triaxial Ellipsoid & Parametric Surface Projection.

Executes across all 174 Batch-2 accepted patient records:
- Consumes inputs directly from outputs/batch2_cardiac_frame/{patient_id}/.
- Derives axis-aligned triaxial ellipsoid parameters (a, b, c).
- Projects RCA, LMCA, LAD, LCX centerlines onto ellipsoid.
- Constructs fixed 42-point representation & 126-D shape vector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from surface_relative.batch3_pipeline import process_patient_batch3


def save_batch3_qc_plot(
    centerlines_cardiac: dict[str, np.ndarray],
    a: float,
    b: float,
    c: float,
    output_path: Path,
    patient_id: str,
) -> None:
    """Generate 3D visual QC plot of canonical cardiac centerlines overlaid on fitted Triaxial Ellipsoid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    colors = {
        "LMCA": "#1A1A1A",   # Black
        "LAD": "#D62828",    # Red
        "LCX": "#1679B8",    # Blue
        "RCA": "#77589A",    # Purple
    }

    for name, pts in centerlines_cardiac.items():
        vname = name.upper()
        if pts is not None and len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors.get(vname, "#555555"), linewidth=2.5, label=vname)

    # Wireframe ellipsoid mesh
    u_grid = np.linspace(0, 2 * np.pi, 30)
    v_grid = np.linspace(0, np.pi, 20)
    ex = a * np.outer(np.cos(u_grid), np.sin(v_grid))
    ey = b * np.outer(np.sin(u_grid), np.sin(v_grid))
    ez = c * np.outer(np.ones_like(u_grid), np.cos(v_grid))

    ax.plot_wireframe(ex, ey, ez, color="cyan", alpha=0.15, linewidth=0.5)

    ax.set_title(f"{patient_id} — Triaxial Ellipsoid (a={a:.1f}, b={b:.1f}, c={c:.1f} mm)", fontsize=10, fontweight="bold")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_batch3(
    batch2_dir: Path,
    output_dir: Path,
    qc_plot_count: int = 20,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    b2_summary_path = batch2_dir / "batch2_summary.json"

    if not b2_summary_path.exists():
        raise FileNotFoundError(f"Batch-2 summary file not found at {b2_summary_path}")

    with open(b2_summary_path, "r", encoding="utf-8") as f:
        batch2_summary = json.load(f)

    patient_summaries = batch2_summary.get("patient_summaries", [])

    total_records = 200
    batch1_failed_count = 24
    lca_valid_count = 176
    rca_unresolved_count = 1
    batch2_attempted_count = 175
    batch2_failed_count = 1
    batch3_eligible_count = 174

    batch3_succeeded_count = 0
    batch3_failed_count = 0

    patient_results = []
    a_list, b_list, c_list = [], [], []
    b_rel_errors = []
    max_surface_eq_errors = []

    for p_info in patient_summaries:
        p_id = p_info["patient_id"]
        b2_status = p_info.get("batch2_status", "")

        if b2_status != "alignment_succeeded":
            patient_results.append({
                "patient_id": p_id,
                "batch3_status": "batch2_excluded",
                "batch3_passed": False,
                "rejection_reason": f"Excluded: Batch 2 status = {b2_status}",
            })
            continue

        p_b2_dir = batch2_dir / p_id
        npz_path = p_b2_dir / "centerlines_cardiac.npz"
        lm_path = p_b2_dir / "landmarks_cardiac.json"

        if not npz_path.exists() or not lm_path.exists():
            batch3_failed_count += 1
            patient_results.append({
                "patient_id": p_id,
                "batch3_status": "missing_inputs",
                "batch3_passed": False,
                "rejection_reason": "Missing centerlines_cardiac.npz or landmarks_cardiac.json",
            })
            continue

        with np.load(npz_path) as c_data:
            centerlines_cardiac = {k: c_data[k] for k in c_data.files}

        with open(lm_path, "r", encoding="utf-8") as lf:
            landmarks_cardiac = {k: np.asarray(v) for k, v in json.load(lf).items()}

        # Run Batch 3 Pipeline
        b3_res = process_patient_batch3(
            centerlines_cardiac=centerlines_cardiac,
            landmarks_cardiac=landmarks_cardiac,
            patient_id=p_id,
        )

        p_out_dir = output_dir / p_id
        p_out_dir.mkdir(parents=True, exist_ok=True)

        if b3_res["batch3_passed"]:
            batch3_succeeded_count += 1
            ell = b3_res["ellipsoid"]
            meta = b3_res["ellipsoid_metadata"]
            projs = b3_res["projections"]
            fixed = b3_res["fixed_representation"]
            val = b3_res["validation"]

            # Export ellipsoid_params.json
            with open(p_out_dir / "ellipsoid_params.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            # Export surface_projection.json
            proj_export = {}
            for vname, vdata in projs.items():
                if vdata is not None:
                    proj_export[vname] = {
                        "u": vdata["u"].tolist(),
                        "v": vdata["v"].tolist(),
                        "offset": vdata["offset"].tolist(),
                        "deviations": vdata["deviations"].tolist(),
                        "surface_points": vdata["surface_points"].tolist(),
                    }
                else:
                    proj_export[vname] = None

            with open(p_out_dir / "surface_projection.json", "w", encoding="utf-8") as f:
                json.dump(proj_export, f, indent=2)

            # Export fixed_representation.json
            fixed_export = {
                "fixed_branches": {
                    k: {
                        "cardiac_points": v["cardiac_points"].tolist(),
                        "u": v["u"].tolist(),
                        "v": v["v"].tolist(),
                        "offset": v["offset"].tolist(),
                        "uvo": v["uvo"].tolist(),
                    } if v is not None else None
                    for k, v in fixed["fixed_branches"].items()
                },
                "shape_vector": fixed["shape_vector"].tolist() if fixed["shape_vector"] is not None else None,
                "shape_vector_dimension": len(fixed["shape_vector"]) if fixed["shape_vector"] is not None else 0,
                "ordering": "RCA -> LMCA -> LAD -> LCX",
            }
            with open(p_out_dir / "fixed_representation.json", "w", encoding="utf-8") as f:
                json.dump(fixed_export, f, indent=2)

            # Export validation.json
            with open(p_out_dir / "validation.json", "w", encoding="utf-8") as f:
                json.dump(val, f, indent=2)

            # Record population stats
            a_list.append(ell.a)
            b_list.append(ell.b)
            c_list.append(ell.c)
            b_rel_errors.append(ell.b_consistency_relative_error)
            max_surface_eq_errors.append(val["max_surface_equation_error"])

            # Save QC Plot
            if batch3_succeeded_count <= qc_plot_count or p_id in {"1.label", "82.label", "103.label"}:
                save_batch3_qc_plot(
                    centerlines_cardiac=centerlines_cardiac,
                    a=ell.a,
                    b=ell.b,
                    c=ell.c,
                    output_path=p_out_dir / "qc_visualization.png",
                    patient_id=p_id,
                )

            patient_results.append({
                "patient_id": p_id,
                "batch3_status": "surface_projection_succeeded",
                "batch3_passed": True,
                "a_mm": ell.a,
                "b_mm": ell.b,
                "c_mm": ell.c,
                "b_consistency_relative_error": ell.b_consistency_relative_error,
            })
        else:
            batch3_failed_count += 1
            patient_results.append({
                "patient_id": p_id,
                "batch3_status": "surface_projection_failed",
                "batch3_passed": False,
                "rejection_reasons": b3_res["rejection_reasons"],
            })

    summary_payload = {
        "dataset_accounting": {
            "total_patient_records": total_records,
            "batch1_failed_excluded": batch1_failed_count,
            "lca_valid_eligible": lca_valid_count,
            "rca_unresolved_skipped": rca_unresolved_count,
            "batch2_attempted": batch2_attempted_count,
            "batch2_accepted_batch3_eligible": batch3_eligible_count,
            "batch2_failed_excluded": batch2_failed_count,
            "batch3_succeeded": batch3_succeeded_count,
            "batch3_failed": batch3_failed_count,
            "success_rate_over_eligible": float(batch3_succeeded_count / batch3_eligible_count) if batch3_eligible_count > 0 else 0.0,
        },
        "population_ellipsoid_statistics": {
            "a_mm_mean": float(np.mean(a_list)) if a_list else 0.0,
            "a_mm_std": float(np.std(a_list)) if a_list else 0.0,
            "b_mm_mean": float(np.mean(b_list)) if b_list else 0.0,
            "b_mm_std": float(np.std(b_list)) if b_list else 0.0,
            "c_mm_mean": float(np.mean(c_list)) if c_list else 0.0,
            "c_mm_std": float(np.std(c_list)) if c_list else 0.0,
            "b_consistency_relative_error_mean": float(np.mean(b_rel_errors)) if b_rel_errors else 0.0,
            "max_surface_equation_error_max": float(np.max(max_surface_eq_errors)) if max_surface_eq_errors else 0.0,
        },
        "patient_results": patient_results,
    }

    with open(output_dir / "batch3_summary.json", "w", encoding="utf-8") as sf:
        json.dump(summary_payload, sf, indent=2)

    return summary_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Batch 3 Ellipsoid Fitting & Parametric Surface Projection")
    parser.add_argument("--batch2-dir", type=Path, default=Path("outputs/batch2_cardiac_frame"), help="Path to Batch 2 outputs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/batch3_surface_projection"), help="Path to save Batch 3 outputs")
    parser.add_argument("--qc-plot-count", type=int, default=20, help="Number of patient QC plots to generate")
    args = parser.parse_args()

    print(f"=== Starting Batch 3 Surface Projection across Batch 2 outputs in {args.batch2_dir} ===")
    res = run_batch3(args.batch2_dir, args.output_dir, args.qc_plot_count)

    ac = res["dataset_accounting"]
    es = res["population_ellipsoid_statistics"]

    print("\n" + "=" * 80)
    print(" BATCH 3 SURFACE PROJECTION SUMMARY REPORT")
    print("=" * 80)
    print(f" Total Patient Records:               {ac['total_patient_records']}")
    print(f" Batch 1 Failed (Excluded):           {ac['batch1_failed_excluded']} ({ac['batch1_failed_excluded']/ac['total_patient_records']*100:.1f}%)")
    print(f" LCA Valid Records:                   {ac['lca_valid_eligible']} ({ac['lca_valid_eligible']/ac['total_patient_records']*100:.1f}%)")
    print(f"   - RCA Unresolved (Skipped):        {ac['rca_unresolved_skipped']} ({ac['rca_unresolved_skipped']/ac['total_patient_records']*100:.1f}%)")
    print(f" Batch 2 Attempted:                   {ac['batch2_attempted']}")
    print(f"   - Batch 2 Failed:                  {ac['batch2_failed_excluded']}")
    print(f" Batch 3 Eligible (Batch 2 Accepted): {ac['batch2_accepted_batch3_eligible']}")
    print(f"   - Batch 3 Succeeded:               {ac['batch3_succeeded']} ({ac['batch3_succeeded']/ac['batch2_accepted_batch3_eligible']*100:.1f}%)")
    print(f"   - Batch 3 Failed:                  {ac['batch3_failed']} ({ac['batch3_failed']/ac['batch2_accepted_batch3_eligible']*100:.1f}%)")
    print("=" * 80)
    print(f" Success Rate (Over Eligible 174):   {ac['success_rate_over_eligible']*100:.1f}%")
    print("=" * 80)
    print(f" Mean Ellipsoid Semi-Axes:")
    print(f"   a (Cardiac X): {es['a_mm_mean']:.2f} mm ± {es['a_mm_std']:.2f} mm")
    print(f"   b (Cardiac Y): {es['b_mm_mean']:.2f} mm ± {es['b_mm_std']:.2f} mm")
    print(f"   c (Cardiac Z): {es['c_mm_mean']:.2f} mm ± {es['c_mm_std']:.2f} mm")
    print(f" Mean b-Axis Consistency Rel Error:   {es['b_consistency_relative_error_mean']*100:.2f}%")
    print(f" Max Surface Equation Error:         {es['max_surface_equation_error_max']:.2e}")
    print("=" * 80)
