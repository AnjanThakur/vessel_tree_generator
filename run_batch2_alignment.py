"""Runner script for Batch 2 — Plane Fitting, Per-Patient Cardiac Frame, & Global Alignment.

Processes all 200 Batch-1 patient records:
- Excludes 24 Batch-1 extraction failures.
- Processes 176 LCA-valid records:
  - 175 RCA-resolved cases attempted for 4-vessel plane fit and rigid cardiac transformation.
  - 1 RCA-unresolved case (82.label) safely skipped with status 'rca_unresolved'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

# Ensure project directory is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from pca_ssm_vessel_tree_generator.alignment.cardiac_frame_pipeline import (
    process_patient_cardiac_frame,
)
from pca_ssm_vessel_tree_generator.alignment.batch2_qc import (
    save_batch2_qc_plot,
)
from pca_ssm_vessel_tree_generator.alignment.batch2_validator import (
    validate_batch2_invariants,
)


def run_batch2(
    batch1_dir: Path,
    output_dir: Path,
    qc_plot_count: int = 20,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = batch1_dir / "batch1_extraction_summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Batch-1 summary file not found at {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        batch1_summary = json.load(f)

    patient_records = batch1_summary.get("results", batch1_summary.get("patient_results", []))
    total_records = len(patient_records)

    batch1_failed_count = 0
    lca_valid_count = 0
    rca_unresolved_skipped_count = 0
    alignment_attempted_count = 0
    alignment_succeeded_count = 0
    alignment_failed_count = 0

    patient_summaries = []
    rms_residuals = []
    lad_z_changes = []
    ring_z_rms_list = []
    max_length_errors = []
    max_landmark_errors = []

    specific_qc_patients = {"1.label", "82.label", "103.label"}

    for idx, p_res in enumerate(patient_records):
        p_id = p_res["patient_id"]
        lca_succeeded = p_res.get("lca_succeeded", False)
        rca_resolved = p_res.get("rca_resolved", False)
        rca_unresolved_reason = p_res.get("rca_unresolved_reason", "")

        if not lca_succeeded:
            batch1_failed_count += 1
            patient_summaries.append({
                "patient_id": p_id,
                "batch2_status": "batch1_failed",
                "batch2_passed": False,
                "rejection_reason": "Excluded: Extraction failed in Batch 1",
            })
            continue

        lca_valid_count += 1

        p_batch1_dir = batch1_dir / p_id
        npz_path = p_batch1_dir / "centerlines_scanner.npz"
        if not npz_path.exists():
            npz_path = p_batch1_dir / "centerlines.npz"

        json_path = p_batch1_dir / "extracted_data.json"
        if not json_path.exists():
            json_path = p_batch1_dir / "landmarks.json"

        if not rca_resolved:
            rca_unresolved_skipped_count += 1
            patient_summaries.append({
                "patient_id": p_id,
                "batch2_status": "rca_unresolved",
                "batch2_passed": False,
                "rejection_reason": f"Skipped: RCA unresolved in Batch 1 ({rca_unresolved_reason})",
            })
            # Generate QC plot for 82.label if needed
            if p_id in specific_qc_patients and npz_path.exists():
                c_data = np.load(npz_path)
                lm_data = {}
                if json_path.exists():
                    with open(json_path, "r", encoding="utf-8") as jf:
                        jcontent = json.load(jf)
                        lm_raw = jcontent.get("landmarks", jcontent)
                        lm_data = {k: np.array(v) for k, v in lm_raw.items()}
                save_batch2_qc_plot(
                    centerlines_scanner={k: c_data[k] for k in c_data.files},
                    landmarks_scanner=lm_data,
                    centerlines_cardiac={},
                    landmarks_cardiac={},
                    output_path=output_dir / p_id / "qc_visualization.png",
                    patient_id=p_id,
                    status_str="rca_unresolved",
                )
            continue

        alignment_attempted_count += 1

        if not npz_path.exists() or not json_path.exists():
            alignment_failed_count += 1
            patient_summaries.append({
                "patient_id": p_id,
                "batch2_status": "missing_inputs",
                "batch2_passed": False,
                "rejection_reason": "Missing centerlines or landmarks file from Batch 1",
            })
            continue

        with np.load(npz_path) as c_data:
            centerlines_scanner = {k: c_data[k] for k in c_data.files}

        with open(json_path, "r", encoding="utf-8") as jf:
            jcontent = json.load(jf)
            lm_raw = jcontent.get("landmarks", jcontent)
            landmarks_scanner = {k: np.asarray(v) for k, v in lm_raw.items()}

        # Run Batch 2 Cardiac Frame Pipeline
        pipe_res = process_patient_cardiac_frame(
            centerlines_scanner=centerlines_scanner,
            landmarks_scanner=landmarks_scanner,
            patient_id=p_id,
            rca_resolved=True,
        )

        p_out_dir = output_dir / p_id
        p_out_dir.mkdir(parents=True, exist_ok=True)

        if pipe_res["batch2_passed"]:
            alignment_succeeded_count += 1
            card_cl = pipe_res["transformed_centerlines"]
            card_lm = pipe_res["transformed_landmarks"]

            # Validate Batch 2 Rigid Invariants
            inv_res = validate_batch2_invariants(
                centerlines_scanner=centerlines_scanner,
                centerlines_cardiac=card_cl,
                landmarks_scanner=landmarks_scanner,
                landmarks_cardiac=card_lm,
                R_total=np.array(pipe_res["R_total"]),
                origin=np.array(pipe_res["origin_scanner_ras_mm"]),
            )

            # Export NPZ
            np.savez_compressed(p_out_dir / "centerlines_cardiac.npz", **card_cl)

            # Export Landmarks JSON
            with open(p_out_dir / "landmarks_cardiac.json", "w", encoding="utf-8") as f:
                json.dump({k: v.tolist() for k, v in card_lm.items()}, f, indent=2)

            # Export Frame JSON
            with open(p_out_dir / "cardiac_frame.json", "w", encoding="utf-8") as f:
                json.dump({
                    "origin_scanner_ras_mm": pipe_res["origin_scanner_ras_mm"],
                    "R_total": pipe_res["R_total"],
                    "coronary_normal": pipe_res["coronary_fit"]["coronary_normal"].tolist(),
                    "iv_normal": pipe_res["iv_fit"]["iv_normal"].tolist(),
                    "coronary_centroid": pipe_res["coronary_fit"]["coronary_centroid"].tolist(),
                }, f, indent=2)

            # Export Plane QC JSON
            with open(p_out_dir / "plane_qc.json", "w", encoding="utf-8") as f:
                json.dump({
                    "rms_residual_mm": pipe_res["coronary_fit"]["rms_residual_mm"],
                    "mean_residual_mm": pipe_res["coronary_fit"]["mean_residual_mm"],
                    "max_residual_mm": pipe_res["coronary_fit"]["max_residual_mm"],
                    "singular_values": pipe_res["coronary_fit"]["singular_values"],
                    "rank": pipe_res["coronary_fit"]["rank"],
                    "iv_orthogonality_error": pipe_res["iv_fit"]["orthogonality_error"],
                    "residuals": pipe_res["residuals"],
                }, f, indent=2)

            # Export Validation JSON
            with open(p_out_dir / "validation.json", "w", encoding="utf-8") as f:
                json.dump(inv_res, f, indent=2)

            # Record stats
            rms_residuals.append(pipe_res["coronary_fit"]["rms_residual_mm"])
            lad_z_changes.append(pipe_res["residuals"]["lad_z_change_mm"])
            ring_z_rms_list.append(pipe_res["residuals"]["ring_z_rms_mm"])
            max_length_errors.append(inv_res["max_segment_length_error_mm"])
            max_landmark_errors.append(inv_res["max_landmark_distance_error_mm"])

            # Save QC Plot
            if alignment_succeeded_count <= qc_plot_count or p_id in specific_qc_patients:
                save_batch2_qc_plot(
                    centerlines_scanner=centerlines_scanner,
                    landmarks_scanner=landmarks_scanner,
                    centerlines_cardiac=card_cl,
                    landmarks_cardiac=card_lm,
                    output_path=p_out_dir / "qc_visualization.png",
                    patient_id=p_id,
                    status_str="alignment_succeeded",
                )

            patient_summaries.append({
                "patient_id": p_id,
                "batch2_status": "alignment_succeeded",
                "batch2_passed": True,
                "coronary_rms_residual_mm": pipe_res["coronary_fit"]["rms_residual_mm"],
                "iv_orthogonality_error": pipe_res["iv_fit"]["orthogonality_error"],
                "lad_z_change_mm": pipe_res["residuals"]["lad_z_change_mm"],
                "canonical_lca_angle_deg": pipe_res["residuals"]["canonical_lca_angle_deg"],
            })
        else:
            alignment_failed_count += 1
            patient_summaries.append({
                "patient_id": p_id,
                "batch2_status": "alignment_failed",
                "batch2_passed": False,
                "rejection_reasons": pipe_res["rejection_reasons"],
            })

    global_summary = {
        "dataset_accounting": {
            "total_patient_records": total_records,
            "batch1_failed_excluded": batch1_failed_count,
            "lca_valid_eligible": lca_valid_count,
            "rca_unresolved_skipped": rca_unresolved_skipped_count,
            "alignment_attempted": alignment_attempted_count,
            "alignment_succeeded": alignment_succeeded_count,
            "alignment_failed": alignment_failed_count,
            "success_rate_over_eligible": float(alignment_succeeded_count / lca_valid_count) if lca_valid_count > 0 else 0.0,
            "success_rate_over_attempted": float(alignment_succeeded_count / alignment_attempted_count) if alignment_attempted_count > 0 else 0.0,
        },
        "population_residuals": {
            "coronary_rms_residual_mm_mean": float(np.mean(rms_residuals)) if rms_residuals else 0.0,
            "coronary_rms_residual_mm_std": float(np.std(rms_residuals)) if rms_residuals else 0.0,
            "lad_z_change_mm_mean": float(np.mean(lad_z_changes)) if lad_z_changes else 0.0,
            "ring_z_rms_mm_mean": float(np.mean(ring_z_rms_list)) if ring_z_rms_list else 0.0,
            "max_segment_length_error_mm_max": float(np.max(max_length_errors)) if max_length_errors else 0.0,
            "max_landmark_distance_error_mm_max": float(np.max(max_landmark_errors)) if max_landmark_errors else 0.0,
        },
        "patient_summaries": patient_summaries,
    }

    with open(output_dir / "batch2_summary.json", "w", encoding="utf-8") as sf:
        json.dump(global_summary, sf, indent=2)

    return global_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Batch 2 Plane Fitting & Cardiac Frame Alignment")
    parser.add_argument("--batch1-dir", type=Path, default=Path("outputs/batch1_extracted"), help="Path to Batch 1 extracted outputs")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/batch2_cardiac_frame"), help="Path to save Batch 2 outputs")
    parser.add_argument("--qc-plot-count", type=int, default=20, help="Number of patient QC plots to generate")
    args = parser.parse_args()

    print(f"=== Starting Batch 2 Alignment across Batch 1 outputs in {args.batch1_dir} ===")
    res = run_batch2(args.batch1_dir, args.output_dir, args.qc_plot_count)

    ac = res["dataset_accounting"]
    pr = res["population_residuals"]

    print("\n" + "=" * 80)
    print(" BATCH 2 ALIGNMENT SUMMARY REPORT")
    print("=" * 80)
    print(f" Total Patient Records:               {ac['total_patient_records']}")
    print(f" Batch 1 Extraction Failed (Excluded): {ac['batch1_failed_excluded']} ({ac['batch1_failed_excluded']/ac['total_patient_records']*100:.1f}%)")
    print(f" LCA Valid Records (Eligible):         {ac['lca_valid_eligible']} ({ac['lca_valid_eligible']/ac['total_patient_records']*100:.1f}%)")
    print(f"   - RCA Unresolved (Skipped):        {ac['rca_unresolved_skipped']} ({ac['rca_unresolved_skipped']/ac['total_patient_records']*100:.1f}%)")
    print(f"   - Attempted for 4-Vessel Alignment: {ac['alignment_attempted']} ({ac['alignment_attempted']/ac['total_patient_records']*100:.1f}%)")
    print(f"     - Alignment Succeeded:           {ac['alignment_succeeded']} ({ac['alignment_succeeded']/ac['total_patient_records']*100:.1f}%)")
    print(f"     - Alignment Failed:              {ac['alignment_failed']} ({ac['alignment_failed']/ac['total_patient_records']*100:.1f}%)")
    print("=" * 80)
    print(f" Success Rate (Over Eligible 176):   {ac['success_rate_over_eligible']*100:.1f}%")
    print(f" Success Rate (Over Attempted 175):  {ac['success_rate_over_attempted']*100:.1f}%")
    print("=" * 80)
    print(f" Mean Coronary Plane RMS Residual:   {pr['coronary_rms_residual_mm_mean']:.3f} mm ± {pr['coronary_rms_residual_mm_std']:.3f} mm")
    print(f" Mean LAD Descent (Z-change):       {pr['lad_z_change_mm_mean']:.2f} mm")
    print(f" Mean Ring Z-Residual (RCA+LCX):     {pr['ring_z_rms_mm_mean']:.3f} mm")
    print(f" Max Segment Length Rigid Error:     {pr['max_segment_length_error_mm_max']:.2e} mm")
    print(f" Max Landmark Distance Rigid Error:  {pr['max_landmark_distance_error_mm_max']:.2e} mm")
    print("=" * 80)
