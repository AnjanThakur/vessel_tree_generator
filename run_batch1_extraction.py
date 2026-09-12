"""Batch 1 Extraction Runner Script.

Executes Batch 1 Extraction & Landmark Identification conforming to TDD Part 1 across all dataset cases.
Generates standardized scanner-frame JSON/NPZ outputs, validation metrics, and 3D visual QC plots.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure project directory is in python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from pca_ssm_vessel_tree_generator.extraction.nifti_loader import (
    load_centerline_archive,
    load_binary_nifti,
    read_nifti_header,
)
from pca_ssm_vessel_tree_generator.extraction.skeleton_graph import (
    build_graph_from_summary,
    build_graph_from_mask,
)
from pca_ssm_vessel_tree_generator.extraction.landmark_detector import (
    detect_ostia_pair,
    label_ostia_lca_rca,
    detect_lmca_bifurcation,
)
from pca_ssm_vessel_tree_generator.extraction.branch_splitter import (
    extract_all_main_branches,
)
from pca_ssm_vessel_tree_generator.extraction.batch1_validator import (
    validate_batch1_extraction,
)
from pca_ssm_vessel_tree_generator.extraction.batch1_qc import (
    save_batch1_qc_plot,
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


def process_patient_archive(
    case_dir: Path,
    patient_id: str,
    output_dir: Path,
    max_ostium_distance_mm: float = 45.0,
    generate_qc_plots: bool = True,
) -> dict[str, Any]:
    """Process a single patient centerline graph archive through Batch 1 extraction."""
    archive = load_centerline_archive(case_dir)
    branches_dict = archive["branches_dict"]
    summary_df = archive["summary_df"]

    # 1. Build Graph
    G, node_coords, endpoint_nodes, endpoint_coords, junction_nodes = build_graph_from_summary(
        summary_df, branches_dict
    )

    if len(endpoint_nodes) < 2:
        return {
            "patient_id": patient_id,
            "lca_succeeded": False,
            "rca_resolved": False,
            "overall_status": "extraction_failed",
            "failures": ["Fewer than 2 endpoints found in graph"],
            "warnings": [],
        }

    # 2. Ostia Detection (min mutual distance + centroid proximity)
    node_a, node_b, coord_a, coord_b, pair_dist, ostia_qc = detect_ostia_pair(
        endpoint_nodes, endpoint_coords, max_ostium_distance_mm=max_ostium_distance_mm
    )

    # 3. LCA vs RCA Ostium Labeling
    lca_ost_node, rca_ost_node, lca_ost_coord, rca_ost_coord, ostia_label_meta = label_ostia_lca_rca(
        G, node_a, node_b, junction_nodes
    )

    # 4. LMCA Bifurcation Detection
    bif_node, bif_coord, lmca_bif_meta = detect_lmca_bifurcation(
        G, lca_ost_node, junction_nodes
    )

    # 5. Main Vessel Path Extraction (LMCA, LAD, LCX, RCA main branch validation)
    branches_res = extract_all_main_branches(
        G, lca_ost_node, rca_ost_node, bif_node, branches_dict
    )

    centerlines = {
        "LMCA": branches_res["LMCA"],
        "LAD": branches_res["LAD"],
        "LCX": branches_res["LCX"],
        "RCA": branches_res["RCA"],
    }

    landmarks = {
        "lca_ostium": lca_ost_coord,
        "rca_ostium": rca_ost_coord,
        "bifurcation": bif_coord,
        "lad_endpoint": branches_res["lad_endpoint"],
        "lcx_endpoint": branches_res["lcx_endpoint"],
        "rca_endpoint": branches_res["rca_endpoint"],
    }

    # 6. Batch 1 Validation (12 checks)
    validation_res = validate_batch1_extraction(
        centerlines=centerlines,
        landmarks=landmarks,
        rca_resolved=branches_res["rca_resolved"],
        rca_unresolved_reason=branches_res["rca_unresolved_reason"],
        qc_metadata=ostia_qc,
    )

    # 7. Standardized Output Export
    patient_out_dir = output_dir / patient_id
    patient_out_dir.mkdir(parents=True, exist_ok=True)

    # Export NPZ centerlines
    valid_centerlines_npz = {
        k: v for k, v in centerlines.items() if v is not None and len(v) > 0
    }
    np.savez_compressed(patient_out_dir / "centerlines_scanner.npz", **valid_centerlines_npz)

    # Export JSON payload
    export_payload = {
        "patient_id": patient_id,
        "coordinate_space": "scanner_ras_physical_mm",
        "landmarks": landmarks,
        "validation": validation_res,
        "ostia_qc": ostia_qc,
        "lmca_bifurcation_meta": lmca_bif_meta,
        "lad_lcx_classification": branches_res["lad_lcx_classification"],
    }
    (patient_out_dir / "extracted_data.json").write_text(
        json.dumps(jsonable(export_payload), indent=2) + "\n", encoding="utf-8"
    )

    # 8. QC Plot Generation
    if generate_qc_plots:
        qc_plot_path = patient_out_dir / "qc_visualization.png"
        save_batch1_qc_plot(
            centerlines=centerlines,
            landmarks=landmarks,
            output_path=qc_plot_path,
            patient_id=patient_id,
            validation_res=validation_res,
        )

    res_summary = {
        "patient_id": patient_id,
        "lca_succeeded": validation_res["lca_succeeded"],
        "rca_resolved": validation_res["rca_resolved"],
        "overall_status": validation_res["overall_status"],
        "ostia_distance_mm": pair_dist,
        "failures": validation_res["failures"],
        "warnings": validation_res["warnings"],
        "patient_output_dir": str(patient_out_dir),
    }

    return res_summary


def run_batch1(args: argparse.Namespace) -> int:
    centerline_root = args.centerlines_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted([d for d in centerline_root.glob("*.label") if d.is_dir()])
    if not case_dirs:
        case_dirs = sorted([d for d in centerline_root.iterdir() if d.is_dir()])

    print(f"=== Starting Batch 1 Extraction across {len(case_dirs)} cases in {centerline_root} ===")

    results = []
    n_succeeded_rca_resolved = 0
    n_succeeded_rca_unresolved = 0
    n_failed = 0

    for idx, case_dir in enumerate(case_dirs, 1):
        patient_id = case_dir.name
        try:
            res = process_patient_archive(
                case_dir=case_dir,
                patient_id=patient_id,
                output_dir=output_dir,
                max_ostium_distance_mm=args.max_ostium_distance_mm,
                generate_qc_plots=(idx <= args.qc_plot_count),
            )
            results.append(res)
            status = res["overall_status"]
            if status == "extraction_succeeded_rca_resolved":
                n_succeeded_rca_resolved += 1
            elif status == "extraction_succeeded_rca_unresolved":
                n_succeeded_rca_unresolved += 1
            else:
                n_failed += 1
        except Exception as exc:
            n_failed += 1
            results.append({
                "patient_id": patient_id,
                "lca_succeeded": False,
                "rca_resolved": False,
                "overall_status": "extraction_failed",
                "failures": [f"Exception: {exc}"],
                "warnings": [],
            })

    total = len(results)
    summary_report = {
        "batch": "Batch 1 — Data Extraction & Landmarks",
        "total_patients_processed": total,
        "extraction_succeeded_rca_resolved": n_succeeded_rca_resolved,
        "extraction_succeeded_rca_unresolved": n_succeeded_rca_unresolved,
        "extraction_failed": n_failed,
        "lca_success_rate_percent": float((n_succeeded_rca_resolved + n_succeeded_rca_unresolved) / max(total, 1) * 100),
        "rca_resolution_rate_percent": float(n_succeeded_rca_resolved / max(total, 1) * 100),
        "results": results,
    }

    (output_dir / "batch1_extraction_summary.json").write_text(
        json.dumps(jsonable(summary_report), indent=2) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 80)
    print(" BATCH 1 EXTRACTION SUMMARY REPORT")
    print("=" * 80)
    print(f" Total Patients Processed:            {total}")
    print(f" Extraction Succeeded + RCA Resolved:   {n_succeeded_rca_resolved} ({n_succeeded_rca_resolved/max(total,1)*100:.1f}%)")
    print(f" Extraction Succeeded + RCA Unresolved: {n_succeeded_rca_unresolved} ({n_succeeded_rca_unresolved/max(total,1)*100:.1f}%)")
    print(f" Extraction Failed:                    {n_failed} ({n_failed/max(total,1)*100:.1f}%)")
    print("=" * 80 + "\n")

    return 0 if (n_succeeded_rca_resolved + n_succeeded_rca_unresolved) > 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch 1 Data Extraction & Landmark Runner")
    parser.add_argument(
        "--centerlines-dir",
        type=Path,
        default=PROJECT_ROOT / "lca_ssm" / "centerlines",
        help="Path to centerlines archive directory containing patient folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "batch1_extracted",
        help="Output directory for Batch 1 extracted dataset",
    )
    parser.add_argument(
        "--max-ostium-distance-mm",
        type=float,
        default=45.0,
        help="QC threshold for maximum ostium-pair distance in mm",
    )
    parser.add_argument(
        "--qc-plot-count",
        type=int,
        default=10,
        help="Number of representative QC plots to generate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run_batch1(parse_args()))
