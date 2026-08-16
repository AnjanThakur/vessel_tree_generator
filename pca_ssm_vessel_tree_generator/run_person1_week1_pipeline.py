#!/usr/bin/env python
"""Person 1 — Week 1 Pipeline Runner.

Converts the completed 191-patient PPT model into the surface-relative statistical representation.

Supports:
  --smoke-test    Run on 3 patients (117.label, 91.label, 1.label) and print detailed verification report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# Import surface_relative package modules
from surface_relative.cardiac_frame import compute_cardiac_frame, transform_to_cardiac_frame, validate_cardiac_frame
from surface_relative.ellipsoid_model import derive_patient_ellipsoid
from surface_relative.fixed_representation import build_patient_fixed_representation
from surface_relative.integrity import verify_protected_assets
from surface_relative.surface_projection import project_point_to_surface
from surface_relative.population_statistics import (
    build_validation_thresholds,
    compute_circular_stats,
    compute_linear_stats,
    fit_deviation_pca,
)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
PPT_OUTPUT_DIR = REPO_ROOT / "outputs" / "lca_ssm" / "ppt_priority_completion"
CENTERLINES_DIR = BASE_DIR / "centerlines"
NII_DIR = BASE_DIR / "nii files"
OUTPUT_DIR = REPO_ROOT / "outputs" / "lca_ssm" / "person1_week1"

SMOKE_TEST_CASES = ["117.label", "91.label", "1.label"]


def jsonable(val: Any) -> Any:
    if isinstance(val, Path):
        return str(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, dict):
        return {str(k): jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [jsonable(v) for v in val]
    return val


def read_ppt_parameters(csv_path: Path) -> dict[str, dict[str, Any]]:
    """Read existing PPT two-plane two-ellipse parameters CSV into case_id dict."""
    rows = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["case_id"]] = row
    return rows


from lca_ssm_label_adapter import extract_lca_from_label_case
from build_lca_population_ssm import infer_rca_candidate


def load_extracted_case_archive(case_id: str) -> dict[str, np.ndarray] | None:
    """Load patient's extracted physical RAS centerlines (lmca, lad, lcx, rca).

    Uses the existing checked pipeline's adapter logic to load centerlines from lca_ssm/centerlines and lca_ssm/nii files.
    """
    clean_id = case_id.replace(".nii.gz", "").replace(".nii", "")
    if not clean_id.endswith(".label"):
        label_id = f"{clean_id}.label"
    else:
        label_id = clean_id

    c_dir = REPO_ROOT / "lca_ssm" / "centerlines" / label_id
    n_path = REPO_ROOT / "lca_ssm" / "nii files" / f"{label_id}.nii.gz"

    if not c_dir.is_dir() or not n_path.is_file():
        # Fallback to without .label suffix
        c_dir = REPO_ROOT / "lca_ssm" / "centerlines" / clean_id
        n_path = REPO_ROOT / "lca_ssm" / "nii files" / f"{clean_id}.nii.gz"
        if not c_dir.is_dir() or not n_path.is_file():
            return None

    try:
        tree, meta = extract_lca_from_label_case(c_dir, n_path)
        root_node = meta["root_selection"]["node_id"]
        rca, rca_meta = infer_rca_candidate(c_dir, n_path, root_node)
        data = {
            "lmca": tree["lmca"],
            "lad": tree["lad"],
            "lcx": tree["lcx"],
        }
        if rca is not None and len(rca) >= 5:
            data["rca"] = rca
        else:
            data["rca"] = None
        return data
    except Exception as err:
        print(f"Error loading case {case_id}: {err}")
        return None


def run_pipeline(smoke_test: bool = True) -> dict[str, Any]:
    """Execute Person 1 Week 1 pipeline."""
    print("=" * 70)
    print(f"Running Person 1 Week 1 Pipeline (Smoke Test Mode: {smoke_test})")
    print("=" * 70)

    # 0. Verify integrity of protected assets
    integrity_res = verify_protected_assets(PPT_OUTPUT_DIR)
    print(f"Protected Assets Check: All Exist = {integrity_res['all_files_exist']}, Hashes Match = {integrity_res['hashes_matched']}")

    ppt_params_csv = PPT_OUTPUT_DIR / "population_two_plane_two_ellipse_parameters.csv"
    if not ppt_params_csv.is_file():
        raise FileNotFoundError(f"Missing authoritative PPT parameters CSV at {ppt_params_csv}")

    ppt_rows = read_ppt_parameters(ppt_params_csv)
    print(f"Loaded {len(ppt_rows)} patient parameter rows from PPT outputs.")

    target_cases = SMOKE_TEST_CASES if smoke_test else sorted(ppt_rows.keys(), key=lambda c: int(c.split('.')[0]) if c.split('.')[0].isdigit() else 999)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cardiac_frames_rows = []
    cardiac_frame_val_rows = []
    ellipsoid_rows = []
    surface_coord_rows = []
    fixed_rep_matrices = []
    shape_vectors_list = []
    complete_case_ids = []

    scaffold_params_list = []
    branch_lengths = {"lmca": [], "lad": [], "lcx": [], "rca": []}
    bifurcation_angles = []

    smoke_reports = {}

    for case_id in target_cases:
        if case_id not in ppt_rows:
            print(f"Warning: {case_id} not found in PPT parameters CSV")
            continue

        ppt_row = ppt_rows[case_id]
        case_data = load_extracted_case_archive(case_id)
        if case_data is None:
            print(f"Warning: Centerline archive missing for {case_id}")
            continue

        raw_lmca = case_data.get("lmca")
        raw_lad = case_data.get("lad")
        raw_lcx = case_data.get("lcx")
        raw_rca = case_data.get("rca")

        if raw_lmca is None or raw_lad is None or raw_lcx is None:
            print(f"Warning: Incomplete mandatory centerlines for {case_id}")
            continue

        lca_ostium = raw_lmca[0]
        coronary_normal = np.array([float(ppt_row["coronary_plane_normal_x"]), float(ppt_row["coronary_plane_normal_y"]), float(ppt_row["coronary_plane_normal_z"])])
        iv_normal = np.array([float(ppt_row["lad_plane_normal_x"]), float(ppt_row["lad_plane_normal_y"]), float(ppt_row["lad_plane_normal_z"])])
        coronary_centroid = np.array([float(ppt_row["coronary_plane_centroid_x"]), float(ppt_row["coronary_plane_centroid_y"]), float(ppt_row["coronary_plane_centroid_z"])])

        # Stage 1: Cardiac Frame
        origin, R_total, frame_meta = compute_cardiac_frame(
            coronary_normal=coronary_normal,
            iv_normal=iv_normal,
            coronary_centroid=coronary_centroid,
            lad_centerline=raw_lad,
            lca_ostium=lca_ostium,
        )

        frame_val = validate_cardiac_frame(
            origin=origin,
            R_total=R_total,
            branches_raw={"lmca": raw_lmca, "lad": raw_lad, "lcx": raw_lcx, "rca": raw_rca},
        )

        cardiac_frames_rows.append({
            "case_id": case_id,
            "frame_origin_x": origin[0], "frame_origin_y": origin[1], "frame_origin_z": origin[2],
            "axis_x_x": R_total[0, 0], "axis_x_y": R_total[0, 1], "axis_x_z": R_total[0, 2],
            "axis_y_x": R_total[1, 0], "axis_y_y": R_total[1, 1], "axis_y_z": R_total[1, 2],
            "axis_z_x": R_total[2, 0], "axis_z_y": R_total[2, 1], "axis_z_z": R_total[2, 2],
            "determinant": frame_val["determinant"],
            "delta_theta_deg": math.degrees(frame_meta["delta_theta_rad"]),
        })

        frame_val_row = {"case_id": case_id}
        frame_val_row.update(frame_val)
        cardiac_frame_val_rows.append(frame_val_row)

        # Stage 2: Ellipsoid Model
        ellipsoid = derive_patient_ellipsoid(ppt_row)
        ellipsoid_rows.append({
            "case_id": case_id,
            "a": ellipsoid.a, "b": ellipsoid.b, "c": ellipsoid.c,
            "ellipsoid_center_x": ellipsoid.center[0],
            "ellipsoid_center_y": ellipsoid.center[1],
            "ellipsoid_center_z": ellipsoid.center[2],
            "ellipse_center_separation": ellipsoid.ellipse_center_separation,
            "quality_flags": ";".join(ellipsoid.quality_flags),
            "is_valid": ellipsoid.is_valid,
        })

        scaffold_params_list.append({"a": ellipsoid.a, "b": ellipsoid.b, "c": ellipsoid.c})

        # Transform centerlines to cardiac frame (WITHOUT modifying raw_data!)
        cardiac_branches = {
            "lmca": transform_to_cardiac_frame(raw_lmca, origin, R_total),
            "lad": transform_to_cardiac_frame(raw_lad, origin, R_total),
            "lcx": transform_to_cardiac_frame(raw_lcx, origin, R_total),
            "rca": transform_to_cardiac_frame(raw_rca, origin, R_total) if raw_rca is not None and len(raw_rca) >= 5 else None,
        }

        # Stage 3: Surface Coordinates (u, v, offset)
        for b_name, pts_c in cardiac_branches.items():
            if pts_c is not None:
                raw_pts = case_data[b_name]
                # Measure lengths
                b_len = float(np.sum(np.linalg.norm(np.diff(raw_pts, axis=0), axis=1)))
                branch_lengths[b_name].append(b_len)

                for idx in range(len(pts_c)):
                    # Compute (u, v, offset)
                    u, v, off, _ = project_point_to_surface(pts_c[idx], ellipsoid.a, ellipsoid.b, ellipsoid.c)
                    surface_coord_rows.append({
                        "case_id": case_id,
                        "branch": b_name,
                        "source_point_index": idx,
                        "source_x": raw_pts[idx][0], "source_y": raw_pts[idx][1], "source_z": raw_pts[idx][2],
                        "cardiac_x": pts_c[idx][0], "cardiac_y": pts_c[idx][1], "cardiac_z": pts_c[idx][2],
                        "u": u, "v": v, "offset": off,
                    })

        # Measure bifurcation angle
        if len(cardiac_branches["lad"]) >= 2 and len(cardiac_branches["lcx"]) >= 2:
            v_lad = cardiac_branches["lad"][1] - cardiac_branches["lad"][0]
            v_lcx = cardiac_branches["lcx"][1] - cardiac_branches["lcx"][0]
            norm_lad = np.linalg.norm(v_lad)
            norm_lcx = np.linalg.norm(v_lcx)
            if norm_lad > 1e-6 and norm_lcx > 1e-6:
                cos_ang = np.clip(np.dot(v_lad, v_lcx) / (norm_lad * norm_lcx), -1.0, 1.0)
                bifurcation_angles.append(math.degrees(math.acos(cos_ang)))

        # Stage 4: Fixed Point Representation & 126-D shape vector
        fixed_rep = build_patient_fixed_representation(cardiac_branches, ellipsoid.a, ellipsoid.b, ellipsoid.c)
        if fixed_rep["is_complete"]:
            fixed_rep_matrices.append(fixed_rep["uvo_matrix_42_3"])
            shape_vectors_list.append(fixed_rep["shape_vector"])
            complete_case_ids.append(case_id)

        # Store smoke report data
        smoke_reports[case_id] = {
            "frame_pass": frame_val["overall_pass"],
            "determinant": frame_val["determinant"],
            "orthogonality_error": frame_val["orthogonality_error"],
            "max_length_error": max([v for k, v in frame_val.items() if k.startswith("length_error_")]),
            "LAD_apex_aligned": frame_val["LAD_apex_aligned"],
            "LAD_apex_z_diff": frame_val["LAD_apex_alignment_z_diff"],
            "ellipsoid_valid": ellipsoid.is_valid,
            "ellipsoid_abc": (ellipsoid.a, ellipsoid.b, ellipsoid.c),
            "has_fixed_points": fixed_rep["uvo_matrix_42_3"] is not None,
            "has_deviation_vector": fixed_rep["shape_vector"] is not None,
            "is_complete": fixed_rep["is_complete"],
        }

    # Write output files (all 11 handoff output files)
    print("\nWriting handoff output files to:", OUTPUT_DIR)

    # 1. population_cardiac_frames.csv
    with (OUTPUT_DIR / "population_cardiac_frames.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cardiac_frames_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cardiac_frames_rows)

    # 2. cardiac_frame_validation.csv
    with (OUTPUT_DIR / "cardiac_frame_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(cardiac_frame_val_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cardiac_frame_val_rows)

    # 3. population_ellipsoid_parameters.csv
    with (OUTPUT_DIR / "population_ellipsoid_parameters.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ellipsoid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ellipsoid_rows)

    # 4. population_surface_coordinates.csv
    with (OUTPUT_DIR / "population_surface_coordinates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(surface_coord_rows[0].keys()))
        writer.writeheader()
        writer.writerows(surface_coord_rows)

    # 5. fixed_surface_representation.npz
    fixed_matrices_arr = np.array(fixed_rep_matrices) if fixed_rep_matrices else np.zeros((0, 42, 3))
    np.savez_compressed(
        OUTPUT_DIR / "fixed_surface_representation.npz",
        fixed_surface_representation=fixed_matrices_arr,
        case_ids=np.array(complete_case_ids),
        branch_order=np.array(["RCA", "LMCA", "LAD", "LCX"]),
        branch_counts=np.array([15, 5, 12, 10]),
    )

    # 6. fixed_surface_representation_summary.json
    fixed_summary = {
        "n_total_cases": len(target_cases),
        "n_complete_cases": len(complete_case_ids),
        "complete_case_ids": complete_case_ids,
        "total_fixed_points": 42,
        "branch_counts": {"RCA": 15, "LMCA": 5, "LAD": 12, "LCX": 10},
        "fixed_matrix_shape": list(fixed_matrices_arr.shape),
    }
    (OUTPUT_DIR / "fixed_surface_representation_summary.json").write_text(json.dumps(fixed_summary, indent=2), encoding="utf-8")

    # Stage 5: PCA & Population Statistics
    pca_results = None
    if len(shape_vectors_list) >= 2:
        shape_matrix = np.array(shape_vectors_list)
        pca_results = fit_deviation_pca(shape_matrix, variance_cutoff=0.95)

        # 8. surface_deviation_pca.npz
        np.savez_compressed(
            OUTPUT_DIR / "surface_deviation_pca.npz",
            mean_vector=pca_results["mean_vector"],
            components=pca_results["components_retained"],
            all_components=pca_results["components_all"],
            singular_values=pca_results["singular_values"],
            eigenvalues=pca_results["eigenvalues"],
            explained_variance_ratio=pca_results["explained_variance_ratio"],
            complete_case_ids=np.array(complete_case_ids),
        )

        # 9. surface_deviation_pca_summary.json
        pca_summary = {
            "n_samples": pca_results["n_samples"],
            "n_features": pca_results["n_features"],
            "k_retained": pca_results["k_retained"],
            "variance_cutoff": pca_results["variance_cutoff"],
            "cumulative_variance_retained": float(pca_results["cumulative_explained_variance"][pca_results["k_retained"] - 1]),
            "eigenvalues": pca_results["eigenvalues"].tolist(),
            "explained_variance_ratio": pca_results["explained_variance_ratio"].tolist(),
        }
        (OUTPUT_DIR / "surface_deviation_pca_summary.json").write_text(json.dumps(pca_summary, indent=2), encoding="utf-8")

    # 7. population_surface_statistics.json
    u_vals = [r["u"] for r in surface_coord_rows]
    v_vals = [r["v"] for r in surface_coord_rows]
    off_vals = [r["offset"] for r in surface_coord_rows]

    pop_stats = {
        "scaffold": {
            "a": compute_linear_stats([p["a"] for p in scaffold_params_list]),
            "b": compute_linear_stats([p["b"] for p in scaffold_params_list]),
            "c": compute_linear_stats([p["c"] for p in scaffold_params_list]),
        },
        "surface_coordinates": {
            "u_circular": compute_circular_stats(u_vals),
            "v": compute_linear_stats(v_vals),
            "offset": compute_linear_stats(off_vals),
        },
        "branch_lengths_mm": {k: compute_linear_stats(v) for k, v in branch_lengths.items()},
        "bifurcation_angle_deg": compute_linear_stats(bifurcation_angles),
    }
    (OUTPUT_DIR / "population_surface_statistics.json").write_text(json.dumps(jsonable(pop_stats), indent=2), encoding="utf-8")

    # 10. population_validation_thresholds.json
    val_thresholds = build_validation_thresholds(scaffold_params_list, branch_lengths, bifurcation_angles, pca_results)
    (OUTPUT_DIR / "population_validation_thresholds.json").write_text(json.dumps(jsonable(val_thresholds), indent=2), encoding="utf-8")

    # 11. week1_manifest.json
    manifest = {
        "implementation": "Person 1 — Week 1 Real Data to Surface-Relative Statistical Representation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "smoke_test": smoke_test,
        "n_cases_processed": len(target_cases),
        "n_complete_cases": len(complete_case_ids),
        "n_cardiac_frame_passes": sum(1 for r in cardiac_frame_val_rows if r["overall_pass"]),
        "pca_dimensions": 126,
        "pca_retained_components": pca_results["k_retained"] if pca_results else 0,
        "output_files": [
            "population_cardiac_frames.csv",
            "cardiac_frame_validation.csv",
            "population_ellipsoid_parameters.csv",
            "population_surface_coordinates.csv",
            "fixed_surface_representation.npz",
            "fixed_surface_representation_summary.json",
            "population_surface_statistics.json",
            "surface_deviation_pca.npz",
            "surface_deviation_pca_summary.json",
            "population_validation_thresholds.json",
            "week1_manifest.json",
        ],
    }
    (OUTPUT_DIR / "week1_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "smoke_reports": smoke_reports,
        "fixed_matrices_shape": fixed_matrices_arr.shape,
        "shape_matrix_shape": (len(shape_vectors_list), 126) if shape_vectors_list else (0, 126),
        "pca_results": pca_results,
        "complete_case_ids": complete_case_ids,
        "manifest": manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Person 1 Week 1 Pipeline Runner")
    parser.add_argument("--smoke-test", action="store_true", default=True, help="Run 3-patient smoke test mode")
    parser.add_argument("--full", action="store_true", help="Run full 191-patient population mode")
    args = parser.parse_args()

    smoke_mode = not args.full
    res = run_pipeline(smoke_test=smoke_mode)

    if smoke_mode:
        print("\n" + "=" * 70)
        print("PERSON 1 — WEEK 1 DETAILED SMOKE TEST REPORT (3 Patients)")
        print("=" * 70)
        for cid in SMOKE_TEST_CASES:
            if cid in res["smoke_reports"]:
                rep = res["smoke_reports"][cid]
                print(f"\nPatient {cid}:")
                print(f"  frame: {'PASS' if rep['frame_pass'] else 'FAIL'}")
                print(f"  determinant: {rep['determinant']:.8f}")
                print(f"  orthogonality error: {rep['orthogonality_error']:.2e}")
                print(f"  length preservation max error: {rep['max_length_error']:.2e} mm")
                print(f"  LAD alignment: {'PASS (descends in -Z)' if rep['LAD_apex_aligned'] else 'FAIL'} (z_diff: {rep['LAD_apex_z_diff']:.2f} mm)")
                print(f"  ellipsoid: {'VALID' if rep['ellipsoid_valid'] else 'INVALID'} (a={rep['ellipsoid_abc'][0]:.2f}, b={rep['ellipsoid_abc'][1]:.2f}, c={rep['ellipsoid_abc'][2]:.2f})")
                print(f"  surface coordinates: COMPUTED")
                print(f"  fixed points: {'PRESENT (42x3)' if rep['has_fixed_points'] else 'MISSING'}")
                print(f"  deviation vector: {'PRESENT (126-D)' if rep['has_deviation_vector'] else 'MISSING'}")

        print("\n" + "-" * 70)
        print(f"Fixed Representation Matrix Shape: {res['fixed_matrices_shape']}")
        print(f"Deviation Matrix Shape: {res['shape_matrix_shape']}")
        if res["pca_results"]:
            print(f"PCA Dimensions: {res['pca_results']['n_features']}")
            print(f"Retained PCA Components (95% variance): {res['pca_results']['k_retained']}")
            print(f"Cumulative Explained Variance: {res['pca_results']['cumulative_explained_variance'][res['pca_results']['k_retained'] - 1]:.4f}")
        print("=" * 70)
    else:
        # Full population report
        print("\n" + "=" * 70)
        print("PERSON 1 — WEEK 1 FULL POPULATION VALIDATION REPORT (191 Patients)")
        print("=" * 70)
        reports = res["smoke_reports"]
        n_total = len(reports)
        n_rca = sum(1 for r in reports.values() if r["has_fixed_points"])
        n_no_rca = n_total - n_rca
        n_valid_ellipsoid = sum(1 for r in reports.values() if r["ellipsoid_valid"])
        n_invalid_ellipsoid = n_total - n_valid_ellipsoid
        n_frame_pass = sum(1 for r in reports.values() if r["frame_pass"])
        n_frame_fail = n_total - n_frame_pass
        n_complete_pca = len(res["complete_case_ids"])

        print(f"Total Accepted Patients Processed: {n_total}")
        print(f"  - Patients with RCA Candidate: {n_rca}")
        print(f"  - Patients without RCA Candidate: {n_no_rca}")
        print(f"  - Patients with Valid Ellipsoids: {n_valid_ellipsoid}")
        print(f"  - Patients with Invalid Ellipsoids: {n_invalid_ellipsoid}")
        print(f"  - Cardiac Frame Passes: {n_frame_pass}")
        print(f"  - Cardiac Frame Failures: {n_frame_fail}")
        print(f"  - Complete 4-Vessel Cases (Used for PCA): {n_complete_pca}")
        print("\nMatrix Dimensions & PCA Statistics:")
        print(f"  - Fixed Representation Matrix Shape: {res['fixed_matrices_shape']}")
        print(f"  - Deviation Matrix Shape: {res['shape_matrix_shape']}")
        if res["pca_results"]:
            pca = res["pca_results"]
            print(f"  - PCA Input Matrix Dimensions: {pca['n_samples']} x {pca['n_features']}")
            print(f"  - Retained PCA Components (95% variance): {pca['k_retained']}")
            print(f"  - Cumulative Explained Variance: {pca['cumulative_explained_variance'][pca['k_retained'] - 1]:.4f}")
            print(f"  - Top 5 Eigenvalues: {[round(float(e), 4) for e in pca['eigenvalues'][:5]]}")

        # Post-run SHA-256 integrity check
        post_integrity = verify_protected_assets(PPT_OUTPUT_DIR)
        print("\nProtected Upstream Assets Post-Run Verification:")
        print(f"  - All Files Exist: {post_integrity['all_files_exist']}")
        print(f"  - SHA-256 Checksums Matched: {post_integrity['hashes_matched']}")
        print("=" * 70)


if __name__ == "__main__":
    main()
