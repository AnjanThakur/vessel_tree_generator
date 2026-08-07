# LCA_vmtk_pipeline/compare_extractions.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Import original pipeline parsing and mapping logic
from lca_vessel_tree_generator.LCA_topology_generator.patient_preprocessor import (
    parse_pth_points,
    parse_ctgr_points,
    parse_vtp_points,
    PATIENT_MAPPING,
)
from lca_vessel_tree_generator.LCA_vmtk_pipeline.vmtk_preprocessor import (
    parse_topology_and_extract_branches,
)

def resample_curve_uniform(points, n_samples=100):
    """Resample a 3D curve to exactly n_samples points spaced evenly along its arc length."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return np.tile(points[0], (n_samples, 1))
    diffs = np.diff(points, axis=0)
    dists = np.concatenate(([0], np.cumsum(np.linalg.norm(diffs, axis=1))))
    total_length = dists[-1]
    if total_length < 1e-9:
        return np.tile(points[0], (n_samples, 1))
    new_dists = np.linspace(0, total_length, n_samples)
    resampled = np.zeros((n_samples, 3))
    for dim in range(3):
        resampled[:, dim] = np.interp(new_dists, dists, points[:, dim])
    return resampled

def calculate_path_length(points):
    """Compute cumulative 3D path length."""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

def calculate_bifurcation_angle(lad, lcx):
    """Calculate takeoff angle in degrees between LAD and LCx takeoff vectors."""
    idx = min(2, len(lad)-1, len(lcx)-1)
    v_lad = lad[idx] - lad[0]
    v_lcx = lcx[idx] - lcx[0]
    norm_lad = np.linalg.norm(v_lad)
    norm_lcx = np.linalg.norm(v_lcx)
    if norm_lad < 1e-6 or norm_lcx < 1e-6:
        return 0.0
    cos_theta = np.dot(v_lad, v_lcx) / (norm_lad * norm_lcx)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def calculate_hausdorff_distance(curve_a, curve_b):
    """Compute the directed Hausdorff distance from curve_a to curve_b."""
    max_d = 0.0
    for p in curve_a:
        dists = np.linalg.norm(curve_b - p, axis=1)
        min_d = np.min(dists)
        if min_d > max_d:
            max_d = min_d
    return max_d

def compute_symmetric_hausdorff(curve_a, curve_b):
    """Compute symmetric Hausdorff distance between two curves."""
    d1 = calculate_hausdorff_distance(curve_a, curve_b)
    d2 = calculate_hausdorff_distance(curve_b, curve_a)
    return float(max(d1, d2))

def compute_curve_metrics(curve_a, curve_b):
    """
    Computes comparative metrics between curve_a and curve_b:
    1. Hausdorff distance
    2. Mean point-to-point distance (after arc-length parameterization)
    3. RMSE (after arc-length parameterization)
    """
    # Hausdorff is coordinate-independent of sampling density
    hausdorff = compute_symmetric_hausdorff(curve_a, curve_b)
    
    # Resample both curves to 100 points for point-to-point comparisons
    a_res = resample_curve_uniform(curve_a, 100)
    b_res = resample_curve_uniform(curve_b, 100)
    
    mean_dist = float(np.mean(np.linalg.norm(a_res - b_res, axis=1)))
    rmse = float(np.sqrt(np.mean(np.sum((a_res - b_res)**2, axis=1))))
    
    return {
        "hausdorff_mm": hausdorff,
        "mean_point_distance_mm": mean_dist,
        "rmse_mm": rmse
    }

def get_original_raw_branches(base_dir, patient_id):
    """Extracts original unnormalized branches from LCA_dataset using original logic."""
    p_path = os.path.join(base_dir, "LCA_dataset", patient_id)
    mapping = PATIENT_MAPPING[patient_id]
    
    if mapping["source"] == "pth_split":
        lmca_raw = parse_pth_points(os.path.join(p_path, mapping["lmca_file"]))
        lad_raw = parse_pth_points(os.path.join(p_path, mapping["lad_file"]))
        lcx_raw = parse_pth_points(os.path.join(p_path, mapping["lcx_file"]))
        
        lmca_arr = np.array(lmca_raw)
        lad_arr = np.array(lad_raw)
        lcx_arr = np.array(lcx_raw)
        
        # Scale to mm
        if max(calculate_path_length(lmca_arr), calculate_path_length(lad_arr), calculate_path_length(lcx_arr)) < 30.0:
            lmca_arr *= 10.0
            lad_arr *= 10.0
            lcx_arr *= 10.0
    else:
        f1 = os.path.join(p_path, mapping["lmca_lad_file"])
        if f1.endswith(".pth"):
            lmca_lad_raw = parse_pth_points(f1)
        elif f1.endswith(".ctgr"):
            lmca_lad_raw = parse_ctgr_points(f1)
        elif f1.endswith(".vtp"):
            lmca_lad_raw = parse_vtp_points(f1)
            
        f2 = os.path.join(p_path, mapping["lcx_file"])
        if f2.endswith(".pth"):
            lcx_raw = parse_pth_points(f2)
        elif f2.endswith(".ctgr"):
            lcx_raw = parse_ctgr_points(f2)
        elif f2.endswith(".vtp"):
            lcx_raw = parse_vtp_points(f2)
            
        lmca_lad_pts = np.array(lmca_lad_raw)
        lcx_pts = np.array(lcx_raw)
        
        if max(calculate_path_length(lmca_lad_pts), calculate_path_length(lcx_pts)) < 30.0:
            lmca_lad_pts *= 10.0
            lcx_pts *= 10.0
            
        start_gap = np.linalg.norm(lmca_lad_pts[0] - lcx_pts[0])
        
        if start_gap < 1.5:
            min_len = min(len(lmca_lad_pts), len(lcx_pts))
            bifurcation_idx = 0
            for idx in range(min_len):
                d = np.linalg.norm(lmca_lad_pts[idx] - lcx_pts[idx])
                if d > 1.5:
                    bifurcation_idx = idx
                    break
            if bifurcation_idx == 0:
                bifurcation_idx = min_len // 4
            lmca_arr = lmca_lad_pts[:bifurcation_idx + 1]
            lad_arr = lmca_lad_pts[bifurcation_idx:]
            lcx_arr = lcx_pts[bifurcation_idx:]
        else:
            proximal_lcx_pts = lcx_pts[:min(len(lcx_pts), 3)]
            best_indices = []
            min_distances = []
            for pt in proximal_lcx_pts:
                dists = np.linalg.norm(lmca_lad_pts - pt, axis=1)
                idx = np.argmin(dists)
                best_indices.append(idx)
                min_distances.append(dists[idx])
            bifurcation_idx = int(np.round(np.mean(best_indices)))
            lmca_arr = lmca_lad_pts[:bifurcation_idx + 1]
            lad_arr = lmca_lad_pts[bifurcation_idx:]
            lcx_arr = lcx_pts.copy()
            lcx_arr[0] = lmca_arr[-1]
            
    # Snap bifurcation
    lad_arr[0] = lmca_arr[-1]
    lcx_arr[0] = lmca_arr[-1]
    
    return lmca_arr, lad_arr, lcx_arr

def plot_overlay_comparison(patient_id, orig_tree, vmtk_tree, output_path):
    """Generates a 3D overlay comparison plot of Original vs. VMTK normalized centerlines."""
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Original (Dashed)
    ax.plot(orig_tree['LMCA'][:, 0], orig_tree['LMCA'][:, 1], orig_tree['LMCA'][:, 2], 
            color='black', linestyle='--', linewidth=2.5, alpha=0.5, label='Original LMCA')
    ax.plot(orig_tree['LAD'][:, 0], orig_tree['LAD'][:, 1], orig_tree['LAD'][:, 2], 
            color='red', linestyle='--', linewidth=2.5, alpha=0.5, label='Original LAD')
    ax.plot(orig_tree['LCX'][:, 0], orig_tree['LCX'][:, 1], orig_tree['LCX'][:, 2], 
            color='blue', linestyle='--', linewidth=2.5, alpha=0.5, label='Original LCX')
            
    # VMTK (Solid)
    ax.plot(vmtk_tree['LMCA'][:, 0], vmtk_tree['LMCA'][:, 1], vmtk_tree['LMCA'][:, 2], 
            color='black', linestyle='-', linewidth=2.0, label='VMTK LMCA')
    ax.plot(vmtk_tree['LAD'][:, 0], vmtk_tree['LAD'][:, 1], vmtk_tree['LAD'][:, 2], 
            color='red', linestyle='-', linewidth=2.0, label='VMTK LAD')
    ax.plot(vmtk_tree['LCX'][:, 0], vmtk_tree['LCX'][:, 1], vmtk_tree['LCX'][:, 2], 
            color='blue', linestyle='-', linewidth=2.0, label='VMTK LCX')
            
    # Bifurcations
    ax.scatter([orig_tree['LMCA'][-1, 0]], [orig_tree['LMCA'][-1, 1]], [orig_tree['LMCA'][-1, 2]], 
               color='orange', marker='X', s=100, edgecolor='black', zorder=10, label='Original Bifurcation')
    ax.scatter([vmtk_tree['LMCA'][-1, 0]], [vmtk_tree['LMCA'][-1, 1]], [vmtk_tree['LMCA'][-1, 2]], 
               color='purple', marker='o', s=80, edgecolor='white', zorder=10, label='VMTK Bifurcation')
               
    ax.set_title(f"Original vs. VMTK Centerline Comparison Overlay\nPatient: {patient_id} (Normalized Alignment Coordinates)", fontsize=11)
    ax.set_xlabel("X (LMCA alignment direction)")
    ax.set_ylabel("Y (Lateral)")
    ax.set_zlabel("Z (Bifurcation Normal)")
    
    # Set equal axes bounds
    pts_all = np.concatenate([
        orig_tree['LMCA'], orig_tree['LAD'], orig_tree['LCX'],
        vmtk_tree['LMCA'], vmtk_tree['LAD'], vmtk_tree['LCX']
    ])
    max_range = np.array([pts_all[:,0].max()-pts_all[:,0].min(), pts_all[:,1].max()-pts_all[:,1].min(), pts_all[:,2].max()-pts_all[:,2].min()]).max() / 2.0
    mid_x = (pts_all[:,0].max()+pts_all[:,0].min()) / 2.0
    mid_y = (pts_all[:,1].max()+pts_all[:,1].min()) / 2.0
    mid_z = (pts_all[:,2].max()+pts_all[:,2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    orig_dataset_dir = os.path.join(base_dir, "processed_dataset")
    vmtk_dataset_dir = os.path.join(base_dir, "processed_vmtk_dataset")
    orig_ctrl_points_dir = os.path.join(base_dir, "LCA_branch_control_points", "generated")
    vmtk_ctrl_points_dir = os.path.join(base_dir, "LCA_branch_control_points", "vmtk_generated")
    comparison_out_dir = os.path.join(base_dir, "outputs", "vmtk_dataset_lca", "comparisons")
    
    os.makedirs(comparison_out_dir, exist_ok=True)
    
    # Load VMTK preprocessing report to identify skipped patients
    vmtk_report_path = os.path.join(vmtk_dataset_dir, "vmtk_preprocessing_report.json")
    vmtk_statuses = {}
    if os.path.exists(vmtk_report_path):
        try:
            with open(vmtk_report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
                for r in report:
                    vmtk_statuses[r["patient_id"]] = {
                        "status": r.get("status", "FAILED"),
                        "error": r.get("error", "")
                    }
        except Exception as e:
            print(f"[WARN] Failed to load VMTK preprocessing report: {e}")
            
    vmtk_folders = sorted(glob(os.path.join(vmtk_dataset_dir, "*_H_CORO_*")))
    comparison_report = []
    
    # Load collective databases for resampled checks
    orig_lmca_stacked = np.load(os.path.join(orig_ctrl_points_dir, "LMCA_patient_ctrl_points.npy"))
    orig_lad_stacked = np.load(os.path.join(orig_ctrl_points_dir, "LAD_patient_ctrl_points.npy"))
    orig_lcx_stacked = np.load(os.path.join(orig_ctrl_points_dir, "LCX_patient_ctrl_points.npy"))
    
    vmtk_lmca_stacked = np.load(os.path.join(vmtk_ctrl_points_dir, "LMCA_patient_ctrl_points.npy"))
    vmtk_lad_stacked = np.load(os.path.join(vmtk_ctrl_points_dir, "LAD_patient_ctrl_points.npy"))
    vmtk_lcx_stacked = np.load(os.path.join(vmtk_ctrl_points_dir, "LCX_patient_ctrl_points.npy"))
    
    # Keep track of active indices mapping to collective files
    orig_patient_ids = sorted([os.path.basename(p) for p in glob(os.path.join(orig_dataset_dir, "*_H_CORO_*"))])
    vmtk_patient_ids = sorted([os.path.basename(p) for p in glob(os.path.join(vmtk_dataset_dir, "*_H_CORO_*"))])
    
    # Filter only VMTK patients that actually PASSED preprocessing
    active_vmtk_patient_ids = []
    for pid in vmtk_patient_ids:
        status_info = vmtk_statuses.get(pid, {"status": "PASSED"})
        if status_info["status"] == "PASSED":
            active_vmtk_patient_ids.append(pid)
            
    print(f"Total Original preprocessed patients: {len(orig_patient_ids)}")
    print(f"Total VMTK preprocessed patients: {len(active_vmtk_patient_ids)}")
    
    representative_patient_data = None
    
    print("\nStarting Detailed Original vs. VMTK Comparative Analysis...")
    for v_folder in vmtk_folders:
        patient_id = os.path.basename(v_folder)
        orig_folder = os.path.join(orig_dataset_dir, patient_id)
        
        status_info = vmtk_statuses.get(patient_id, {"status": "PASSED", "error": ""})
        if status_info["status"] == "Classification Failed":
            print(f"Skipping Patient {patient_id}: Flagged as 'Classification Failed' (Reason: {status_info['error']})")
            comparison_report.append({
                "patient_id": patient_id,
                "status": "Classification Failed",
                "error": status_info["error"]
            })
            continue
            
        if not os.path.exists(orig_folder):
            print(f"Skipping Patient {patient_id}: Not found in original processed dataset.")
            comparison_report.append({
                "patient_id": patient_id,
                "status": "Missing Original Preprocessed Data"
            })
            continue
            
        print(f"Comparing Patient {patient_id}...")
        
        try:
            # -------------------------------------------------------------
            # STAGE 1: LOAD RAW CENTERLINES (BEFORE NORMALIZATION)
            # -------------------------------------------------------------
            raw_orig_lmca, raw_orig_lad, raw_orig_lcx = get_original_raw_branches(base_dir, patient_id)
            
            raw_vmtk_file = os.path.join(base_dir, "lca_raw_branches", f"{patient_id}_raw_branches.npz")
            raw_vmtk_lmca_br, raw_vmtk_lad_br, raw_vmtk_lcx_br = parse_topology_and_extract_branches(raw_vmtk_file)
            
            raw_vmtk_lmca = raw_vmtk_lmca_br['xyz']
            raw_vmtk_lad = raw_vmtk_lad_br['xyz']
            raw_vmtk_lcx = raw_vmtk_lcx_br['xyz']
            
            metrics_raw = {
                "LMCA": compute_curve_metrics(raw_orig_lmca, raw_vmtk_lmca),
                "LAD": compute_curve_metrics(raw_orig_lad, raw_vmtk_lad),
                "LCX": compute_curve_metrics(raw_orig_lcx, raw_vmtk_lcx)
            }
            
            # -------------------------------------------------------------
            # STAGE 2: LOAD NORMALIZED CENTERLINES (AFTER ALIGNMENT, BEFORE RESAMPLING)
            # -------------------------------------------------------------
            norm_orig_lmca = np.load(os.path.join(orig_folder, "lmca.npy"))
            norm_orig_lad = np.load(os.path.join(orig_folder, "lad.npy"))
            norm_orig_lcx = np.load(os.path.join(orig_folder, "lcx.npy"))
            
            norm_vmtk_lmca = np.load(os.path.join(v_folder, "lmca.npy"))
            norm_vmtk_lad = np.load(os.path.join(v_folder, "lad.npy"))
            norm_vmtk_lcx = np.load(os.path.join(v_folder, "lcx.npy"))
            
            metrics_norm = {
                "LMCA": compute_curve_metrics(norm_orig_lmca, norm_vmtk_lmca),
                "LAD": compute_curve_metrics(norm_orig_lad, norm_vmtk_lad),
                "LCX": compute_curve_metrics(norm_orig_lcx, norm_vmtk_lcx)
            }
            
            # -------------------------------------------------------------
            # STAGE 3: LOAD RESAMPLED CONTROL POINTS
            # -------------------------------------------------------------
            # Locate index mapping in collective stacked databases
            orig_db_idx = orig_patient_ids.index(patient_id)
            vmtk_db_idx = active_vmtk_patient_ids.index(patient_id)
            
            res_orig_lmca = orig_lmca_stacked[orig_db_idx]
            res_orig_lad = orig_lad_stacked[orig_db_idx]
            res_orig_lcx = orig_lcx_stacked[orig_db_idx]
            
            res_vmtk_lmca = vmtk_lmca_stacked[vmtk_db_idx]
            res_vmtk_lad = vmtk_lad_stacked[vmtk_db_idx]
            res_vmtk_lcx = vmtk_lcx_stacked[vmtk_db_idx]
            
            metrics_res = {
                "LMCA": compute_curve_metrics(res_orig_lmca, res_vmtk_lmca),
                "LAD": compute_curve_metrics(res_orig_lad, res_vmtk_lad),
                "LCX": compute_curve_metrics(res_orig_lcx, res_vmtk_lcx)
            }
            
            # -------------------------------------------------------------
            # LENGTH AND BIFURCATION ANGLE COMPARISONS
            # -------------------------------------------------------------
            len_lmca_orig, len_lmca_vmtk = calculate_path_length(norm_orig_lmca), calculate_path_length(norm_vmtk_lmca)
            len_lad_orig, len_lad_vmtk = calculate_path_length(norm_orig_lad), calculate_path_length(norm_vmtk_lad)
            len_lcx_orig, len_lcx_vmtk = calculate_path_length(norm_orig_lcx), calculate_path_length(norm_vmtk_lcx)
            
            angle_orig = calculate_bifurcation_angle(norm_orig_lad, norm_orig_lcx)
            angle_vmtk = calculate_bifurcation_angle(norm_vmtk_lad, norm_vmtk_lcx)
            
            patient_results = {
                "patient_id": patient_id,
                "status": "COMPARED",
                "metrics_raw": metrics_raw,
                "metrics_normalized": metrics_norm,
                "metrics_resampled": metrics_res,
                "lengths_mm": {
                    "LMCA": {"original": len_lmca_orig, "vmtk": len_lmca_vmtk, "diff": len_lmca_vmtk - len_lmca_orig},
                    "LAD": {"original": len_lad_orig, "vmtk": len_lad_vmtk, "diff": len_lad_vmtk - len_lad_orig},
                    "LCX": {"original": len_lcx_orig, "vmtk": len_lcx_vmtk, "diff": len_lcx_vmtk - len_lcx_orig}
                },
                "bifurcation_angle_deg": {
                    "original": angle_orig,
                    "vmtk": angle_vmtk,
                    "diff": angle_vmtk - angle_orig
                }
            }
            comparison_report.append(patient_results)
            
            # Save representative patient data
            if patient_id == "0070_H_CORO_KD":
                representative_patient_data = {
                    "patient_id": patient_id,
                    "raw": {"orig": raw_orig_lmca, "vmtk": raw_vmtk_lmca},
                    "norm": {"orig": norm_orig_lmca, "vmtk": norm_vmtk_lmca},
                    "res": {"orig": res_orig_lmca, "vmtk": res_vmtk_lmca}
                }
            
            # Plot comparison overlay
            plot_path = os.path.join(comparison_out_dir, f"{patient_id}_comparison.png")
            plot_overlay_comparison(
                patient_id,
                {'LMCA': norm_orig_lmca, 'LAD': norm_orig_lad, 'LCX': norm_orig_lcx},
                {'LMCA': norm_vmtk_lmca, 'LAD': norm_vmtk_lad, 'LCX': norm_vmtk_lcx},
                plot_path
            )
            
        except Exception as e:
            print(f"[ERROR] Failed to compare patient {patient_id}: {e}")
            comparison_report.append({
                "patient_id": patient_id,
                "status": "ERROR",
                "error": str(e)
            })
            
    # Save the detailed comparative report
    report_json_path = os.path.join(base_dir, "outputs", "vmtk_dataset_lca", "extraction_comparison_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=6)
        
    print(f"\nAnalysis completed successfully.")
    print(f"Detailed comparison report saved: {report_json_path}")
    print(f"Comparison overlays saved:         {comparison_out_dir}")
    
    # Print the representative patient coordinates at each stage
    if representative_patient_data:
        p_id = representative_patient_data["patient_id"]
        print(f"\n==========================================================================")
        print(f"REPRESENTATIVE PATIENT COORDINATES STAGES COMPARISON: {p_id}")
        print(f"==========================================================================")
        
        stages = [
            ("RAW COORDINATES (BEFORE NORMALIZATION)", representative_patient_data["raw"]),
            ("NORMALIZED COORDINATES (AFTER ALIGNMENT, BEFORE RESAMPLING)", representative_patient_data["norm"]),
            ("RESAMPLED CONTROL POINTS", representative_patient_data["res"])
        ]
        
        for name, data in stages:
            print(f"\n--- {name} ---")
            print("  Original (First 5 points):")
            for pt in data["orig"][:5]:
                print(f"    [{pt[0]:.8f}, {pt[1]:.8f}, {pt[2]:.8f}]")
                
            print("  VMTK (First 5 points):")
            for pt in data["vmtk"][:5]:
                print(f"    [{pt[0]:.8f}, {pt[1]:.8f}, {pt[2]:.8f}]")
        print(f"==========================================================================\n")

if __name__ == "__main__":
    main()
