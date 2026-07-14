# LCA_vmtk_pipeline/vmtk_preprocessor.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Import downstream pipeline functions directly from the original preprocessor
from lca_vessel_tree_generator.LCA_topology_generator.patient_preprocessor import (
    normalize_artery_tree,
    resample_polyline,
)

def calculate_bifurcation_angle(lad, lcx):
    """Calculate the 3D angle (in degrees) between LAD and LCx takeoff vectors."""
    idx = min(2, len(lad)-1, len(lcx)-1)
    v_lad = lad[idx] - lad[0]
    v_lcx = lcx[idx] - lcx[0]
    
    norm_lad = np.linalg.norm(v_lad)
    norm_lcx = np.linalg.norm(v_lcx)
    
    if norm_lad < 1e-6 or norm_lcx < 1e-6:
        return 0.0
        
    cos_theta = np.dot(v_lad, v_lcx) / (norm_lad * norm_lcx)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def calculate_max_deviation(original, resampled):
    """Calculate the maximum point-to-point Euclidean deviation between original and resampled curves."""
    max_d = 0.0
    for p in resampled:
        dists = np.linalg.norm(original - p, axis=1)
        min_d = np.min(dists)
        if min_d > max_d:
            max_d = min_d
    return max_d

def parse_topology_and_extract_branches(raw_branches_file):
    """
    Loads raw branch curves and dynamically identifies LMCA, LAD, and LCx
    using topology first (connectivity graph), then local takeoff geometry.
    If confidence is low or connectivity is ambiguous, raises ValueError ("Classification Failed").
    """
    data = np.load(raw_branches_file, allow_pickle=True)
    n_branches = int(data['n_branches'])
    
    branches = []
    for i in range(n_branches):
        branches.append({
            'xyz': data[f'xyz_{i}'],
            'radius': data[f'radius_{i}']
        })
        
    if len(branches) < 3:
        raise ValueError("Classification Failed: Fewer than 3 branches extracted by VMTK.")
        
    # 1. Build Directed Connectivity Graph based on endpoint distances
    parent_map = {i: [] for i in range(n_branches)}
    child_map = {i: [] for i in range(n_branches)}
    
    threshold = 1.0 # mm threshold for connection
    
    for i in range(n_branches):
        end_pt = branches[i]['xyz'][-1]
        for j in range(n_branches):
            if i == j:
                continue
            start_pt = branches[j]['xyz'][0]
            dist = np.linalg.norm(end_pt - start_pt)
            if dist < threshold:
                child_map[i].append(j)
                parent_map[j].append(i)
                
    # 2. Identify the Root Branch (LMCA)
    # The LMCA has no parent (starts at the ostium) and splits at the first bifurcation
    candidate_roots = [i for i in range(n_branches) if len(parent_map[i]) == 0 and len(child_map[i]) > 0]
    
    if not candidate_roots:
        candidate_roots = sorted(range(n_branches), key=lambda i: len(child_map[i]), reverse=True)
        
    lmca_idx = candidate_roots[0]
    daughter_indices = child_map[lmca_idx]
    
    # Strict Topology Check
    if len(daughter_indices) < 2:
        # Fallback: check if there's any other bifurcation node in the graph
        fork_nodes = [i for i in range(n_branches) if len(child_map[i]) >= 2]
        if fork_nodes:
            lmca_idx = fork_nodes[0]
            daughter_indices = child_map[lmca_idx]
        else:
            raise ValueError("Classification Failed: Could not identify first bifurcation with 2+ children via connectivity.")
            
    # Restrict bifurcation daughters to the primary ones (if extra wiggles exist, pick the 2 longest)
    if len(daughter_indices) > 2:
        daughter_lengths = []
        for idx in daughter_indices:
            pts = branches[idx]['xyz']
            l = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
            daughter_lengths.append((l, idx))
        daughter_lengths.sort(reverse=True)
        daughter_indices = [daughter_lengths[0][1], daughter_lengths[1][1]]
        
    d1_idx, d2_idx = daughter_indices[0], daughter_indices[1]
    
    # 3. Classify daughter branches into LAD vs. LCx using Takeoff Angles (Geometry)
    lmca_pts = branches[lmca_idx]['xyz']
    lmca_vec = lmca_pts[-1] - lmca_pts[0]
    lmca_norm = np.linalg.norm(lmca_vec)
    if lmca_norm < 1e-6:
        raise ValueError("Classification Failed: Zero-length LMCA trunk (overlapping ostium/bifurcation).")
        
    pts_d1 = branches[d1_idx]['xyz']
    pts_d2 = branches[d2_idx]['xyz']
    
    # Local takeoff tangent vectors over first 3 points
    idx_d1 = min(2, len(pts_d1)-1)
    idx_d2 = min(2, len(pts_d2)-1)
    
    v_d1 = pts_d1[idx_d1] - pts_d1[0]
    v_d2 = pts_d2[idx_d2] - pts_d2[0]
    
    norm_v_d1 = np.linalg.norm(v_d1)
    norm_v_d2 = np.linalg.norm(v_d2)
    if norm_v_d1 < 1e-6 or norm_v_d2 < 1e-6:
        raise ValueError("Classification Failed: Zero-length takeoff vector in daughter branches.")
        
    cos_d1 = np.dot(lmca_vec, v_d1) / (lmca_norm * norm_v_d1)
    cos_d2 = np.dot(lmca_vec, v_d2) / (lmca_norm * norm_v_d2)
    
    theta_d1 = np.degrees(np.arccos(np.clip(cos_d1, -1.0, 1.0)))
    theta_d2 = np.degrees(np.arccos(np.clip(cos_d2, -1.0, 1.0)))
    
    # Strict Geometry Takeoff Angle Gate (Confidence difference must be >= 15 degrees)
    angle_difference = abs(theta_d1 - theta_d2)
    if angle_difference < 15.0:
        raise ValueError(f"Classification Failed: Ambiguous takeoff takeoff angles (diff={angle_difference:.1f}° < 15.0°).")
        
    # Map daughters based on takeoff deflection angle (LAD runs straighter, smaller deflection)
    if theta_d1 < theta_d2:
        lad_idx = d1_idx
        lcx_idx = d2_idx
    else:
        lad_idx = d2_idx
        lcx_idx = d1_idx
        
    # Length validation is kept strictly for validation metrics (no rejection gate)
    return branches[lmca_idx], branches[lad_idx], branches[lcx_idx]

def generate_patient_overlay_plot(patient_id, lmca_orig, lad_orig, lcx_orig, 
                                  lmca_proc, lad_proc, lcx_proc, output_path):
    """
    Saves a 3D comparison plot of the VMTK continuous curves vs. the VMTK resampled control points.
    Both are plotted in the normalized coordinate system to verify resampling fidelity.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 1. Plot continuous alignment as thick semi-transparent lines
    ax.plot(lmca_orig[:, 0], lmca_orig[:, 1], lmca_orig[:, 2], color='black', linewidth=4.5, alpha=0.35, label='Extracted LMCA')
    ax.plot(lad_orig[:, 0], lad_orig[:, 1], lad_orig[:, 2], color='red', linewidth=4.5, alpha=0.35, label='Extracted LAD')
    ax.plot(lcx_orig[:, 0], lcx_orig[:, 1], lcx_orig[:, 2], color='blue', linewidth=4.5, alpha=0.35, label='Extracted LCX')
    
    # 2. Plot resampled/processed centerlines as dashed lines with markers
    ax.plot(lmca_proc[:, 0], lmca_proc[:, 1], lmca_proc[:, 2], 
            color='black', linewidth=1.5, linestyle='--', marker='o', markersize=4, alpha=0.85, label='Resampled LMCA')
    ax.plot(lad_proc[:, 0], lad_proc[:, 1], lad_proc[:, 2], 
            color='red', linewidth=1.5, linestyle='--', marker='o', markersize=4, alpha=0.85, label='Resampled LAD')
    ax.plot(lcx_proc[:, 0], lcx_proc[:, 1], lcx_proc[:, 2], 
            color='blue', linewidth=1.5, linestyle='--', marker='o', markersize=4, alpha=0.85, label='Resampled LCX')
            
    # Highlight bifurcation point
    ax.scatter([lmca_proc[-1, 0]], [lmca_proc[-1, 1]], [lmca_proc[-1, 2]], 
               color='purple', s=80, edgecolor='white', zorder=10, label='Bifurcation')
               
    ax.set_title(f"Patient {patient_id} - Extracted vs. Resampled Centerline Overlay\n(Normalized Alignment Coordinates)", fontsize=10)
    ax.set_xlabel("X (LMCA alignment)")
    ax.set_ylabel("Y (Lateral)")
    ax.set_zlabel("Z (Bifurcation Normal)")
    
    # Set equal scale
    pts = np.concatenate([lmca_orig, lad_orig, lcx_orig])
    max_range = np.array([pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min()]).max() / 2.0
    mid_x = (pts[:,0].max()+pts[:,0].min()) / 2.0
    mid_y = (pts[:,1].max()+pts[:,1].min()) / 2.0
    mid_z = (pts[:,2].max()+pts[:,2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "lca_raw_branches")
    output_base_dir = os.path.join(base_dir, "processed_vmtk_dataset")
    ctrl_points_dir = os.path.join(base_dir, "LCA_branch_control_points", "vmtk_generated")
    overlay_dir = os.path.join(base_dir, "outputs", "vmtk_dataset_lca", "overlays")
    
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(ctrl_points_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    
    raw_files = sorted(glob(os.path.join(raw_dir, "*_raw_branches.npz")))
    if not raw_files:
        print(f"No raw branch files (.npz) found in {raw_dir}. Please run vmtk_extractor.py first.")
        return
        
    validation_reports = []
    collective_lmca = []
    collective_lad = []
    collective_lcx = []
    
    for raw_file in raw_files:
        patient_id = os.path.basename(raw_file).replace("_raw_branches.npz", "")
        print(f"Preprocessing Patient {patient_id}...")
        
        try:
            # 1. Parse topology & map branches (will throw ValueError if classification fails)
            lmca_branch, lad_branch, lcx_branch = parse_topology_and_extract_branches(raw_file)
            
            raw_lmca = lmca_branch['xyz']
            raw_lad = lad_branch['xyz']
            raw_lcx = lcx_branch['xyz']
            
            raw_lmca_r = lmca_branch['radius']
            raw_lad_r = lad_branch['radius']
            raw_lcx_r = lcx_branch['radius']
            
            # 2. Snapping Bifurcation
            # Snap start coordinates of LAD/LCx to end coordinate of LMCA to preserve topology
            bifurcation_xyz = raw_lmca[-1].copy()
            raw_lad_snapped = raw_lad.copy()
            raw_lcx_snapped = raw_lcx.copy()
            raw_lad_snapped[0] = bifurcation_xyz
            raw_lcx_snapped[0] = bifurcation_xyz
            
            # 3. Normalize Artery Tree Coordinates (Ostium at 0, LMCA along X, Bifurcation Plane aligned to Y/Z)
            # Reuses the exact normalize_artery_tree function imported from the original pipeline
            lmca_norm, lad_norm, lcx_norm = normalize_artery_tree(raw_lmca, raw_lad_snapped, raw_lcx_snapped)
            
            # 4. Resample branches to target control point resolutions
            # Reuses the exact resample_polyline function imported from the original pipeline
            lmca_ctrl = resample_polyline(lmca_norm, 5)
            lad_ctrl = resample_polyline(lad_norm, 12)
            lcx_ctrl = resample_polyline(lcx_norm, 10)
            
            # 5. Compute original vs processed geometry metrics (in aligned coordinates)
            l_lmca_orig = np.sum(np.linalg.norm(np.diff(lmca_norm, axis=0), axis=1))
            l_lad_orig = np.sum(np.linalg.norm(np.diff(lad_norm, axis=0), axis=1))
            l_lcx_orig = np.sum(np.linalg.norm(np.diff(lcx_norm, axis=0), axis=1))
            bif_angle_orig = calculate_bifurcation_angle(lad_norm, lcx_norm)
            
            l_lmca_proc = np.sum(np.linalg.norm(np.diff(lmca_ctrl, axis=0), axis=1))
            l_lad_proc = np.sum(np.linalg.norm(np.diff(lad_ctrl, axis=0), axis=1))
            l_lcx_proc = np.sum(np.linalg.norm(np.diff(lcx_ctrl, axis=0), axis=1))
            bif_angle_proc = calculate_bifurcation_angle(lad_ctrl, lcx_ctrl)
            
            # Max deviation check between continuous aligned geometry and resampled control points
            max_dev_lmca = calculate_max_deviation(lmca_norm, lmca_ctrl)
            max_dev_lad = calculate_max_deviation(lad_norm, lad_ctrl)
            max_dev_lcx = calculate_max_deviation(lcx_norm, lcx_ctrl)
            
            # 6. Save Patient Normalizations
            p_out_dir = os.path.join(output_base_dir, patient_id)
            os.makedirs(p_out_dir, exist_ok=True)
            np.save(os.path.join(p_out_dir, "lmca.npy"), lmca_norm)
            np.save(os.path.join(p_out_dir, "lad.npy"), lad_norm)
            np.save(os.path.join(p_out_dir, "lcx.npy"), lcx_norm)
            
            # Extract and resample radii
            lmca_r_ctrl = np.interp(np.linspace(0, len(raw_lmca_r)-1, 5), np.arange(len(raw_lmca_r)), raw_lmca_r)
            lad_r_ctrl = np.interp(np.linspace(0, len(raw_lad_r)-1, 12), np.arange(len(raw_lad_r)), raw_lad_r)
            lcx_r_ctrl = np.interp(np.linspace(0, len(raw_lcx_r)-1, 10), np.arange(len(raw_lcx_r)), raw_lcx_r)
            
            np.save(os.path.join(p_out_dir, "lmca_radius.npy"), lmca_r_ctrl)
            np.save(os.path.join(p_out_dir, "lad_radius.npy"), lad_r_ctrl)
            np.save(os.path.join(p_out_dir, "lcx_radius.npy"), lcx_r_ctrl)
            
            # 7. Append to Collective Database List
            collective_lmca.append(lmca_ctrl)
            collective_lad.append(lad_ctrl)
            collective_lcx.append(lcx_ctrl)
            
            # 8. Generate Original vs Processed Visual Overlay (in normalized coordinates)
            overlay_plot_path = os.path.join(overlay_dir, f"{patient_id}_overlay.png")
            generate_patient_overlay_plot(patient_id, lmca_norm, lad_norm, lcx_norm, 
                                          lmca_ctrl, lad_ctrl, lcx_ctrl, 
                                          overlay_plot_path)
            
            # 9. Compile Validation Report
            report = {
                "patient_id": patient_id,
                "status": "PASSED",
                "original": {
                    "lmca_length_mm": float(l_lmca_orig),
                    "lad_length_mm": float(l_lad_orig),
                    "lcx_length_mm": float(l_lcx_orig),
                    "bifurcation_angle_deg": float(bif_angle_orig)
                },
                "processed": {
                    "lmca_length_mm": float(l_lmca_proc),
                    "lad_length_mm": float(l_lad_proc),
                    "lcx_length_mm": float(l_lcx_proc),
                    "bifurcation_angle_deg": float(bif_angle_proc)
                },
                "deviation_error_mm": {
                    "LMCA": float(max_dev_lmca),
                    "LAD": float(max_dev_lad),
                    "LCX": float(max_dev_lcx)
                }
            }
            validation_reports.append(report)
            print(f"Status: PASSED. Bifurcation Angle: {bif_angle_proc:.1f}° (Orig: {bif_angle_orig:.1f}°)")
            
        except ValueError as ve:
            print(f"Status: FAILED for patient {patient_id}. Reason: {ve}")
            validation_reports.append({
                "patient_id": patient_id,
                "status": "Classification Failed",
                "error": str(ve)
            })
        except Exception as e:
            print(f"Status: FAILED for patient {patient_id}. Error: {e}")
            validation_reports.append({
                "patient_id": patient_id,
                "status": "FAILED",
                "error": str(e)
            })
            
    # Save Collective Databases
    if collective_lmca:
        lmca_stacked = np.stack(collective_lmca, axis=0)
        lad_stacked = np.stack(collective_lad, axis=0)
        lcx_stacked = np.stack(collective_lcx, axis=0)
        
        np.save(os.path.join(ctrl_points_dir, "LMCA_patient_ctrl_points.npy"), lmca_stacked)
        np.save(os.path.join(ctrl_points_dir, "LAD_patient_ctrl_points.npy"), lad_stacked)
        np.save(os.path.join(ctrl_points_dir, "LCX_patient_ctrl_points.npy"), lcx_stacked)
        
        all_tree_ctrl_points = np.concatenate([lmca_stacked, lad_stacked, lcx_stacked], axis=1)
        tree_mean = np.mean(all_tree_ctrl_points, axis=0)
        tree_std = np.std(all_tree_ctrl_points, axis=0)
        
        np.save(os.path.join(ctrl_points_dir, "LCA_tree_ctrl_points.npy"), all_tree_ctrl_points)
        np.save(os.path.join(ctrl_points_dir, "LCA_tree_mean.npy"), tree_mean)
        np.save(os.path.join(ctrl_points_dir, "LCA_tree_std.npy"), tree_std)
        print("\nSuccessfully exported VMTK collective control point databases & statistics.")
    else:
        print("\nWARNING: No patients passed VMTK preprocessing!")

    # Write Validation Report JSON
    val_report_path = os.path.join(output_base_dir, "vmtk_preprocessing_report.json")
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(validation_reports, f, indent=2)
    print(f"Validation report saved to: {val_report_path}")
    print(f"Visual overlays saved under: {overlay_dir}")

if __name__ == "__main__":
    main()
