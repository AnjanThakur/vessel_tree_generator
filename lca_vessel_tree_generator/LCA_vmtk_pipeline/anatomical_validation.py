# LCA_vmtk_pipeline/anatomical_validation.py

import os
import json
import base64
import zlib
import struct
import re
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.spatial import KDTree

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

# ==============================================================================
# CONFIGURABLE MANUAL REVIEW THRESHOLDS
# ==============================================================================
# Patients will be flagged as "Needs Manual Review" if the absolute difference
# between VMTK and Original centerlines exceeds any of these values.
REVIEW_THRESHOLDS = {
    "takeoff_angle_diff_deg": 15.0,        # Landing takeoff angle difference
    "mean_wall_distance_diff_mm": 0.5,      # Wall distance difference threshold
    "symmetry_deviation_diff_mm": 0.3,      # Off-centeredness symmetry difference
    "curvature_diff": 0.5,                  # Discrete Menger curvature difference
    "tortuosity_diff": 0.1                  # Arc-to-chord length difference
}

def find_vessel_surface_mesh(patient_dir):
    """
    Recursively scans the patient folder for the largest .vtp surface mesh file,
    ignoring centerlines and caps.
    """
    vtp_candidates = []
    for root, dirs, files in os.walk(patient_dir):
        for f in files:
            if f.endswith('.vtp'):
                path = os.path.join(root, f)
                # Ignore centerlines and caps
                if 'centerline' in path.lower() or 'cap' in path.lower():
                    continue
                vtp_candidates.append((os.path.getsize(path), path))
    if not vtp_candidates:
        raise FileNotFoundError(f"Missing surface mesh (.vtp) in {patient_dir}")
    vtp_candidates.sort(reverse=True)
    return vtp_candidates[0][1]

def load_vtp_surface_points(vtp_path):
    """
    Parses a VTK XML PolyData (.vtp) file in pure Python.
    Extracts the 3D surface vertices array. Supports base64 and raw binary encodings.
    """
    with open(vtp_path, "rb") as f:
        content = f.read()

    # Find the AppendedData tag
    tag_match = re.search(rb'<AppendedData[^>]*>', content)
    if not tag_match:
        raise ValueError("AppendedData tag not found in VTP XML.")
    
    start_tag = tag_match.group(0)
    start_idx = content.find(start_tag)
    
    # Determine the encoding
    is_raw = b'encoding="raw"' in start_tag or b"encoding='raw'" in start_tag

    points_match = re.search(rb'<DataArray[^>]*Name="Points"[^>]*offset="(\d+)"', content[:start_idx])
    if not points_match:
        points_match = re.search(rb'<DataArray[^>]*offset="(\d+)"[^>]*Name="Points"', content[:start_idx])
        
    if not points_match:
        raise ValueError(f"Could not find Points array offset in VTP XML: {vtp_path}")
        
    points_offset = int(points_match.group(1))

    offset_matches = re.findall(rb'offset="(\d+)"', content[:start_idx])
    offsets = sorted(list(set(int(x) for x in offset_matches)))
    
    points_idx = offsets.index(points_offset)
    next_offset = offsets[points_idx+1]
        
    end_tag = b'</AppendedData>'
    end_idx = content.find(end_tag, start_idx)
    
    raw_data = content[start_idx + len(start_tag):end_idx].strip()
    if raw_data.startswith(b"_"):
        raw_data = raw_data[1:]

    if is_raw:
        # For raw encoding, the offsets are exact byte offsets from the start of the raw data.
        points_decoded = raw_data[points_offset:next_offset]
    else:
        # Base64 encoding
        raw_b64 = raw_data.replace(b'\n', b'').replace(b'\r', b'').replace(b'\t', b'').replace(b' ', b'')
        slice_b64 = raw_b64[points_offset:next_offset]
        parts = slice_b64.split(b'=')
        decoded_parts = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            rem = len(p) % 4
            if rem > 0:
                p += b'=' * (4 - rem)
            decoded_parts.append(base64.b64decode(p))
        points_decoded = b"".join(decoded_parts)

    num_blocks, block_size, last_block_size = struct.unpack("<III", points_decoded[:12])
    compressed_sizes = struct.unpack(f"<{num_blocks}I", points_decoded[12:12+4*num_blocks])

    block_data_start = 12 + 4 * num_blocks
    decompressed = b""
    curr = block_data_start
    for sz in compressed_sizes:
        comp_block = points_decoded[curr:curr+sz]
        decompressed += zlib.decompress(comp_block)
        curr += sz

    points = np.frombuffer(decompressed, dtype=np.float32).reshape(-1, 3)
    return points.copy()

def get_original_raw_branches(base_dir, patient_id):
    """Extracts original unnormalized branches from LCA_dataset in raw scanner coordinates."""
    p_path = os.path.join(base_dir, "LCA_dataset", patient_id)
    mapping = PATIENT_MAPPING[patient_id]
    
    if mapping["source"] == "pth_split":
        lmca_raw = parse_pth_points(os.path.join(p_path, mapping["lmca_file"]))
        lad_raw = parse_pth_points(os.path.join(p_path, mapping["lad_file"]))
        lcx_raw = parse_pth_points(os.path.join(p_path, mapping["lcx_file"]))
        
        lmca_arr = np.array(lmca_raw)
        lad_arr = np.array(lad_raw)
        lcx_arr = np.array(lcx_raw)
        
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

def calculate_path_length(points):
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

def calculate_bifurcation_angle(lad, lcx):
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

def compute_curvature(points):
    """Computes mean circumcircle (Menger) curvature along a discrete polyline."""
    if len(points) < 3:
        return 0.0
    curvatures = []
    for i in range(1, len(points)-1):
        p_prev = points[i-1]
        p_curr = points[i]
        p_next = points[i+1]
        
        a = p_curr - p_prev
        b = p_next - p_curr
        c = p_next - p_prev
        
        len_a = np.linalg.norm(a)
        len_b = np.linalg.norm(b)
        len_c = np.linalg.norm(c)
        
        if len_a < 1e-6 or len_b < 1e-6 or len_c < 1e-6:
            continue
            
        cross_prod = np.cross(a, b)
        area = 0.5 * np.linalg.norm(cross_prod)
        kappa = (4.0 * area) / (len_a * len_b * len_c)
        curvatures.append(kappa)
        
    return float(np.mean(curvatures)) if curvatures else 0.0

def compute_tortuosity(points):
    if len(points) < 2:
        return 0.0
    path_len = calculate_path_length(points)
    straight_dist = np.linalg.norm(points[-1] - points[0])
    if straight_dist < 1e-6:
        return 0.0
    return float((path_len / straight_dist) - 1.0)

def compute_centeredness_metrics(points, mesh_tree, mesh_points):
    """
    Computes lumen centering metrics:
    1. Mean Wall Distance (distance to closest surface vertex)
    2. Symmetry Deviation (centeredness): average projected offset from the center of the local wall slice
    3. Radius Tap Consistency (Standard Deviation of local wall distances)
    """
    if len(points) < 2:
        return {"mean_wall_distance_mm": 0.0, "symmetry_deviation_mm": 0.0, "radius_std_mm": 0.0}

    wall_distances = []
    symmetry_offsets = []

    # Get local tangents
    tangents = np.zeros_like(points)
    tangents[0] = points[1] - points[0]
    tangents[-1] = points[-1] - points[-2]
    for i in range(1, len(points)-1):
        tangents[i] = (points[i+1] - points[i-1]) / 2.0

    # Query closest wall points
    dists, closest_indices = mesh_tree.query(points)
    
    for i in range(len(points)):
        p = points[i]
        r = dists[i]
        t = tangents[i]
        norm_t = np.linalg.norm(t)
        if norm_t < 1e-6:
            continue
        t_hat = t / norm_t

        wall_distances.append(float(r))

        # Query all surface points in the local lumen slice around point p
        nearby_indices = mesh_tree.query_ball_point(p, r * 1.5)
        if len(nearby_indices) < 3:
            continue
            
        nearby_pts = mesh_points[nearby_indices]
        u = nearby_pts - p
        projections = np.dot(u, t_hat)
        mask = np.abs(projections) < 1.0
        if np.any(mask):
            proj_normals = u[mask] - projections[mask, np.newaxis] * t_hat
            mean_offset_vec = np.mean(proj_normals, axis=0)
            symmetry_offsets.append(float(np.linalg.norm(mean_offset_vec)))

    mean_wall_dist = float(np.mean(wall_distances)) if wall_distances else 0.0
    mean_symmetry = float(np.mean(symmetry_offsets)) if symmetry_offsets else 0.0
    radius_std = float(np.std(wall_distances)) if wall_distances else 0.0

    return {
        "mean_wall_distance_mm": mean_wall_dist,
        "symmetry_deviation_mm": mean_symmetry,
        "radius_std_mm": radius_std
    }

def evaluate_better(orig_val, vmtk_val, metric_name):
    """Determines which pipeline performed better (Original or VMTK) for a given metric."""
    if metric_name in ["symmetry_deviation_mm", "radius_std_mm", "curvature", "tortuosity"]:
        if vmtk_val < orig_val:
            return "VMTK"
        elif orig_val < vmtk_val:
            return "Original"
        else:
            return "Tie"
    elif metric_name == "mean_wall_distance_mm":
        if vmtk_val > orig_val:
            return "VMTK"
        elif orig_val > vmtk_val:
            return "Original"
        else:
            return "Tie"
    return "N/A"

def generate_3d_anatomical_plot(patient_id, mesh_pts, orig_tree, vmtk_tree, output_path):
    """Generates a 3D visualization showing surface mesh point cloud and both centerlines."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    skip = max(1, len(mesh_pts) // 8000)
    mesh_dec = mesh_pts[::skip]
    ax.scatter(mesh_dec[:, 0], mesh_dec[:, 1], mesh_dec[:, 2], 
               color='grey', s=1.0, alpha=0.035, depthshade=False, label='Vessel Surface Mesh')
               
    ax.plot(orig_tree['LMCA'][:, 0], orig_tree['LMCA'][:, 1], orig_tree['LMCA'][:, 2], 
            color='royalblue', linewidth=3.0, label='Original Centerlines')
    ax.plot(orig_tree['LAD'][:, 0], orig_tree['LAD'][:, 1], orig_tree['LAD'][:, 2], 
            color='royalblue', linewidth=3.0)
    ax.plot(orig_tree['LCX'][:, 0], orig_tree['LCX'][:, 1], orig_tree['LCX'][:, 2], 
            color='royalblue', linewidth=3.0)
            
    ax.plot(vmtk_tree['LMCA'][:, 0], vmtk_tree['LMCA'][:, 1], vmtk_tree['LMCA'][:, 2], 
            color='crimson', linewidth=2.5, linestyle='-', label='VMTK Centerlines')
    ax.plot(vmtk_tree['LAD'][:, 0], vmtk_tree['LAD'][:, 1], vmtk_tree['LAD'][:, 2], 
            color='crimson', linewidth=2.5, linestyle='-')
    ax.plot(vmtk_tree['LCX'][:, 0], vmtk_tree['LCX'][:, 1], vmtk_tree['LCX'][:, 2], 
            color='crimson', linewidth=2.5, linestyle='-')
            
    ax.scatter([orig_tree['LMCA'][-1, 0]], [orig_tree['LMCA'][-1, 1]], [orig_tree['LMCA'][-1, 2]], 
               color='darkblue', marker='X', s=100, zorder=10, label='Original Bifurcation')
    ax.scatter([vmtk_tree['LMCA'][-1, 0]], [vmtk_tree['LMCA'][-1, 1]], [vmtk_tree['LMCA'][-1, 2]], 
               color='darkred', marker='o', s=80, zorder=10, label='VMTK Bifurcation')
               
    ax.set_title(f"Patient {patient_id} - Anatomical Centerlines vs. Vessel Surface Mesh\n(Original Scanner Coordinates)", fontsize=11)
    ax.set_xlabel("X (Scanner)")
    ax.set_ylabel("Y (Scanner)")
    ax.set_zlabel("Z (Scanner)")
    
    pts_all = np.concatenate([orig_tree['LMCA'], orig_tree['LAD'], orig_tree['LCX'],
                              vmtk_tree['LMCA'], vmtk_tree['LAD'], vmtk_tree['LCX']])
    max_range = np.array([pts_all[:,0].max()-pts_all[:,0].min(), pts_all[:,1].max()-pts_all[:,1].min(), pts_all[:,2].max()-pts_all[:,2].min()]).max() / 2.0
    mid_x = (pts_all[:,0].max()+pts_all[:,0].min()) / 2.0
    mid_y = (pts_all[:,1].max()+pts_all[:,1].min()) / 2.0
    mid_z = (pts_all[:,2].max()+pts_all[:,2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

def get_stats(vals):
    """Computes Mean, Std, Median, Min, and Max from a list."""
    if not vals:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(vals)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr))
    }

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, "LCA_dataset")
    raw_vmtk_dir = os.path.join(base_dir, "lca_raw_branches")
    
    output_dir = os.path.join(base_dir, "outputs", "vmtk_dataset_lca")
    report_dir = os.path.join(output_dir, "anatomical_overlays")
    os.makedirs(report_dir, exist_ok=True)
    
    # 1. Dynamically discover all patient folders
    patient_folders = sorted(glob(os.path.join(dataset_dir, "*_H_CORO_*")))
    
    # Load Preprocessing report to identify failure reasons
    vmtk_report_path = os.path.join(base_dir, "processed_vmtk_dataset", "vmtk_preprocessing_report.json")
    vmtk_statuses = {}
    vmtk_errors = {}
    if os.path.exists(vmtk_report_path):
        try:
            with open(vmtk_report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
                for r in report:
                    vmtk_statuses[r["patient_id"]] = r.get("status", "FAILED")
                    vmtk_errors[r["patient_id"]] = r.get("error", "Preprocessor failed")
        except Exception as e:
            print(f"[WARN] Failed to load preprocessor report: {e}")

    # Load extraction comparison report for raw coordinates differences (Hausdorff & RMSE)
    comp_report_path = os.path.join(output_dir, "extraction_comparison_report.json")
    comp_map = {}
    if os.path.exists(comp_report_path):
        try:
            with open(comp_report_path, "r", encoding="utf-8") as f:
                comp_data = json.load(f)
                for item in comp_data:
                    comp_map[item["patient_id"]] = item
        except Exception as e:
            print(f"[WARN] Failed to load comparison report: {e}")

    anatomical_report = []
    failed_patients = []
    manual_review_patients = []
    
    # Accumulators for population-level statistics
    stat_metrics = {
        "mean_wall_distance_mm": {"Original": [], "VMTK": []},
        "symmetry_deviation_mm": {"Original": [], "VMTK": []},
        "radius_std_mm": {"Original": [], "VMTK": []},
        "curvature": {"Original": [], "VMTK": []},
        "tortuosity": {"Original": [], "VMTK": []},
        "hausdorff_mm": [],
        "rmse_mm": [],
        "bifurcation_angle_diff_deg": []
    }

    # Win counters
    wins = {"symmetry_deviation_mm": {"Original": 0, "VMTK": 0, "Tie": 0},
            "radius_std_mm": {"Original": 0, "VMTK": 0, "Tie": 0},
            "curvature": {"Original": 0, "VMTK": 0, "Tie": 0},
            "tortuosity": {"Original": 0, "VMTK": 0, "Tie": 0},
            "mean_wall_distance_mm": {"Original": 0, "VMTK": 0, "Tie": 0}}
            
    print("Starting Anatomical Validation against Original Vessel Meshes...")
    for p_folder in patient_folders:
        patient_id = os.path.basename(p_folder)
        
        # A. Find surface mesh dynamically
        vtp_file = None
        try:
            vtp_file = find_vessel_surface_mesh(p_folder)
        except FileNotFoundError as fnf:
            print(f"[FAIL] Patient {patient_id}: {fnf}")
            failed_patients.append({"patient_id": patient_id, "reason": "Missing surface mesh (.vtp)"})
            continue
            
        # B. Check raw branches file
        raw_vmtk_file = os.path.join(raw_vmtk_dir, f"{patient_id}_raw_branches.npz")
        if not os.path.exists(raw_vmtk_file):
            print(f"[FAIL] Patient {patient_id}: Missing VMTK raw branches NPZ")
            failed_patients.append({"patient_id": patient_id, "reason": "Missing VMTK raw branches (.npz)"})
            continue
            
        # C. Check preprocessor status
        v_status = vmtk_statuses.get(patient_id, "FAILED")
        if v_status == "Classification Failed":
            err_msg = vmtk_errors.get(patient_id, "Classification Failed")
            print(f"[FAIL] Patient {patient_id}: Flagged as 'Classification Failed' ({err_msg})")
            failed_patients.append({"patient_id": patient_id, "reason": err_msg})
            continue
        elif v_status == "FAILED":
            err_msg = vmtk_errors.get(patient_id, "Preprocessor crashed")
            print(f"[FAIL] Patient {patient_id}: Preprocessor failed ({err_msg})")
            failed_patients.append({"patient_id": patient_id, "reason": err_msg})
            continue
            
        print(f"Analyzing Patient {patient_id}...")
        
        try:
            # 1. Load surface points (scaled to mm)
            mesh_points = load_vtp_surface_points(vtp_file) * 10.0
            mesh_tree = KDTree(mesh_points)
            
            # 2. Extract original raw branches
            orig_lmca, orig_lad, orig_lcx = get_original_raw_branches(base_dir, patient_id)
            
            # 3. Extract raw VMTK branches
            vmtk_lmca_br, vmtk_lad_br, vmtk_lcx_br = parse_topology_and_extract_branches(raw_vmtk_file)
            vmtk_lmca = vmtk_lmca_br['xyz']
            vmtk_lad = vmtk_lad_br['xyz']
            vmtk_lcx = vmtk_lcx_br['xyz']
            
            patient_branches_metrics = {}
            review_flags = []
            
            for branch_name, o_pts, v_pts in [("LMCA", orig_lmca, vmtk_lmca), 
                                               ("LAD", orig_lad, vmtk_lad), 
                                               ("LCX", orig_lcx, vmtk_lcx)]:
                                               
                o_center = compute_centeredness_metrics(o_pts, mesh_tree, mesh_points)
                o_curvature = compute_curvature(o_pts)
                o_tortuosity = compute_tortuosity(o_pts)
                o_len = calculate_path_length(o_pts)
                
                v_center = compute_centeredness_metrics(v_pts, mesh_tree, mesh_points)
                v_curvature = compute_curvature(v_pts)
                v_tortuosity = compute_tortuosity(v_pts)
                v_len = calculate_path_length(v_pts)
                
                metrics_to_compare = {
                    "mean_wall_distance_mm": (o_center["mean_wall_distance_mm"], v_center["mean_wall_distance_mm"]),
                    "symmetry_deviation_mm": (o_center["symmetry_deviation_mm"], v_center["symmetry_deviation_mm"]),
                    "radius_std_mm": (o_center["radius_std_mm"], v_center["radius_std_mm"]),
                    "curvature": (o_curvature, v_curvature),
                    "tortuosity": (o_tortuosity, v_tortuosity)
                }
                
                comparisons = {}
                for m_name, (o_val, v_val) in metrics_to_compare.items():
                    winner = evaluate_better(o_val, v_val, m_name)
                    diff = v_val - o_val
                    comparisons[m_name] = {
                        "original": o_val,
                        "vmtk": v_val,
                        "diff": diff,
                        "better": winner
                    }
                    if winner != "N/A":
                        wins[m_name][winner] += 1
                        
                    # Accumulate for stats spread
                    stat_metrics[m_name]["Original"].append(o_val)
                    stat_metrics[m_name]["VMTK"].append(v_val)
                    
                    # Threshold checks for manual review
                    abs_diff = abs(diff)
                    if m_name == "mean_wall_distance_mm" and abs_diff > REVIEW_THRESHOLDS["mean_wall_distance_diff_mm"]:
                        review_flags.append(f"{branch_name} Wall Distance Diff ({abs_diff:.3f} mm) > threshold ({REVIEW_THRESHOLDS['mean_wall_distance_diff_mm']} mm)")
                    elif m_name == "symmetry_deviation_mm" and abs_diff > REVIEW_THRESHOLDS["symmetry_deviation_diff_mm"]:
                        review_flags.append(f"{branch_name} Centeredness Diff ({abs_diff:.3f} mm) > threshold ({REVIEW_THRESHOLDS['symmetry_deviation_diff_mm']} mm)")
                    elif m_name == "curvature" and abs_diff > REVIEW_THRESHOLDS["curvature_diff"]:
                        review_flags.append(f"{branch_name} Curvature Diff ({abs_diff:.3f}) > threshold ({REVIEW_THRESHOLDS['curvature_diff']})")
                    elif m_name == "tortuosity" and abs_diff > REVIEW_THRESHOLDS["tortuosity_diff"]:
                        review_flags.append(f"{branch_name} Tortuosity Diff ({abs_diff:.3f}) > threshold ({REVIEW_THRESHOLDS['tortuosity_diff']})")
                        
                patient_branches_metrics[branch_name] = {
                    "comparisons": comparisons,
                    "lengths_mm": {"original": o_len, "vmtk": v_len, "diff": v_len - o_len}
                }
                
            angle_orig = calculate_bifurcation_angle(orig_lad, orig_lcx)
            angle_vmtk = calculate_bifurcation_angle(vmtk_lad, vmtk_lcx)
            angle_diff = angle_vmtk - angle_orig
            
            if abs(angle_diff) > REVIEW_THRESHOLDS["takeoff_angle_diff_deg"]:
                review_flags.append(f"Takeoff Bifurcation Angle Diff ({abs(angle_diff):.2f}°) > threshold ({REVIEW_THRESHOLDS['takeoff_angle_diff_deg']}°)")

            # Accumulate patient-level stats
            stat_metrics["bifurcation_angle_diff_deg"].append(abs(angle_diff))
            c_data = comp_map.get(patient_id)
            if c_data:
                for br in ["LMCA", "LAD", "LCX"]:
                    stat_metrics["hausdorff_mm"].append(c_data["metrics_raw"][br]["hausdorff_mm"])
                    stat_metrics["rmse_mm"].append(c_data["metrics_raw"][br]["rmse_mm"])

            # Save patient results
            patient_results = {
                "patient_id": patient_id,
                "status": "VALIDATED",
                "needs_manual_review": len(review_flags) > 0,
                "review_reasons": review_flags,
                "bifurcation_angle_deg": {
                    "original": angle_orig,
                    "vmtk": angle_vmtk,
                    "diff": angle_diff
                },
                "branches": patient_branches_metrics
            }
            anatomical_report.append(patient_results)
            
            # Automatically generate overlays for manual review, or all successfully validated
            plot_path = os.path.join(report_dir, f"{patient_id}_anatomical_comparison.png")
            generate_3d_anatomical_plot(
                patient_id, 
                mesh_points, 
                {'LMCA': orig_lmca, 'LAD': orig_lad, 'LCX': orig_lcx}, 
                {'LMCA': vmtk_lmca, 'LAD': vmtk_lad, 'LCX': vmtk_lcx}, 
                plot_path
            )
            
            if len(review_flags) > 0:
                print(f"  [FLAG] Patient {patient_id} flagged for review: {len(review_flags)} reasons.")
                manual_review_patients.append({"patient_id": patient_id, "reasons": review_flags, "plot": f"outputs/vmtk_dataset_lca/anatomical_overlays/{patient_id}_anatomical_comparison.png"})
                
        except Exception as e:
            print(f"[ERROR] Anatomical validation failed for patient {patient_id}: {e}")
            failed_patients.append({"patient_id": patient_id, "reason": f"Execution error: {str(e)}"})
            
    # Save detailed JSON report
    report_json_path = os.path.join(output_dir, "anatomical_validation_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(anatomical_report, f, indent=6)
        
    print(f"\nAnatomical validation completed.")
    print(f"Report JSON saved to: {report_json_path}")
    print(f"Overlays saved to: {report_dir}")

    # ==============================================================================
    # POPULATION SUMMARY COMPILATION
    # ==============================================================================
    summary_path = os.path.join(output_dir, "anatomical_validation_summary.md")
    
    num_validated = len(anatomical_report)
    
    # Process failures and manual review lists
    failed_table = ""
    if failed_patients:
        failed_table = "| Patient ID | Failure Reason |\n| :--- | :--- |\n"
        for fp in failed_patients:
            failed_table += f"| `{fp['patient_id']}` | {fp['reason']} |\n"
    else:
        failed_table = "*No failures occurred. All discoverable patients were validated successfully.*"
        
    review_table = ""
    if manual_review_patients:
        review_table = "| Patient ID | Triggering Reason(s) | 3D Visual Overlay Link |\n| :--- | :--- | :--- |\n"
        for mr in manual_review_patients:
            reasons_str = "<br>".join(mr["reasons"])
            review_table += f"| `{mr['patient_id']}` | {reasons_str} | [View Overlay]({mr['plot']}) |\n"
    else:
        review_table = "*No patients exceeded manual review thresholds.*"

    # Compute Statistical spreads
    stats_spread = {}
    for key, val in stat_metrics.items():
        if isinstance(val, dict):
            stats_spread[key] = {
                "Original": get_stats(val["Original"]),
                "VMTK": get_stats(val["VMTK"])
            }
        else:
            stats_spread[key] = get_stats(val)

    # Winner definitions based on computed statistics
    overall_lumen_follow = "Tie"
    if wins["mean_wall_distance_mm"]["VMTK"] > wins["mean_wall_distance_mm"]["Original"]:
        overall_lumen_follow = "VMTK"
    elif wins["mean_wall_distance_mm"]["Original"] > wins["mean_wall_distance_mm"]["VMTK"]:
        overall_lumen_follow = "Original"
        
    overall_centered = "Tie"
    if wins["symmetry_deviation_mm"]["VMTK"] > wins["symmetry_deviation_mm"]["Original"]:
        overall_centered = "VMTK"
    elif wins["symmetry_deviation_mm"]["Original"] > wins["symmetry_deviation_mm"]["VMTK"]:
        overall_centered = "Original"
        
    overall_smoothness = "Tie"
    vmtk_smooth_score = wins["curvature"]["VMTK"] + wins["tortuosity"]["VMTK"]
    orig_smooth_score = wins["curvature"]["Original"] + wins["tortuosity"]["Original"]
    if vmtk_smooth_score > orig_smooth_score:
        overall_smoothness = "VMTK"
    elif orig_smooth_score > vmtk_smooth_score:
        overall_smoothness = "Original"

    # Patient details markdown compilation
    patient_details_list = []
    for pat in anatomical_report:
        pid = pat["patient_id"]
        pat_md = f"### Patient {pid}\n\n"
        pat_md += f"* **Bifurcation Takeoff Angle**:\n"
        pat_md += f"  * Original: {pat['bifurcation_angle_deg']['original']:.4f}°\n"
        pat_md += f"  * VMTK: {pat['bifurcation_angle_deg']['vmtk']:.4f}°\n"
        pat_md += f"  * Absolute Difference: {abs(pat['bifurcation_angle_deg']['diff']):.4f}°\n"
        
        status_review = "**Needs Manual Review** ⚠️" if pat["needs_manual_review"] else "Passed Auto-validation"
        pat_md += f"* **Validation Status**: {status_review}\n"
        if pat["needs_manual_review"]:
            pat_md += "  * Trigger(s): " + ", ".join(pat["review_reasons"]) + "\n"
        pat_md += "\n"
        
        pat_md += "| Branch | Metric | Original | VMTK | Winner | Reason |\n"
        pat_md += "| :--- | :--- | :---: | :---: | :---: | :--- |\n"
        
        for branch_name in ["LMCA", "LAD", "LCX"]:
            b_data = pat["branches"][branch_name]
            comps = b_data["comparisons"]
            
            # Wall distance
            orig_wd = comps["mean_wall_distance_mm"]["original"]
            vmtk_wd = comps["mean_wall_distance_mm"]["vmtk"]
            winner_wd = comps["mean_wall_distance_mm"]["better"]
            reason_wd = "VMTK sits inside larger inscribed spheres (closer to center)." if winner_wd == "VMTK" else "Original runs through wider lumen regions."
            if winner_wd == "Tie": reason_wd = "Equivalent radius tracking."
            pat_md += f"| **{branch_name}** | Mean Wall Distance (mm) | {orig_wd:.4f} | {vmtk_wd:.4f} | **{winner_wd}** | {reason_wd} |\n"
            
            # Symmetry Deviation
            orig_sd = comps["symmetry_deviation_mm"]["original"]
            vmtk_sd = comps["symmetry_deviation_mm"]["vmtk"]
            winner_sd = comps["symmetry_deviation_mm"]["better"]
            reason_sd = "VMTK has lower offset from cross-sectional lumen centroid." if winner_sd == "VMTK" else "Original centerline is closer to the geometric center."
            if winner_sd == "Tie": reason_sd = "Equally centered."
            pat_md += f"| | Symmetry Deviation (mm) | {orig_sd:.4f} | {vmtk_sd:.4f} | **{winner_sd}** | {reason_sd} |\n"
            
            # Radius Std
            orig_rs = comps["radius_std_mm"]["original"]
            vmtk_rs = comps["radius_std_mm"]["vmtk"]
            winner_rs = comps["radius_std_mm"]["better"]
            reason_rs = "VMTK has more consistent tapering (lower radius std)." if winner_rs == "VMTK" else "Original has fewer tapering wiggles."
            if winner_rs == "Tie": reason_rs = "Identical radius spread."
            pat_md += f"| | Radius Consistency (mm) | {orig_rs:.4f} | {vmtk_rs:.4f} | **{winner_rs}** | {reason_rs} |\n"
            
            # Curvature/Tortuosity (Smoothness score)
            orig_c = comps["curvature"]["original"]
            vmtk_c = comps["curvature"]["vmtk"]
            orig_t = comps["tortuosity"]["original"]
            vmtk_t = comps["tortuosity"]["vmtk"]
            
            # Smoothness score evaluation
            if (vmtk_c + vmtk_t) < (orig_c + orig_t):
                winner_sm = "VMTK"
                reason_sm = "VMTK is mathematically smoother (lower curvature/tortuosity)."
            else:
                winner_sm = "Original"
                reason_sm = "Original has fewer high-frequency wiggles (lower curvature)."
            pat_md += f"| | Curvature | {orig_c:.4f} | {vmtk_c:.4f} | **{winner_sm}** | {reason_sm} |\n"
            pat_md += f"| | Tortuosity | {orig_t:.4f} | {vmtk_t:.4f} | - | - |\n"
            
        pat_md += "\n---\n"
        patient_details_list.append(pat_md)

    # Compile Dynamic Evidence-Based Recommendation
    total_branches = num_validated * 3
    final_win_centeredness = wins["symmetry_deviation_mm"]["VMTK"]
    final_win_taper = wins["radius_std_mm"]["VMTK"]
    final_win_smoothness = wins["curvature"]["Original"] + wins["tortuosity"]["Original"]
    
    recommendation_block = ""
    if num_validated > 0:
        if final_win_centeredness > (total_branches // 2) and final_win_taper > (total_branches // 2):
            recommendation_block = f"""**Recommendation**: **VMTK (Mesh-Derived)** should be preferred for downstream statistical modeling. 
The quantitative evidence across all successfully validated patient datasets ({num_validated} patients) shows that VMTK centerlines sit significantly closer to the geometric lumen center (winning centeredness on {final_win_centeredness} out of {total_branches} branches) and show higher radius tapering consistency (winning on {final_win_taper} out of {total_branches} branches). The original manual annotations contain minor tracking wiggles that result in larger offsets and noise, although they are smoother overall."""
        else:
            recommendation_block = f"""**Recommendation**: **Original (Human-Annotated)** centerlines are preferred.
The quantitative evidence shows that VMTK centerlines do not satisfy the majority of geometric criteria, specifically regarding curvature/tortuosity (Original wins on {final_win_smoothness} comparisons), which introduces significant wiggles that degrade shape analysis."""
    else:
        recommendation_block = "**Recommendation**: No patients were successfully validated, so no centerline modeling recommendation can be made."

    formula_centeredness = r"$$\vec{c}_i = \frac{1}{M}\sum_{j=1}^M \left[ \vec{u}_j - (\vec{u}_j \cdot \hat{t}_i)\hat{t}_i \right]$$"
    formula_radius = r"$$\sigma_r = \sqrt{\frac{1}{N}\sum_{i=1}^N (r_i - \bar{r})^2}$$"
    formula_curvature = r"$$\kappa_i = \frac{2 \|a \times b\|}{\|a\| \|b\| \|a + b\|}$$"
    formula_tortuosity = r"$$T = \frac{\text{Path Length}}{\text{Straight-Line Distance}} - 1$$"

    summary_md = f"""# Anatomical Centerline Validation Summary Report

This report summarizes the objective, quantitative validation between the **Original (Human-Annotated)** centerlines and the **VMTK (Mesh-Derived)** centerlines, using the patient-specific 3D coronary artery surface meshes (`.vtp`) as the absolute reference.

---

## 1. Documentation of Anatomical Metrics

### A. Lumen Centeredness (Symmetry Deviation)
* **Mathematical Definition**: The average magnitude of the projected vectors from each centerline point to the nearby vessel wall points on the normal cross-sectional plane.
  {formula_centeredness}
  where $\\vec{{u}}_j = V_j - P_i$ is the vector to surface vertex $V_j$, and $\\hat{{t}}_i$ is the unit local tangent.
* **Unit**: Millimeters (mm).
* **Indication**: A perfectly centered path has a symmetry deviation of $0.0$ mm. A lower value indicates a more anatomically centered path inside the vessel lumen.

### B. Radius Consistency (Tapering)
* **Mathematical Definition**: The standard deviation of the local wall distance (closest distance to the vessel mesh) along the path of each branch.
  {formula_radius}
* **Unit**: Millimeters (mm).
* **Indication**: Healthy coronary artery branches taper smoothly. A lower standard deviation indicates a consistent radius profile, free from artificial wiggles or localized diameter fluctuations.

### C. Curvature
* **Mathematical Definition**: Mean discrete Menger curvature calculated circumcircularly using three consecutive points:
  {formula_curvature}
  where $a = P_i - P_{{i-1}}$ and $b = P_{{i+1}} - P_i$.
* **Unit**: Dimensionless ($mm^{{-1}}$).
* **Indication**: Evaluates path smoothness. A lower curvature indicates a smoother mathematical curve that is less prone to tracking wiggles.

### D. Tortuosity
* **Mathematical Definition**: The arc-to-chord length ratio minus one:
  {formula_tortuosity}
* **Unit**: Dimensionless.
* **Indication**: Evaluates wiggles along length. A lower tortuosity indicates a more direct, physiologically natural centerline path.

---

## 2. Failure Analysis
Every discoverable patient folder in `LCA_dataset` was processed. Failures and skips are documented below:

{failed_table}

---

## 3. Cases Flagged for Manual Review
The following patients exceeded one or more thresholds defined in `REVIEW_THRESHOLDS`:
* Takeoff angle difference $> {REVIEW_THRESHOLDS['takeoff_angle_diff_deg']}°$
* Mean wall distance difference $> {REVIEW_THRESHOLDS['mean_wall_distance_diff_mm']}$ mm
* Symmetry deviation difference $> {REVIEW_THRESHOLDS['symmetry_deviation_diff_mm']}$ mm
* Curvature difference $> {REVIEW_THRESHOLDS['curvature_diff']}$
* Tortuosity difference $> {REVIEW_THRESHOLDS['tortuosity_diff']}$

{review_table}

---

## 4. Dataset-Level Statistical Spreads
Statistical averages computed over all successfully validated patients ({num_validated} patients):

| Metric | Origin / VMTK | Mean | Std Dev | Median | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Lumen Centeredness (mm)** | Original | {stats_spread["symmetry_deviation_mm"]["Original"]["mean"]:.6f} | {stats_spread["symmetry_deviation_mm"]["Original"]["std"]:.6f} | {stats_spread["symmetry_deviation_mm"]["Original"]["median"]:.6f} | {stats_spread["symmetry_deviation_mm"]["Original"]["min"]:.6f} | {stats_spread["symmetry_deviation_mm"]["Original"]["max"]:.6f} |
| | VMTK | {stats_spread["symmetry_deviation_mm"]["VMTK"]["mean"]:.6f} | {stats_spread["symmetry_deviation_mm"]["VMTK"]["std"]:.6f} | {stats_spread["symmetry_deviation_mm"]["VMTK"]["median"]:.6f} | {stats_spread["symmetry_deviation_mm"]["VMTK"]["min"]:.6f} | {stats_spread["symmetry_deviation_mm"]["VMTK"]["max"]:.6f} |
| **Radius Consistency (mm)** | Original | {stats_spread["radius_std_mm"]["Original"]["mean"]:.6f} | {stats_spread["radius_std_mm"]["Original"]["std"]:.6f} | {stats_spread["radius_std_mm"]["Original"]["median"]:.6f} | {stats_spread["radius_std_mm"]["Original"]["min"]:.6f} | {stats_spread["radius_std_mm"]["Original"]["max"]:.6f} |
| | VMTK | {stats_spread["radius_std_mm"]["VMTK"]["mean"]:.6f} | {stats_spread["radius_std_mm"]["VMTK"]["std"]:.6f} | {stats_spread["radius_std_mm"]["VMTK"]["median"]:.6f} | {stats_spread["radius_std_mm"]["VMTK"]["min"]:.6f} | {stats_spread["radius_std_mm"]["VMTK"]["max"]:.6f} |
| **Curvature** | Original | {stats_spread["curvature"]["Original"]["mean"]:.6f} | {stats_spread["curvature"]["Original"]["std"]:.6f} | {stats_spread["curvature"]["Original"]["median"]:.6f} | {stats_spread["curvature"]["Original"]["min"]:.6f} | {stats_spread["curvature"]["Original"]["max"]:.6f} |
| | VMTK | {stats_spread["curvature"]["VMTK"]["mean"]:.6f} | {stats_spread["curvature"]["VMTK"]["std"]:.6f} | {stats_spread["curvature"]["VMTK"]["median"]:.6f} | {stats_spread["curvature"]["VMTK"]["min"]:.6f} | {stats_spread["curvature"]["VMTK"]["max"]:.6f} |
| **Tortuosity** | Original | {stats_spread["tortuosity"]["Original"]["mean"]:.6f} | {stats_spread["tortuosity"]["Original"]["std"]:.6f} | {stats_spread["tortuosity"]["Original"]["median"]:.6f} | {stats_spread["tortuosity"]["Original"]["min"]:.6f} | {stats_spread["tortuosity"]["Original"]["max"]:.6f} |
| | VMTK | {stats_spread["tortuosity"]["VMTK"]["mean"]:.6f} | {stats_spread["tortuosity"]["VMTK"]["std"]:.6f} | {stats_spread["tortuosity"]["VMTK"]["median"]:.6f} | {stats_spread["tortuosity"]["VMTK"]["min"]:.6f} | {stats_spread["tortuosity"]["VMTK"]["max"]:.6f} |
| **Mean Wall Distance (mm)** | Original | {stats_spread["mean_wall_distance_mm"]["Original"]["mean"]:.6f} | {stats_spread["mean_wall_distance_mm"]["Original"]["std"]:.6f} | {stats_spread["mean_wall_distance_mm"]["Original"]["median"]:.6f} | {stats_spread["mean_wall_distance_mm"]["Original"]["min"]:.6f} | {stats_spread["mean_wall_distance_mm"]["Original"]["max"]:.6f} |
| | VMTK | {stats_spread["mean_wall_distance_mm"]["VMTK"]["mean"]:.6f} | {stats_spread["mean_wall_distance_mm"]["VMTK"]["std"]:.6f} | {stats_spread["mean_wall_distance_mm"]["VMTK"]["median"]:.6f} | {stats_spread["mean_wall_distance_mm"]["VMTK"]["min"]:.6f} | {stats_spread["mean_wall_distance_mm"]["VMTK"]["max"]:.6f} |
| **Hausdorff Curve Diff (mm)**| VMTK vs Orig| {stats_spread["hausdorff_mm"]["mean"]:.6f} | {stats_spread["hausdorff_mm"]["std"]:.6f} | {stats_spread["hausdorff_mm"]["median"]:.6f} | {stats_spread["hausdorff_mm"]["min"]:.6f} | {stats_spread["hausdorff_mm"]["max"]:.6f} |
| **RMSE Curve Diff (mm)** | VMTK vs Orig| {stats_spread["rmse_mm"]["mean"]:.6f} | {stats_spread["rmse_mm"]["std"]:.6f} | {stats_spread["rmse_mm"]["median"]:.6f} | {stats_spread["rmse_mm"]["min"]:.6f} | {stats_spread["rmse_mm"]["max"]:.6f} |
| **Bifurcation angle diff (°)**| VMTK vs Orig| {stats_spread["bifurcation_angle_diff_deg"]["mean"]:.6f} | {stats_spread["bifurcation_angle_diff_deg"]["std"]:.6f} | {stats_spread["bifurcation_angle_diff_deg"]["median"]:.6f} | {stats_spread["bifurcation_angle_diff_deg"]["min"]:.6f} | {stats_spread["bifurcation_angle_diff_deg"]["max"]:.6f} |

---

## 5. Patient-by-Patient Breakdown

"""
    
    summary_md += "\n".join(patient_details_list)
    
    # Final Cohort Summary Table Block at the very end
    overall_winner = "Tie"
    vmtk_wins = wins["symmetry_deviation_mm"]["VMTK"] + wins["radius_std_mm"]["VMTK"]
    orig_wins = wins["symmetry_deviation_mm"]["Original"] + wins["radius_std_mm"]["Original"]
    if vmtk_wins > orig_wins:
        overall_winner = "VMTK"
    elif orig_wins > vmtk_wins:
        overall_winner = "Original"

    summary_md += f"""

---

## 6. Final Overall Cohort Summary

| Metric Evaluation | Original Better Wins | VMTK Better Wins | Tie Wins | Metric Superiority Winner |
| :--- | :--- | :---: | :---: | :---: |
| **Lumen Centeredness** | {wins["symmetry_deviation_mm"]["Original"]} | {wins["symmetry_deviation_mm"]["VMTK"]} | {wins["symmetry_deviation_mm"]["Tie"]} | **{"VMTK" if wins["symmetry_deviation_mm"]["VMTK"] > wins["symmetry_deviation_mm"]["Original"] else "Original"}** |
| **Radius Consistency** | {wins["radius_std_mm"]["Original"]} | {wins["radius_std_mm"]["VMTK"]} | {wins["radius_std_mm"]["Tie"]} | **{"VMTK" if wins["radius_std_mm"]["VMTK"] > wins["radius_std_mm"]["Original"] else "Original"}** |
| **Curvature** | {wins["curvature"]["Original"]} | {wins["curvature"]["VMTK"]} | {wins["curvature"]["Tie"]} | **{"VMTK" if wins["curvature"]["VMTK"] > wins["curvature"]["Original"] else "Original"}** |
| **Tortuosity** | {wins["tortuosity"]["Original"]} | {wins["tortuosity"]["VMTK"]} | {wins["tortuosity"]["Tie"]} | **{"VMTK" if wins["tortuosity"]["VMTK"] > wins["tortuosity"]["Original"] else "Original"}** |
| **Overall Cohort Winner** | — | **{overall_winner}** | — | — |

* **Total Patients Processed**: {len(patient_folders)}
* **Successfully Validated**: {num_validated}
* **Failed / Skipped**: {len(failed_patients)}
* **Needs Manual Review**: {len(manual_review_patients)}

### Final Recommendation
{recommendation_block}
"""

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_md)
        
    print(f"\nUnified summary validation report written: {summary_path}")

if __name__ == "__main__":
    main()
