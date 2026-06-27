# LCA_topology_generator/patient_preprocessor.py

import os
import xml.etree.ElementTree as ET
import base64
import zlib
import struct
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Mapping from patient folder to its anatomical centerline files
PATIENT_MAPPING = {
    "0070_H_CORO_KD": {
        "source": "mix",
        "lmca_lad_file": "Paths/left_coronary.pth",
        "lcx_file": "Segmentations/left(2).ctgr"
    },
    "0075_H_CORO_CAD": {
        "source": "pth",
        "lmca_lad_file": "Paths/lc1.pth",
        "lcx_file": "Paths/lc2.pth"
    },
    "0136_H_CORO_KD": {
        "source": "pth",
        "lmca_lad_file": "Paths/left1.pth",
        "lcx_file": "Paths/left2.pth"
    },
    "0138_H_CORO_KD": {
        "source": "pth",
        "lmca_lad_file": "Paths/lca1.pth",
        "lcx_file": "Paths/lca2.pth"
    },
    "0145_H_CORO_KD": {
        "source": "pth",
        "lmca_lad_file": "Paths/lca1.pth",
        "lcx_file": "Paths/lca2.pth"
    },
    "0169_H_CORO_KD": {
        "source": "ctgr",
        "lmca_lad_file": "Segmentations/left_lad.ctgr",
        "lcx_file": "Segmentations/lcx.ctgr"
    },
    "0170_H_CORO_KD": {
        "source": "ctgr",
        "lmca_lad_file": "Segmentations/left_lad.ctgr",
        "lcx_file": "Segmentations/left_lcx.ctgr"
    },
    "0177_H_CORO_KD": {
        "source": "vtp",
        "lmca_lad_file": "Models/Centerlines/lca1.vtp",
        "lcx_file": "Models/Centerlines/lca3.vtp"
    },
    "0178_H_CORO_KD": {
        "source": "vtp",
        "lmca_lad_file": "Models/Centerlines/lca1.vtp",
        "lcx_file": "Models/Centerlines/lca2.vtp"
    }
}

def parse_pth_points(pth_path):
    if not os.path.exists(pth_path):
        return []
    try:
        with open(pth_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace('<format version="1.0" />', '')
        root = ET.fromstring(content)
        points = []
        path_points = root.findall(".//path_points/path_point/pos")
        for p in path_points:
            x = float(p.attrib['x'])
            y = float(p.attrib['y'])
            z = float(p.attrib['z'])
            points.append([x, y, z])
        if not points:
            control_points = root.findall(".//control_points/point")
            for p in control_points:
                x = float(p.attrib['x'])
                y = float(p.attrib['y'])
                z = float(p.attrib['z'])
                points.append([x, y, z])
        return points
    except Exception:
        return []

def parse_ctgr_points(ctgr_path):
    if not os.path.exists(ctgr_path):
        return []
    try:
        with open(ctgr_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = content.replace('<format version="1.0" />', '')
        root = ET.fromstring(content)
        points = []
        contours = root.findall(".//contour")
        for c in contours:
            pos = c.find(".//path_point/pos")
            if pos is not None:
                x = float(pos.attrib['x'])
                y = float(pos.attrib['y'])
                z = float(pos.attrib['z'])
                points.append([x, y, z])
        return points
    except Exception:
        return []

def parse_vtp_points(vtp_path):
    if not os.path.exists(vtp_path):
        return []
    try:
        tree = ET.parse(vtp_path)
        root = tree.getroot()
        points_elem = root.find(".//Points/DataArray")
        if points_elem is None:
            return []
        
        text = "".join(points_elem.text.split())
        header_b64 = text[:24]
        data_b64 = text[24:]
        
        header_decoded = base64.b64decode(header_b64)
        data_decoded = base64.b64decode(data_b64)
            
        try:
            decompressed = zlib.decompress(data_decoded)
        except Exception:
            decompressed = zlib.decompress(data_decoded, -zlib.MAX_WBITS)
                
        point_format = f"<{len(decompressed)//4}f"
        coords = struct.unpack(point_format, decompressed)
        
        points = []
        for i in range(0, len(coords), 3):
            if i + 2 < len(coords):
                points.append([coords[i], coords[i+1], coords[i+2]])
                
        # Ensure it starts from ostium region (high Z value for KD datasets)
        if len(points) > 1 and points[0][2] < points[-1][2]:
            points.reverse()
            
        return points
    except Exception:
        return []

def resample_polyline(points, n):
    points = np.asarray(points, dtype=float)
    diffs = np.diff(points, axis=0)
    distances = np.zeros(len(points))
    distances[1:] = np.cumsum(np.linalg.norm(diffs, axis=1))
    
    total = distances[-1]
    if total <= 1e-9:
        return np.tile(points[0], (n, 1))
        
    target = np.linspace(0.0, total, n)
    result = np.zeros((n, 3), dtype=float)
    for dim in range(3):
        result[:, dim] = np.interp(target, distances, points[:, dim])
    return result

def get_rotation_matrix_align_vectors(u, v):
    """Returns the 3x3 rotation matrix that rotates vector u to vector v."""
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    cos_theta = np.dot(u, v)
    if cos_theta > 0.999999:
        return np.eye(3)
    if cos_theta < -0.999999:
        axis = np.array([0.0, 1.0, 0.0]) if abs(u[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        axis -= np.dot(axis, u) * u
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * np.dot(K, K)
        
    axis = np.cross(u, v)
    sin_theta = np.linalg.norm(axis)
    axis /= sin_theta
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + sin_theta * K + (1.0 - cos_theta) * np.dot(K, K)
    return R

def normalize_artery_tree(lmca, lad, lcx):
    """
    Translates ostium to (0,0,0), aligns LMCA vector to +X, and rotates
    around X so the bifurcation plane normal aligns to +Z.
    """
    ostium = lmca[0].copy()
    lmca_trans = lmca - ostium
    lad_trans = lad - ostium
    lcx_trans = lcx - ostium
    
    bifurcation = lmca_trans[-1]
    lmca_vec = bifurcation
    lmca_vec_len = np.linalg.norm(lmca_vec)
    if lmca_vec_len > 1e-6:
        lmca_dir = lmca_vec / lmca_vec_len
        R1 = get_rotation_matrix_align_vectors(lmca_dir, np.array([1.0, 0.0, 0.0]))
    else:
        R1 = np.eye(3)
    
    lmca_rot1 = np.dot(lmca_trans, R1.T)
    lad_rot1 = np.dot(lad_trans, R1.T)
    lcx_rot1 = np.dot(lcx_trans, R1.T)
    
    v_lad = lad_rot1[min(len(lad_rot1)-1, 3)] - lad_rot1[0]
    v_lcx = lcx_rot1[min(len(lcx_rot1)-1, 3)] - lcx_rot1[0]
    
    plane_normal = np.cross(v_lad, v_lcx)
    normal_yz = np.array([0.0, plane_normal[1], plane_normal[2]])
    norm_yz_len = np.linalg.norm(normal_yz)
    
    if norm_yz_len > 1e-6:
        normal_yz /= norm_yz_len
        cos_phi = normal_yz[2]
        sin_phi = -normal_yz[1]
        
        R2 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_phi, -sin_phi],
            [0.0, sin_phi, cos_phi]
        ])
        
        lmca_normalized = np.dot(lmca_rot1, R2.T)
        lad_normalized = np.dot(lad_rot1, R2.T)
        lcx_normalized = np.dot(lcx_rot1, R2.T)
    else:
        lmca_normalized = lmca_rot1
        lad_normalized = lad_rot1
        lcx_normalized = lcx_rot1
        
    return lmca_normalized, lad_normalized, lcx_normalized

def main():
    dataset_root = r"D:\Projects\lca_vessel\LCA_dataset"
    output_base_dir = r"D:\Projects\lca_vessel\latest_lca\vessel_tree_generator\processed_dataset"
    ctrl_points_dir = r"D:\Projects\lca_vessel\latest_lca\vessel_tree_generator\LCA_branch_control_points\generated"
    
    os.makedirs(output_base_dir, exist_ok=True)
    os.makedirs(ctrl_points_dir, exist_ok=True)
    
    qc_reports = []
    collective_lmca = []
    collective_lad = []
    collective_lcx = []
    
    for patient_id, mapping in sorted(PATIENT_MAPPING.items()):
        p_path = os.path.join(dataset_root, patient_id)
        if not os.path.exists(p_path):
            continue
            
        report = {
            "patient_id": patient_id,
            "status": "FAILED",
            "errors": [],
            "separation_type": mapping["source"],
            "bifurcation_distance_error_mm": 0.0
        }
        
        lmca_raw = []
        lad_raw = []
        lcx_raw = []
        
        try:
            if mapping["source"] == "pth_split":
                lmca_raw = parse_pth_points(os.path.join(p_path, mapping["lmca_file"]))
                lad_raw = parse_pth_points(os.path.join(p_path, mapping["lad_file"]))
                lcx_raw = parse_pth_points(os.path.join(p_path, mapping["lcx_file"]))
                
                lmca_arr = np.array(lmca_raw)
                lad_arr = np.array(lad_raw)
                lcx_arr = np.array(lcx_raw)
                
                l_lmca = np.sum(np.linalg.norm(np.diff(lmca_arr, axis=0), axis=1))
                l_lad = np.sum(np.linalg.norm(np.diff(lad_arr, axis=0), axis=1))
                l_lcx = np.sum(np.linalg.norm(np.diff(lcx_arr, axis=0), axis=1))
                
                if max(l_lmca, l_lad, l_lcx) < 30.0:
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
                    
                if not lmca_lad_raw or not lcx_raw:
                    raise ValueError("Raw path files are empty or missing.")
                    
                lmca_lad_pts = np.array(lmca_lad_raw)
                lcx_pts = np.array(lcx_raw)
                
                # Scale raw coordinates to millimeters immediately if they are in centimeters
                raw_l_lmca_lad = np.sum(np.linalg.norm(np.diff(lmca_lad_pts, axis=0), axis=1))
                raw_l_lcx = np.sum(np.linalg.norm(np.diff(lcx_pts, axis=0), axis=1))
                if max(raw_l_lmca_lad, raw_l_lcx) < 30.0:
                    lmca_lad_pts *= 10.0
                    lcx_pts *= 10.0
                
                # Check if paths are overlapping at the start
                start_gap = np.linalg.norm(lmca_lad_pts[0] - lcx_pts[0])
                
                # If they start close (e.g. within 1.5 mm), they overlap on the LMCA trunk
                if start_gap < 1.5:
                    # Find the deviation index (where they split)
                    min_len = min(len(lmca_lad_pts), len(lcx_pts))
                    bifurcation_idx = 0
                    for idx in range(min_len):
                        d = np.linalg.norm(lmca_lad_pts[idx] - lcx_pts[idx])
                        if d > 1.5: # separation threshold (1.5 mm)
                            bifurcation_idx = idx
                            break
                    if bifurcation_idx == 0:
                        bifurcation_idx = min_len // 4 # fallback split
                        
                    report["bifurcation_distance_error_mm"] = 0.0
                    
                    lmca_arr = lmca_lad_pts[:bifurcation_idx + 1]
                    lad_arr = lmca_lad_pts[bifurcation_idx:]
                    lcx_arr = lcx_pts[bifurcation_idx:]
                else:
                    # Non-overlapping: LCx branches off mid-path
                    proximal_lcx_pts = lcx_pts[:min(len(lcx_pts), 3)]
                    best_indices = []
                    min_distances = []
                    for pt in proximal_lcx_pts:
                        dists = np.linalg.norm(lmca_lad_pts - pt, axis=1)
                        idx = np.argmin(dists)
                        best_indices.append(idx)
                        min_distances.append(dists[idx])
                        
                    bifurcation_idx = int(np.round(np.mean(best_indices)))
                    report["bifurcation_distance_error_mm"] = float(min_distances[0])
                    
                    # Threshold verification (12 mm maximum gap)
                    if min_distances[0] > 12.0:
                        raise ValueError(f"LCx ostium too far from LMCA trunk: {min_distances[0]:.3f} mm")
                        
                    lmca_arr = lmca_lad_pts[:bifurcation_idx + 1]
                    lad_arr = lmca_lad_pts[bifurcation_idx:]
                    
                    lcx_arr = lcx_pts.copy()
                    lcx_arr[0] = lmca_arr[-1]
                
            if len(lmca_arr) == 0 or len(lad_arr) == 0 or len(lcx_arr) == 0:
                raise ValueError("Extraction yielded empty coordinates.")
                
            # Compute branch lengths
            l_lmca = np.sum(np.linalg.norm(np.diff(lmca_arr, axis=0), axis=1))
            l_lad = np.sum(np.linalg.norm(np.diff(lad_arr, axis=0), axis=1))
            l_lcx = np.sum(np.linalg.norm(np.diff(lcx_arr, axis=0), axis=1))
            
            gap_lad = np.linalg.norm(lad_arr[0] - lmca_arr[-1])
            gap_lcx = np.linalg.norm(lcx_arr[0] - lmca_arr[-1])
            
            # Allow minor numerical snapping if gap is small, otherwise raise topology error
            if gap_lad > 2.0 or gap_lcx > 2.0:
                raise ValueError(f"Topology disconnected! LAD gap={gap_lad:.3f}mm, LCx gap={gap_lcx:.3f}mm")
                
            # Perform topological snapping for exact coordinate alignment
            lad_arr[0] = lmca_arr[-1]
            lcx_arr[0] = lmca_arr[-1]
            
            if l_lmca < 0.0 or l_lad < 10.0 or l_lcx < 10.0:
                raise ValueError(f"Suspiciously short lengths: LMCA={l_lmca:.1f}mm, LAD={l_lad:.1f}mm, LCx={l_lcx:.1f}mm")
                
            lmca_norm, lad_norm, lcx_norm = normalize_artery_tree(lmca_arr, lad_arr, lcx_arr)
            
            p_out_dir = os.path.join(output_base_dir, patient_id)
            os.makedirs(p_out_dir, exist_ok=True)
            np.save(os.path.join(p_out_dir, "lmca.npy"), lmca_norm)
            np.save(os.path.join(p_out_dir, "lad.npy"), lad_norm)
            np.save(os.path.join(p_out_dir, "lcx.npy"), lcx_norm)
            
            lmca_ctrl = resample_polyline(lmca_norm, 5)
            lad_ctrl = resample_polyline(lad_norm, 12)
            lcx_ctrl = resample_polyline(lcx_norm, 10)
            
            collective_lmca.append(lmca_ctrl)
            collective_lad.append(lad_ctrl)
            collective_lcx.append(lcx_ctrl)
            
            report["status"] = "PASSED"
            
        except Exception as e:
            report["errors"].append(str(e))
            print(f"ERROR processing {patient_id}: {e}")
            
        qc_reports.append(report)
        print(f"Processed Patient {patient_id:20}: Status={report['status']}")
        
    if collective_lmca:
        np.save(os.path.join(ctrl_points_dir, "LMCA_patient_ctrl_points.npy"), np.stack(collective_lmca, axis=0))
        np.save(os.path.join(ctrl_points_dir, "LAD_patient_ctrl_points.npy"), np.stack(collective_lad, axis=0))
        np.save(os.path.join(ctrl_points_dir, "LCX_patient_ctrl_points.npy"), np.stack(collective_lcx, axis=0))
        print("Successfully exported collective patient control point databases.")
    else:
        print("WARNING: No patients passed preprocessing! Collective databases were not exported.")
        
    with open(os.path.join(output_base_dir, "preprocessing_qc_report.json"), "w", encoding="utf-8") as qc_file:
        json.dump(qc_reports, qc_file, indent=2)
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(projection='3d')
    
    for patient_id in PATIENT_MAPPING.keys():
        p_dir = os.path.join(output_base_dir, patient_id)
        if os.path.exists(os.path.join(p_dir, "lmca.npy")):
            lmca = np.load(os.path.join(p_dir, "lmca.npy"))
            lad = np.load(os.path.join(p_dir, "lad.npy"))
            lcx = np.load(os.path.join(p_dir, "lcx.npy"))
            
            ax.plot(lmca[:, 0], lmca[:, 1], lmca[:, 2], color='black', alpha=0.7)
            ax.plot(lad[:, 0], lad[:, 1], lad[:, 2], color='red', alpha=0.7)
            ax.plot(lcx[:, 0], lcx[:, 1], lcx[:, 2], color='blue', alpha=0.7)
            
    ax.set_title("LCA Patient Centerlines Normalized Alignment Overlay")
    ax.set_xlabel("X (LMCA Direction)")
    ax.set_ylabel("Y (Lateral)")
    ax.set_zlabel("Z (Bifurcation Normal)")
    plt.savefig(os.path.join(output_base_dir, "all_normalized_centerlines.png"))
    plt.close()
    
    print("\nPreprocessing completed successfully.")
    print(f"QC report: {os.path.join(output_base_dir, 'preprocessing_qc_report.json')}")
    print(f"Alignment overlay plot: {os.path.join(output_base_dir, 'all_normalized_centerlines.png')}")

if __name__ == "__main__":
    main()
