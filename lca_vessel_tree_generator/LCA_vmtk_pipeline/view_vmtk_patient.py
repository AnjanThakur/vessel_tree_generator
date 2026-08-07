# LCA_vmtk_pipeline/view_vmtk_patient.py

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def get_valid_patients(trees_dir, report_path):
    # Scan the trees folder for exported valid patients (e.g. patient_0000)
    tree_folders = sorted(glob(os.path.join(trees_dir, "patient_*")))
    
    # Load the preprocessing report to map index to original patient ID
    patient_mapping = {}
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
                for idx, r in enumerate(report):
                    patient_mapping[idx] = r.get("patient_id", f"Patient_{idx}")
        except Exception as e:
            print(f"[WARN] Failed to load mapping report: {e}")
            
    valid_list = []
    for f in tree_folders:
        folder_name = os.path.basename(f)
        idx_str = folder_name.replace("patient_", "")
        try:
            idx = int(idx_str)
        except ValueError:
            continue
            
        original_name = patient_mapping.get(idx, f"patient_{idx:04d}")
        valid_list.append({
            "folder_name": folder_name,
            "original_name": original_name,
            "path": f
        })
    return valid_list

def main():
    # Resolve directories
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trees_dir = os.path.join(base_dir, "outputs", "vmtk_dataset_lca", "trees")
    report_path = os.path.join(base_dir, "processed_vmtk_dataset", "vmtk_preprocessing_report.json")
    
    valid_patients = get_valid_patients(trees_dir, report_path)
    
    if not valid_patients:
        print(f"[ERROR] No valid VMTK patient trees found in: {trees_dir}")
        print("Please run the preprocessor and generator first:\n")
        print("    python -m lca_vessel_tree_generator.LCA_vmtk_pipeline.vmtk_preprocessor")
        print("    python -m lca_vessel_tree_generator.LCA_vmtk_pipeline.generate_lca_vmtk_dataset")
        sys.exit(1)
        
    print("Available Valid VMTK Patients (6/9 passed validation):")
    for idx, p in enumerate(valid_patients):
        print(f"  [{idx}] {p['original_name']} ({p['folder_name']})")
        
    # Get user selection
    try:
        selection = input(f"\nSelect a patient index (0-{len(valid_patients)-1}) [Default: 0]: ").strip()
        selected_idx = int(selection) if selection else 0
        if selected_idx < 0 or selected_idx >= len(valid_patients):
            raise ValueError()
    except (ValueError, KeyboardInterrupt):
        print("Invalid selection or exit. Defaulting to patient index 0.")
        selected_idx = 0
        
    p = valid_patients[selected_idx]
    patient_folder = p['path']
    
    print(f"\nLoading and visualizing: {p['original_name']} ({p['folder_name']})")
    try:
        lmca = np.load(os.path.join(patient_folder, "lmca_centerline.npy"))
        lad = np.load(os.path.join(patient_folder, "lad_centerline.npy"))
        lcx = np.load(os.path.join(patient_folder, "lcx_centerline.npy"))
    except Exception as e:
        print(f"[ERROR] Failed to load centerline files from {patient_folder}: {e}")
        sys.exit(1)
        
    # Create the 3D plot
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot curves
    ax.plot(lmca[:, 0], lmca[:, 1], lmca[:, 2], color='black', linewidth=3.5, label='LMCA')
    ax.plot(lad[:, 0], lad[:, 1], lad[:, 2], color='red', linewidth=3.5, label='LAD')
    ax.plot(lcx[:, 0], lcx[:, 1], lcx[:, 2], color='blue', linewidth=3.5, label='LCX')
    
    # Highlight bifurcation
    ax.scatter([lmca[-1, 0]], [lmca[-1, 1]], [lmca[-1, 2]], color='purple', s=80, edgecolor='white', zorder=10, label='Bifurcation')
    
    ax.set_title(f"Interactive 3D View - Patient: {p['original_name']} ({p['folder_name']})\n(Click & drag to rotate, right-click & drag to zoom)", fontsize=11)
    ax.set_xlabel("X (LMCA alignment)")
    ax.set_ylabel("Y (Lateral)")
    ax.set_zlabel("Z (Bifurcation Normal)")
    
    # Set equal scale
    pts = np.concatenate([lmca, lad, lcx])
    max_range = np.array([pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min(), pts[:,2].max()-pts[:,2].min()]).max() / 2.0
    mid_x = (pts[:,0].max()+pts[:,0].min()) / 2.0
    mid_y = (pts[:,1].max()+pts[:,1].min()) / 2.0
    mid_z = (pts[:,2].max()+pts[:,2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.legend(fontsize=9, loc='upper left')
    plt.tight_layout()
    
    print("\n>>> Click and drag the Matplotlib window to rotate the 3D model! Close the window when finished.")
    plt.show()

if __name__ == "__main__":
    main()
