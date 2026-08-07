# lca_vessel_tree_generator/LCA_topology_generator/view_patient.py

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that shapes aren't distorted.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def get_valid_patients(trees_dir, report_path):
    tree_folders = sorted(glob(os.path.join(trees_dir, "patient_*")))
    
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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trees_dir = os.path.join(base_dir, "outputs", "dataset_lca", "trees")
    report_path = os.path.join(base_dir, "processed_dataset", "preprocessing_qc_report.json")
    
    valid_patients = get_valid_patients(trees_dir, report_path)
    
    if not valid_patients:
        print(f"[ERROR] No valid patient trees found in: {trees_dir}")
        print("Please run the preprocessor and generator first:\n")
        print("    python -m lca_vessel_tree_generator.LCA_topology_generator.generate_lca_dataset")
        sys.exit(1)
        
    print("Available Valid Patient Outputs (First Approach):")
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
    
    print("\nVisualization modes:")
    print("  [1] Centerlines only (Fast)")
    print("  [2] Centerlines + Tube Surface (Plot individual branch tubes)")
    print("  [3] Centerlines + Connected Tight Mesh (Plot fully merged tight mesh) [Default]")
    print("  [4] Render all elements together")
    try:
        mode_select = input("Select visualization mode (1-4) [Default: 3]: ").strip()
        mode = int(mode_select) if mode_select else 3
        if mode not in [1, 2, 3, 4]:
            raise ValueError()
    except (ValueError, KeyboardInterrupt):
        print("Invalid choice or exit. Defaulting to mode 3.")
        mode = 3

    print(f"\nLoading and visualizing: {p['original_name']} ({p['folder_name']}) in Mode {mode}")
    
    # Create the 3D plot
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    all_coords = []
    
    # 1. Load and plot centerlines
    try:
        lmca = np.load(os.path.join(patient_folder, "lmca_centerline.npy"))
        lad = np.load(os.path.join(patient_folder, "lad_centerline.npy"))
        lcx = np.load(os.path.join(patient_folder, "lcx_centerline.npy"))
        
        # Centerlines plotting
        ax.plot(lmca[:, 0], lmca[:, 1], lmca[:, 2], color='black', linewidth=3.5, label='LMCA Centerline', zorder=5)
        ax.plot(lad[:, 0], lad[:, 1], lad[:, 2], color='red', linewidth=3.5, label='LAD Centerline', zorder=5)
        ax.plot(lcx[:, 0], lcx[:, 1], lcx[:, 2], color='blue', linewidth=3.5, label='LCX Centerline', zorder=5)
        
        # Highlight bifurcation
        ax.scatter([lmca[-1, 0]], [lmca[-1, 1]], [lmca[-1, 2]], color='purple', s=80, edgecolor='white', zorder=10, label='Bifurcation')
        
        all_coords.append(lmca)
        all_coords.append(lad)
        all_coords.append(lcx)
    except Exception as e:
        print(f"[ERROR] Failed to load centerline files: {e}")
        sys.exit(1)

    # 2. Tube Surface
    if mode in [2, 4]:
        try:
            tube_path = os.path.join(patient_folder, "tree_tube_surface.npz")
            if os.path.exists(tube_path):
                tube_surfaces = np.load(tube_path)
                surface_colors = {"LMCA": "#555555", "LAD": "#ff7f0e", "LCX": "#1f77b4"}
                for branch_name in ["LMCA", "LAD", "LCX"]:
                    if branch_name in tube_surfaces:
                        surf = tube_surfaces[branch_name]
                        ax.plot_surface(
                            surf[:, :, 0],
                            surf[:, :, 1],
                            surf[:, :, 2],
                            color=surface_colors[branch_name],
                            alpha=0.6,
                            linewidth=0,
                            antialiased=True,
                            shade=True
                        )
                        all_coords.append(surf.reshape(-1, 3))
            else:
                print(f"[WARN] tree_tube_surface.npz not found in {patient_folder}")
        except Exception as e:
            print(f"[ERROR] Failed to plot tube surface: {e}")

    # 3. Tight Mesh
    if mode in [3, 4]:
        try:
            mesh_path = os.path.join(patient_folder, "tree_tight_mesh.npz")
            if os.path.exists(mesh_path):
                tight_mesh = np.load(mesh_path)
                vertices = tight_mesh["vertices"]
                faces = tight_mesh["faces"]
                
                # Limit faces for visualization performance if too large
                max_faces = 12000
                draw_faces = faces
                if len(faces) > max_faces:
                    step = int(np.ceil(len(faces) / max_faces))
                    draw_faces = faces[::step]
                
                mesh_coll = Poly3DCollection(
                    vertices[draw_faces],
                    facecolor="#2f6f9f",
                    edgecolor="#1d3342",
                    linewidth=0.03,
                    alpha=0.45 if mode == 4 else 0.75,
                )
                ax.add_collection3d(mesh_coll)
                all_coords.append(vertices)
            else:
                print(f"[WARN] tree_tight_mesh.npz not found in {patient_folder}")
        except Exception as e:
            print(f"[ERROR] Failed to plot tight mesh: {e}")

    # Configure axes
    ax.set_title(f"Interactive 3D View (First Approach) - Patient: {p['original_name']} ({p['folder_name']})\n(Click & drag to rotate, right-click & drag to zoom)", fontsize=11)
    ax.set_xlabel("X (LMCA alignment)")
    ax.set_ylabel("Y (Lateral)")
    ax.set_zlabel("Z (Bifurcation Normal)")
    
    # Calculate equal bounds
    if all_coords:
        pts = np.concatenate(all_coords, axis=0)
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