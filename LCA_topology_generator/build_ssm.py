# LCA_topology_generator/build_ssm.py

import os
import numpy as np

def build_ssm(ctrl_points_dir, output_path):
    print("Building Statistical Shape Model (SSM) for Unified LCA Tree...")
    
    # Load collective control points
    lmca_pts = np.load(os.path.join(ctrl_points_dir, "LMCA_patient_ctrl_points.npy"))
    lad_pts = np.load(os.path.join(ctrl_points_dir, "LAD_patient_ctrl_points.npy"))
    lcx_pts = np.load(os.path.join(ctrl_points_dir, "LCX_patient_ctrl_points.npy"))
    
    print(f"Loaded LMCA control points: {lmca_pts.shape}")
    print(f"Loaded LAD control points: {lad_pts.shape}")
    print(f"Loaded LCX control points: {lcx_pts.shape}")
    
    # Concatenate branches along axis 1 (points axis): 5 + 12 + 10 = 27 points
    # Shape: (M, 27, 3)
    tree_pts = np.concatenate([lmca_pts, lad_pts, lcx_pts], axis=1)
    M, N_pts, D = tree_pts.shape
    print(f"Combined LCA Tree shape: {tree_pts.shape}")
    
    # Flatten control points for each patient
    # Shape: (M, 81)
    X = tree_pts.reshape(M, N_pts * D)
    
    # Calculate mean shape of the unified tree
    mean_shape = np.mean(X, axis=0)
    X_centered = X - mean_shape
    
    # SVD of centered data
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Eigenvalues (variance) and standard deviation of each component
    eigenvalues = (S ** 2) / (max(1, M - 1))
    std_devs = np.sqrt(eigenvalues)
    
    # Store in model data
    model_data = {
        "LCA_mean": mean_shape,
        "LCA_eigenvectors": Vt,  # rows are eigenvectors
        "LCA_eigenvalues": eigenvalues,
        "LCA_std_devs": std_devs,
    }
    
    # Print variance explained
    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_var if total_var > 0 else np.zeros_like(eigenvalues)
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("\nSSM PCA Analysis (LCA Tree):")
    print(f"  Mean shape: {mean_shape.reshape(N_pts, D).shape}")
    print(f"  Eigenvectors: {Vt.shape}")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Explained Variance Ratio: {explained_variance_ratio}")
    print(f"  Cumulative Variance: {cumulative_variance}")
    
    np.savez(output_path, **model_data)
    print(f"\nUnified SSM prior model saved successfully to: {output_path}")


def main():
    ctrl_points_dir = r"D:\Projects\lca_vessel\latest_lca\vessel_tree_generator\LCA_branch_control_points\generated"
    output_path = os.path.join(ctrl_points_dir, "lca_ssm_prior.npz")
    build_ssm(ctrl_points_dir, output_path)

if __name__ == "__main__":
    main()
