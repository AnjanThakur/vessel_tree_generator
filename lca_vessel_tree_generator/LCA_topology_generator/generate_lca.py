# LCA_topology_generator/generate_lca.py

import argparse
import os
import numpy as np

from .statistics import load_tree_statistics
from .control_point_sampler import sample_lca_control_points
from .augmentation import apply_tree_shear, apply_tree_warp
from .bspline import interpolate_lca_tree
from .visualize import (
    plot_original_control_points,
    plot_sampled_control_points,
    plot_augmented_control_points,
    plot_bspline_centerlines,
    plot_debug_comparison
)

def main():
    parser = argparse.ArgumentParser(description="Synthetic Left Coronary Artery (LCA) Centerline Generator")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator")
    parser.add_argument("--shear", action=argparse.BooleanOptionalAction, default=True, help="Apply random shear augmentation")
    parser.add_argument("--warp", action=argparse.BooleanOptionalAction, default=True, help="Apply random warp augmentation")
    
    # Configurable B-spline sampling densities
    parser.add_argument("--lmca-points", type=int, default=150, help="Sampling density for LMCA centerline")
    parser.add_argument("--lad-points", type=int, default=300, help="Sampling density for LAD centerline")
    parser.add_argument("--lcx-points", type=int, default=250, help="Sampling density for LCX centerline")
    
    parser.add_argument("--stats-dir", type=str, default="LCA_branch_control_points/generated", help="Directory containing statistical npy files")
    parser.add_argument("--output", type=str, default="LCA_branch_control_points/generated", help="Directory to save output files and plots")
    
    args = parser.parse_args()
    
    rng = np.random.default_rng(args.seed)
    
    # 1. Load Statistical Model
    print(f"Loading tree statistics from: {args.stats_dir}")
    mean, std = load_tree_statistics(args.stats_dir)
    
    # 2. Generate Synthetic Control Points
    print("Sampling synthetic control points with bifurcation snapping...")
    sampled_tree = sample_lca_control_points(mean, std, rng)
    
    # 3. Apply Optional Augmentations
    augmented_tree = sampled_tree.copy()
    if args.shear:
        print("Applying random shear augmentation (strength=0.12)...")
        augmented_tree = apply_tree_shear(augmented_tree, strength=0.12, rng=rng)
    if args.warp:
        print("Applying random warp augmentation (strength=0.10)...")
        augmented_tree = apply_tree_warp(augmented_tree, strength=0.10, rng=rng)
        
    # 4. Cubic B-Spline Interpolation
    print("Performing B-spline interpolation on branches...")
    centerlines = interpolate_lca_tree(
        augmented_tree, 
        lmca_points=args.lmca_points, 
        lad_points=args.lad_points, 
        lcx_points=args.lcx_points
    )
    
    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)
    
    # 5. Save Outputs (Intermediate files for future debugging and generation)
    print(f"Saving generated files to: {args.output}")
    np.save(os.path.join(args.output, "sampled_tree.npy"), sampled_tree)
    np.save(os.path.join(args.output, "augmented_tree.npy"), augmented_tree)
    np.save(os.path.join(args.output, "lmca_centerline.npy"), centerlines["LMCA"])
    np.save(os.path.join(args.output, "lad_centerline.npy"), centerlines["LAD"])
    np.save(os.path.join(args.output, "lcx_centerline.npy"), centerlines["LCX"])
    
    # 6. Visualization Plots
    print("Generating visualization plots...")
    
    # Load original patient dataset for comparison (if available)
    orig_path = os.path.join(args.stats_dir, "LCA_tree_ctrl_points.npy")
    if os.path.exists(orig_path):
        original_pts = np.load(orig_path)
    else:
        original_pts = mean # Fallback to mean if patient data not present
        
    plot_original_control_points(original_pts, os.path.join(args.output, "original_control_points.png"))
    plot_sampled_control_points(sampled_tree, os.path.join(args.output, "sampled_control_points.png"))
    plot_augmented_control_points(augmented_tree, os.path.join(args.output, "augmented_control_points.png"))
    plot_bspline_centerlines(centerlines, os.path.join(args.output, "bspline_centerlines.png"))
    
    # Comparison overlay/grid plot
    plot_debug_comparison(
        original_pts, 
        sampled_tree, 
        augmented_tree, 
        centerlines, 
        os.path.join(args.output, "debug_comparison.png")
    )
    
    print("\nLCA Generation Pipeline completed successfully.")
    print(f"Check output plots and files in: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
