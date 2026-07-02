# LCA_topology_generator/visualize.py

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

def _plot_lca_tree_points(ax, points: np.ndarray, alpha=0.8, marker='o', linestyle='-'):
    """
    Helper function to plot a 27-point LCA control point tree.
    """
    lmca = points[0:5]
    lad = points[5:17]
    lcx = points[17:27]
    
    ax.plot(lmca[:, 0], lmca[:, 1], lmca[:, 2], color='black', alpha=alpha, linestyle=linestyle, label='LMCA')
    ax.plot(lad[:, 0], lad[:, 1], lad[:, 2], color='red', alpha=alpha, linestyle=linestyle, label='LAD')
    ax.plot(lcx[:, 0], lcx[:, 1], lcx[:, 2], color='blue', alpha=alpha, linestyle=linestyle, label='LCX')
    
    if marker:
        ax.scatter(lmca[:, 0], lmca[:, 1], lmca[:, 2], color='black', alpha=alpha, marker=marker)
        ax.scatter(lad[:, 0], lad[:, 1], lad[:, 2], color='red', alpha=alpha, marker=marker)
        ax.scatter(lcx[:, 0], lcx[:, 1], lcx[:, 2], color='blue', alpha=alpha, marker=marker)

def plot_original_control_points(original_points: np.ndarray, output_path: str):
    """
    Plots preprocessed patient control points.
    original_points can be shape (M, 27, 3) or (27, 3).
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    
    if original_points.ndim == 3:
        for idx in range(len(original_points)):
            _plot_lca_tree_points(ax, original_points[idx], alpha=0.4, marker=None)
        # Highlight first one
        _plot_lca_tree_points(ax, original_points[0], alpha=0.9, marker='o')
    else:
        _plot_lca_tree_points(ax, original_points, alpha=0.9, marker='o')
        
    ax.set_title("Original Patient LCA Control Points")
    ax.set_xlabel("X (LMCA Direction)")
    ax.set_ylabel("Y (Lateral)")
    ax.set_zlabel("Z (Bifurcation Normal)")
    set_axes_equal(ax)
    
    # Remove duplicates from legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_sampled_control_points(sampled_points: np.ndarray, output_path: str):
    """
    Plots synthetic sampled control points.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    
    _plot_lca_tree_points(ax, sampled_points, alpha=0.9, marker='o')
    
    ax.set_title("Sampled Synthetic LCA Control Points")
    ax.set_xlabel("X (LMCA Direction)")
    ax.set_ylabel("Y (Lateral)")
    ax.set_zlabel("Z (Bifurcation Normal)")
    set_axes_equal(ax)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_augmented_control_points(augmented_points: np.ndarray, output_path: str):
    """
    Plots control points after shearing/warping.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    
    _plot_lca_tree_points(ax, augmented_points, alpha=0.9, marker='o')
    
    ax.set_title("Augmented Synthetic LCA Control Points")
    ax.set_xlabel("X (LMCA Direction)")
    ax.set_ylabel("Y (Lateral)")
    ax.set_zlabel("Z (Bifurcation Normal)")
    set_axes_equal(ax)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_bspline_centerlines(centerlines: dict, output_path: str):
    """
    Plots interpolated B-spline branches.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(projection='3d')
    
    lmca = centerlines["LMCA"]
    lad = centerlines["LAD"]
    lcx = centerlines["LCX"]
    
    ax.plot(lmca[:, 0], lmca[:, 1], lmca[:, 2], color='black', linewidth=2, label='LMCA Centerline')
    ax.plot(lad[:, 0], lad[:, 1], lad[:, 2], color='red', linewidth=2, label='LAD Centerline')
    ax.plot(lcx[:, 0], lcx[:, 1], lcx[:, 2], color='blue', linewidth=2, label='LCX Centerline')
    
    ax.set_title("Interpolated Cubic B-Spline LCA Centerlines")
    ax.set_xlabel("X (LMCA Direction)")
    ax.set_ylabel("Y (Lateral)")
    ax.set_zlabel("Z (Bifurcation Normal)")
    set_axes_equal(ax)
    ax.legend()
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def plot_debug_comparison(original_pts: np.ndarray, sampled_pts: np.ndarray, augmented_pts: np.ndarray, centerlines: dict, output_path: str):
    """
    Renders an overlay or grid comparison of all stages for detailed visualization and debugging.
    Creates a 2x2 grid of subplots.
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Subplot 1: Patient CPs
    ax1 = fig.add_subplot(221, projection='3d')
    if original_pts.ndim == 3:
        for idx in range(len(original_pts)):
            _plot_lca_tree_points(ax1, original_pts[idx], alpha=0.3, marker=None)
        _plot_lca_tree_points(ax1, original_pts[0], alpha=0.9, marker='o')
    else:
        _plot_lca_tree_points(ax1, original_pts, alpha=0.9, marker='o')
    ax1.set_title("1. Original Patient CPs")
    set_axes_equal(ax1)
    
    # Subplot 2: Sampled CPs
    ax2 = fig.add_subplot(222, projection='3d')
    _plot_lca_tree_points(ax2, sampled_pts, alpha=0.9, marker='o')
    ax2.set_title("2. Sampled Synthetic CPs")
    set_axes_equal(ax2)
    
    # Subplot 3: Augmented CPs
    ax3 = fig.add_subplot(223, projection='3d')
    _plot_lca_tree_points(ax3, augmented_pts, alpha=0.9, marker='o')
    ax3.set_title("3. Augmented (Shear + Warp) CPs")
    set_axes_equal(ax3)
    
    # Subplot 4: Centerlines Overlay
    ax4 = fig.add_subplot(224, projection='3d')
    # Draw CPs as semi-transparent dots
    _plot_lca_tree_points(ax4, augmented_pts, alpha=0.3, marker='o', linestyle=':')
    # Draw dense centerlines as thick curves
    lmca = centerlines["LMCA"]
    lad = centerlines["LAD"]
    lcx = centerlines["LCX"]
    ax4.plot(lmca[:, 0], lmca[:, 1], lmca[:, 2], color='black', linewidth=2.5, label='LMCA Spline')
    ax4.plot(lad[:, 0], lad[:, 1], lad[:, 2], color='red', linewidth=2.5, label='LAD Spline')
    ax4.plot(lcx[:, 0], lcx[:, 1], lcx[:, 2], color='blue', linewidth=2.5, label='LCX Spline')
    ax4.set_title("4. Final Cubic B-Spline Centerlines Overlay")
    set_axes_equal(ax4)
    
    # Clean legends
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left')
        
    plt.suptitle("LCA Synthetic Generation Debug Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close()
