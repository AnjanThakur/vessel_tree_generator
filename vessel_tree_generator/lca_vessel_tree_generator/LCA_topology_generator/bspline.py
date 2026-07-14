# LCA_topology_generator/bspline.py

import numpy as np
from geomdl import BSpline, utilities

def interpolate_branch(ctrl_pts: np.ndarray, sample_size: int) -> np.ndarray:
    """
    Performs cubic (degree-3) B-spline interpolation on a branch's control points
    following the exact logic from the RCA reference project.
    
    :param ctrl_pts: Control points of shape (P, 3).
    :param sample_size: Number of centerline points to interpolate.
    :return: Centerline coordinates of shape (sample_size, 3).
    """
    curve = BSpline.Curve()
    curve.degree = 3
    curve.ctrlpts = ctrl_pts.tolist()
    
    # Generate clamped uniform knot vector automatically based on degree and point count
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.delta = 0.01
    curve.sample_size = sample_size
    
    # Evaluate curve points and convert back to numpy array
    return np.array(curve.evalpts)

def interpolate_lca_tree(tree_ctrl_points: np.ndarray, lmca_points: int = 150, lad_points: int = 300, lcx_points: int = 250) -> dict:
    """
    Performs cubic B-spline interpolation on the three anatomical centerline paths
    extracted from the connected LCA control-point tree while preserving the shared LMCA bifurcation.
    
    LMCA control points: indices 0-4 (5 points)
    LAD control points: indices 5-16 (12 points)
    LCX control points: indices 17-26 (10 points)
    
    :param tree_ctrl_points: Connected LCA control point tree of shape (27, 3).
    :param lmca_points: Centerline density for the LMCA branch. Default is 150.
    :param lad_points: Centerline density for the LAD branch. Default is 300.
    :param lcx_points: Centerline density for the LCX branch. Default is 250.
    :return: Dictionary containing the three centerline paths: "LMCA", "LAD", "LCX".
    """
    # Extract control points for each branch
    lmca_ctrl = tree_ctrl_points[0:5]    # Indices 0 to 4
    lad_ctrl = tree_ctrl_points[5:17]   # Indices 5 to 16
    lcx_ctrl = tree_ctrl_points[17:27]  # Indices 17 to 26
    
    # Interpolate each branch using cubic B-splines
    lmca_centerline = interpolate_branch(lmca_ctrl, lmca_points)
    lad_centerline = interpolate_branch(lad_ctrl, lad_points)
    lcx_centerline = interpolate_branch(lcx_ctrl, lcx_points)
    
    return {
        "LMCA": lmca_centerline,
        "LAD": lad_centerline,
        "LCX": lcx_centerline
    }
