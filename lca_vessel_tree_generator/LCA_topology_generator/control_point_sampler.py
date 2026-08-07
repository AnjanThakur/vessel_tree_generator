# LCA_topology_generator/control_point_sampler.py

import numpy as np

def sample_lca_control_points(mean: np.ndarray, std: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Samples a (27, 3) array of control points using the uniform sampling range:
    rng.uniform(mean - 1.5 * std, mean + std + 1.5 * std)
    
    Then snaps index 5 (LAD start) and index 17 (LCX start) to index 4 (LMCA terminal) 
    to guarantee they share the exact same bifurcation node coordinates.
    
    :param mean: Mean control point positions of shape (27, 3).
    :param std: Standard deviation of control point positions of shape (27, 3).
    :param rng: A numpy random generator instance.
    :return: A sampled control points array of shape (27, 3).
    """
    # Sample coordinate-wise independently using the RCA-matching range formula:
    # low = mean - 1.5 * std
    # high = mean + 2.5 * std (which is mean + std + 1.5 * std)
    low = mean - 1.5 * std
    high = mean + 2.5 * std
    
    sampled_points = rng.uniform(low, high)
    
    # --- Topological Snapping ---
    # The LMCA terminal control point (index 4) represents the bifurcation node. 
    # The LAD and LCX proximal control points are initialized to the same coordinates 
    # as the LMCA terminal control point, ensuring that all three branches share a 
    # common bifurcation before B-spline interpolation.
    sampled_points[5] = sampled_points[4].copy()
    sampled_points[17] = sampled_points[4].copy()
    
    return sampled_points
