# LCA_topology_generator/statistics.py

import os
import numpy as np

def load_tree_statistics(dir_path: str):
    """
    Loads LCA_tree_mean.npy and LCA_tree_std.npy from the specified directory path.
    
    :param dir_path: Directory path where the npy files are stored.
    :return: A tuple of (tree_mean, tree_std) as numpy arrays.
    """
    mean_path = os.path.join(dir_path, "LCA_tree_mean.npy")
    std_path = os.path.join(dir_path, "LCA_tree_std.npy")
    
    if not os.path.exists(mean_path):
        raise FileNotFoundError(f"LCA mean statistics file not found at: {mean_path}")
    if not os.path.exists(std_path):
        raise FileNotFoundError(f"LCA standard deviation statistics file not found at: {std_path}")
        
    tree_mean = np.load(mean_path)
    tree_std = np.load(std_path)
    
    return tree_mean, tree_std
