"""Skeletonization and NetworkX graph construction for Batch 1."""

from __future__ import annotations

from typing import Any
import networkx as nx
import numpy as np
import pandas as pd
from scipy.ndimage import label


def build_graph_from_summary(
    summary_df: pd.DataFrame, branches_dict: dict[str, np.ndarray]
) -> tuple[nx.Graph, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a weighted NetworkX Graph from pre-extracted summary CSV and branches NPZ.

    Extracts node coordinates directly from summary CSV columns ('coord-src-0', 'coord-dst-0')
    or from branches_dict arrays.

    Returns:
        G: networkx.Graph with Euclidean edge weights and 3D node coordinates
        node_coords: (N, 3) physical RAS coordinates for all graph nodes
        endpoint_nodes: 1D array of degree-1 node indices
        endpoint_coords: (K, 3) physical RAS coordinates for degree-1 nodes
        junction_nodes: 1D array of degree-3+ node indices
    """
    G = nx.Graph()
    all_nodes = set()
    node_positions = {}

    src_col = "node-id-src" if "node-id-src" in summary_df.columns else "node_id_src"
    dst_col = "node-id-dst" if "node-id-dst" in summary_df.columns else "node_id_dst"
    dist_col = "branch-distance" if "branch-distance" in summary_df.columns else "branch_distance"

    # Check for direct coordinate columns in summary_df
    has_direct_coords = ("coord-src-0" in summary_df.columns or "coord_src_0" in summary_df.columns)

    for idx, row in summary_df.iterrows():
        u = int(row[src_col])
        v = int(row[dst_col])
        w = float(row[dist_col])
        path_id = idx
        path_key = f"path_{path_id}"

        coords = branches_dict.get(path_key)
        if coords is not None and len(coords) >= 2:
            node_positions[u] = coords[0]
            node_positions[v] = coords[-1]
        elif has_direct_coords:
            c_src_x = "coord-src-0" if "coord-src-0" in summary_df.columns else "coord_src_0"
            c_src_y = "coord-src-1" if "coord-src-1" in summary_df.columns else "coord_src_1"
            c_src_z = "coord-src-2" if "coord-src-2" in summary_df.columns else "coord_src_2"

            c_dst_x = "coord-dst-0" if "coord-dst-0" in summary_df.columns else "coord_dst_0"
            c_dst_y = "coord-dst-1" if "coord-dst-1" in summary_df.columns else "coord_dst_1"
            c_dst_z = "coord-dst-2" if "coord-dst-2" in summary_df.columns else "coord_dst_2"

            node_positions[u] = np.array([row[c_src_x], row[c_src_y], row[c_src_z]], dtype=float)
            node_positions[v] = np.array([row[c_dst_x], row[c_dst_y], row[c_dst_z]], dtype=float)

        all_nodes.add(u)
        all_nodes.add(v)

        if G.has_edge(u, v):
            if w < G[u][v]["weight"]:
                G[u][v]["weight"] = w
                G[u][v]["path_id"] = path_id
                G[u][v]["node_id_src"] = u
                G[u][v]["node_id_dst"] = v
        else:
            G.add_edge(u, v, weight=w, path_id=path_id, node_id_src=u, node_id_dst=v)

    # Populate node attributes for ALL graph nodes
    max_node_id = max(all_nodes) if all_nodes else 0
    node_coords = np.zeros((max_node_id + 1, 3), dtype=float)
    for node_id in all_nodes:
        pos = node_positions.get(node_id, np.zeros(3, dtype=float))
        node_coords[node_id] = pos
        G.nodes[node_id]["coord"] = pos

    degrees = dict(G.degree())
    endpoint_nodes = np.array([node for node, deg in degrees.items() if deg == 1], dtype=int)
    junction_nodes = np.array([node for node, deg in degrees.items() if deg >= 3], dtype=int)

    endpoint_coords = node_coords[endpoint_nodes] if len(endpoint_nodes) > 0 else np.empty((0, 3))

    return G, node_coords, endpoint_nodes, endpoint_coords, junction_nodes


def build_graph_from_mask(
    mask: np.ndarray, affine: np.ndarray, spacing_mm: np.ndarray
) -> tuple[nx.Graph, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Skeletonize 3D binary label mask using Lee method and build NetworkX Graph via skan."""
    from skimage.morphology import skeletonize
    from skan import Skeleton

    # Keep largest connected component
    labeled, num_features = label(mask)
    if num_features > 1:
        sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
        largest_label = int(np.argmax(sizes) + 1)
        mask = (labeled == largest_label)

    # Skeletonize using Lee method (reusing existing codebase convention)
    skel = skeletonize(mask, method="lee")
    sk = Skeleton(skel.astype(np.uint8), spacing=spacing_mm)

    node_coords_voxel = sk.coordinates  # (N, 3) voxel coords
    # Convert voxel to physical scanner RAS via affine transform
    homog = np.column_stack([node_coords_voxel, np.ones(len(node_coords_voxel))])
    node_coords = (homog @ affine.T)[:, :3]

    degrees = sk.degrees
    endpoint_nodes = np.where(degrees == 1)[0]
    junction_nodes = np.where(degrees >= 3)[0]
    endpoint_coords = node_coords[endpoint_nodes]

    # Build NetworkX graph from skeleton paths
    G = nx.Graph()
    for i in range(sk.n_paths):
        path_nodes = sk.path_nodes(i)
        for u, v in zip(path_nodes[:-1], path_nodes[1:]):
            u_coord = node_coords[u]
            v_coord = node_coords[v]
            dist = float(np.linalg.norm(u_coord - v_coord))
            G.add_edge(u, v, weight=dist)

    for n in G.nodes():
        G.nodes[n]["coord"] = node_coords[n]

    return G, node_coords, endpoint_nodes, endpoint_coords, junction_nodes
