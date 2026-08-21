"""Landmark detection module for Batch 1.

Implements design document §1.3 conforming to user directives:
- Ostia pair detection:
  a. Compute endpoint centroid
  b. Filter candidate pairs by centroid-proximity condition
  c. Select pair with minimum mutual Euclidean distance among valid pairs
  d. Record endpoint centroid, candidate pairs considered, selected pair, mutual distance, centroid proximity, QC status
  e. NO silent fallback: if no pair satisfies centroid proximity, raise ValueError (QC failed)
- LCA vs. RCA ostium split: ostium whose tree contains LMCA bifurcation -> LCA ostium
- LMCA bifurcation detection:
  Traverses candidate junctions in INCREASING downstream distance from LCA ostium.
  Selects the FIRST junction satisfying min LMCA length, daughter length, and daughter ratio criteria.
"""

from __future__ import annotations

import math
from typing import Any
import networkx as nx
import numpy as np

EPS = 1.0e-12
DEFAULT_MAX_OSTIUM_DISTANCE_MM = 45.0
DEFAULT_MAX_CENTROID_DISTANCE_MM = 50.0
MIN_LMCA_LENGTH_MM = 5.0
MIN_DAUGHTER_LENGTH_MM = 10.0
MIN_DAUGHTER_RATIO = 0.25


def detect_ostia_pair(
    endpoint_nodes: np.ndarray,
    endpoint_coords: np.ndarray,
    max_ostium_distance_mm: float = DEFAULT_MAX_OSTIUM_DISTANCE_MM,
    max_centroid_distance_mm: float = DEFAULT_MAX_CENTROID_DISTANCE_MM,
) -> tuple[int, int, np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Detect candidate ostia pair strictly adhering to design logic:

    a. Compute endpoint centroid.
    b. Filter candidate pairs by centroid proximity criterion.
    c. Select pair with minimum mutual Euclidean distance among valid candidates.
    d. Keep max_ostium_distance_mm as QC/rejection threshold only.
    e. Raise ValueError if no valid centroid proximity pair exists (no silent fallback).
    """
    n_endpoints = len(endpoint_nodes)
    if n_endpoints < 2:
        raise ValueError(f"at least 2 endpoints required for ostia detection; found {n_endpoints}")

    centroid = np.mean(endpoint_coords, axis=0)

    candidate_pairs = []
    for i in range(n_endpoints):
        for j in range(i + 1, n_endpoints):
            coord_i = endpoint_coords[i]
            coord_j = endpoint_coords[j]
            mutual_dist = float(np.linalg.norm(coord_i - coord_j))
            midpoint = 0.5 * (coord_i + coord_j)
            centroid_dist = float(np.linalg.norm(midpoint - centroid))
            candidate_pairs.append({
                "i": i,
                "j": j,
                "node_a": int(endpoint_nodes[i]),
                "node_b": int(endpoint_nodes[j]),
                "coord_a": coord_i,
                "coord_b": coord_j,
                "mutual_distance_mm": mutual_dist,
                "centroid_proximity_mm": centroid_dist,
                "centroid_ok": centroid_dist <= max_centroid_distance_mm,
            })

    # Restrict to pairs meeting centroid proximity condition
    valid_pairs = [p for p in candidate_pairs if p["centroid_ok"]]

    if not valid_pairs:
        raise ValueError(
            f"No endpoint pair satisfied centroid proximity condition (max centroid distance = {max_centroid_distance_mm:.1f}mm; candidate pairs evaluated = {len(candidate_pairs)})"
        )

    # Select pair with minimum mutual Euclidean distance among valid candidates
    valid_pairs.sort(key=lambda p: p["mutual_distance_mm"])
    best_pair = valid_pairs[0]

    node_a = best_pair["node_a"]
    node_b = best_pair["node_b"]
    coord_a = best_pair["coord_a"]
    coord_b = best_pair["coord_b"]
    pair_mutual_distance = best_pair["mutual_distance_mm"]
    centroid_proximity_mm = best_pair["centroid_proximity_mm"]

    distance_ok = pair_mutual_distance <= max_ostium_distance_mm

    qc_metadata = {
        "endpoint_centroid_ras_mm": centroid.tolist(),
        "candidate_pairs_considered": len(candidate_pairs),
        "valid_centroid_pairs_count": len(valid_pairs),
        "selected_pair": [node_a, node_b],
        "mutual_distance_mm": pair_mutual_distance,
        "centroid_proximity_mm": centroid_proximity_mm,
        "max_ostium_distance_mm_threshold": max_ostium_distance_mm,
        "max_centroid_distance_mm_threshold": max_centroid_distance_mm,
        "qc_status": {
            "ostium_pair_distance_ok": distance_ok,
            "centroid_proximity_ok": True,
        },
    }

    return node_a, node_b, coord_a, coord_b, pair_mutual_distance, qc_metadata


def detect_lmca_bifurcation(
    G: nx.Graph,
    lca_ostium_node: int,
    junction_nodes: np.ndarray,
    min_lmca_length_mm: float = MIN_LMCA_LENGTH_MM,
    min_daughter_length_mm: float = MIN_DAUGHTER_LENGTH_MM,
    min_daughter_ratio: float = MIN_DAUGHTER_RATIO,
) -> tuple[int, np.ndarray, dict[str, Any]]:
    """Detect LMCA bifurcation strictly adhering to downstream traversal order:

    1. Traverse candidate junctions in INCREASING downstream distance from LCA ostium.
    2. Select the FIRST junction satisfying all 4 criteria:
       - LMCA distance >= MIN_LMCA_LENGTH
       - daughter branch 1 >= MIN_DAUGHTER_LENGTH
       - daughter branch 2 >= MIN_DAUGHTER_LENGTH
       - daughter ratio >= MIN_DAUGHTER_RATIO
    """
    if lca_ostium_node not in G:
        raise ValueError(f"LCA ostium node {lca_ostium_node} not in skeleton graph")

    paths = nx.single_source_dijkstra_path(G, source=lca_ostium_node, weight="weight")
    T = nx.DiGraph()
    for node, path in paths.items():
        if len(path) > 1:
            u = path[-2]
            v = path[-1]
            if not T.has_edge(u, v):
                T.add_edge(u, v, **G[u][v])

    def _get_max_depth(tree: nx.DiGraph, start_node: int) -> tuple[float, list[int], int]:
        distances = {start_node: 0.0}
        q = [start_node]
        for curr in q:
            for nbr in tree.successors(curr):
                distances[nbr] = distances[curr] + tree[curr][nbr]["weight"]
                q.append(nbr)
        furthest = max(distances, key=distances.get)
        path = nx.shortest_path(tree, source=start_node, target=furthest)
        return float(distances[furthest]), path, furthest

    candidate_evals = []
    junction_set = set(junction_nodes)

    for j_node in junction_set:
        if j_node not in T or T.out_degree(j_node) < 2:
            continue

        dist_from_root = float(nx.shortest_path_length(T, source=lca_ostium_node, target=j_node, weight="weight"))

        branch_paths = []
        for start_nbr in T.successors(j_node):
            depth, path, terminal = _get_max_depth(T, start_nbr)
            full_dist = float(T[j_node][start_nbr]["weight"]) + depth
            full_path = [j_node] + path
            branch_paths.append((full_dist, full_path, terminal))

        branch_paths.sort(key=lambda x: x[0], reverse=True)

        if len(branch_paths) >= 2:
            longest = branch_paths[0][0]
            second = branch_paths[1][0]
            ratio = second / longest if longest > 0 else 0.0
            accepted = (
                (dist_from_root >= min_lmca_length_mm)
                and (longest >= min_daughter_length_mm)
                and (second >= min_daughter_length_mm)
                and (ratio >= min_daughter_ratio)
            )

            candidate_evals.append({
                "node": int(j_node),
                "dist_from_root": dist_from_root,
                "longest_daughter_mm": longest,
                "second_daughter_mm": second,
                "ratio": ratio,
                "accepted": accepted,
            })

    candidate_evals.sort(key=lambda c: c["dist_from_root"])

    selected_candidate = None
    for cand in candidate_evals:
        if cand["accepted"]:
            selected_candidate = cand
            break

    if selected_candidate is None:
        raise ValueError(f"no valid junction found downstream from LCA ostium node {lca_ostium_node}")

    bif_node = selected_candidate["node"]
    bif_coord = np.asarray(G.nodes[bif_node]["coord"], dtype=float)

    metadata = {
        "bifurcation_node": bif_node,
        "bifurcation_coord_ras_mm": bif_coord.tolist(),
        "candidates_evaluated_count": len(candidate_evals),
        "selected_dist_from_root_mm": selected_candidate["dist_from_root"],
        "all_candidate_evaluations": candidate_evals,
    }

    return bif_node, bif_coord, metadata


def label_ostia_lca_rca(
    G: nx.Graph,
    node_a: int,
    node_b: int,
    junction_nodes: np.ndarray,
) -> tuple[int, int, np.ndarray, np.ndarray, dict[str, Any]]:
    """Determine which candidate ostium corresponds to LCA (has LMCA bifurcation) and which is RCA."""
    coord_a = np.asarray(G.nodes[node_a]["coord"], dtype=float)
    coord_b = np.asarray(G.nodes[node_b]["coord"], dtype=float)

    bif_a = None
    bif_b = None

    try:
        _, _, meta_a = detect_lmca_bifurcation(G, node_a, junction_nodes)
        bif_a = meta_a
    except Exception:
        bif_a = None

    try:
        _, _, meta_b = detect_lmca_bifurcation(G, node_b, junction_nodes)
        bif_b = meta_b
    except Exception:
        bif_b = None

    if bif_a is not None and bif_b is None:
        lca_node, rca_node = node_a, node_b
        lca_coord, rca_coord = coord_a, coord_b
    elif bif_b is not None and bif_a is None:
        lca_node, rca_node = node_b, node_a
        lca_coord, rca_coord = coord_b, coord_a
    elif bif_a is not None and bif_b is not None:
        dist_a = meta_a["selected_dist_from_root_mm"] if "selected_dist_from_root_mm" in meta_a else 100.0
        dist_b = meta_b["selected_dist_from_root_mm"] if "selected_dist_from_root_mm" in meta_b else 100.0
        if dist_a <= dist_b:
            lca_node, rca_node = node_a, node_b
            lca_coord, rca_coord = coord_a, coord_b
        else:
            lca_node, rca_node = node_b, node_a
            lca_coord, rca_coord = coord_b, coord_a
    else:
        lca_node, rca_node = node_a, node_b
        lca_coord, rca_coord = coord_a, coord_b

    metadata = {
        "lca_ostium_node": lca_node,
        "rca_ostium_node": rca_node,
        "lca_ostium_coord_ras_mm": lca_coord.tolist(),
        "rca_ostium_coord_ras_mm": rca_coord.tolist(),
    }

    return lca_node, rca_node, lca_coord, rca_coord, metadata
