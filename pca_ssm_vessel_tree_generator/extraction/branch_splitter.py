"""Branch extraction and classification module for Batch 1."""

from __future__ import annotations

from typing import Any
import networkx as nx
import numpy as np

EPS = 1.0e-12
MIN_RCA_PATH_LENGTH_MM = 15.0
MAX_RCA_TORTUOSITY = 3.5


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= EPS or not np.isfinite(norm):
        raise ValueError("zero length vector")
    return vec / norm


def extract_path_coords(
    G: nx.Graph, path_nodes: list[int], branches_dict: dict[str, np.ndarray] | None = None
) -> np.ndarray:
    """Extract ordered 3D physical RAS coordinates for a graph node path."""
    if len(path_nodes) < 2:
        return np.empty((0, 3), dtype=float)

    if branches_dict is not None:
        full_coords = []
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            if G.has_edge(u, v):
                edge_data = G[u][v]
                path_id = edge_data.get("path_id")
                path_key = f"path_{path_id}" if path_id is not None else None
                if path_key and path_key in branches_dict:
                    coords = branches_dict[path_key].copy()
                    src_node = edge_data.get("node_id_src", u)
                    dst_node = edge_data.get("node_id_dst", v)
                    if u == dst_node and v == src_node:
                        coords = coords[::-1]
                    if len(full_coords) > 0:
                        coords = coords[1:]
                    full_coords.append(coords)
                    continue

            u_c = np.asarray(G.nodes[u]["coord"], dtype=float)
            v_c = np.asarray(G.nodes[v]["coord"], dtype=float)
            if len(full_coords) == 0:
                full_coords.append(u_c.reshape(1, 3))
            full_coords.append(v_c.reshape(1, 3))

        if full_coords:
            return np.vstack(full_coords)

    coords = [np.asarray(G.nodes[n]["coord"], dtype=float) for n in path_nodes]
    return np.vstack(coords)


def classify_lad_lcx(
    daughter_a_coords: np.ndarray, daughter_b_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """Classify the two bifurcation daughter paths into LAD (descending) and LCX (circumflex).

    Batch 1 modular classification: Uses multi-signal RAS anatomical rules
    (inferior Z descent, lateral X/Y extent). Prepared for Batch 2 coronary plane dot-product augmentation.

    Returns:
        lad_coords: 3D array for LAD
        lcx_coords: 3D array for LCX
        winner_role: "A_is_LAD" or "B_is_LAD"
    """
    if len(daughter_a_coords) < 2 or len(daughter_b_coords) < 2:
        return daughter_a_coords, daughter_b_coords, "A_is_LAD"

    z_min_a = float(np.min(daughter_a_coords[:, 2]))
    z_min_b = float(np.min(daughter_b_coords[:, 2]))

    z_disp_a = float(daughter_a_coords[-1, 2] - daughter_a_coords[0, 2])
    z_disp_b = float(daughter_b_coords[-1, 2] - daughter_b_coords[0, 2])

    len_a = float(np.sum(np.linalg.norm(np.diff(daughter_a_coords, axis=0), axis=1)))
    len_b = float(np.sum(np.linalg.norm(np.diff(daughter_b_coords, axis=0), axis=1)))

    score_a = -z_min_a - z_disp_a + 0.1 * len_a
    score_b = -z_min_b - z_disp_b + 0.1 * len_b

    if score_a >= score_b:
        return daughter_a_coords, daughter_b_coords, "A_is_LAD"
    else:
        return daughter_b_coords, daughter_a_coords, "B_is_LAD"


def extract_rca_main_path(
    G: nx.Graph,
    rca_ostium_node: int,
    branches_dict: dict[str, np.ndarray] | None = None,
    min_rca_length_mm: float = MIN_RCA_PATH_LENGTH_MM,
    max_tortuosity: float = MAX_RCA_TORTUOSITY,
) -> tuple[np.ndarray | None, np.ndarray | None, bool, str]:
    """Strengthened main RCA path extraction and topological validation.

    Evaluates candidate distal paths from rca_ostium_node using graph topology,
    length, tortuosity, and continuity criteria. Returns rca_resolved = False if no reliable RCA exists.
    """
    if rca_ostium_node not in G:
        return None, None, False, f"rca_ostium_node {rca_ostium_node} not in skeleton graph"

    component = nx.node_connected_component(G, rca_ostium_node)
    if len(component) < 3:
        return None, None, False, f"RCA component contains only {len(component)} nodes (disconnected stub)"

    lengths, paths = nx.single_source_dijkstra(G, source=rca_ostium_node, weight="weight")

    endpoints_in_comp = [n for n in component if G.degree(n) == 1 and n != rca_ostium_node]
    if not endpoints_in_comp:
        endpoints_in_comp = [n for n in component if n != rca_ostium_node]

    if not endpoints_in_comp:
        return None, None, False, "no distal endpoints found in RCA connected component"

    ostium_coord = np.asarray(G.nodes[rca_ostium_node]["coord"], dtype=float)

    candidate_paths_eval = []
    for ep in endpoints_in_comp:
        path_length = lengths[ep]
        ep_coord = np.asarray(G.nodes[ep]["coord"], dtype=float)
        euclidean_disp = float(np.linalg.norm(ep_coord - ostium_coord))
        tortuosity = path_length / euclidean_disp if euclidean_disp > EPS else 999.0

        valid_anatomical = (path_length >= min_rca_length_mm) and (tortuosity <= max_tortuosity)

        candidate_paths_eval.append({
            "endpoint_node": ep,
            "path_nodes": paths[ep],
            "path_length_mm": path_length,
            "euclidean_disp_mm": euclidean_disp,
            "tortuosity": tortuosity,
            "valid": valid_anatomical,
        })

    valid_candidates = [c for c in candidate_paths_eval if c["valid"]]

    if not valid_candidates:
        return None, None, False, f"No candidate RCA path met length (>={min_rca_length_mm}mm) and tortuosity (<={max_tortuosity}) criteria"

    # Select candidate path with maximum anatomically valid length
    valid_candidates.sort(key=lambda c: c["path_length_mm"], reverse=True)
    best_candidate = valid_candidates[0]

    rca_path_nodes = best_candidate["path_nodes"]
    rca_coords = extract_path_coords(G, rca_path_nodes, branches_dict)

    if len(rca_coords) < 2:
        return None, None, False, "extracted RCA coordinate array has fewer than 2 points"

    rca_endpoint = rca_coords[-1]
    return rca_coords, rca_endpoint, True, f"RCA main path successfully resolved (length={best_candidate['path_length_mm']:.1f}mm, tortuosity={best_candidate['tortuosity']:.2f})"


def extract_all_main_branches(
    G: nx.Graph,
    lca_ostium_node: int,
    rca_ostium_node: int,
    bifurcation_node: int,
    branches_dict: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Extract ordered 3D main paths for LMCA, LAD, LCX, and RCA."""
    # 1. LMCA path
    lmca_nodes = nx.shortest_path(G, source=lca_ostium_node, target=bifurcation_node, weight="weight")
    lmca_coords = extract_path_coords(G, lmca_nodes, branches_dict)

    # 2. Daughter branches from bifurcation
    neighbors = list(G.neighbors(bifurcation_node))
    lmca_neighbor = lmca_nodes[-2] if len(lmca_nodes) >= 2 else None
    daughter_neighbors = [n for n in neighbors if n != lmca_neighbor]

    def _get_furthest_path(start_nbr: int) -> tuple[float, list[int]]:
        sub_g = G.copy()
        if sub_g.has_node(bifurcation_node):
            sub_g.remove_node(bifurcation_node)
        if start_nbr not in sub_g:
            return 0.0, [start_nbr]
        lens, pths = nx.single_source_dijkstra(sub_g, source=start_nbr, weight="weight")
        furthest = max(lens, key=lens.get)
        return float(lens[furthest]), [bifurcation_node] + pths[furthest]

    daughter_evals = []
    for nbr in daughter_neighbors:
        d_len, d_path = _get_furthest_path(nbr)
        daughter_evals.append((d_len, d_path))

    daughter_evals.sort(key=lambda x: x[0], reverse=True)

    if len(daughter_evals) >= 2:
        path_a = daughter_evals[0][1]
        path_b = daughter_evals[1][1]
    elif len(daughter_evals) == 1:
        path_a = daughter_evals[0][1]
        path_b = [bifurcation_node]
    else:
        path_a = [bifurcation_node]
        path_b = [bifurcation_node]

    coords_a = extract_path_coords(G, path_a, branches_dict)
    coords_b = extract_path_coords(G, path_b, branches_dict)

    lad_coords, lcx_coords, winner_role = classify_lad_lcx(coords_a, coords_b)

    lad_endpoint = lad_coords[-1] if len(lad_coords) >= 1 else np.zeros(3)
    lcx_endpoint = lcx_coords[-1] if len(lcx_coords) >= 1 else np.zeros(3)

    # 3. RCA path extraction (strengthened)
    rca_coords, rca_endpoint, rca_resolved, rca_reason = extract_rca_main_path(
        G, rca_ostium_node, branches_dict
    )

    return {
        "LMCA": lmca_coords,
        "LAD": lad_coords,
        "LCX": lcx_coords,
        "RCA": rca_coords,
        "lad_endpoint": lad_endpoint,
        "lcx_endpoint": lcx_endpoint,
        "rca_endpoint": rca_endpoint,
        "rca_resolved": rca_resolved,
        "rca_unresolved_reason": rca_reason,
        "lad_lcx_classification": winner_role,
    }
