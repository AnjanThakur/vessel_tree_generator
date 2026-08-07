#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import networkx as nx
import napari
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# CONFIGURATION
# ==========================================
MINIMUM_LENGTH = 10.0
MINIMUM_RATIO = 0.25
MIN_LMCA_LENGTH = 5.0
# ==========================================

# 1. Load Data
print("Loading data...")
landmarks = np.load(BASE_DIR / "landmarks.npy", allow_pickle=True).item()
summary = pd.read_csv(BASE_DIR / "summary.csv")
branches = np.load(BASE_DIR / "branches.npz")

root_node = landmarks["candidate_root_node"]
junction_nodes = landmarks["junction_nodes"]

# 2. Build NetworkX Graph
print("Building graph...")
G = nx.Graph()
for idx, row in summary.iterrows():
    u = int(row['node_id_src'])
    v = int(row['node_id_dst'])
    w = row['branch_distance']
    path_id = idx

    if G.has_edge(u, v):
        if w < G[u][v]['weight']:
            G[u][v]['weight'] = w
            G[u][v]['path_id'] = path_id
            G[u][v]['node_id_src'] = u
            G[u][v]['node_id_dst'] = v
    else:
        G.add_edge(u, v, weight=w, path_id=path_id, node_id_src=u, node_id_dst=v)

# 3. Build Directed Shortest Path Tree (SPT) to enforce downstream traversal
print("Building shortest path tree to prevent cycle leakage...")
T = nx.DiGraph()
paths = nx.single_source_dijkstra_path(G, source=root_node, weight='weight')
for node, path in paths.items():
    if len(path) > 1:
        u = path[-2]
        v = path[-1]
        if not T.has_edge(u, v):
            T.add_edge(u, v, **G[u][v])

def get_max_depth_in_tree(tree, start_node):
    distances = {start_node: 0}
    q = [start_node]
    for curr in q:
        for nbr in tree.successors(curr):
            distances[nbr] = distances[curr] + tree[curr][nbr]['weight']
            q.append(nbr)
    furthest = max(distances, key=distances.get)
    # The path is guaranteed unique in a directed tree
    path = nx.shortest_path(tree, source=start_node, target=furthest)
    return distances[furthest], path, furthest

# 4. Evaluate EVERY Junction in the SPT
print("Evaluating candidate junctions...")
candidate_evals = []

for j_node in junction_nodes:
    if j_node not in T or T.out_degree(j_node) < 2:
        continue

    dist_from_root = nx.shortest_path_length(T, source=root_node, target=j_node, weight='weight')

    branch_paths = []
    for start_nbr in T.successors(j_node):
        dist, path, terminal = get_max_depth_in_tree(T, start_nbr)
        full_dist = T[j_node][start_nbr]['weight'] + dist
        full_path = [j_node] + path
        branch_paths.append((full_dist, full_path, terminal))

    branch_paths.sort(key=lambda x: x[0], reverse=True)

    if len(branch_paths) >= 2:
        longest = branch_paths[0][0]
        second = branch_paths[1][0]
        ratio = second / longest if longest > 0 else 0
        accepted = (second >= MINIMUM_LENGTH) and (ratio >= MINIMUM_RATIO) and (dist_from_root >= MIN_LMCA_LENGTH)

        candidate_evals.append({
            'node': j_node,
            'dist_from_root': dist_from_root,
            'longest': longest,
            'second': second,
            'ratio': ratio,
            'accepted': accepted,
            'all_branch_lengths': [b[0] for b in branch_paths],
            'branchA': branch_paths[0],
            'branchB': branch_paths[1]
        })

# Debug Mode Table
print("\n" + "="*90)
print(f"{'Junction':<10} | {'Dist from Root':<15} | {'Longest':<10} | {'Second':<10} | {'Ratio':<10} | {'Accepted?':<10}")
print("-" * 90)
for c in sorted(candidate_evals, key=lambda x: x['dist_from_root']):
    acc_str = "Accepted" if c['accepted'] else "Rejected"
    print(f"Node {c['node']:<5} | {c['dist_from_root']:<15.2f} | {c['longest']:<10.2f} | {c['second']:<10.2f} | {c['ratio']:<10.3f} | {acc_str:<10}")
    if c['accepted']:
        branches_str = ", ".join(f"{length:.1f} mm" for length in c['all_branch_lengths'])
        print(f"  -> Outgoing branches: {branches_str}")
print("="*90 + "\n")

# 5. Selection Phase
valid_candidates = [c for c in candidate_evals if c['accepted']]

if not valid_candidates:
    raise ValueError("No valid bifurcation candidates found! Consider lowering MINIMUM_LENGTH or MINIMUM_RATIO.")

# Sort by largest second_longest_branch (descending), then smallest distance_from_root (ascending)
valid_candidates.sort(key=lambda x: (x['second'], -x['dist_from_root']), reverse=True)
best_candidate = valid_candidates[0]

lmca_bifurcation_node = best_candidate['node']
print(f"--> Selected LMCA Bifurcation: Node {lmca_bifurcation_node}\n")

# 6 & 7. Extract LMCA and Daughter Branches
lmca_nodes = nx.shortest_path(G, source=root_node, target=lmca_bifurcation_node, weight='weight')
branchA_nodes = best_candidate['branchA'][1]
branchA_terminal = best_candidate['branchA'][2]
branchB_nodes = best_candidate['branchB'][1]
branchB_terminal = best_candidate['branchB'][2]

def extract_coords(path_nodes):
    if len(path_nodes) < 2:
        return np.empty((0, 3))

    full_coords = []
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        edge_data = G[u][v]
        path_id = edge_data['path_id']
        coords = branches[f"path_{path_id}"]

        src_node = edge_data['node_id_src']
        dst_node = edge_data['node_id_dst']

        if u == dst_node and v == src_node:
            coords = coords[::-1]

        if len(full_coords) > 0:
            coords = coords[1:]

        full_coords.append(coords)

    return np.vstack(full_coords) if full_coords else np.empty((0, 3))

lmca_coords = extract_coords(lmca_nodes)
branchA_coords = extract_coords(branchA_nodes)
branchB_coords = extract_coords(branchB_nodes)

# Gather entire centerline for Cyan background
all_coords = []
for key in branches.files:
    all_coords.append(branches[key])

# 8. Save Files
print("Saving outputs...")
np.save(BASE_DIR / "LMCA.npy", lmca_coords)
np.save(BASE_DIR / "BranchA.npy", branchA_coords)
np.save(BASE_DIR / "BranchB.npy", branchB_coords)

np.save(BASE_DIR / "LMCA_nodes.npy", np.array(lmca_nodes))
np.save(BASE_DIR / "BranchA_nodes.npy", np.array(branchA_nodes))
np.save(BASE_DIR / "BranchB_nodes.npy", np.array(branchB_nodes))

selected_landmarks = {
    "candidate_root_node": root_node,
    "lmca_bifurcation_node": lmca_bifurcation_node,
    "branchA_terminal": branchA_terminal,
    "branchB_terminal": branchB_terminal
}
np.save(BASE_DIR / "selected_landmarks.npy", selected_landmarks, allow_pickle=True)

# 9. Napari Visualization
print("Visualizing in Napari...")
viewer = napari.Viewer(ndisplay=3)

# Background entire skeleton
viewer.add_shapes(all_coords, shape_type='path', edge_color='cyan', edge_width=0.3, opacity=0.3, name='Entire Centerline')

if len(lmca_coords) > 0:
    viewer.add_shapes(lmca_coords, shape_type='path', edge_color='red', edge_width=1.0, name='LMCA')
if len(branchA_coords) > 0:
    viewer.add_shapes(branchA_coords, shape_type='path', edge_color='green', edge_width=1.0, name='Branch A')
if len(branchB_coords) > 0:
    viewer.add_shapes(branchB_coords, shape_type='path', edge_color='blue', edge_width=1.0, name='Branch B')

root_coord = landmarks["candidate_root_coord"]
bif_idx = np.where(landmarks["junction_nodes"] == lmca_bifurcation_node)[0]
bif_coord = landmarks["junction_coords"][bif_idx[0]].reshape(1, 3) if len(bif_idx) > 0 else []

viewer.add_points(root_coord, size=2.0, face_color='white', name='Candidate Root')
if len(bif_coord) > 0:
    viewer.add_points(bif_coord, size=2.0, face_color='yellow', name='Bifurcation')

viewer.show()
napari.run()
