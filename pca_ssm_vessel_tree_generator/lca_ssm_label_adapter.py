"""Adapter from the 200 PCA label/centerline cases to an inferred LCA tree.

The source cases contain an unlabelled skeleton graph rather than explicit
LMCA/LAD/LCX arrays.  This module follows the repository's existing landmark
selection rules to find the root and bifurcation, then uses the common RAS scan
orientation to classify the two selected daughters as LAD and LCX candidates.
The classification is recorded as inferred, never as source ground truth.
"""

from __future__ import annotations

import csv
import gzip
import struct
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt


EPS = 1.0e-12
MINIMUM_DAUGHTER_LENGTH_MM = 10.0
MINIMUM_DAUGHTER_RATIO = 0.25
MINIMUM_LMCA_LENGTH_MM = 5.0
LOCAL_RADIUS_WINDOW_MM = 8.0

NIFTI_DTYPES = {
    2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1",
    512: "u2", 768: "u4", 1024: "i8", 1280: "u8",
}


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError("cannot normalize a zero or non-finite direction")
    return vector / norm


def read_nifti_header(path: Path) -> dict[str, Any]:
    """Read the small subset of a NIfTI-1 header required by this adapter."""
    with gzip.open(path, "rb") as handle:
        header = handle.read(352)
    if len(header) < 352:
        raise ValueError(f"truncated NIfTI header: {path.name}")
    if struct.unpack("<i", header[:4])[0] == 348:
        endian = "<"
    elif struct.unpack(">i", header[:4])[0] == 348:
        endian = ">"
    else:
        raise ValueError(f"not a NIfTI-1 file: {path.name}")
    dimensions = struct.unpack(endian + "8h", header[40:56])
    if dimensions[0] != 3:
        raise ValueError(f"expected a 3-D label volume; got {dimensions[0]} dimensions")
    shape = tuple(int(value) for value in dimensions[1:4])
    spacing = np.asarray(struct.unpack(endian + "8f", header[76:108])[1:4], dtype=float)
    datatype = int(struct.unpack(endian + "h", header[70:72])[0])
    if datatype not in NIFTI_DTYPES:
        raise ValueError(f"unsupported NIfTI datatype code {datatype}")
    voxel_offset = int(round(struct.unpack(endian + "f", header[108:112])[0]))
    sform_code = int(struct.unpack(endian + "h", header[254:256])[0])
    if sform_code <= 0:
        raise ValueError("NIfTI sform is required to establish the common RAS anatomy axes")
    affine = np.eye(4, dtype=float)
    affine[:3, :] = np.asarray(
        [struct.unpack(endian + "4f", header[offset:offset + 16]) for offset in (280, 296, 312)]
    )
    if not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) <= EPS:
        raise ValueError("NIfTI sform is non-finite or singular")
    return {
        "endian": endian, "shape": shape, "spacing_mm": spacing,
        "datatype": datatype, "voxel_offset": voxel_offset, "affine": affine,
        "sform_code": sform_code,
    }


def load_binary_nifti(path: Path, header: dict[str, Any]) -> np.ndarray:
    """Stream a gzipped NIfTI into a compact Boolean array one z-plane at a time."""
    shape = header["shape"]
    dtype = np.dtype(header["endian"] + NIFTI_DTYPES[header["datatype"]])
    values_per_plane = shape[0] * shape[1]
    bytes_per_plane = values_per_plane * dtype.itemsize
    mask = np.empty(shape, dtype=bool)
    with gzip.open(path, "rb") as handle:
        handle.seek(header["voxel_offset"])
        for z_index in range(shape[2]):
            raw = handle.read(bytes_per_plane)
            if len(raw) != bytes_per_plane:
                raise ValueError(f"truncated NIfTI image data at z={z_index}")
            plane = np.frombuffer(raw, dtype=dtype, count=values_per_plane)
            mask[:, :, z_index] = plane.reshape(shape[:2], order="F") > 0
    if not np.any(mask):
        raise ValueError("label volume is empty")
    return mask


def load_graph(centerline_dir: Path) -> tuple[nx.Graph, list[dict[str, str]], Any]:
    """Load the skan summary and its corresponding ordered branch arrays."""
    summary_path = centerline_dir / "summary.csv"
    branches_path = centerline_dir / "branches.npz"
    if not summary_path.is_file() or not branches_path.is_file():
        raise ValueError("summary.csv and branches.npz are both required")
    with summary_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("summary.csv contains no skeleton branches")
    branches = np.load(branches_path, allow_pickle=False)
    graph = nx.Graph()
    for path_id, row in enumerate(rows):
        source = int(row["node-id-src"])
        destination = int(row["node-id-dst"])
        distance = float(row["branch-distance"])
        data = {
            "weight": distance, "path_id": path_id,
            "node_id_src": source, "node_id_dst": destination,
        }
        if graph.has_edge(source, destination):
            if distance < graph[source][destination]["weight"]:
                graph[source][destination].update(data)
        else:
            graph.add_edge(source, destination, **data)
    return graph, rows, branches


def node_image_coordinates(rows: list[dict[str, str]]) -> dict[int, np.ndarray]:
    coordinates: dict[int, np.ndarray] = {}
    for row in rows:
        source = int(row["node-id-src"])
        destination = int(row["node-id-dst"])
        coordinates[source] = np.array([int(row[f"image-coord-src-{axis}"]) for axis in range(3)])
        coordinates[destination] = np.array([int(row[f"image-coord-dst-{axis}"]) for axis in range(3)])
    return coordinates


def select_root_by_endpoint_radius(
    graph: nx.Graph, rows: list[dict[str, str]], mask: np.ndarray, spacing: np.ndarray
) -> tuple[int, dict]:
    """Match stage 1: select the skeleton endpoint with the largest mask radius."""
    image_coordinates = node_image_coordinates(rows)
    endpoints = sorted(node for node, degree in graph.degree if degree == 1)
    if len(endpoints) < 3:
        raise ValueError(f"at least three graph endpoints are required; got {len(endpoints)}")
    window = np.ceil(LOCAL_RADIUS_WINDOW_MM / spacing).astype(int)
    estimates = []
    for node in endpoints:
        center = image_coordinates[node]
        if np.any(center < 0) or np.any(center >= np.asarray(mask.shape)):
            raise ValueError(f"endpoint node {node} lies outside its label volume")
        lower = np.maximum(center - window, 0)
        upper = np.minimum(center + window + 1, mask.shape)
        crop = mask[
            lower[0]:upper[0], lower[1]:upper[1], lower[2]:upper[2]
        ]
        distance = distance_transform_edt(crop, sampling=spacing)
        radius = float(distance[tuple(center - lower)])
        estimates.append((radius, node, center.copy()))
    estimates.sort(key=lambda item: (-item[0], item[1]))
    best = estimates[0]
    second_radius = estimates[1][0]
    return best[1], {
        "method": "largest_endpoint_radius_from_label_mask",
        "node_id": best[1], "image_coordinate": best[2], "radius_mm": best[0],
        "second_largest_radius_mm": second_radius,
        "radius_margin_mm": best[0] - second_radius,
        "endpoint_count": len(endpoints),
    }


def build_shortest_path_tree(graph: nx.Graph, root: int) -> nx.DiGraph:
    if root not in graph:
        raise ValueError(f"root node {root} is absent from the centerline graph")
    tree = nx.DiGraph()
    paths = nx.single_source_dijkstra_path(graph, source=root, weight="weight")
    tree.add_node(root)
    for path in paths.values():
        if len(path) > 1:
            source, destination = path[-2:]
            if not tree.has_edge(source, destination):
                tree.add_edge(source, destination, **graph[source][destination])
    return tree


def maximum_tree_depth(tree: nx.DiGraph, start: int) -> tuple[float, list[int], int]:
    distances = {start: 0.0}
    queue = [start]
    for current in queue:
        for neighbor in tree.successors(current):
            distances[neighbor] = distances[current] + float(tree[current][neighbor]["weight"])
            queue.append(neighbor)
    furthest = max(distances, key=distances.get)
    return distances[furthest], nx.shortest_path(tree, start, furthest), furthest


def select_lmca_bifurcation(graph: nx.Graph, root: int) -> dict:
    """Apply the existing stage-2 candidate thresholds and ranking unchanged."""
    tree = build_shortest_path_tree(graph, root)
    candidates = []
    for junction in sorted(node for node, degree in graph.degree if degree >= 3):
        if junction not in tree or tree.out_degree(junction) < 2:
            continue
        distance_from_root = float(nx.shortest_path_length(tree, root, junction, weight="weight"))
        daughter_paths = []
        for neighbor in tree.successors(junction):
            depth, path, terminal = maximum_tree_depth(tree, neighbor)
            total = float(tree[junction][neighbor]["weight"]) + depth
            daughter_paths.append((total, [junction] + path, terminal))
        daughter_paths.sort(key=lambda item: item[0], reverse=True)
        if len(daughter_paths) < 2:
            continue
        longest, second = daughter_paths[0][0], daughter_paths[1][0]
        ratio = second / longest if longest > 0 else 0.0
        if (
            second >= MINIMUM_DAUGHTER_LENGTH_MM
            and ratio >= MINIMUM_DAUGHTER_RATIO
            and distance_from_root >= MINIMUM_LMCA_LENGTH_MM
        ):
            candidates.append({
                "node": junction, "distance_from_root_mm": distance_from_root,
                "longest_daughter_mm": longest, "second_daughter_mm": second,
                "daughter_ratio": ratio, "branch_a": daughter_paths[0],
                "branch_b": daughter_paths[1],
            })
    if not candidates:
        raise ValueError("no bifurcation satisfies the existing stage-2 LCA selection thresholds")
    candidates.sort(
        key=lambda candidate: (candidate["second_daughter_mm"], -candidate["distance_from_root_mm"]),
        reverse=True,
    )
    selected = candidates[0]
    selected["lmca_nodes"] = nx.shortest_path(graph, root, selected["node"], weight="weight")
    selected["valid_candidate_count"] = len(candidates)
    return selected


def extract_path_coordinates(graph: nx.Graph, path_nodes: list[int], branches: Any) -> np.ndarray:
    if len(path_nodes) < 2:
        raise ValueError("selected graph path contains fewer than two nodes")
    pieces = []
    for source, destination in zip(path_nodes[:-1], path_nodes[1:]):
        edge = graph[source][destination]
        key = f"branch_{edge['path_id']}"
        if key not in branches.files:
            raise ValueError(f"branches.npz is missing {key}")
        coordinates = np.asarray(branches[key], dtype=float)
        if source == edge["node_id_dst"] and destination == edge["node_id_src"]:
            coordinates = coordinates[::-1]
        if pieces:
            coordinates = coordinates[1:]
        pieces.append(coordinates)
    result = np.vstack(pieces)
    if result.ndim != 2 or result.shape[1] != 3 or not np.all(np.isfinite(result)):
        raise ValueError("selected branch coordinates are malformed or non-finite")
    return result


def coordinates_to_world(points_mm: np.ndarray, header: dict[str, Any]) -> np.ndarray:
    voxel_coordinates = np.asarray(points_mm, dtype=float) / header["spacing_mm"]
    return voxel_coordinates @ header["affine"][:3, :3].T + header["affine"][:3, 3]


def initial_direction(points: np.ndarray) -> np.ndarray:
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    segments = segments[lengths > EPS]
    lengths = lengths[lengths > EPS]
    if not len(segments):
        raise ValueError("selected daughter branch has zero length")
    count = min(5, len(segments))
    return _normalize(np.mean(segments[:count] / lengths[:count, None], axis=0))


def infer_daughter_labels(branch_a: np.ndarray, branch_b: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Infer LAD/LCX candidates from documented RAS anatomy axes and length prior."""
    candidates = {"branch_a": branch_a, "branch_b": branch_b}
    lad_axis = _normalize(np.array([0.0, 1.0, -1.0]))  # anterior and inferior in RAS
    lcx_axis = np.array([-1.0, 0.0, 0.0])  # patient-left in RAS
    lengths = {
        name: float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        for name, points in candidates.items()
    }
    features = {}
    for name, points in candidates.items():
        overall = _normalize(points[-1] - points[0])
        initial = initial_direction(points)
        features[name] = {
            "overall_direction_ras": overall, "initial_direction_ras": initial,
            "lad_axis_score": float(0.75 * np.dot(overall, lad_axis) + 0.25 * np.dot(initial, lad_axis)),
            "lcx_axis_score": float(0.75 * np.dot(overall, lcx_axis) + 0.25 * np.dot(initial, lcx_axis)),
            "length_mm": lengths[name],
        }
    length_bonus_a = 0.10 if lengths["branch_a"] >= lengths["branch_b"] else -0.10
    assignment_scores = {
        "branch_a_as_lad": (
            features["branch_a"]["lad_axis_score"] + features["branch_b"]["lcx_axis_score"] + length_bonus_a
        ),
        "branch_b_as_lad": (
            features["branch_b"]["lad_axis_score"] + features["branch_a"]["lcx_axis_score"] - length_bonus_a
        ),
    }
    if assignment_scores["branch_a_as_lad"] >= assignment_scores["branch_b_as_lad"]:
        lad_name, lcx_name = "branch_a", "branch_b"
    else:
        lad_name, lcx_name = "branch_b", "branch_a"
    margin = abs(assignment_scores["branch_a_as_lad"] - assignment_scores["branch_b_as_lad"])
    return {"lad": candidates[lad_name], "lcx": candidates[lcx_name]}, {
        "status": "anatomically_inferred_not_ground_truth",
        "coordinate_system": "NIfTI sform RAS",
        "lad_reference_axis": lad_axis,
        "lcx_reference_axis": lcx_axis,
        "selected_lad_source": lad_name,
        "selected_lcx_source": lcx_name,
        "assignment_scores": assignment_scores,
        "assignment_margin": float(margin),
        "candidate_features": features,
        "rule": (
            "maximize combined LAD anterior-inferior alignment and LCX patient-left alignment; "
            "include a small prior favoring the longer daughter as LAD"
        ),
    }


def extract_lca_from_label_case(centerline_dir: Path, nifti_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Extract inferred LMCA/LAD/LCX centerlines from one PCA label case."""
    if not nifti_path.is_file():
        raise ValueError(f"missing matching label volume: {nifti_path.name}")
    header = read_nifti_header(nifti_path)
    graph, rows, branches_archive = load_graph(centerline_dir)
    try:
        mask = load_binary_nifti(nifti_path, header)
        root, root_metadata = select_root_by_endpoint_radius(
            graph, rows, mask, header["spacing_mm"]
        )
        del mask
        selected = select_lmca_bifurcation(graph, root)
        lmca = extract_path_coordinates(graph, selected["lmca_nodes"], branches_archive)
        branch_a = extract_path_coordinates(graph, selected["branch_a"][1], branches_archive)
        branch_b = extract_path_coordinates(graph, selected["branch_b"][1], branches_archive)
    finally:
        branches_archive.close()
    world_lmca = coordinates_to_world(lmca, header)
    world_a = coordinates_to_world(branch_a, header)
    world_b = coordinates_to_world(branch_b, header)
    labelled, label_metadata = infer_daughter_labels(world_a, world_b)
    result = {"lmca": world_lmca, **labelled}
    junction = np.vstack((result["lmca"][-1], result["lad"][0], result["lcx"][0]))
    offsets = np.linalg.norm(junction - junction.mean(axis=0), axis=1)
    metadata = {
        "source_patient": centerline_dir.name,
        "coordinate_system": "world millimetres from NIfTI sform (RAS)",
        "nifti_shape": header["shape"], "nifti_spacing_mm": header["spacing_mm"],
        "nifti_affine": header["affine"], "root_selection": root_metadata,
        "bifurcation_selection": {
            key: value for key, value in selected.items()
            if key not in {"branch_a", "branch_b", "lmca_nodes"}
        },
        "daughter_classification": label_metadata,
        "junction_coordinate_mm": junction.mean(axis=0),
        "branch_junction_offsets_mm": dict(zip(("lmca", "lad", "lcx"), offsets)),
        "maximum_junction_offset_mm": float(offsets.max()),
    }
    return result, metadata
