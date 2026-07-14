from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .visualize import set_axes_equal


BRANCH_ORDER = ["LMCA", "LAD", "LCX"]


def _ring_indices(start_index: int, num_rings: int, ring_points: int) -> np.ndarray:
    return np.arange(start_index, start_index + num_rings * ring_points).reshape(num_rings, ring_points)


def _add_tube_faces(rings: np.ndarray, faces: list):
    num_rings, ring_points = rings.shape
    for ring_idx in range(num_rings - 1):
        current_ring = rings[ring_idx]
        next_ring = rings[ring_idx + 1]
        for point_idx in range(ring_points):
            point_next = (point_idx + 1) % ring_points
            faces.append([current_ring[point_idx], next_ring[point_idx], next_ring[point_next]])
            faces.append([current_ring[point_idx], next_ring[point_next], current_ring[point_next]])


def _add_cap(vertices: list, faces: list, ring: np.ndarray, reverse: bool = False) -> int:
    center = np.mean(np.asarray(vertices, dtype=float)[ring], axis=0)
    center_idx = len(vertices)
    vertices.append(center)

    ring_points = len(ring)
    for point_idx in range(ring_points):
        point_next = (point_idx + 1) % ring_points
        if reverse:
            faces.append([center_idx, ring[point_next], ring[point_idx]])
        else:
            faces.append([center_idx, ring[point_idx], ring[point_next]])

    return center_idx


def _add_ring_bridge(faces: list, source_ring: np.ndarray, target_ring: np.ndarray):
    if len(source_ring) != len(target_ring):
        raise ValueError("Bifurcation bridge rings must use the same number of points.")

    ring_points = len(source_ring)
    for point_idx in range(ring_points):
        point_next = (point_idx + 1) % ring_points
        faces.append([source_ring[point_idx], target_ring[point_idx], target_ring[point_next]])
        faces.append([source_ring[point_idx], target_ring[point_next], source_ring[point_next]])


def _ring_center(vertices: list, ring: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(vertices, dtype=float)[ring], axis=0)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return np.zeros_like(vector, dtype=float)
    return vector / norm


def _add_scaled_ring(vertices: list, source_ring: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    source_points = np.asarray(vertices, dtype=float)[source_ring]
    source_center = np.mean(source_points, axis=0)
    scaled_points = center + (source_points - source_center) * scale
    start_index = len(vertices)
    vertices.extend(scaled_points)
    return np.arange(start_index, start_index + len(source_ring), dtype=np.int64)


def _add_hub_cap(vertices: list, faces: list, ring: np.ndarray, hub_center_idx: int):
    ring_points = len(ring)
    for point_idx in range(ring_points):
        point_next = (point_idx + 1) % ring_points
        faces.append([hub_center_idx, ring[point_next], ring[point_idx]])


def _add_bifurcation_hub(vertices: list, faces: list, branch_rings: dict) -> dict:
    """
    Adds a small MVP bifurcation hub without reusing one ring edge for 3+ faces.

    Each branch boundary connects to its own smaller transition ring. The three
    transition rings meet through a shared central hub point. This keeps the
    mesh connected and closed at the edge level while staying intentionally
    simple rather than boolean-unioning the bifurcation.
    """
    boundary_rings = {
        "LMCA": branch_rings["LMCA"][-1],
        "LAD": branch_rings["LAD"][0],
        "LCX": branch_rings["LCX"][0],
    }
    boundary_centers = {
        branch_name: _ring_center(vertices, ring)
        for branch_name, ring in boundary_rings.items()
    }
    hub_center = np.mean(np.vstack(list(boundary_centers.values())), axis=0)

    branch_directions = {}
    if len(branch_rings["LMCA"]) >= 2:
        branch_directions["LMCA"] = _unit(
            boundary_centers["LMCA"] - _ring_center(vertices, branch_rings["LMCA"][-2])
        )
    else:
        branch_directions["LMCA"] = _unit(boundary_centers["LMCA"] - hub_center)

    for branch_name in ["LAD", "LCX"]:
        if len(branch_rings[branch_name]) >= 2:
            branch_directions[branch_name] = _unit(
                _ring_center(vertices, branch_rings[branch_name][1]) - boundary_centers[branch_name]
            )
        else:
            branch_directions[branch_name] = _unit(boundary_centers[branch_name] - hub_center)

    transition_rings = {}
    transition_scale = 0.58
    transition_offset_scale = 0.35
    for branch_name, boundary_ring in boundary_rings.items():
        boundary_points = np.asarray(vertices, dtype=float)[boundary_ring]
        radius_estimate = float(np.mean(np.linalg.norm(boundary_points - boundary_centers[branch_name], axis=1)))
        transition_center = hub_center + branch_directions[branch_name] * radius_estimate * transition_offset_scale
        transition_ring = _add_scaled_ring(
            vertices,
            boundary_ring,
            transition_center,
            scale=transition_scale,
        )
        transition_rings[branch_name] = transition_ring
        _add_ring_bridge(faces, boundary_ring, transition_ring)

    hub_center_idx = len(vertices)
    vertices.append(hub_center)
    for branch_name in BRANCH_ORDER:
        _add_hub_cap(vertices, faces, transition_rings[branch_name], hub_center_idx)

    return {
        "hub_center": hub_center.tolist(),
        "hub_center_vertex_index": int(hub_center_idx),
        "transition_scale": transition_scale,
        "transition_offset_scale": transition_offset_scale,
        "transition_ring_vertex_indices": {
            branch_name: [int(value) for value in ring]
            for branch_name, ring in transition_rings.items()
        },
    }


def _edge_counts(faces: np.ndarray) -> dict:
    counts = defaultdict(int)
    for face in faces:
        for start, end in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]:
            edge = tuple(sorted((int(start), int(end))))
            counts[edge] += 1
    return counts


def _connected_component_count(vertex_count: int, faces: np.ndarray) -> int:
    if vertex_count == 0:
        return 0

    parent = list(range(vertex_count))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        root_a = find(int(a))
        root_b = find(int(b))
        if root_a != root_b:
            parent[root_b] = root_a

    used_vertices = set()
    for face in faces:
        used_vertices.update(int(value) for value in face)
        union(face[0], face[1])
        union(face[1], face[2])
        union(face[2], face[0])

    return len({find(vertex) for vertex in used_vertices})


def build_lca_tight_mesh(tube_surfaces: dict) -> tuple:
    """
    Builds a simple connected LCA mesh from branch-wise tube surfaces.

    This is an MVP stitch: branch surfaces are preserved, free ends are capped,
    and simple ring bridges connect LMCA to LAD/LCX at the bifurcation.
    """
    missing = [branch for branch in BRANCH_ORDER if branch not in tube_surfaces]
    if missing:
        raise ValueError(f"Missing tube surfaces: {missing}")

    ring_points = None
    vertices = []
    faces = []
    branch_rings = {}
    branch_shapes = {}

    for branch_name in BRANCH_ORDER:
        surface = np.asarray(tube_surfaces[branch_name], dtype=float)
        if surface.ndim != 3 or surface.shape[2] != 3:
            raise ValueError(f"{branch_name} surface must have shape (N, ring_points, 3).")
        if not np.all(np.isfinite(surface)):
            raise ValueError(f"{branch_name} surface contains NaN or Inf values.")
        if ring_points is None:
            ring_points = surface.shape[1]
        elif surface.shape[1] != ring_points:
            raise ValueError("All branch surfaces must use the same ring count.")

        start_index = len(vertices)
        vertices.extend(surface.reshape(-1, 3))
        rings = _ring_indices(start_index, surface.shape[0], surface.shape[1])
        branch_rings[branch_name] = rings
        branch_shapes[branch_name] = list(surface.shape)
        _add_tube_faces(rings, faces)

    cap_indices = {
        "LMCA_proximal": _add_cap(vertices, faces, branch_rings["LMCA"][0], reverse=True),
        "LAD_distal": _add_cap(vertices, faces, branch_rings["LAD"][-1], reverse=False),
        "LCX_distal": _add_cap(vertices, faces, branch_rings["LCX"][-1], reverse=False),
    }

    hub_metadata = _add_bifurcation_hub(vertices, faces, branch_rings)

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    metadata = {
        "method": "MVP hub-transition LCA mesh from branch-wise tube surfaces",
        "ring_points": int(ring_points),
        "branch_order": BRANCH_ORDER,
        "branch_surface_shapes": branch_shapes,
        "caps": {
            "LMCA_proximal_start": True,
            "LAD_distal_end": True,
            "LCX_distal_end": True,
        },
        "cap_vertex_indices": {key: int(value) for key, value in cap_indices.items()},
        "bifurcation_connectors": {
            "LMCA_last_ring_to_hub": True,
            "hub_to_LAD_first_ring": True,
            "hub_to_LCX_first_ring": True,
            "shared_edge_reuse_avoided": True,
        },
        "bifurcation_hub": hub_metadata,
        "limitations": "Simple hub transition faces only; not a boolean union, CFD-grade mesh, or clinical-grade anatomical bifurcation.",
    }
    return vertices, faces, metadata


def validate_tight_mesh(vertices: np.ndarray, faces: np.ndarray, metadata: dict) -> dict:
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)
    errors = []
    warnings = []

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        errors.append("vertices must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        errors.append("faces must have shape (F, 3)")

    if vertices.size == 0:
        errors.append("vertex count must be greater than 0")
    if faces.size == 0:
        errors.append("face count must be greater than 0")
    if vertices.size > 0 and not np.all(np.isfinite(vertices)):
        errors.append("vertices contain NaN or Inf values")
    if faces.size > 0 and not np.all(np.isfinite(faces)):
        errors.append("faces contain NaN or Inf values")

    if faces.size > 0 and vertices.size > 0:
        if int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices):
            errors.append("faces reference vertices outside the vertex array")

    branch_shapes = metadata.get("branch_surface_shapes", {})
    branches_included = all(branch in branch_shapes for branch in BRANCH_ORDER)
    if not branches_included:
        errors.append("all three branches were not included in the tight mesh")

    caps = metadata.get("caps", {})
    free_ends_capped = all(caps.get(key) for key in ["LMCA_proximal_start", "LAD_distal_end", "LCX_distal_end"])
    if not free_ends_capped:
        errors.append("not all free vessel ends were capped")

    connectors = metadata.get("bifurcation_connectors", {})
    bifurcation_connector_created = all(
        connectors.get(key)
        for key in ["LMCA_last_ring_to_hub", "hub_to_LAD_first_ring", "hub_to_LCX_first_ring"]
    )
    if not bifurcation_connector_created:
        errors.append("bifurcation connector was not created")

    boundary_edge_count = None
    nonmanifold_edge_count = None
    connected_component_count = None
    if faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0:
        edge_counts = _edge_counts(faces)
        boundary_edge_count = sum(1 for count in edge_counts.values() if count == 1)
        nonmanifold_edge_count = sum(1 for count in edge_counts.values() if count > 2)
        connected_component_count = _connected_component_count(len(vertices), faces)
        if connected_component_count != 1:
            errors.append(f"mesh has {connected_component_count} connected components")
        if boundary_edge_count > 0:
            warnings.append(f"mesh has {boundary_edge_count} boundary edges")
        if nonmanifold_edge_count > 0:
            warnings.append(f"mesh has {nonmanifold_edge_count} non-manifold edges")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "vertices_shape": list(vertices.shape),
        "faces_shape": list(faces.shape),
        "vertex_count": int(len(vertices)) if vertices.ndim >= 1 else 0,
        "face_count": int(len(faces)) if faces.ndim >= 1 else 0,
        "branches_included": bool(branches_included),
        "free_ends_capped": bool(free_ends_capped),
        "bifurcation_connector_created": bool(bifurcation_connector_created),
        "boundary_edge_count": boundary_edge_count,
        "nonmanifold_edge_count": nonmanifold_edge_count,
        "connected_component_count": connected_component_count,
        "metadata": metadata,
    }


def save_tight_mesh_preview(path, vertices: np.ndarray, faces: np.ndarray, max_faces: int = 7000):
    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_subplot(projection="3d")

    draw_faces = faces
    if len(faces) > max_faces:
        step = int(np.ceil(len(faces) / max_faces))
        draw_faces = faces[::step]

    mesh = Poly3DCollection(
        vertices[draw_faces],
        facecolor="#2f6f9f",
        edgecolor="#1d3342",
        linewidth=0.03,
        alpha=0.78,
    )
    ax.add_collection3d(mesh)
    ax.auto_scale_xyz(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    set_axes_equal(ax)
    ax.set_title("MVP connected tight LCA mesh")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_tight_mesh_ply(path, vertices: np.ndarray, faces: np.ndarray):
    with open(path, "w", encoding="utf-8") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(vertices)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write(f"element face {len(faces)}\n")
        ply_file.write("property list uchar int vertex_indices\n")
        ply_file.write("end_header\n")
        for vertex in vertices:
            ply_file.write(f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        for face in faces:
            ply_file.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")


def save_tight_mesh_stl(path, vertices: np.ndarray, faces: np.ndarray):
    with open(path, "w", encoding="utf-8") as stl_file:
        stl_file.write("solid lca_tight_mesh\n")
        for face in faces:
            triangle = vertices[face]
            normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            normal_norm = np.linalg.norm(normal)
            if normal_norm > 1e-12:
                normal = normal / normal_norm
            else:
                normal = np.zeros(3)
            stl_file.write(f"  facet normal {normal[0]:.8e} {normal[1]:.8e} {normal[2]:.8e}\n")
            stl_file.write("    outer loop\n")
            for vertex in triangle:
                stl_file.write(f"      vertex {vertex[0]:.8e} {vertex[1]:.8e} {vertex[2]:.8e}\n")
            stl_file.write("    endloop\n")
            stl_file.write("  endfacet\n")
        stl_file.write("endsolid lca_tight_mesh\n")
