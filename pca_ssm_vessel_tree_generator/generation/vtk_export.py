"""Modular VTK/ParaView export for generated coronary trees."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyvista as pv

from generation.tree_assembler import SyntheticTree
from surface_relative.surface_projection import ellipsoid_point


BRANCH_IDS = {"LMCA": 1, "LAD": 2, "LCX": 3, "RCA": 4}


def normalized_arc_position(points: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    return cumulative / max(float(cumulative[-1]), 1.0e-12)


def polyline(points: np.ndarray, branch_name: str, surface_path=None) -> pv.PolyData:
    points = np.asarray(points, dtype=np.float64)
    mesh = pv.PolyData()
    mesh.points = points
    mesh.lines = np.concatenate(([len(points)], np.arange(len(points), dtype=np.int64)))
    mesh.point_data["point_index"] = np.arange(len(points), dtype=np.int64)
    mesh.point_data["normalized_arc_position"] = normalized_arc_position(points)
    if surface_path is not None:
        mesh.point_data["surface_u_rad"] = np.asarray(surface_path.u, dtype=float)
        mesh.point_data["surface_v_rad"] = np.asarray(surface_path.v, dtype=float)
        mesh.point_data["surface_normal_offset_mm"] = np.asarray(surface_path.normal_offset, dtype=float)
        mesh.point_data["deviation_tangent_u_mm"] = np.asarray(surface_path.local_deviation[:, 0], dtype=float)
        mesh.point_data["deviation_tangent_v_mm"] = np.asarray(surface_path.local_deviation[:, 1], dtype=float)
        mesh.point_data["deviation_normal_mm"] = np.asarray(surface_path.local_deviation[:, 2], dtype=float)
    mesh.field_data["branch_id"] = np.array([BRANCH_IDS[branch_name]], dtype=np.int32)
    mesh.field_data["synthetic_geometry"] = np.array([1], dtype=np.uint8)
    return mesh


def ellipsoid_mesh(a: float, b: float, c: float, n_u: int = 96, n_v: int = 48) -> pv.PolyData:
    """Build a display surface from the shared parametric ellipsoid API."""
    u_values = np.linspace(-np.pi, np.pi, n_u, endpoint=False)
    v_values = np.linspace(0.0, np.pi, n_v)
    points = np.vstack([ellipsoid_point(float(u), float(v), a, b, c) for v in v_values for u in u_values])
    faces: list[int] = []
    for v_index in range(n_v - 1):
        for u_index in range(n_u):
            next_u = (u_index + 1) % n_u
            lower = v_index * n_u
            upper = (v_index + 1) * n_u
            faces.extend((4, lower + u_index, lower + next_u, upper + next_u, upper + u_index))
    mesh = pv.PolyData(points, faces=np.asarray(faces, dtype=np.int64))
    mesh.field_data["ellipsoid_a_mm"] = np.array([a], dtype=float)
    mesh.field_data["ellipsoid_b_mm"] = np.array([b], dtype=float)
    mesh.field_data["ellipsoid_c_mm"] = np.array([c], dtype=float)
    mesh.field_data["synthetic_support_surface"] = np.array([1], dtype=np.uint8)
    return mesh


def landmark_vertices(tree: SyntheticTree) -> pv.PolyData:
    names = list(tree.landmarks)
    points = []
    exported_names = []
    for name in names:
        landmark = tree.landmarks[name]
        # Export the exact topological bifurcation from the assembled tree.
        if name == "bifurcation":
            point = tree.branches["LMCA"][-1]
        elif name == "lca_ostium":
            point = tree.branches["LMCA"][0]
        elif name == "lad_endpoint":
            point = tree.branches["LAD"][-1]
        elif name == "lcx_endpoint":
            point = tree.branches["LCX"][-1]
        elif name == "rca_ostium" and "RCA" in tree.branches:
            point = tree.branches["RCA"][0]
        elif name == "rca_endpoint" and "RCA" in tree.branches:
            point = tree.branches["RCA"][-1]
        else:
            continue
        points.append(point)
        exported_names.append(name)
    mesh = pv.PolyData()
    mesh.points = np.asarray(points, dtype=np.float64)
    mesh.verts = np.column_stack((np.ones(len(points), dtype=np.int64), np.arange(len(points), dtype=np.int64))).ravel()
    mesh.point_data["landmark_name"] = np.asarray(exported_names)
    return mesh


def write_vtm(path: Path, blocks: list[tuple[str, Path]]) -> None:
    root = ET.Element("VTKFile", type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian")
    multiblock = ET.SubElement(root, "vtkMultiBlockDataSet")
    for index, (name, target) in enumerate(blocks):
        relative = Path(os.path.relpath(target, path.parent)).as_posix()
        ET.SubElement(multiblock, "DataSet", index=str(index), name=name, file=relative)
    ET.indent(root, space="  ")
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def export_tree_vtk(tree: SyntheticTree, output_directory: Path) -> dict[str, str]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    ellipsoid_path = output_directory / "synthetic_ellipsoid.vtp"
    ellipsoid_mesh(tree.ellipsoid.a, tree.ellipsoid.b, tree.ellipsoid.c).save(ellipsoid_path, binary=True)
    paths["SYNTHETIC_ELLIPSOID"] = ellipsoid_path
    for name, points in tree.branches.items():
        path = output_directory / f"{name}.vtp"
        polyline(points, name, tree.surface_paths.get(name)).save(path, binary=True)
        paths[f"SYNTHETIC_{name}"] = path
    landmarks_path = output_directory / "landmarks.vtp"
    landmark_vertices(tree).save(landmarks_path, binary=True)
    paths["SYNTHETIC_LANDMARKS"] = landmarks_path
    combined_path = output_directory / "synthetic_tree.vtm"
    write_vtm(combined_path, list(paths.items()))
    # Reopen the multiblock now so broken relative references fail immediately.
    reopened = pv.read(combined_path)
    if len(reopened) != len(paths):
        raise RuntimeError(f"VTK read-back returned {len(reopened)} blocks; expected {len(paths)}")
    result = {name: str(path.resolve()) for name, path in paths.items()}
    result["SYNTHETIC_TREE_VTM"] = str(combined_path.resolve())
    return result
