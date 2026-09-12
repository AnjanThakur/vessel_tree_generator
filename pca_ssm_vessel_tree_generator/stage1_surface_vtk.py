"""VTK export helpers for the Stage-1 parametric anatomical support surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from heart_support_surface import HeartSupportSurface


def canonical_transform(points_ras: np.ndarray, origin_ras: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    points_ras = np.asarray(points_ras, dtype=float)
    return (np.asarray(rotation, dtype=float) @ (points_ras - origin_ras).T).T


def polyline(points: np.ndarray, *, closed: bool = False) -> pv.PolyData:
    points = np.asarray(points, dtype=float)
    mesh = pv.PolyData(points)
    indices = np.arange(len(points) + int(closed), dtype=np.int64)
    if closed:
        indices[-1] = 0
    mesh.lines = np.concatenate(([len(indices)], indices))
    return mesh


def point_vertex(point: np.ndarray) -> pv.PolyData:
    mesh = pv.PolyData(np.asarray(point, dtype=float).reshape(1, 3))
    mesh.verts = np.array([1, 0], dtype=np.int64)
    return mesh


def transform_polydata(mesh_ras: pv.PolyData, origin_ras: np.ndarray, rotation: np.ndarray) -> pv.PolyData:
    transformed = mesh_ras.copy(deep=True)
    transformed.points = canonical_transform(mesh_ras.points, origin_ras, rotation)
    return transformed


def axis_polyline(start: np.ndarray, end: np.ndarray) -> pv.PolyData:
    return polyline(np.vstack((start, end)))


def surface_polydata(surface: HeartSupportSurface) -> pv.PolyData:
    vertices, triangles, metadata = surface.mesh_arrays()
    faces = np.hstack((np.full((len(triangles), 1), 3, dtype=np.int64), triangles)).ravel()
    mesh = pv.PolyData(vertices, faces=faces)
    for name, values in metadata.items():
        mesh.point_data[name] = values
    mesh.field_data["support_surface_exponent_n"] = np.array([surface.exponent], dtype=float)
    mesh.field_data["dataset_derived_reference"] = np.array([1], dtype=np.uint8)
    mesh.field_data["myocardial_reconstruction"] = np.array([0], dtype=np.uint8)
    return mesh


def source_polyline(points: np.ndarray) -> pv.PolyData:
    mesh = polyline(points)
    differences = np.diff(points, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(differences, axis=1))))
    mesh.point_data["normalized_source_arc_length"] = cumulative / max(float(cumulative[-1]), 1.0e-12)
    mesh.field_data["immutable_source_geometry"] = np.array([1], dtype=np.uint8)
    return mesh


def save_polydata(path: Path, mesh: pv.PolyData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(path, binary=True)


def save_multiblock(path: Path, blocks: dict[str, pv.DataSet]) -> None:
    multiblock = pv.MultiBlock()
    for name, block in blocks.items():
        multiblock[name] = block
    path.parent.mkdir(parents=True, exist_ok=True)
    multiblock.save(path, binary=True)


def build_vtk_objects(
    *,
    source_canonical: dict[str, np.ndarray],
    plane_meshes_canonical: dict[str, pv.PolyData],
    ellipse_meshes_canonical: dict[str, pv.PolyData],
    surface: HeartSupportSurface,
) -> dict[str, pv.DataSet]:
    a, b, c = surface.semi_axes
    center = surface.origin
    x_axis = surface.crown_axis
    y_axis = surface.secondary_crown_axis
    superior_axis = -surface.apex_axis
    return {
        "LMCA": source_polyline(source_canonical["lmca"]),
        "LAD": source_polyline(source_canonical["lad"]),
        "LCX": source_polyline(source_canonical["lcx"]),
        "INFERRED_RCA_CANDIDATE": source_polyline(source_canonical["rca"]),
        "BIFURCATION": point_vertex(np.zeros(3)),
        "CORONARY_CENTROID_SVD_MEASUREMENT_PLANE": plane_meshes_canonical["coronary"],
        "LAD_CENTROID_SVD_MEASUREMENT_PLANE": plane_meshes_canonical["lad"],
        "CORONARY_CROWN_REFERENCE_ELLIPSE": ellipse_meshes_canonical["crown"],
        "LONG_AXIS_APICAL_REFERENCE_ELLIPSE": ellipse_meshes_canonical["long_axis"],
        "CROWN_AXIS": axis_polyline(center - a * x_axis, center + a * x_axis),
        "SECONDARY_CROWN_AXIS": axis_polyline(center - b * y_axis, center + b * y_axis),
        "APEX_AXIS": axis_polyline(center + c * superior_axis, center + c * surface.apex_axis),
        "PARAMETRIC_ANATOMICAL_SUPPORT_SURFACE": surface_polydata(surface),
    }


def export_stage1_vtk(case_root: Path, objects: dict[str, pv.DataSet]) -> dict[str, str]:
    mapping = {
        "01_source_geometry/LMCA_centerline.vtp": "LMCA",
        "01_source_geometry/LAD_centerline.vtp": "LAD",
        "01_source_geometry/LCX_centerline.vtp": "LCX",
        "01_source_geometry/RCA_candidate_centerline.vtp": "INFERRED_RCA_CANDIDATE",
        "01_source_geometry/bifurcation.vtp": "BIFURCATION",
        "02_measurement_planes/coronary_centroid_SVD_plane.vtp": "CORONARY_CENTROID_SVD_MEASUREMENT_PLANE",
        "02_measurement_planes/LAD_centroid_SVD_plane.vtp": "LAD_CENTROID_SVD_MEASUREMENT_PLANE",
        "03_reference_ellipses/coronary_crown_reference_ellipse.vtp": "CORONARY_CROWN_REFERENCE_ELLIPSE",
        "03_reference_ellipses/long_axis_apical_reference_ellipse.vtp": "LONG_AXIS_APICAL_REFERENCE_ELLIPSE",
        "03_reference_ellipses/crown_axis.vtp": "CROWN_AXIS",
        "03_reference_ellipses/secondary_crown_axis.vtp": "SECONDARY_CROWN_AXIS",
        "03_reference_ellipses/apex_axis.vtp": "APEX_AXIS",
        "04_parametric_surface/parametric_heart_support_surface.vtp": "PARAMETRIC_ANATOMICAL_SUPPORT_SURFACE",
    }
    for relative_path, object_name in mapping.items():
        save_polydata(case_root / relative_path, objects[object_name])

    source_names = ["LMCA", "LAD", "LCX", "INFERRED_RCA_CANDIDATE", "BIFURCATION"]
    scaffold_names = [
        "BIFURCATION",
        "CORONARY_CROWN_REFERENCE_ELLIPSE",
        "LONG_AXIS_APICAL_REFERENCE_ELLIPSE",
        "CROWN_AXIS",
        "SECONDARY_CROWN_AXIS",
        "APEX_AXIS",
        "PARAMETRIC_ANATOMICAL_SUPPORT_SURFACE",
    ]
    measurement_names = [
        "CORONARY_CENTROID_SVD_MEASUREMENT_PLANE",
        "LAD_CENTROID_SVD_MEASUREMENT_PLANE",
    ]
    multiblocks = {
        "03_reference_ellipses/heart_scaffold.vtm": scaffold_names,
        "01_source_geometry/source_centerlines.vtm": source_names,
        "05_overlay/source_plus_scaffold.vtm": source_names + scaffold_names[1:-1],
        "05_overlay/source_plus_parametric_surface.vtm": source_names
        + ["PARAMETRIC_ANATOMICAL_SUPPORT_SURFACE"],
        "05_overlay/stage1_complete_anatomical_model.vtm": source_names
        + measurement_names
        + scaffold_names[1:],
    }
    for relative_path, names in multiblocks.items():
        save_multiblock(case_root / relative_path, {name: objects[name] for name in names})

    paths = {name: str((case_root / relative).resolve()) for relative, name in mapping.items()}
    for relative, names in multiblocks.items():
        key = Path(relative).stem.upper()
        paths[key] = str((case_root / relative).resolve())
    return paths


def vtk_block_manifest() -> dict[str, list[str]]:
    return {
        "heart_scaffold": [
            "BIFURCATION",
            "CORONARY_CROWN_REFERENCE_ELLIPSE",
            "LONG_AXIS_APICAL_REFERENCE_ELLIPSE",
            "CROWN_AXIS",
            "SECONDARY_CROWN_AXIS",
            "APEX_AXIS",
            "PARAMETRIC_ANATOMICAL_SUPPORT_SURFACE",
        ],
        "source_centerlines": ["LMCA", "LAD", "LCX", "INFERRED_RCA_CANDIDATE", "BIFURCATION"],
        "stage1_complete_anatomical_model": [
            "LMCA",
            "LAD",
            "LCX",
            "INFERRED_RCA_CANDIDATE",
            "BIFURCATION",
            "CORONARY_CENTROID_SVD_MEASUREMENT_PLANE",
            "LAD_CENTROID_SVD_MEASUREMENT_PLANE",
            "CORONARY_CROWN_REFERENCE_ELLIPSE",
            "LONG_AXIS_APICAL_REFERENCE_ELLIPSE",
            "CROWN_AXIS",
            "SECONDARY_CROWN_AXIS",
            "APEX_AXIS",
            "PARAMETRIC_ANATOMICAL_SUPPORT_SURFACE",
        ],
    }
