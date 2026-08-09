"""Export the completed PPT two-plane/two-ellipse measurements for ParaView.

This module is intentionally visualization-only.  It reads immutable source
centerlines and saved measurement results; it does not refit a plane or ellipse.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


COLORS = {
    "LMCA": "#222222",
    "LAD": "#d62728",
    "LCX": "#1f77b4",
    "RCA": "#9467bd",
    "CORONARY_PLANE": "#4c78a8",
    "LAD_PLANE": "#e45756",
    "CROWN_ELLIPSE": "#00a087",
    "LAD_ELLIPSE": "#f28e2b",
    "LANDMARK": "#111111",
    "RESIDUAL": "#777777",
}
ELLIPSE_SAMPLES = 720
RESIDUAL_STRIDE = 10


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def directory_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(root)).replace("\\", "/"): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def arc_position(points: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    return cumulative / max(float(cumulative[-1]), 1.0e-12)


def source_polyline(points: np.ndarray, branch_id: int) -> pv.PolyData:
    points = np.asarray(points, dtype=np.float64)
    mesh = pv.PolyData()
    mesh.points = points
    mesh.lines = np.concatenate(([len(points)], np.arange(len(points), dtype=np.int64)))
    mesh.point_data["point_index"] = np.arange(len(points), dtype=np.int64)
    mesh.point_data["normalized_arc_position"] = arc_position(points)
    mesh.field_data["branch_id"] = np.array([branch_id], dtype=np.int32)
    mesh.field_data["immutable_source_geometry"] = np.array([1], dtype=np.uint8)
    return mesh


def point_vertex(point: np.ndarray) -> pv.PolyData:
    mesh = pv.PolyData(np.asarray(point, dtype=np.float64).reshape(1, 3))
    mesh.verts = np.array([1, 0], dtype=np.int64)
    mesh.field_data["shared_bifurcation"] = np.array([1], dtype=np.uint8)
    return mesh


def plane_patch(plane: dict, half_extent: float, plane_id: int) -> pv.PolyData:
    center = np.asarray(plane["centroid_ras_mm"], dtype=float)
    u = np.asarray(plane["basis_u_ras"], dtype=float)
    v = np.asarray(plane["basis_v_ras"], dtype=float)
    points = np.asarray([
        center - half_extent * u - half_extent * v,
        center + half_extent * u - half_extent * v,
        center + half_extent * u + half_extent * v,
        center - half_extent * u + half_extent * v,
    ])
    mesh = pv.PolyData(points, faces=np.array([4, 0, 1, 2, 3], dtype=np.int64))
    mesh.field_data["plane_id"] = np.array([plane_id], dtype=np.int32)
    mesh.field_data["saved_centroid_ras_mm"] = center
    mesh.field_data["saved_normal_ras"] = np.asarray(plane["normal_ras"], dtype=float)
    mesh.field_data["saved_basis_u_ras"] = u
    mesh.field_data["saved_basis_v_ras"] = v
    mesh.field_data["saved_plane_rmse_mm"] = np.array([plane["rmse_mm"]], dtype=float)
    mesh.field_data["finite_display_patch_not_ellipse"] = np.array([1], dtype=np.uint8)
    return mesh


def ellipse_points(ellipse: dict, plane: dict, sample_count: int = ELLIPSE_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    u = np.asarray(plane["basis_u_ras"], dtype=float)
    v = np.asarray(plane["basis_v_ras"], dtype=float)
    tilt = float(ellipse["tilt_rad"])
    major = np.cos(tilt) * u + np.sin(tilt) * v
    minor = -np.sin(tilt) * u + np.cos(tilt) * v
    center = np.asarray(ellipse["center_3d_ras_mm"], dtype=float)
    points = (
        center[None, :]
        + float(ellipse["a_mm"]) * np.cos(theta)[:, None] * major[None, :]
        + float(ellipse["b_mm"]) * np.sin(theta)[:, None] * minor[None, :]
    )
    return points, theta


def ellipse_polyline(ellipse: dict, plane: dict, ellipse_id: int) -> pv.PolyData:
    points, theta = ellipse_points(ellipse, plane)
    mesh = pv.PolyData()
    mesh.points = points
    indices = np.concatenate((np.arange(len(points), dtype=np.int64), [0]))
    mesh.lines = np.concatenate(([len(indices)], indices))
    mesh.point_data["theta_rad"] = theta
    mesh.point_data["theta_deg"] = np.degrees(theta)
    mesh.field_data["ellipse_id"] = np.array([ellipse_id], dtype=np.int32)
    mesh.field_data["saved_center_ras_mm"] = np.asarray(ellipse["center_3d_ras_mm"], dtype=float)
    mesh.field_data["saved_a_mm"] = np.array([ellipse["a_mm"]], dtype=float)
    mesh.field_data["saved_b_mm"] = np.array([ellipse["b_mm"]], dtype=float)
    mesh.field_data["saved_tilt_rad"] = np.array([ellipse["tilt_rad"]], dtype=float)
    mesh.field_data["saved_fit_rmse_mm"] = np.array([ellipse["in_plane_rmse_mm"]], dtype=float)
    mesh.field_data["statistical_reference_not_replacement"] = np.array([1], dtype=np.uint8)
    return mesh


def landmark_polydata(landmarks: dict, has_rca: bool) -> tuple[pv.PolyData, list[dict]]:
    angle = landmarks.get("ellipse_angles_rad", {})
    records = [
        ("LMCA_START", "LMCA_start_ras_mm", 1, np.nan),
        ("BIFURCATION", "LMCA_bifurcation_ras_mm", 4, angle.get("bifurcation_crown_theta_rad", np.nan)),
        ("LAD_START", "LAD_start_ras_mm", 2, angle.get("bifurcation_lad_theta_rad", np.nan)),
        ("LAD_TERMINAL", "LAD_terminal_ras_mm", 2, angle.get("lad_terminal_theta_rad", np.nan)),
        ("LCX_START", "LCX_start_ras_mm", 3, angle.get("bifurcation_crown_theta_rad", np.nan)),
        ("LCX_TERMINAL", "LCX_terminal_ras_mm", 3, angle.get("lcx_terminal_theta_rad", np.nan)),
    ]
    if has_rca:
        records.extend([
            ("RCA_CANDIDATE_START", "RCA_candidate_start_ras_mm", 4, angle.get("rca_start_theta_rad", np.nan)),
            ("RCA_CANDIDATE_TERMINAL", "RCA_candidate_terminal_ras_mm", 4, angle.get("rca_terminal_theta_rad", np.nan)),
        ])
    records = [record for record in records if record[1] in landmarks]
    points = np.asarray([landmarks[key] for _, key, _, _ in records], dtype=np.float64)
    mesh = pv.PolyData(points)
    mesh.verts = np.column_stack((np.ones(len(points), dtype=np.int64), np.arange(len(points), dtype=np.int64))).ravel()
    mesh.point_data["landmark_name"] = np.asarray([name for name, _, _, _ in records])
    mesh.point_data["associated_branch_id"] = np.asarray([branch for _, _, branch, _ in records], dtype=np.int32)
    mesh.point_data["ellipse_theta_deg"] = np.asarray([np.degrees(value) for _, _, _, value in records])
    metadata = [
        {
            "landmark_name": name,
            "associated_branch_id": branch,
            "ras_mm": landmarks[key],
            "ellipse_theta_deg": None if not np.isfinite(value) else float(np.degrees(value)),
        }
        for name, key, branch, value in records
    ]
    return mesh, metadata


def residual_polydata(rows: list[dict], source: np.ndarray, branch_id: int) -> tuple[pv.PolyData, dict]:
    selected = list(range(0, len(rows), RESIDUAL_STRIDE))
    if rows and selected[-1] != len(rows) - 1:
        selected.append(len(rows) - 1)
    points: list[list[float]] = []
    lines: list[int] = []
    attrs = {name: [] for name in (
        "source_point_index", "normalized_arc_position", "euclidean_residual_mm",
        "signed_in_plane_residual_mm", "signed_out_of_plane_residual_mm",
    )}
    max_csv_source_difference = 0.0
    for cell_index, row_index in enumerate(selected):
        row = rows[row_index]
        source_csv = np.asarray([row["source_x"], row["source_y"], row["source_z"]], dtype=float)
        reference = np.asarray([row["reference_x"], row["reference_y"], row["reference_z"]], dtype=float)
        point_index = int(row["point_index"])
        max_csv_source_difference = max(max_csv_source_difference, float(np.max(np.abs(source_csv - source[point_index]))))
        points.extend((source_csv.tolist(), reference.tolist()))
        lines.extend((2, 2 * cell_index, 2 * cell_index + 1))
        attrs["source_point_index"].append(point_index)
        attrs["normalized_arc_position"].append(float(row["s"]))
        attrs["euclidean_residual_mm"].append(float(row["euclidean_residual"]))
        attrs["signed_in_plane_residual_mm"].append(float(row["in_plane_residual"]))
        attrs["signed_out_of_plane_residual_mm"].append(float(row["out_of_plane_residual"]))
    mesh = pv.PolyData()
    mesh.points = np.asarray(points, dtype=np.float64)
    mesh.lines = np.asarray(lines, dtype=np.int64)
    for name, values in attrs.items():
        dtype = np.int64 if name == "source_point_index" else float
        mesh.cell_data[name] = np.asarray(values, dtype=dtype)
    mesh.cell_data["branch_id"] = np.full(len(selected), branch_id, dtype=np.int32)
    mesh.field_data["correspondence_source"] = np.array([1], dtype=np.int32)
    return mesh, {
        "saved_correspondence_rows": len(rows),
        "exported_connector_count": len(selected),
        "subsample_stride": RESIDUAL_STRIDE,
        "max_csv_to_source_coordinate_difference_mm": max_csv_source_difference,
    }


def save_polydata(path: Path, mesh: pv.PolyData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(path, binary=True)


def write_vtm(path: Path, blocks: Iterable[tuple[str, Path]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("VTKFile", type="vtkMultiBlockDataSet", version="1.0", byte_order="LittleEndian")
    dataset = ET.SubElement(root, "vtkMultiBlockDataSet")
    for index, (name, target) in enumerate(blocks):
        relative = Path(os.path.relpath(target, path.parent)).as_posix()
        ET.SubElement(dataset, "DataSet", index=str(index), name=name, file=relative)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read_residual_rows(path: Path, wanted_cases: set[str]) -> dict[tuple[str, str], list[dict]]:
    result: dict[tuple[str, str], list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["case_id"] in wanted_cases and row["branch"] in {"LAD", "LCX"}:
                result.setdefault((row["case_id"], row["branch"]), []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: int(row["point_index"]))
    return result


def plane_corners(mesh: pv.PolyData) -> np.ndarray:
    return np.asarray(mesh.points)


def equal_axes(ax, arrays: Iterable[np.ndarray]) -> None:
    points = np.vstack([np.asarray(value) for value in arrays if len(value)])
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    center = 0.5 * (lower + upper)
    radius = max(float(np.max(upper - lower)) * 0.55, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def preview(path: Path, title: str, source: dict[str, np.ndarray], planes: dict[str, pv.PolyData],
            ellipses: dict[str, pv.PolyData], landmarks: pv.PolyData | None,
            residuals: dict[str, pv.PolyData], show_source: bool, show_planes: bool,
            show_ellipses: bool, show_landmarks: bool, show_residuals: bool) -> None:
    fig = plt.figure(figsize=(9.5, 8.2))
    ax = fig.add_subplot(111, projection="3d")
    bounds: list[np.ndarray] = []
    if show_source:
        for key in ("lmca", "lad", "lcx", "rca"):
            if key not in source:
                continue
            label = "RCA candidate" if key == "rca" else key.upper()
            ax.plot(*source[key].T, color=COLORS["RCA" if key == "rca" else key.upper()], lw=2.1, label=label)
            bounds.append(source[key])
    if show_planes:
        for key, mesh in planes.items():
            color = COLORS["CORONARY_PLANE" if key == "coronary" else "LAD_PLANE"]
            label = "Coronary SVD plane" if key == "coronary" else "LAD SVD plane"
            vertices = plane_corners(mesh)
            ax.add_collection3d(Poly3DCollection([vertices], alpha=0.18, facecolor=color, edgecolor=color, label=label))
            bounds.append(vertices)
    if show_ellipses:
        for key, mesh in ellipses.items():
            color = COLORS["CROWN_ELLIPSE" if key == "crown" else "LAD_ELLIPSE"]
            label = "Crown fitted ellipse" if key == "crown" else "LAD fitted ellipse"
            ax.plot(*mesh.points.T, color=color, lw=2.5, label=label)
            bounds.append(mesh.points)
    if show_landmarks and landmarks is not None:
        ax.scatter(*landmarks.points.T, color=COLORS["LANDMARK"], s=34, depthshade=False, label="Landmarks")
        bounds.append(landmarks.points)
    if show_residuals:
        first = True
        for mesh in residuals.values():
            for cell in range(mesh.n_cells):
                ids = mesh.get_cell(cell).point_ids
                segment = mesh.points[ids]
                ax.plot(*segment.T, color=COLORS["RESIDUAL"], alpha=0.55, lw=0.65,
                        label="Source-to-reference residual" if first else None)
                first = False
            bounds.append(mesh.points)
    equal_axes(ax, bounds)
    ax.view_init(elev=24, azim=-56)
    ax.set_xlabel("RAS-X (mm)")
    ax.set_ylabel("RAS-Y (mm)")
    ax.set_zlabel("RAS-Z (mm)")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def expected_paths(case_root: Path, has_rca: bool) -> list[Path]:
    paths = [
        case_root / "01_source/LMCA_source.vtp", case_root / "01_source/LAD_source.vtp",
        case_root / "01_source/LCX_source.vtp", case_root / "01_source/bifurcation.vtp",
        case_root / "01_source/source_centerlines.vtm",
        case_root / "02_planes/coronary_SVD_plane.vtp", case_root / "02_planes/LAD_SVD_plane.vtp",
        case_root / "02_planes/measurement_planes.vtm",
        case_root / "03_ellipses/crown_fitted_ellipse.vtp", case_root / "03_ellipses/LAD_fitted_ellipse.vtp",
        case_root / "03_ellipses/fitted_ellipses.vtm",
        case_root / "04_landmarks/landmark_points.vtp", case_root / "04_landmarks/landmarks.vtm",
        case_root / "05_residuals/LCX_source_to_ellipse_residuals.vtp",
        case_root / "05_residuals/LAD_source_to_ellipse_residuals.vtp", case_root / "05_residuals/residuals.vtm",
        case_root / "06_combined/ppt_two_plane_two_ellipse_model.vtm",
        case_root / "measurements.json", case_root / "vtk_validation.json", case_root / "README.md",
    ]
    paths.extend(case_root / f"preview_{index:02d}_{name}.png" for index, name in enumerate([
        "source_only", "measurement_planes", "fitted_ellipses", "source_plus_planes",
        "source_plus_ellipses", "complete_model", "residual_connectors"], 1))
    if has_rca:
        paths.append(case_root / "01_source/RCA_candidate_source.vtp")
    return paths


def validate_vtm(path: Path) -> dict:
    tree = ET.parse(path)
    datasets = tree.findall(".//DataSet")
    missing = []
    for dataset in datasets:
        target = (path.parent / dataset.attrib["file"]).resolve()
        if not target.is_file():
            missing.append(str(target))
    read_error = None
    try:
        pv.read(path)
    except Exception as exc:  # pragma: no cover - reported in JSON
        read_error = str(exc)
    return {
        "block_names": [dataset.attrib.get("name") for dataset in datasets],
        "reference_count": len(datasets),
        "missing_references": missing,
        "pyvista_read_error": read_error,
        "pass": not missing and read_error is None,
    }


def export_case(repo: Path, output: Path, case_id: str, selection: dict,
                residual_rows: dict[tuple[str, str], list[dict]], protected_before: dict) -> dict:
    case_root = output / case_id
    saved_path = repo / "outputs/lca_ssm/ppt_priority_completion/patients" / case_id / "two_plane_two_ellipse.json"
    raw_path = repo / "outputs/lca_ssm/raw_cases" / case_id / "original_centerlines.npz"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    with np.load(raw_path) as archive:
        source = {name: np.asarray(archive[name]).copy() for name in archive.files}
    source_before = {name: points.copy() for name, points in source.items()}
    has_rca = "rca" in source

    combined = np.vstack(list(source.values()))
    half_extent = max(0.58 * float(np.max(np.ptp(combined, axis=0))), 20.0)
    planes_saved = saved["planes"]
    ellipses_saved = saved["ellipses"]
    plane_meshes = {
        "coronary": plane_patch(planes_saved["coronary_AV_groove"], half_extent, 1),
        "lad": plane_patch(planes_saved["interventricular_LAD"], half_extent, 2),
    }
    ellipse_meshes = {
        "crown": ellipse_polyline(ellipses_saved["crown_AV_groove"], planes_saved["coronary_AV_groove"], 1),
        "lad": ellipse_polyline(ellipses_saved["interventricular_LAD"], planes_saved["interventricular_LAD"], 2),
    }
    landmark_mesh, landmark_metadata = landmark_polydata(saved["landmarks"], has_rca)
    residual_meshes: dict[str, pv.PolyData] = {}
    residual_metadata: dict[str, dict] = {}
    for branch, branch_id in (("LCX", 3), ("LAD", 2)):
        residual_meshes[branch], residual_metadata[branch] = residual_polydata(
            residual_rows[(case_id, branch)], source[branch.lower()], branch_id
        )

    source_paths = {
        "SOURCE_LMCA": case_root / "01_source/LMCA_source.vtp",
        "SOURCE_LAD": case_root / "01_source/LAD_source.vtp",
        "SOURCE_LCX": case_root / "01_source/LCX_source.vtp",
    }
    if has_rca:
        source_paths["SOURCE_RCA_CANDIDATE"] = case_root / "01_source/RCA_candidate_source.vtp"
    bifurcation_path = case_root / "01_source/bifurcation.vtp"
    source_meshes = {name: source_polyline(source[key], number) for name, key, number in (
        ("SOURCE_LMCA", "lmca", 1), ("SOURCE_LAD", "lad", 2), ("SOURCE_LCX", "lcx", 3)
    )}
    if has_rca:
        source_meshes["SOURCE_RCA_CANDIDATE"] = source_polyline(source["rca"], 4)
    for name, path in source_paths.items():
        save_polydata(path, source_meshes[name])
    save_polydata(bifurcation_path, point_vertex(np.asarray(saved["landmarks"]["LMCA_bifurcation_ras_mm"])))

    plane_paths = {
        "CORONARY_SVD_PLANE": case_root / "02_planes/coronary_SVD_plane.vtp",
        "LAD_SVD_PLANE": case_root / "02_planes/LAD_SVD_plane.vtp",
    }
    save_polydata(plane_paths["CORONARY_SVD_PLANE"], plane_meshes["coronary"])
    save_polydata(plane_paths["LAD_SVD_PLANE"], plane_meshes["lad"])
    ellipse_paths = {
        "CROWN_FITTED_ELLIPSE": case_root / "03_ellipses/crown_fitted_ellipse.vtp",
        "LAD_FITTED_ELLIPSE": case_root / "03_ellipses/LAD_fitted_ellipse.vtp",
    }
    save_polydata(ellipse_paths["CROWN_FITTED_ELLIPSE"], ellipse_meshes["crown"])
    save_polydata(ellipse_paths["LAD_FITTED_ELLIPSE"], ellipse_meshes["lad"])
    landmark_path = case_root / "04_landmarks/landmark_points.vtp"
    save_polydata(landmark_path, landmark_mesh)
    residual_paths = {
        "LCX_ELLIPSE_RESIDUALS": case_root / "05_residuals/LCX_source_to_ellipse_residuals.vtp",
        "LAD_ELLIPSE_RESIDUALS": case_root / "05_residuals/LAD_source_to_ellipse_residuals.vtp",
    }
    save_polydata(residual_paths["LCX_ELLIPSE_RESIDUALS"], residual_meshes["LCX"])
    save_polydata(residual_paths["LAD_ELLIPSE_RESIDUALS"], residual_meshes["LAD"])

    write_vtm(case_root / "01_source/source_centerlines.vtm", list(source_paths.items()) + [("BIFURCATION", bifurcation_path)])
    write_vtm(case_root / "02_planes/measurement_planes.vtm", plane_paths.items())
    write_vtm(case_root / "03_ellipses/fitted_ellipses.vtm", ellipse_paths.items())
    write_vtm(case_root / "04_landmarks/landmarks.vtm", [("LANDMARKS", landmark_path)])
    write_vtm(case_root / "05_residuals/residuals.vtm", residual_paths.items())
    combined_blocks = (
        list(source_paths.items()) + [("BIFURCATION", bifurcation_path)] + list(plane_paths.items())
        + list(ellipse_paths.items()) + [("LANDMARKS", landmark_path)] + list(residual_paths.items())
    )
    combined_path = case_root / "06_combined/ppt_two_plane_two_ellipse_model.vtm"
    write_vtm(combined_path, combined_blocks)

    preview_specs = [
        ("source_only", True, False, False, True, False),
        ("measurement_planes", False, True, False, False, False),
        ("fitted_ellipses", False, False, True, True, False),
        ("source_plus_planes", True, True, False, True, False),
        ("source_plus_ellipses", True, False, True, True, False),
        ("complete_model", True, True, True, True, False),
        ("residual_connectors", True, False, True, False, True),
    ]
    for index, (name, *flags) in enumerate(preview_specs, 1):
        preview(case_root / f"preview_{index:02d}_{name}.png", f"{case_id}: {name.replace('_', ' ')}",
                source, plane_meshes, ellipse_meshes, landmark_mesh, residual_meshes, *flags)

    measurements = {
        "case_id": case_id,
        "selection": selection,
        "coordinate_system": "NIfTI RAS millimetres",
        "source_geometry_immutable": True,
        "input_files": {
            "source_centerlines": str(raw_path.relative_to(repo)).replace("\\", "/"),
            "saved_measurements": str(saved_path.relative_to(repo)).replace("\\", "/"),
            "saved_residual_correspondences": "outputs/lca_ssm/ppt_priority_completion/population_pointwise_ellipse_residuals.csv",
        },
        "source_point_counts": {name.upper(): int(len(points)) for name, points in source.items()},
        "source_array_sha256": {name: array_sha256(points) for name, points in source.items()},
        "raw_archive_sha256": file_sha256(raw_path),
        "planes": planes_saved,
        "ellipses": ellipses_saved,
        "landmarks": landmark_metadata,
        "residual_connectors": residual_metadata,
        "visualization": {
            "plane_patch_half_extent_mm": half_extent,
            "ellipse_sample_count": ELLIPSE_SAMPLES,
            "preview_camera": {"elevation_deg": 24, "azimuth_deg": -56},
            "colors": COLORS,
        },
    }
    json_write(case_root / "measurements.json", measurements)
    # Reserve the two self-describing files before the completeness check; both
    # are replaced below with their final content.
    json_write(case_root / "vtk_validation.json", {"overall_status": "PENDING"})
    (case_root / "README.md").write_text("Validation and usage notes are being finalized.\n", encoding="utf-8")

    source_checks = {}
    for name, points in source.items():
        vtk_path = source_paths[f"SOURCE_{'RCA_CANDIDATE' if name == 'rca' else name.upper()}"]
        readback = pv.read(vtk_path)
        source_checks[name] = {
            "point_count": int(readback.n_points),
            "polyline_cell_count": int(readback.n_lines),
            "max_exported_coordinate_difference_mm": float(np.max(np.abs(readback.points - points))),
            "max_source_coordinate_change_mm": float(np.max(np.abs(source[name] - source_before[name]))),
            "max_segment_length_change_mm": float(np.max(np.abs(
                np.linalg.norm(np.diff(source[name], axis=0), axis=1)
                - np.linalg.norm(np.diff(source_before[name], axis=0), axis=1)
            ))) if len(points) > 1 else 0.0,
            "saved_array_hash_match": array_sha256(points) == saved["source_integrity"]["source_hashes"][name],
        }
    ellipse_checks = {}
    for name, key, plane_key, vtk_key in (
        ("crown", "crown_AV_groove", "coronary_AV_groove", "CROWN_FITTED_ELLIPSE"),
        ("lad", "interventricular_LAD", "interventricular_LAD", "LAD_FITTED_ELLIPSE"),
    ):
        expected, _ = ellipse_points(ellipses_saved[key], planes_saved[plane_key])
        exported = pv.read(ellipse_paths[vtk_key])
        ellipse_checks[name] = {
            "sample_count": int(exported.n_points),
            "max_saved_parameter_reconstruction_difference_mm": float(np.max(np.abs(exported.points - expected))),
            "center_difference_mm": float(np.max(np.abs(exported.field_data["saved_center_ras_mm"] - ellipses_saved[key]["center_3d_ras_mm"]))),
            "a_difference_mm": abs(float(exported.field_data["saved_a_mm"][0]) - float(ellipses_saved[key]["a_mm"])),
            "b_difference_mm": abs(float(exported.field_data["saved_b_mm"][0]) - float(ellipses_saved[key]["b_mm"])),
            "tilt_difference_rad": abs(float(exported.field_data["saved_tilt_rad"][0]) - float(ellipses_saved[key]["tilt_rad"])),
        }
    plane_checks = {}
    for name, key, vtk_key in (
        ("coronary", "coronary_AV_groove", "CORONARY_SVD_PLANE"),
        ("lad", "interventricular_LAD", "LAD_SVD_PLANE"),
    ):
        exported = pv.read(plane_paths[vtk_key])
        plane_checks[name] = {
            "centroid_difference_mm": float(np.max(np.abs(exported.field_data["saved_centroid_ras_mm"] - planes_saved[key]["centroid_ras_mm"]))),
            "normal_difference": float(np.max(np.abs(exported.field_data["saved_normal_ras"] - planes_saved[key]["normal_ras"]))),
            "basis_u_difference": float(np.max(np.abs(exported.field_data["saved_basis_u_ras"] - planes_saved[key]["basis_u_ras"]))),
            "basis_v_difference": float(np.max(np.abs(exported.field_data["saved_basis_v_ras"] - planes_saved[key]["basis_v_ras"]))),
            "rmse_difference_mm": abs(float(exported.field_data["saved_plane_rmse_mm"][0]) - float(planes_saved[key]["rmse_mm"])),
        }
    vtm_files = sorted(case_root.rglob("*.vtm"))
    vtm_checks = {str(path.relative_to(case_root)).replace("\\", "/"): validate_vtm(path) for path in vtm_files}
    bifurcation_difference = float(np.max(np.abs(
        pv.read(bifurcation_path).points[0] - np.asarray(saved["landmarks"]["LMCA_bifurcation_ras_mm"])
    )))
    missing_files = [str(path.relative_to(case_root)).replace("\\", "/") for path in expected_paths(case_root, has_rca) if not path.is_file()]
    protected_after = protected_snapshot(repo)
    ellipse_numeric_pass = all(
        value["max_saved_parameter_reconstruction_difference_mm"] == 0.0
        and value["center_difference_mm"] == 0.0
        and value["a_difference_mm"] == 0.0
        and value["b_difference_mm"] == 0.0
        and value["tilt_difference_rad"] == 0.0
        for value in ellipse_checks.values()
    )
    checks_pass = (
        not missing_files
        and all(value["max_exported_coordinate_difference_mm"] == 0.0 for value in source_checks.values())
        and all(value["max_source_coordinate_change_mm"] == 0.0 for value in source_checks.values())
        and all(value["max_segment_length_change_mm"] == 0.0 for value in source_checks.values())
        and all(value["saved_array_hash_match"] for value in source_checks.values())
        and ellipse_numeric_pass
        and all(max(value.values()) == 0.0 for value in plane_checks.values())
        and bifurcation_difference == 0.0
        and all(value["pass"] for value in vtm_checks.values())
        and protected_after == protected_before
        and file_sha256(raw_path) == saved["source_integrity"]["raw_archive_sha256"]
    )
    validation = {
        "case_id": case_id,
        "overall_status": "PASS" if checks_pass else "FAIL",
        "source_geometry_checks": source_checks,
        "ellipse_saved_parameter_checks": ellipse_checks,
        "plane_saved_parameter_checks": plane_checks,
        "bifurcation_coordinate_difference_mm": bifurcation_difference,
        "expected_file_checks": {"missing_files": missing_files, "pass": not missing_files},
        "vtm_reference_checks": vtm_checks,
        "source_archive_hash_unchanged": file_sha256(raw_path) == saved["source_integrity"]["raw_archive_sha256"],
        "protected_and_population_input_hashes_unchanged": protected_after == protected_before,
    }
    json_write(case_root / "vtk_validation.json", validation)
    readme = f"""# {case_id} — PPT two-plane + two-ellipse visualization

Open `06_combined/ppt_two_plane_two_ellipse_model.vtm` in ParaView.

The SVD plane determines the anatomical plane orientation. The finite plane polygons are display patches only; they are not fitted ellipses. The ellipse is independently fitted inside that plane. The source artery is measured but never replaced. The fitted ellipse is a statistical reference. The residual is the source artery's deviation from that reference.

The RCA component is an inferred candidate, not annotated ground truth. {"This case contains that inferred candidate." if has_rca else "This LCX-only crown-support case has no exported RCA candidate."}

All coordinates are the saved NIfTI RAS coordinates in millimetres. The source `.vtp` files contain exactly one polyline with the original point order and point count. Residual connectors are a display subsample (every {RESIDUAL_STRIDE}th saved correspondence plus the terminal row); their scalar arrays retain the saved residual values.

Selection note: {selection['reason']}
"""
    (case_root / "README.md").write_text(readme, encoding="utf-8")
    return {"case_id": case_id, "validation": validation, "combined_vtm": combined_path, "blocks": combined_blocks}


def protected_snapshot(repo: Path) -> dict:
    completion = repo / "outputs/lca_ssm/ppt_priority_completion"
    files = [
        completion / "population_two_plane_two_ellipse_parameters.csv",
        completion / "population_pointwise_ellipse_residuals.csv",
        completion / "population_statistics.json",
        completion / "population_statistics.csv",
        completion / "population_case_status.csv",
    ]
    return {
        "population_input_files": {str(path.relative_to(repo)).replace("\\", "/"): file_sha256(path) for path in files},
        "protected_stage1_outputs": {
            "stage1_heart_scaffold_vtk": directory_hashes(repo / "outputs/lca_ssm/stage1_heart_scaffold_vtk"),
            "stage1_parametric_heart_surface": directory_hashes(repo / "outputs/lca_ssm/stage1_parametric_heart_surface"),
        },
    }


def choose_cases(parameter_csv: Path) -> tuple[list[str], dict[str, dict]]:
    with parameter_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    eligible = [row for row in rows if row["coronary_support"] == "LCX_only_crown_approximation_insufficient_RCA_support"]
    if len(eligible) != 2:
        raise RuntimeError(f"Expected exactly two LCX-only crown-support cases; found {len(eligible)}")
    for row in eligible:
        row["selection_score"] = float(row["crown_fit_rmse"]) + float(row["lad_fit_rmse"])
    chosen = min(eligible, key=lambda row: row["selection_score"])
    other = max(eligible, key=lambda row: row["selection_score"])
    selections = {
        "91.label": {"role": "required_representative", "reason": "Explicitly required representative case 91.label."},
        "189.label": {"role": "required_representative", "reason": "Explicitly required representative case 189.label."},
        chosen["case_id"]: {
            "role": "automatically_selected_LCX_only_crown_support",
            "rule": "minimum saved crown_fit_rmse + saved lad_fit_rmse among the two LCX-only crown-support cases",
            "score_mm": chosen["selection_score"],
            "alternative_case_id": other["case_id"],
            "alternative_score_mm": other["selection_score"],
            "reason": (
                f"Automatically selected {chosen['case_id']} because its saved combined crown-plus-LAD fit RMSE "
                f"({chosen['selection_score']:.6f} mm) is lower than {other['case_id']} "
                f"({other['selection_score']:.6f} mm)."
            ),
        },
    }
    return ["91.label", "189.label", chosen["case_id"]], selections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--clean", action="store_true", help="Remove only the dedicated VTK visualization output before export.")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    completion = repo / "outputs/lca_ssm/ppt_priority_completion"
    output = completion / "vtk_visualization"
    if output.exists():
        if not args.clean:
            raise FileExistsError(f"Refusing to overwrite existing visualization output: {output}. Use --clean explicitly.")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    cases, selections = choose_cases(completion / "population_two_plane_two_ellipse_parameters.csv")
    protected_before = protected_snapshot(repo)
    residual_rows = read_residual_rows(completion / "population_pointwise_ellipse_residuals.csv", set(cases))
    results = [export_case(repo, output, case, selections[case], residual_rows, protected_before) for case in cases]

    demo = output / "presentation_demo"
    demo.mkdir()
    case_91 = next(result for result in results if result["case_id"] == "91.label")
    demo_vtm = demo / "91_label_ppt_two_plane_two_ellipse_model.vtm"
    write_vtm(demo_vtm, case_91["blocks"])
    demo_text = """# ParaView demo steps

1. Open `91_label_ppt_two_plane_two_ellipse_model.vtm` in ParaView and click **Apply**.
2. In the Pipeline Browser, expand the multiblock dataset and use **Block Colors Distinct Values** if desired.
3. Show the three source branches first; use line width 4–6 and keep the original RAS axes visible.
4. Add `CORONARY_SVD_PLANE` and `LAD_SVD_PLANE` with opacity around 0.15–0.25.
5. Add the two fitted ellipses with line width 5–7.
6. Show `LANDMARKS` using Point Gaussian or Glyph representation.
7. Finally show the residual blocks, color by `euclidean_residual_mm`, and use a thin tube only for presentation visibility.

Narration: The SVD plane determines the anatomical plane orientation. The ellipse is independently fitted inside that plane. The source artery is measured but never replaced. The fitted ellipse is a statistical reference. The residual is the source artery's deviation from that reference. The RCA component is an inferred candidate, not annotated ground truth.
"""
    (demo / "PARAVIEW_DEMO_STEPS.md").write_text(demo_text, encoding="utf-8")

    protected_after = protected_snapshot(repo)
    global_pass = all(result["validation"]["overall_status"] == "PASS" for result in results)
    global_pass = global_pass and protected_before == protected_after and validate_vtm(demo_vtm)["pass"]
    manifest = {
        "purpose": "Visualization of the completed PPT-priority two-plane plus two-ellipse measurement model only.",
        "cases": cases,
        "case_selection": selections,
        "source_refit_performed": False,
        "plane_refit_performed": False,
        "ellipse_refit_performed": False,
        "population_statistics_updated": False,
        "synthetic_vessels_generated": False,
        "protected_snapshot_before": protected_before,
        "protected_snapshot_after": protected_after,
        "protected_outputs_unchanged": protected_before == protected_after,
        "presentation_demo_vtm": str(demo_vtm.relative_to(output)).replace("\\", "/"),
        "presentation_demo_validation": validate_vtm(demo_vtm),
        "overall_status": "PASS" if global_pass else "FAIL",
    }
    json_write(output / "export_manifest.json", manifest)
    if not global_pass:
        raise RuntimeError("VTK export validation failed; inspect vtk_validation.json and export_manifest.json")
    print(json.dumps({"output": str(output), "cases": cases, "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
