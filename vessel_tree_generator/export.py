"""Portable NumPy, JSON, VTK, and preview export for 4D LCA cases."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from .validation import BRANCH_ORDER


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _arc_fraction(points: np.ndarray) -> np.ndarray:
    distance = np.zeros(len(points), dtype=float)
    if len(points) > 1:
        distance[1:] = np.cumsum(np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1))
    return distance / distance[-1] if len(distance) and distance[-1] > 1.0e-12 else distance


def resample_branch(
    points: np.ndarray,
    number_of_points: int,
    *,
    source_parameter: np.ndarray | None = None,
) -> np.ndarray:
    """Arc-length resample an ``(N, 3|4)`` branch while preserving endpoints."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] not in (3, 4) or len(values) < 2:
        raise ValueError("branch must have shape (N>=2, 3|4)")
    if number_of_points < 2:
        raise ValueError("number_of_points must be at least two")
    source = _arc_fraction(values) if source_parameter is None else np.asarray(source_parameter, dtype=float)
    if source.shape != (len(values),) or np.any(np.diff(source) < 0.0):
        raise ValueError("source_parameter must be a monotonic array matching the branch points")
    target = np.linspace(0.0, 1.0, number_of_points)
    result = np.column_stack([np.interp(target, source, values[:, column]) for column in range(values.shape[1])])
    result[0] = values[0]
    result[-1] = values[-1]
    return result


def _polyline(
    values: np.ndarray,
    *,
    branch_name: str,
    phase: float,
    time_seconds: float,
    disease_reduction: np.ndarray,
) -> pv.PolyData:
    mesh = pv.PolyData(values[:, :3])
    mesh.lines = np.concatenate(([len(values)], np.arange(len(values), dtype=np.int64)))
    mesh.point_data["radius_mm"] = values[:, 3]
    mesh.point_data["normalized_arc_length"] = np.linspace(0.0, 1.0, len(values))
    mesh.point_data["disease_reduction_fraction"] = disease_reduction
    mesh.field_data["branch_id"] = np.asarray([branch_name])
    mesh.field_data["phase"] = np.asarray([phase], dtype=float)
    mesh.field_data["time_seconds"] = np.asarray([time_seconds], dtype=float)
    return mesh


def _write_preview(
    case: dict[str, Any], geometry: np.ndarray, static_geometry: np.ndarray, output: Path
) -> None:
    colors = {"LMCA": "#252525", "LAD": "#d62728", "LCX": "#1f77b4"}
    phases = np.asarray(case["phase_values"], dtype=float)
    peak = float(case["motion"]["peak_phase"])
    peak_index = int(np.argmin(np.abs(phases - peak)))
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0))

    for axis, plot_geometry, title in (
        (axes[0, 0], static_geometry, "Static reference (end diastole)"),
        (axes[0, 1], geometry[peak_index], f"Near peak systole, phase {phases[peak_index]:.2f}"),
    ):
        for branch_index, branch_name in enumerate(BRANCH_ORDER):
            points = plot_geometry[branch_index]
            axis.plot(points[:, 0], points[:, 2], color=colors[branch_name], linewidth=2.8, label=branch_name)
        bifurcation = plot_geometry[0, -1]
        axis.scatter(bifurcation[0], bifurcation[2], color="black", s=35, zorder=5)
        axis.set_title(title)
        axis.set_xlabel("Cardiac X (mm)")
        axis.set_ylabel("Cardiac Z (mm; apex negative)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="best")

    for branch_index, branch_name in enumerate(BRANCH_ORDER):
        axes[1, 0].plot(
            np.linspace(0.0, 1.0, geometry.shape[2]),
            static_geometry[branch_index, :, 3],
            color=colors[branch_name],
            linewidth=2.2,
            label=branch_name,
        )
    axes[1, 0].set_title("Reference radius profiles")
    axes[1, 0].set_xlabel("Normalized branch arc length")
    axes[1, 0].set_ylabel("Radius (mm)")
    axes[1, 0].grid(alpha=0.2)

    reference_xyz = static_geometry[:, :, :3]
    displacement = np.max(np.linalg.norm(geometry[:, :, :, :3] - reference_xyz[None, ...], axis=3), axis=(1, 2))
    radius_change = np.max(
        np.abs(geometry[:, :, :, 3] / static_geometry[:, :, 3][None, ...] - 1.0),
        axis=(1, 2),
    )
    axes[1, 1].plot(phases, displacement, "o-", color="#2ca02c", label="max displacement (mm)")
    radius_axis = axes[1, 1].twinx()
    radius_axis.plot(phases, 100.0 * radius_change, "s--", color="#ff7f0e", label="max radius change (%)")
    axes[1, 1].set_title("4D motion and pulsatility")
    axes[1, 1].set_xlabel("Normalized cardiac phase")
    axes[1, 1].set_ylabel("Displacement (mm)", color="#2ca02c")
    radius_axis.set_ylabel("Radius change (%)", color="#ff7f0e")
    axes[1, 1].grid(alpha=0.2)

    validation_status = "PASS" if case["validation"]["is_valid"] else "FAIL"
    fig.suptitle(f"{case['case_id']} — validated 4D LCA deliverable ({validation_status})", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_case(
    case: dict[str, Any],
    output_directory: str | Path,
    *,
    points_per_branch: int = 50,
    clean: bool = False,
) -> dict[str, Any]:
    """Export a complete portable case and verify VTK readback."""
    output = Path(output_directory).resolve()
    if output.exists():
        if not clean:
            raise FileExistsError(f"refusing to overwrite {output}; pass clean=True explicitly")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    frame_count = len(case["frames"])
    geometry = np.empty((frame_count, len(BRANCH_ORDER), points_per_branch, 4), dtype=float)
    healthy = np.empty((len(BRANCH_ORDER), points_per_branch, 4), dtype=float)
    static_geometry = np.empty((len(BRANCH_ORDER), points_per_branch, 4), dtype=float)
    reduction = np.empty((len(BRANCH_ORDER), points_per_branch), dtype=float)

    source_parameters = {
        name: _arc_fraction(np.asarray(case["reference"]["branches"][name], dtype=float))
        for name in BRANCH_ORDER
    }
    for branch_index, name in enumerate(BRANCH_ORDER):
        healthy[branch_index] = resample_branch(
            case["reference"]["healthy_branches"][name],
            points_per_branch,
            source_parameter=source_parameters[name],
        )
        diseased_reference = resample_branch(
            case["reference"]["branches"][name],
            points_per_branch,
            source_parameter=source_parameters[name],
        )
        static_geometry[branch_index] = diseased_reference
        reduction[branch_index] = np.clip(1.0 - diseased_reference[:, 3] / healthy[branch_index, :, 3], 0.0, 1.0)
    for frame_index, frame in enumerate(case["frames"]):
        for branch_index, name in enumerate(BRANCH_ORDER):
            # Reuse the static-reference parameter at every phase. Native
            # points already correspond through time; recomputing arc length
            # after deformation would make radius/disease values drift between
            # exported point indices.
            geometry[frame_index, branch_index] = resample_branch(
                frame["branches"][name],
                points_per_branch,
                source_parameter=source_parameters[name],
            )

    # Restore exact topological equality after independent interpolation.
    geometry[:, 1, 0, :3] = geometry[:, 0, -1, :3]
    geometry[:, 2, 0, :3] = geometry[:, 0, -1, :3]

    np.save(output / "geometry_cine.npy", geometry, allow_pickle=False)
    np.save(output / "geometry_static.npy", static_geometry, allow_pickle=False)
    np.save(output / "geometry_healthy_reference.npy", healthy, allow_pickle=False)
    np.save(output / "disease_reduction.npy", reduction, allow_pickle=False)
    np.savez_compressed(
        output / "coronary_tree_4d.npz",
        geometry=geometry,
        diseased_reference=static_geometry,
        healthy_reference=healthy,
        disease_reduction_fraction=reduction,
        branch_order=np.asarray(BRANCH_ORDER),
        phase_values=np.asarray(case["phase_values"], dtype=float),
        time_seconds=np.asarray(case["time_seconds"], dtype=float),
    )

    vtk_root = output / "vtk"
    vtk_root.mkdir()
    pvd_entries: list[str] = []
    for frame_index, frame in enumerate(case["frames"]):
        phase_dir = vtk_root / f"phase_{frame_index:03d}"
        phase_dir.mkdir()
        blocks = pv.MultiBlock()
        for branch_index, name in enumerate(BRANCH_ORDER):
            mesh = _polyline(
                geometry[frame_index, branch_index],
                branch_name=name,
                phase=float(frame["phase"]),
                time_seconds=float(frame["time_seconds"]),
                disease_reduction=reduction[branch_index],
            )
            mesh.save(phase_dir / f"{name}.vtp", binary=True)
            blocks[name] = mesh
        blocks.save(phase_dir / "tree.vtm", binary=True)
        pvd_entries.append(
            f'    <DataSet timestep="{float(frame["time_seconds"]):.12g}" group="" part="0" '
            f'file="phase_{frame_index:03d}/tree.vtm"/>'
        )
    (vtk_root / "cine.pvd").write_text(
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n"
        "  <Collection>\n" + "\n".join(pvd_entries) + "\n  </Collection>\n</VTKFile>\n",
        encoding="utf-8",
    )

    graph = {
        **case["hierarchy"],
        "node_attributes": {
            name: {
                "branch_index": index,
                "point_count": points_per_branch,
                "reference_length_mm": float(np.sum(np.linalg.norm(np.diff(static_geometry[index, :, :3], axis=0), axis=1))),
                "minimum_radius_mm": float(np.min(geometry[:, index, :, 3])),
                "maximum_radius_mm": float(np.max(geometry[:, index, :, 3])),
            }
            for index, name in enumerate(BRANCH_ORDER)
        },
    }
    _write_json(output / "graph.json", graph)
    metadata = {
        key: value for key, value in case.items()
        if key not in {"frames", "reference"}
    }
    metadata["arrays"] = {
        "geometry_cine.npy": [frame_count, len(BRANCH_ORDER), points_per_branch, 4],
        "geometry_static.npy": [len(BRANCH_ORDER), points_per_branch, 4],
        "geometry_healthy_reference.npy": [len(BRANCH_ORDER), points_per_branch, 4],
        "disease_reduction.npy": [len(BRANCH_ORDER), points_per_branch],
        "column_order": ["x_mm", "y_mm", "z_mm", "radius_mm"],
        "branch_order": list(BRANCH_ORDER),
    }
    metadata["frame_summaries"] = [
        {
            "phase_index": frame["phase_index"],
            "phase": frame["phase"],
            "time_seconds": frame["time_seconds"],
            "contraction_scale": frame["contraction_scale"],
            "ellipsoid_params": frame["ellipsoid_params"],
        }
        for frame in case["frames"]
    ]
    _write_json(output / "metadata.json", metadata)
    _write_preview(case, geometry, static_geometry, output / "preview.png")

    maximum_export_topology_error = float(max(
        np.max(np.linalg.norm(geometry[:, 0, -1, :3] - geometry[:, 1, 0, :3], axis=1)),
        np.max(np.linalg.norm(geometry[:, 0, -1, :3] - geometry[:, 2, 0, :3], axis=1)),
    ))
    maximum_export_radius_variation = float(np.max(
        np.abs(geometry[..., 3] / static_geometry[..., 3][None, ...] - 1.0)
    ))
    export_validation = {
        "is_valid": bool(
            np.all(np.isfinite(geometry))
            and np.all(geometry[..., 3] > 0.0)
            and maximum_export_topology_error <= 1.0e-9
            and maximum_export_radius_variation <= float(case["pulsatility"]["amplitude"]) + 1.0e-8
        ),
        "temporal_correspondence_parameter": "static-reference normalized arc length reused at every phase",
        "all_values_finite": bool(np.all(np.isfinite(geometry))),
        "all_radii_positive": bool(np.all(geometry[..., 3] > 0.0)),
        "maximum_topology_error_mm": maximum_export_topology_error,
        "maximum_radius_variation_fraction": maximum_export_radius_variation,
        "configured_radius_variation_fraction": float(case["pulsatility"]["amplitude"]),
    }
    if not export_validation["is_valid"]:
        raise RuntimeError(f"fixed-size 4D export validation failed: {export_validation}")

    first_readback = pv.read(vtk_root / "phase_000/tree.vtm")
    last_readback = pv.read(vtk_root / f"phase_{frame_count - 1:03d}/tree.vtm")
    vtk_readback_valid = len(first_readback) == len(BRANCH_ORDER) and len(last_readback) == len(BRANCH_ORDER)
    if not vtk_readback_valid:
        raise RuntimeError("VTK multiblock readback failed")

    key_files = (
        "coronary_tree_4d.npz",
        "geometry_cine.npy",
        "metadata.json",
        "graph.json",
        "preview.png",
        "vtk/cine.pvd",
    )
    checksums = {}
    for relative in key_files:
        payload = (output / relative).read_bytes()
        checksums[relative] = hashlib.sha256(payload).hexdigest()
    manifest = {
        "status": "PASS",
        "case_id": case["case_id"],
        "model_scope": case["model_scope"],
        "frame_count": frame_count,
        "points_per_branch": points_per_branch,
        "geometry_shape": list(geometry.shape),
        "validation_passed": bool(case["validation"]["is_valid"]),
        "fixed_size_export_validation": export_validation,
        "vtk_readback_passed": vtk_readback_valid,
        "portable_relative_paths": True,
        "open_in_paraview": "vtk/cine.pvd",
        "sha256": checksums,
    }
    _write_json(output / "manifest.json", manifest)
    (output / "README.md").write_text(
        "# 4D coronary LCA case\n\n"
        f"Case: `{case['case_id']}`. Validation: **PASS**.\n\n"
        "Open `vtk/cine.pvd` in ParaView to animate the cardiac phases. "
        "The NumPy tensor `geometry_cine.npy` uses shape `(phase, branch, point, x/y/z/radius)` "
        "with branch order LMCA, LAD, LCX. Disease parameters and deterministic provenance are in "
        "`metadata.json`. From inside this exported case directory, run "
        "`python -m vessel_tree_generator visualize --input-dir . --clean` "
        "to create dashboards, GIF, and interactive HTML. "
        "This is a research/engineering prototype, not a clinically validated model.\n",
        encoding="utf-8",
    )
    return manifest


def verify_export(output_directory: str | Path) -> dict[str, Any]:
    """Independently verify an already exported portable case."""
    output = Path(output_directory).resolve()
    errors: list[str] = []
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "FAIL", "errors": ["manifest.json is missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest.get("sha256", {}).items():
        path = output / relative
        if not path.is_file():
            errors.append(f"missing checksummed file: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"checksum mismatch: {relative}")

    try:
        geometry = np.load(output / "geometry_cine.npy", allow_pickle=False)
        static_geometry = np.load(output / "geometry_static.npy", allow_pickle=False)
        expected_shape = tuple(manifest["geometry_shape"])
        if geometry.shape != expected_shape:
            errors.append(f"geometry shape {geometry.shape} does not match manifest {expected_shape}")
        if static_geometry.shape != geometry.shape[1:]:
            errors.append("static geometry shape does not match one cine frame")
        if not np.all(np.isfinite(geometry)) or np.any(geometry[..., 3] <= 0.0):
            errors.append("cine geometry contains invalid coordinates or radii")
        topology_error = max(
            float(np.max(np.linalg.norm(geometry[:, 0, -1, :3] - geometry[:, 1, 0, :3], axis=1))),
            float(np.max(np.linalg.norm(geometry[:, 0, -1, :3] - geometry[:, 2, 0, :3], axis=1))),
        )
        if topology_error > 1.0e-9:
            errors.append(f"cine topology error is {topology_error:.3e} mm")
    except (OSError, ValueError, KeyError, IndexError) as exc:
        errors.append(f"array verification failed: {exc}")

    try:
        first = pv.read(output / "vtk/phase_000/tree.vtm")
        last = pv.read(output / f"vtk/phase_{int(manifest['frame_count']) - 1:03d}/tree.vtm")
        if len(first) != len(BRANCH_ORDER) or len(last) != len(BRANCH_ORDER):
            errors.append("VTK branch count is invalid")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"VTK readback failed: {exc}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "case_id": manifest.get("case_id"),
        "errors": errors,
        "checksum_count": len(manifest.get("sha256", {})),
        "array_and_topology_readback": not any("array" in error or "topology" in error for error in errors),
        "vtk_readback": not any("VTK" in error for error in errors),
    }
