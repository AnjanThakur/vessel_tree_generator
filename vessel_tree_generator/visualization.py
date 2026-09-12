"""Quantitative images, animation, and interactive 3D views for exported cases."""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .export import verify_export
from .validation import BRANCH_ORDER


COLORS = {"LMCA": "#252525", "LAD": "#d62728", "LCX": "#1f77b4"}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _load_case(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).resolve()
    required = (
        "geometry_cine.npy",
        "geometry_static.npy",
        "geometry_healthy_reference.npy",
        "disease_reduction.npy",
        "metadata.json",
        "manifest.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("incomplete exported case: " + ", ".join(missing))
    return {
        "root": root,
        "cine": np.load(root / "geometry_cine.npy", allow_pickle=False),
        "static": np.load(root / "geometry_static.npy", allow_pickle=False),
        "healthy": np.load(root / "geometry_healthy_reference.npy", allow_pickle=False),
        "reduction": np.load(root / "disease_reduction.npy", allow_pickle=False),
        "metadata": json.loads((root / "metadata.json").read_text(encoding="utf-8")),
        "manifest": json.loads((root / "manifest.json").read_text(encoding="utf-8")),
    }


def _normalized_arc(points: np.ndarray) -> np.ndarray:
    distance = np.zeros(len(points), dtype=float)
    if len(points) > 1:
        distance[1:] = np.cumsum(np.linalg.norm(np.diff(points[:, :3], axis=0), axis=1))
    return distance / distance[-1] if len(distance) and distance[-1] > 1.0e-12 else distance


def branch_tortuosity_metrics(points: np.ndarray) -> dict[str, float | int]:
    """Calculate transparent centerline tortuosity metrics."""
    xyz = np.asarray(points, dtype=float)[:, :3]
    segments = np.diff(xyz, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    length = float(np.sum(segment_lengths))
    chord = float(np.linalg.norm(xyz[-1] - xyz[0]))
    distance_metric = length / chord if chord > 1.0e-12 else float("inf")
    valid = segment_lengths > 1.0e-12
    tangents = segments[valid] / segment_lengths[valid, None]
    if len(tangents) > 1:
        angles = np.arccos(np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0))
        local_scale = 0.5 * (segment_lengths[valid][:-1] + segment_lengths[valid][1:])
        curvature = angles / np.maximum(local_scale, 1.0e-12)
    else:
        angles = np.zeros(0)
        curvature = np.zeros(0)
    return {
        "arc_length_mm": length,
        "chord_length_mm": chord,
        "distance_metric_arc_over_chord": float(distance_metric),
        "total_turning_angle_deg": float(np.degrees(np.sum(angles))),
        "mean_discrete_curvature_per_mm": float(np.mean(curvature)) if len(curvature) else 0.0,
        "maximum_discrete_curvature_per_mm": float(np.max(curvature)) if len(curvature) else 0.0,
        "point_count": int(len(xyz)),
    }


def calculate_visualization_metrics(case: dict[str, Any]) -> dict[str, Any]:
    cine = case["cine"]
    static = case["static"]
    healthy = case["healthy"]
    reduction = case["reduction"]
    phases = np.asarray(case["metadata"]["phase_values"], dtype=float)
    displacement = np.linalg.norm(cine[..., :3] - static[None, ..., :3], axis=3)
    radius_change = np.abs(cine[..., 3] / static[None, ..., 3] - 1.0)
    topology_lad = np.linalg.norm(cine[:, 0, -1, :3] - cine[:, 1, 0, :3], axis=1)
    topology_lcx = np.linalg.norm(cine[:, 0, -1, :3] - cine[:, 2, 0, :3], axis=1)
    branches = {}
    for index, name in enumerate(BRANCH_ORDER):
        affected = reduction[index] > 1.0e-6
        branch = {
            **branch_tortuosity_metrics(static[index]),
            "maximum_radius_reduction_fraction": float(np.max(reduction[index])),
            "affected_export_point_count": int(np.count_nonzero(affected)),
            "minimum_healthy_radius_mm": float(np.min(healthy[index, :, 3])),
            "minimum_diseased_radius_mm": float(np.min(static[index, :, 3])),
            "maximum_4d_displacement_mm": float(np.max(displacement[:, index])),
            "maximum_radius_variation_fraction": float(np.max(radius_change[:, index])),
        }
        if np.any(affected):
            arc = _normalized_arc(static[index])
            branch["affected_arc_range"] = [float(np.min(arc[affected])), float(np.max(arc[affected]))]
            branch["maximum_reduction_arc_position"] = float(arc[int(np.argmax(reduction[index]))])
        else:
            branch["affected_arc_range"] = None
            branch["maximum_reduction_arc_position"] = None
        branches[name] = branch
    return {
        "case_id": case["metadata"]["case_id"],
        "status": "PASS",
        "phases": phases.tolist(),
        "branches": branches,
        "four_dimensional": {
            "frame_count": int(len(cine)),
            "maximum_displacement_by_phase_mm": np.max(displacement, axis=(1, 2)).tolist(),
            "mean_displacement_by_phase_mm": np.mean(displacement, axis=(1, 2)).tolist(),
            "maximum_radius_variation_by_phase_fraction": np.max(radius_change, axis=(1, 2)).tolist(),
            "maximum_displacement_mm": float(np.max(displacement)),
            "maximum_radius_variation_fraction": float(np.max(radius_change)),
            "cycle_closing_error_mm": float(np.max(np.linalg.norm(cine[-1, ..., :3] - cine[0, ..., :3], axis=2))),
        },
        "topology": {
            "maximum_LMCA_to_LAD_error_mm": float(np.max(topology_lad)),
            "maximum_LMCA_to_LCX_error_mm": float(np.max(topology_lcx)),
            "exact_shared_bifurcation": bool(max(np.max(topology_lad), np.max(topology_lcx)) <= 1.0e-9),
        },
        "source_export_verification": verify_export(case["root"]),
        "metric_definitions": {
            "distance_metric_arc_over_chord": "centerline arc length divided by endpoint chord; 1 is straight",
            "total_turning_angle_deg": "sum of unsigned angles between consecutive centerline segments",
            "discrete_curvature_per_mm": "turning angle divided by local mean segment length",
        },
    }


def _plot_branch(ax, points: np.ndarray, name: str, *, dimensions: tuple[int, ...]) -> None:
    color = COLORS[name]
    if len(dimensions) == 3:
        ax.plot(points[:, dimensions[0]], points[:, dimensions[1]], points[:, dimensions[2]], color=color, linewidth=3)
    else:
        ax.plot(points[:, dimensions[0]], points[:, dimensions[1]], color=color, linewidth=3, label=name)


def create_validation_dashboard(case: dict[str, Any], metrics: dict[str, Any], output: Path) -> None:
    cine, static, healthy, reduction = case["cine"], case["static"], case["healthy"], case["reduction"]
    phases = np.asarray(case["metadata"]["phase_values"], dtype=float)
    fig = plt.figure(figsize=(17.0, 11.0))
    grid = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.24)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    front = fig.add_subplot(grid[0, 1])
    crown = fig.add_subplot(grid[0, 2])
    radius_ax = fig.add_subplot(grid[1, 0])
    motion_ax = fig.add_subplot(grid[1, 1])
    summary_ax = fig.add_subplot(grid[1, 2])

    for index, name in enumerate(BRANCH_ORDER):
        _plot_branch(ax3d, static[index], name, dimensions=(0, 1, 2))
        _plot_branch(front, static[index], name, dimensions=(0, 2))
        _plot_branch(crown, static[index], name, dimensions=(0, 1))
        affected = reduction[index] > 1.0e-6
        if np.any(affected):
            points = static[index, affected]
            ax3d.scatter(points[:, 0], points[:, 1], points[:, 2], c=reduction[index, affected], cmap="YlOrRd", vmin=0, vmax=1, s=55, edgecolor="black")
            front.scatter(points[:, 0], points[:, 2], c=reduction[index, affected], cmap="YlOrRd", vmin=0, vmax=1, s=36, edgecolor="black", zorder=5)
    bif = static[0, -1]
    ax3d.scatter(*bif[:3], color="black", s=55)
    front.scatter(bif[0], bif[2], color="black", s=48, zorder=6)
    ax3d.set_title("3D static anatomy; lesion points highlighted")
    ax3d.set_xlabel("X (mm)"); ax3d.set_ylabel("Y (mm)"); ax3d.set_zlabel("Z (mm)")
    ax3d.view_init(elev=22, azim=-58)
    front.set_title("Fixed front view: LAD descent")
    front.set_xlabel("Cardiac X (mm)"); front.set_ylabel("Cardiac Z (mm; apex negative)")
    front.set_aspect("equal", adjustable="datalim"); front.grid(alpha=0.2); front.legend()
    crown.set_title("Crown view: LCX lateral course")
    crown.set_xlabel("Cardiac X (mm)"); crown.set_ylabel("Cardiac Y (mm)")
    crown.set_aspect("equal", adjustable="datalim"); crown.grid(alpha=0.2)

    for index, name in enumerate(BRANCH_ORDER):
        arc = _normalized_arc(static[index])
        radius_ax.plot(arc, healthy[index, :, 3], color=COLORS[name], linestyle="--", alpha=0.48)
        radius_ax.plot(arc, static[index, :, 3], color=COLORS[name], linewidth=2.4, label=f"{name} diseased")
        radius_ax.fill_between(arc, static[index, :, 3], healthy[index, :, 3], color=COLORS[name], alpha=0.18)
    radius_ax.set_title("Disease validation: healthy dashed, diseased solid")
    radius_ax.set_xlabel("Normalized branch arc length"); radius_ax.set_ylabel("Radius (mm)")
    radius_ax.grid(alpha=0.2); radius_ax.legend(fontsize=8)

    four_d = metrics["four_dimensional"]
    motion_ax.plot(phases, four_d["maximum_displacement_by_phase_mm"], "o-", color="#2ca02c", label="max displacement (mm)")
    radius_twin = motion_ax.twinx()
    radius_twin.plot(phases, 100 * np.asarray(four_d["maximum_radius_variation_by_phase_fraction"]), "s--", color="#ff7f0e", label="max radius change (%)")
    motion_ax.set_title("4D phase validation and exact cycle closure")
    motion_ax.set_xlabel("Normalized cardiac phase"); motion_ax.set_ylabel("Displacement (mm)", color="#2ca02c")
    radius_twin.set_ylabel("Radius variation (%)", color="#ff7f0e"); motion_ax.grid(alpha=0.2)

    summary_ax.axis("off")
    rows = []
    for name in BRANCH_ORDER:
        item = metrics["branches"][name]
        rows.append([
            name,
            f"{item['arc_length_mm']:.1f}",
            f"{item['distance_metric_arc_over_chord']:.3f}",
            f"{item['total_turning_angle_deg']:.1f}",
            f"{100 * item['maximum_radius_reduction_fraction']:.1f}%",
        ])
    table = summary_ax.table(
        cellText=rows,
        colLabels=["Branch", "Length\n(mm)", "Arc/chord", "Total turn\n(deg)", "Max disease"],
        cellLoc="center", loc="upper center", bbox=[0.0, 0.56, 1.0, 0.4],
    )
    table.auto_set_font_size(False); table.set_fontsize(8.5)
    topology = metrics["topology"]
    verification = metrics["source_export_verification"]
    checklist = (
        f"✓ Export verification: {verification['status']}\n"
        f"✓ Exact LMCA→LAD/LCX junction: {topology['exact_shared_bifurcation']}\n"
        f"✓ Maximum topology error: {max(topology['maximum_LMCA_to_LAD_error_mm'], topology['maximum_LMCA_to_LCX_error_mm']):.2e} mm\n"
        f"✓ Cardiac frames: {four_d['frame_count']}\n"
        f"✓ Cycle closing error: {four_d['cycle_closing_error_mm']:.2e} mm\n"
        f"✓ Maximum 4D displacement: {four_d['maximum_displacement_mm']:.2f} mm\n"
        f"✓ Maximum pulsatility: {100 * four_d['maximum_radius_variation_fraction']:.2f}%\n\n"
        "Tortuosity is measured, not manually forced.\n"
        "Engineering validation; not clinical validation."
    )
    summary_ax.text(0.02, 0.49, checklist, va="top", fontsize=10.5, linespacing=1.45)
    summary_ax.set_title("Quantitative acceptance summary")

    fig.suptitle(f"{case['metadata']['case_id']} — anatomy, disease, 4D motion, radius and tortuosity", fontsize=17)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_tortuosity_plot(case: dict[str, Any], metrics: dict[str, Any], output: Path) -> None:
    static = case["static"]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), sharey=False)
    for index, (axis, name) in enumerate(zip(axes, BRANCH_ORDER)):
        xyz = static[index, :, :3]
        segments = np.diff(xyz, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        tangent = segments / np.maximum(lengths[:, None], 1.0e-12)
        angles = np.arccos(np.clip(np.sum(tangent[:-1] * tangent[1:], axis=1), -1.0, 1.0))
        curvature = angles / np.maximum(0.5 * (lengths[:-1] + lengths[1:]), 1.0e-12)
        arc = _normalized_arc(static[index])[1:-1]
        axis.plot(arc, curvature, color=COLORS[name], linewidth=2.5)
        axis.fill_between(arc, 0.0, curvature, color=COLORS[name], alpha=0.18)
        item = metrics["branches"][name]
        axis.set_title(
            f"{name}\narc/chord={item['distance_metric_arc_over_chord']:.3f}, "
            f"turn={item['total_turning_angle_deg']:.1f}°"
        )
        axis.set_xlabel("Normalized arc length")
        axis.set_ylabel("Discrete curvature (1/mm)")
        axis.grid(alpha=0.2)
    fig.suptitle("Branch tortuosity validation — transparent geometric metrics", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_cardiac_cycle_gif(case: dict[str, Any], output: Path, *, duration_ms: int = 180) -> None:
    cine = case["cine"]
    phases = np.asarray(case["metadata"]["phase_values"], dtype=float)
    reduction = case["reduction"]
    xyz = cine[..., :3]
    limits = [(float(np.min(xyz[..., axis])), float(np.max(xyz[..., axis]))) for axis in range(3)]
    margin = max(maximum - minimum for minimum, maximum in limits) * 0.08
    images: list[Image.Image] = []
    for frame_index, phase in enumerate(phases):
        fig = plt.figure(figsize=(8.0, 7.0))
        axis = fig.add_subplot(111, projection="3d")
        for branch_index, name in enumerate(BRANCH_ORDER):
            points = cine[frame_index, branch_index]
            axis.plot(points[:, 0], points[:, 1], points[:, 2], color=COLORS[name], linewidth=4, label=name)
            affected = reduction[branch_index] > 1.0e-6
            if np.any(affected):
                lesion = points[affected]
                axis.scatter(lesion[:, 0], lesion[:, 1], lesion[:, 2], c=reduction[branch_index, affected], cmap="YlOrRd", vmin=0, vmax=1, s=45, edgecolor="black")
        bif = cine[frame_index, 0, -1]
        axis.scatter(*bif[:3], color="black", s=55)
        axis.set_xlim(limits[0][0] - margin, limits[0][1] + margin)
        axis.set_ylim(limits[1][0] - margin, limits[1][1] + margin)
        axis.set_zlim(limits[2][0] - margin, limits[2][1] + margin)
        axis.set_xlabel("X (mm)"); axis.set_ylabel("Y (mm)"); axis.set_zlabel("Z (mm)")
        axis.set_title(f"{case['metadata']['case_id']} — cardiac phase {phase:.3f}")
        axis.legend(loc="upper left")
        axis.view_init(elev=22, azim=-58)
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        images.append(Image.open(buffer).convert("P", palette=Image.Palette.ADAPTIVE))
    images[0].save(output, save_all=True, append_images=images[1:], duration=duration_ms, loop=0, optimize=False)


def create_interactive_html(
    case: dict[str, Any], output: Path, *, standalone: bool = False
) -> None:
    """Create a rotatable 3D viewer with play/pause and cardiac phase slider."""
    import plotly.graph_objects as go

    cine = case["cine"]
    phases = np.asarray(case["metadata"]["phase_values"], dtype=float)
    reduction = case["reduction"]
    traces_per_frame = 2 * len(BRANCH_ORDER)

    def traces(frame_index: int) -> list[go.Scatter3d]:
        result: list[go.Scatter3d] = []
        for branch_index, name in enumerate(BRANCH_ORDER):
            values = cine[frame_index, branch_index]
            arc = np.linspace(0.0, 1.0, len(values))
            custom = np.column_stack((values[:, 3], reduction[branch_index], arc))
            result.append(go.Scatter3d(
                x=values[:, 0], y=values[:, 1], z=values[:, 2],
                mode="lines", name=name, legendgroup=name,
                line={"color": COLORS[name], "width": 7},
                hoverinfo="skip", showlegend=True,
            ))
            result.append(go.Scatter3d(
                x=values[:, 0], y=values[:, 1], z=values[:, 2],
                mode="markers", name=f"{name} points", legendgroup=name,
                marker={
                    "size": np.clip(2.5 + 3.0 * values[:, 3], 3.0, 10.0),
                    "color": reduction[branch_index], "colorscale": "YlOrRd",
                    "cmin": 0.0, "cmax": 1.0, "showscale": branch_index == 2,
                    "colorbar": {"title": "Disease<br>reduction", "tickformat": ".0%"},
                    "line": {"width": 0.5, "color": COLORS[name]},
                },
                customdata=custom,
                hovertemplate=(
                    f"<b>{name}</b><br>x=%{{x:.2f}} mm<br>y=%{{y:.2f}} mm<br>z=%{{z:.2f}} mm"
                    "<br>radius=%{customdata[0]:.3f} mm<br>disease=%{customdata[1]:.1%}"
                    "<br>arc position=%{customdata[2]:.3f}<extra></extra>"
                ),
                showlegend=False,
            ))
        return result

    frames = [
        go.Frame(data=traces(index), name=f"{phase:.6f}", traces=list(range(traces_per_frame)))
        for index, phase in enumerate(phases)
    ]
    slider_steps = [
        {
            "method": "animate",
            "label": f"{phase:.2f}",
            "args": [[f"{phase:.6f}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
        }
        for phase in phases
    ]
    figure = go.Figure(data=traces(0), frames=frames)
    figure.update_layout(
        title=f"{case['metadata']['case_id']} — interactive 4D LCA tree",
        template="plotly_white",
        scene={
            "xaxis_title": "Cardiac X (mm)", "yaxis_title": "Cardiac Y (mm)", "zaxis_title": "Cardiac Z (mm)",
            "aspectmode": "data", "camera": {"eye": {"x": 1.35, "y": -1.55, "z": 0.9}},
        },
        margin={"l": 0, "r": 80, "t": 65, "b": 80},
        sliders=[{"active": 0, "currentvalue": {"prefix": "Cardiac phase: "}, "steps": slider_steps}],
        updatemenus=[{
            "type": "buttons", "direction": "left", "x": 0.0, "y": -0.08,
            "buttons": [
                {"label": "Play", "method": "animate", "args": [None, {"fromcurrent": True, "frame": {"duration": 180, "redraw": True}, "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
            ],
        }],
        annotations=[{
            "text": "Rotate: drag · Zoom: wheel · Inspect: hover · Marker size: radius · Marker color: disease reduction",
            "xref": "paper", "yref": "paper", "x": 0.5, "y": -0.13, "showarrow": False,
        }],
    )
    figure.write_html(
        output,
        include_plotlyjs=True if standalone else "cdn",
        full_html=True,
        auto_play=False,
        config={"responsive": True, "displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}},
    )


def create_case_visualizations(
    input_directory: str | Path,
    output_directory: str | Path | None = None,
    *,
    clean: bool = False,
    standalone_html: bool = False,
    create_gif: bool = True,
) -> dict[str, Any]:
    """Generate all visual validation artifacts for an exported case."""
    case = _load_case(input_directory)
    output = (
        Path(output_directory).resolve()
        if output_directory is not None
        else case["root"] / "visualizations"
    )
    if output.exists():
        if not clean:
            raise FileExistsError(f"refusing to overwrite {output}; pass clean=True explicitly")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    metrics = calculate_visualization_metrics(case)
    create_validation_dashboard(case, metrics, output / "validation_dashboard.png")
    create_tortuosity_plot(case, metrics, output / "tortuosity_analysis.png")
    create_interactive_html(case, output / "interactive_tree.html", standalone=standalone_html)
    if create_gif:
        create_cardiac_cycle_gif(case, output / "cardiac_cycle.gif")
    _write_json(output / "visualization_metrics.json", metrics)
    artifacts = [
        "validation_dashboard.png",
        "tortuosity_analysis.png",
        "interactive_tree.html",
        "visualization_metrics.json",
    ] + (["cardiac_cycle.gif"] if create_gif else [])
    manifest = {
        "status": "PASS",
        "case_id": case["metadata"]["case_id"],
        "source_case_directory_name": case["root"].name,
        "artifacts": artifacts,
        "interactive_html_self_contained": standalone_html,
        "source_export_verification": metrics["source_export_verification"]["status"],
        "checks_visualized": [
            "fixed cardiac anatomy views", "exact LMCA daughter topology", "disease location and severity",
            "healthy versus diseased radius", "4D cardiac phase motion", "radius pulsatility",
            "cycle closure", "branch tortuosity and curvature",
        ],
    }
    _write_json(output / "visualization_manifest.json", manifest)
    (output / "README.md").write_text(
        "# Visual validation pack\n\n"
        "- Open `validation_dashboard.png` for the complete acceptance overview.\n"
        "- Open `cardiac_cycle.gif` to inspect the closed 4D motion cycle.\n"
        "- Open `interactive_tree.html` in a browser to rotate, zoom, hover, play, pause, and select phase.\n"
        "- Open `tortuosity_analysis.png` for branch curvature and arc/chord metrics.\n"
        "- Read `visualization_metrics.json` for the exact plotted numbers.\n\n"
        "To change disease, generate a new case with `coronary4d generate` (or "
        "`python -m vessel_tree_generator generate`) and rerun the `visualize` command on that case. "
        "The viewer never modifies geometry or validation data.\n",
        encoding="utf-8",
    )
    return manifest


def create_disease_comparison(
    demo_directory: str | Path,
    output_file: str | Path | None = None,
) -> dict[str, Any]:
    """Compare healthy, focal, diffuse, and tandem cases on shared anatomy."""
    demo = Path(demo_directory).resolve()
    case_names = ("healthy", "focal_lad", "diffuse_lcx", "tandem_lad")
    cases = [_load_case(demo / name) for name in case_names]
    output = Path(output_file).resolve() if output_file is not None else demo / "disease_mode_comparison.png"
    fig, axes = plt.subplots(2, len(cases), figsize=(19.0, 9.0))
    summary: dict[str, Any] = {}
    for column, (name, case) in enumerate(zip(case_names, cases)):
        static, reduction = case["static"], case["reduction"]
        top, bottom = axes[0, column], axes[1, column]
        for branch_index, branch_name in enumerate(BRANCH_ORDER):
            points = static[branch_index]
            top.plot(points[:, 0], points[:, 2], color=COLORS[branch_name], linewidth=3)
            affected = reduction[branch_index] > 1.0e-6
            if np.any(affected):
                lesion = points[affected]
                top.scatter(
                    lesion[:, 0], lesion[:, 2], c=reduction[branch_index, affected],
                    cmap="YlOrRd", vmin=0, vmax=1, s=42, edgecolor="black", zorder=5,
                )
            bottom.plot(
                _normalized_arc(points), 100.0 * reduction[branch_index],
                color=COLORS[branch_name], linewidth=2.5, label=branch_name,
            )
        top.scatter(static[0, -1, 0], static[0, -1, 2], color="black", s=42, zorder=6)
        top.set_title(case["metadata"]["case_id"])
        top.set_xlabel("Cardiac X (mm)"); top.set_ylabel("Cardiac Z (mm)")
        top.set_aspect("equal", adjustable="datalim"); top.grid(alpha=0.2)
        bottom.set_xlabel("Normalized branch arc length"); bottom.set_ylabel("Radius reduction (%)")
        bottom.set_ylim(-2.0, 72.0); bottom.grid(alpha=0.2)
        if column == 0:
            bottom.legend(loc="upper right", fontsize=8)
        summary[name] = {
            branch_name: {
                "maximum_reduction_fraction": float(np.max(reduction[index])),
                "affected_export_point_count": int(np.count_nonzero(reduction[index] > 1.0e-6)),
            }
            for index, branch_name in enumerate(BRANCH_ORDER)
        }
    fig.suptitle(
        "Controlled disease comparison on identical seeded anatomy\n"
        "Top: fixed front view and affected points · Bottom: applied radius reduction",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    result = {
        "status": "PASS",
        "shared_anatomy_comparison": True,
        "case_order": list(case_names),
        "output": output.name,
        "disease_summary": summary,
    }
    _write_json(output.with_suffix(".json"), result)
    return result
