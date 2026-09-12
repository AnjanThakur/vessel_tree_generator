"""Publication-quality visualization for the Stage-1 support-surface evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import pyvista as pv

from heart_support_surface import HeartSupportSurface


COLORS = {
    "lmca": "#202124",
    "lad": "#D62828",
    "lcx": "#1976B9",
    "rca": "#7656A5",
    "bifurcation": "#2CA02C",
    "crown": "#00A6C7",
    "long_axis": "#E87500",
    "surface": "#C96F67",
    "coronary_plane": "#8ECAE6",
    "lad_plane": "#F4C27A",
    "ink": "#17222D",
    "muted": "#5C6873",
    "grid": "#D9E0E6",
}


FIGURE_NAMES = [
    "01_original_dataset_centerlines",
    "02_centroid_svd_measurement_planes",
    "03_two_ellipse_anatomical_scaffold",
    "04_3d_parametric_anatomical_support_surface",
    "05_unchanged_centerlines_on_support_surface",
    "06_superior_crown_view",
    "07_front_long_axis_view",
    "08_lateral_view",
    "09_observed_vs_extrapolated_reference_geometry",
    "10_source_to_surface_distance_diagnostics",
    "11_stage1_technical_validation_summary",
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": COLORS["muted"],
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, directory: Path, stem: str, dpi: int) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    svg = directory / f"{stem}.svg"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return {"name": stem, "png": str(png.resolve()), "svg": str(svg.resolve()), "dpi": dpi}


def equal_3d(axis: Any, groups: Iterable[np.ndarray]) -> None:
    available = [np.asarray(group) for group in groups if len(group)]
    points = np.vstack(available)
    low, high = points.min(axis=0), points.max(axis=0)
    center = 0.5 * (low + high)
    radius = max(0.55 * float(np.max(high - low)), 1.0)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))


def configure_3d(axis: Any, title: str, *, elev: float = 19.0, azim: float = -62.0) -> None:
    axis.set_proj_type("ortho")
    axis.view_init(elev=elev, azim=azim)
    axis.set_xlabel("Canonical X · crown")
    axis.set_ylabel("Canonical Y · crown depth")
    axis.set_zlabel("Canonical Z · superior (+) / apex (−)")
    axis.set_title(title, pad=16)
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_alpha(0.04)


def draw_sources_3d(axis: Any, source: dict[str, np.ndarray], *, linewidth: float = 2.8) -> None:
    labels = {"lmca": "LMCA", "lad": "LAD", "lcx": "LCX", "rca": "inferred RCA candidate"}
    for name in ("lmca", "lad", "lcx", "rca"):
        axis.plot(
            *source[name].T,
            color=COLORS[name],
            linewidth=linewidth if name != "rca" else linewidth - 0.6,
            linestyle="--" if name == "rca" else "-",
            label=labels[name],
            zorder=5,
        )
    axis.scatter(0.0, 0.0, 0.0, s=68, color=COLORS["bifurcation"], edgecolor="white", linewidth=0.8, label="LMCA bifurcation", zorder=10)


def draw_surface_3d(axis: Any, surface: HeartSupportSurface, *, alpha: float = 0.24) -> np.ndarray:
    theta = np.linspace(-np.pi, np.pi, 100)
    phi = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 54)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    points = surface.surface_point(theta_grid, phi_grid)
    axis.plot_surface(
        points[..., 0],
        points[..., 1],
        points[..., 2],
        color=COLORS["surface"],
        alpha=alpha,
        linewidth=0.15,
        edgecolor="#A95650",
        rstride=3,
        cstride=4,
        shade=True,
        zorder=1,
    )
    return points.reshape(-1, 3)


def masked_ellipse_points(mesh: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(mesh.points)
    mask = np.asarray(mesh.point_data["observed_source_supported_arc"], dtype=bool)
    return points, mask


def draw_masked_ellipse_3d(axis: Any, mesh: pv.PolyData, color: str, label: str) -> None:
    points, mask = masked_ellipse_points(mesh)
    axis.plot(*points.T, color=color, linewidth=1.7, linestyle="--", alpha=0.72, label=f"{label} · extrapolated")
    supported = points.copy()
    supported[~mask] = np.nan
    axis.plot(*supported.T, color=color, linewidth=4.1, label=f"{label} · source-supported")


def draw_axes_3d(axis: Any, surface: HeartSupportSurface) -> list[np.ndarray]:
    a, b, c = surface.semi_axes
    center = surface.origin
    x = surface.crown_axis
    y = surface.secondary_crown_axis
    apex = surface.apex_axis
    crown_line = np.vstack((center - a * x, center + a * x))
    depth_line = np.vstack((center - b * y, center + b * y))
    apex_line = np.vstack((center - c * apex, center + c * apex))
    axis.plot(*crown_line.T, color="#0072B2", linewidth=2.3, label="crown axis")
    axis.plot(*depth_line.T, color="#4C8C4A", linewidth=1.9, label="secondary crown axis")
    axis.plot(*apex_line.T, color="#D55E00", linewidth=2.3, label="superior–apex axis")
    return [crown_line, depth_line, apex_line]


def add_external_legend(axis: Any, *, columns: int = 1) -> None:
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5, ncol=columns)


def plane_polygon(axis: Any, mesh: pv.PolyData, color: str, label: str) -> np.ndarray:
    points = np.asarray(mesh.points)
    if len(mesh.faces) >= 5:
        face = mesh.faces[1 : 1 + int(mesh.faces[0])]
        polygon = points[face]
    else:
        polygon = points
    collection = Poly3DCollection([polygon], facecolor=color, edgecolor=COLORS["muted"], linewidth=0.8, alpha=0.34, label=label)
    axis.add_collection3d(collection)
    return polygon


def line_segments_with_mask(points: np.ndarray, mask: np.ndarray) -> tuple[LineCollection, LineCollection]:
    segments = np.stack((points[:-1], points[1:]), axis=1)
    segment_mask = mask[:-1] & mask[1:]
    solid = LineCollection(segments[segment_mask], linewidths=4.0)
    dashed = LineCollection(segments[~segment_mask], linewidths=1.7, linestyles="--", alpha=0.72)
    return solid, dashed


def generate_stage1_figures(
    *,
    case_root: Path,
    case_id: str,
    source: dict[str, np.ndarray],
    crown_support: dict[str, np.ndarray],
    planes: dict[str, pv.PolyData],
    plane_records: dict[str, dict[str, np.ndarray | float]],
    ellipses: dict[str, pv.PolyData],
    surface: HeartSupportSurface,
    distance_diagnostics: dict[str, dict[str, Any]],
    validation_summary: dict[str, Any],
    dpi: int = 300,
) -> list[dict[str, Any]]:
    configure_style()
    output = case_root / "06_diagnostics"
    records: list[dict[str, Any]] = []

    # 01 · immutable source proof.
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    draw_sources_3d(ax, source)
    equal_3d(ax, source.values())
    configure_3d(ax, "Original Dataset Centerlines")
    fig.text(0.5, 0.02, f"{case_id} · one shared rigid canonical display transform · no vessel deformation", ha="center", color=COLORS["muted"])
    add_external_legend(ax)
    records.append(save_figure(fig, output, FIGURE_NAMES[0], dpi))

    # 02 · PDF-aligned anatomical plane proof.  The ovals are bounded,
    # support-derived displays of the infinite centroid-SVD planes.
    fig = plt.figure(figsize=(12.8, 7.2))
    ax_crown = fig.add_subplot(121, projection="3d")
    ax_lad = fig.add_subplot(122, projection="3d")

    crown_polygon = plane_polygon(
        ax_crown,
        planes["coronary"],
        COLORS["coronary_plane"],
        "coronary plane · LCX + inferred RCA crown ring",
    )
    ax_crown.plot(*crown_support["lcx"].T, color=COLORS["lcx"], linewidth=2.8, label="LCX · AV-groove/crown support")
    ax_crown.plot(*crown_support["rca"].T, color=COLORS["rca"], linewidth=2.5, linestyle="--", label="inferred RCA crown support")
    crown_centroid = np.asarray(plane_records["coronary"]["centroid"])
    crown_normal = np.asarray(plane_records["coronary"]["normal"])
    ax_crown.scatter(*crown_centroid, s=78, marker="X", color="#0077A8", edgecolor="white", linewidth=0.8, label="fitted centroid")
    ax_crown.quiver(*crown_centroid, *(20.0 * crown_normal), color="#0077A8", linewidth=2.2, arrow_length_ratio=0.15)
    equal_3d(ax_crown, [crown_polygon, crown_support["rca"], crown_support["lcx"]])
    configure_3d(ax_crown, "Coronary plane · crown / AV-groove role", elev=25.0, azim=-58.0)
    ax_crown.title.set_fontsize(13.5)
    ax_crown.set(xlabel="", ylabel="", zlabel="")
    ax_crown.set_xticklabels([])
    ax_crown.set_yticklabels([])
    ax_crown.set_zticklabels([])
    ax_crown.legend(loc="lower center", bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=7.2)

    crown_polygon_right = plane_polygon(
        ax_lad,
        planes["coronary"],
        COLORS["coronary_plane"],
        "coronary plane",
    )
    lad_polygon = plane_polygon(
        ax_lad,
        planes["lad"],
        COLORS["lad_plane"],
        "interventricular plane · LAD descent",
    )
    ax_lad.plot(*source["lad"].T, color=COLORS["lad"], linewidth=2.7, label="LAD · descent toward apex")
    ax_lad.plot(*crown_support["lcx"].T, color=COLORS["lcx"], linewidth=1.8, alpha=0.75, label="LCX crown support")
    ax_lad.plot(*crown_support["rca"].T, color=COLORS["rca"], linewidth=1.6, linestyle="--", alpha=0.75, label="inferred RCA crown support")
    for key, color in (("coronary", "#0077A8"), ("lad", "#C06400")):
        centroid = np.asarray(plane_records[key]["centroid"])
        normal = np.asarray(plane_records[key]["normal"])
        ax_lad.scatter(*centroid, s=66, marker="X", color=color, edgecolor="white", linewidth=0.7)
        ax_lad.quiver(*centroid, *(18.0 * normal), color=color, linewidth=1.9, arrow_length_ratio=0.15)
    equal_3d(ax_lad, [crown_polygon_right, lad_polygon, source["lad"], crown_support["rca"], crown_support["lcx"]])
    configure_3d(ax_lad, "Interventricular plane · LAD to apex", elev=22.0, azim=-62.0)
    ax_lad.title.set_fontsize(13.5)
    ax_lad.set(xlabel="", ylabel="", zlabel="")
    ax_lad.set_xticklabels([])
    ax_lad.set_yticklabels([])
    ax_lad.set_zticklabels([])
    ax_lad.legend(loc="lower center", bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=7.0)

    fig.suptitle("Centroid-SVD Anatomical Measurement Planes", fontsize=22, fontweight="bold", y=0.992)
    fig.text(
        0.5,
        0.012,
        "Bounded ovals show support-derived plane extent · mathematical plane: C = mean(pᵢ), n = Vᵀ[-1], n · (x − C) = 0",
        ha="center",
        color=COLORS["muted"],
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.16, wspace=0.05)
    records.append(save_figure(fig, output, FIGURE_NAMES[1], dpi))

    # 03 · two reference curves only.
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    draw_masked_ellipse_3d(ax, ellipses["crown"], COLORS["crown"], "coronary/crown reference")
    draw_masked_ellipse_3d(ax, ellipses["long_axis"], COLORS["long_axis"], "long-axis/apical reference")
    axis_lines = draw_axes_3d(ax, surface)
    ax.scatter(0, 0, 0, s=68, color=COLORS["bifurcation"], edgecolor="white", label="bifurcation")
    ellipse_points = [ellipses["crown"].points, ellipses["long_axis"].points]
    equal_3d(ax, ellipse_points + axis_lines)
    configure_3d(ax, "Two-Ellipse Anatomical Scaffold")
    add_external_legend(ax)
    records.append(save_figure(fig, output, FIGURE_NAMES[2], dpi))

    # 04 · curves converted to a full 3D surface.
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surface_points = draw_surface_3d(ax, surface, alpha=0.32)
    draw_masked_ellipse_3d(ax, ellipses["crown"], COLORS["crown"], "coronary/crown reference")
    draw_masked_ellipse_3d(ax, ellipses["long_axis"], COLORS["long_axis"], "long-axis/apical reference")
    axis_lines = draw_axes_3d(ax, surface)
    equal_3d(ax, [surface_points, ellipses["crown"].points, ellipses["long_axis"].points] + axis_lines)
    configure_3d(ax, "3D Parametric Anatomical Support Surface")
    fig.text(0.5, 0.02, "Dataset-derived parametric anatomical support scaffold · not a patient-specific myocardial reconstruction", ha="center", color=COLORS["muted"])
    add_external_legend(ax)
    records.append(save_figure(fig, output, FIGURE_NAMES[3], dpi))

    # 05 · unchanged source trajectories relative to surface.
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    surface_points = draw_surface_3d(ax, surface, alpha=0.19)
    draw_sources_3d(ax, source, linewidth=2.9)
    equal_3d(ax, [surface_points] + list(source.values()))
    configure_3d(ax, "Unchanged Coronary Centerlines on the Parametric Support Surface")
    fig.text(0.5, 0.02, "Distances are diagnostic only; no centerline is snapped, projected, or deformed", ha="center", color=COLORS["muted"])
    add_external_legend(ax)
    records.append(save_figure(fig, output, FIGURE_NAMES[4], dpi))

    # 06 · fixed superior/crown orthographic view.
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    a, b, c = surface.semi_axes
    center = surface.origin
    ax.add_patch(Ellipse(center[:2], 2 * a, 2 * b, facecolor=COLORS["surface"], edgecolor="#A95650", alpha=0.18, linewidth=1.2, label="support-surface crown projection"))
    for name, label in (("lcx", "LCX"), ("rca", "inferred RCA candidate")):
        ax.plot(source[name][:, 0], source[name][:, 1], color=COLORS[name], linewidth=3.0 if name == "lcx" else 2.2, linestyle="--" if name == "rca" else "-", label=label)
    crown_points, crown_mask = masked_ellipse_points(ellipses["crown"])
    ax.plot(crown_points[:, 0], crown_points[:, 1], color=COLORS["crown"], linestyle="--", alpha=0.72, linewidth=1.6)
    supported = crown_points.copy(); supported[~crown_mask] = np.nan
    ax.plot(supported[:, 0], supported[:, 1], color=COLORS["crown"], linewidth=3.8, label="coronary/crown reference")
    ax.scatter(0, 0, s=62, color=COLORS["bifurcation"], edgecolor="white", label="bifurcation", zorder=10)
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("Canonical X · crown"); ax.set_ylabel("Canonical Y · crown depth")
    ax.set_title("Superior / Crown View")
    add_external_legend(ax)
    records.append(save_figure(fig, output, FIGURE_NAMES[5], dpi))

    # 07 · fixed front/long-axis view.
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    ax.add_patch(Ellipse((center[0], center[2]), 2 * a, 2 * c, facecolor=COLORS["surface"], edgecolor="#A95650", alpha=0.18, linewidth=1.2, label="support-surface front projection"))
    ax.plot(source["lad"][:, 0], source["lad"][:, 2], color=COLORS["lad"], linewidth=3.2, label="LAD")
    long_points, long_mask = masked_ellipse_points(ellipses["long_axis"])
    ax.plot(long_points[:, 0], long_points[:, 2], color=COLORS["long_axis"], linestyle="--", alpha=0.72, linewidth=1.6)
    supported = long_points.copy(); supported[~long_mask] = np.nan
    ax.plot(supported[:, 0], supported[:, 2], color=COLORS["long_axis"], linewidth=3.8, label="long-axis/apical reference")
    ax.annotate("apex direction", xy=(center[0], center[2] - c), xytext=(center[0] + 0.30 * a, center[2] - 0.68 * c), arrowprops={"arrowstyle": "->", "color": COLORS["long_axis"], "lw": 1.8}, color=COLORS["long_axis"], weight="bold")
    ax.scatter(0, 0, s=62, color=COLORS["bifurcation"], edgecolor="white", label="bifurcation", zorder=10)
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("Canonical X · crown"); ax.set_ylabel("Canonical Z · superior (+) / apex (−)")
    ax.set_title("Front / Long-Axis View")
    add_external_legend(ax)
    records.append(save_figure(fig, output, FIGURE_NAMES[6], dpi))

    # 08 · fixed lateral view.
    fig, ax = plt.subplots(figsize=(12.0, 7.2))
    ax.add_patch(Ellipse((center[1], center[2]), 2 * b, 2 * c, facecolor=COLORS["surface"], edgecolor="#A95650", alpha=0.18, linewidth=1.2, label="support-surface lateral projection"))
    for name, label in (("lmca", "LMCA"), ("lad", "LAD"), ("lcx", "LCX"), ("rca", "inferred RCA candidate")):
        ax.plot(source[name][:, 1], source[name][:, 2], color=COLORS[name], linewidth=2.8 if name != "rca" else 2.0, linestyle="--" if name == "rca" else "-", label=label)
    ax.scatter(0, 0, s=62, color=COLORS["bifurcation"], edgecolor="white", label="bifurcation", zorder=10)
    ax.set_aspect("equal", adjustable="datalim"); ax.set_xlabel("Canonical Y · crown depth"); ax.set_ylabel("Canonical Z · superior (+) / apex (−)")
    ax.set_title("Lateral View")
    add_external_legend(ax)
    records.append(save_figure(fig, output, FIGURE_NAMES[7], dpi))

    # 09 · direct visual grammar for observed vs extrapolated arcs.
    fig = plt.figure(figsize=(12.8, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    draw_masked_ellipse_3d(ax, ellipses["crown"], COLORS["crown"], "coronary/crown reference")
    draw_masked_ellipse_3d(ax, ellipses["long_axis"], COLORS["long_axis"], "long-axis/apical reference")
    equal_3d(ax, [ellipses["crown"].points, ellipses["long_axis"].points])
    configure_3d(ax, "Observed vs Extrapolated Reference Geometry")
    fig.text(0.5, 0.02, "solid = source-supported interval  ·  dashed = extrapolated parametric reference", ha="center", weight="bold", color=COLORS["ink"])
    add_external_legend(ax)
    records.append(save_figure(fig, output, FIGURE_NAMES[8], dpi))

    # 10 · source-to-surface diagnostic only.
    fig, ax = plt.subplots(figsize=(12.4, 7.0))
    labels = {"lmca": "LMCA", "lad": "LAD", "lcx": "LCX", "rca": "inferred RCA candidate"}
    for name in ("lmca", "lad", "lcx", "rca"):
        record = distance_diagnostics[name]
        ax.plot(record["normalized_arc_length"], record["distances_mm"], color=COLORS[name], linewidth=2.5 if name != "rca" else 2.0, linestyle="--" if name == "rca" else "-", label=f"{labels[name]} · mean {record['statistics']['mean_mm']:.1f} mm")
    ax.set_xlim(0, 1); ax.set_xlabel("Normalized immutable source arc length"); ax.set_ylabel("Euclidean nearest-surface distance (mm)")
    ax.set_title("Source Centerline Distance to Parametric Support Surface")
    ax.text(0.01, 0.02, "Diagnostic only · distances do not alter source geometry", transform=ax.transAxes, color=COLORS["muted"])
    add_external_legend(ax)
    records.append(save_figure(fig, output, FIGURE_NAMES[9], dpi))

    # 11 · compact validation proof.
    fig, ax = plt.subplots(figsize=(13.2, 7.4)); ax.axis("off")
    integrity = validation_summary["source_integrity"]
    frame = validation_summary["anatomical_frame"]
    planes_record = validation_summary["planes"]
    surface_record = validation_summary["surface"]
    rows = [
        ("Source coordinate change", f"{integrity['max_coordinate_change_mm']:.3e} mm"),
        ("Original segment-length change", f"{integrity['max_original_segment_length_change_mm']:.3e} mm"),
        ("Rigid-transform segment error", f"{integrity['max_rigid_segment_length_error_mm']:.3e} mm"),
        ("Frame determinant", f"{frame['determinant']:.12f}"),
        ("Frame orthogonality error", f"{frame['orthogonality_error']:.3e}"),
        ("Coronary plane RMSE", f"{planes_record['coronary_rmse_mm']:.3f} mm"),
        ("LAD plane RMSE", f"{planes_record['lad_rmse_mm']:.3f} mm"),
        ("Plane separation", f"{planes_record['separation_angle_deg']:.2f}°"),
        ("Scaffold crown span", f"{surface_record['crown_span_mm']:.2f} mm"),
        ("Scaffold long-axis span", f"{surface_record['long_axis_span_mm']:.2f} mm"),
        ("Surface semi-axes a / b / c", f"{surface_record['semi_axes_mm']['a_crown']:.2f} / {surface_record['semi_axes_mm']['b_depth']:.2f} / {surface_record['semi_axes_mm']['c_long_axis']:.2f} mm"),
        ("VTK coordinate agreement", f"{integrity['max_vtk_coordinate_difference_mm']:.3e} mm"),
    ]
    ax.add_patch(plt.Rectangle((0.03, 0.88), 0.94, 0.09, transform=ax.transAxes, facecolor=COLORS["ink"], edgecolor="none"))
    ax.text(0.055, 0.925, "STAGE-1 TECHNICAL VALIDATION", transform=ax.transAxes, color="white", fontsize=18, weight="bold", va="center")
    status_color = "#197A47" if validation_summary["pass"] else "#B3261E"
    ax.text(0.94, 0.925, "PASS" if validation_summary["pass"] else "FAIL", transform=ax.transAxes, color="white", fontsize=17, weight="bold", va="center", ha="right", bbox={"boxstyle": "round,pad=0.35", "facecolor": status_color, "edgecolor": "none"})
    y = 0.82
    for index, (label, value) in enumerate(rows):
        if index == 6:
            y = 0.82
        column = 0 if index < 6 else 1
        x0 = 0.06 if column == 0 else 0.54
        ax.text(x0, y, label, transform=ax.transAxes, color=COLORS["muted"], fontsize=10, va="top")
        ax.text(x0, y - 0.035, value, transform=ax.transAxes, color=COLORS["ink"], fontsize=14, weight="bold", va="top")
        ax.plot([x0, x0 + 0.39], [y - 0.075, y - 0.075], transform=ax.transAxes, color=COLORS["grid"], linewidth=0.8)
        y -= 0.125
    ax.text(0.5, 0.045, "Dataset-derived parametric anatomical support scaffold · not a patient-specific myocardial reconstruction", transform=ax.transAxes, ha="center", color=COLORS["muted"], fontsize=10)
    records.append(save_figure(fig, output, FIGURE_NAMES[10], dpi))
    return records
