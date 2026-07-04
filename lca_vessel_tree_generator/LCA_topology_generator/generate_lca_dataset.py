import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from .bspline import interpolate_lca_tree
from .lca_validation import DEFAULT_VALIDATION_CONFIG, validate_lca_tree
from .radius_model import (
    DEFAULT_STATIC_RADIUS_MODEL,
    build_lca_radius_tree,
    validate_lca_radius_tree,
)
from .tight_mesh import (
    build_lca_tight_mesh,
    save_tight_mesh_ply,
    save_tight_mesh_preview,
    save_tight_mesh_stl,
    validate_tight_mesh,
)
from .tortuosity import calculate_lca_tortuosity, calculate_tortuosity
from .tortuosity_augmentation import make_target_tortuosity_variant
from .tube_surface import build_lca_tube_surfaces, validate_lca_tube_surfaces
from .visualize import set_axes_equal


VESSEL_BLUE = "#2938b8"
VESSEL_BLUE_LIGHT = "#5b66d5"
BRANCH_COLORS = {
    "LMCA": "black",
    "LAD": "red",
    "LCX": "blue",
}
BRANCH_SLICES = {
    "LMCA": slice(0, 5),
    "LAD": slice(5, 17),
    "LCX": slice(17, 27),
}


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)


def _fmt_tortuosity(value):
    return "NA" if value is None else f"{value:.2f}"


def _project_points(points: np.ndarray, projection: np.ndarray, scale: float = 1.0) -> np.ndarray:
    projected = np.asarray(points, dtype=float) @ projection
    projected = projected - projected[0]
    projected *= scale
    projected[:, 1] *= -1.0
    return projected


def _global_projection(all_points: np.ndarray) -> np.ndarray:
    centered = all_points - np.mean(all_points, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh[:2].T


def _equal_2d(ax, margin: float = 0.08):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    radius = 0.5 * max(x_max - x_min, y_max - y_min)
    radius *= 1.0 + margin
    ax.set_xlim(x_mid - radius, x_mid + radius)
    ax.set_ylim(y_mid - radius, y_mid + radius)
    ax.set_aspect("equal", adjustable="box")


def _set_bounds_from_xy(ax, xy: np.ndarray):
    x_min = float(np.min(xy[:, 0]))
    x_max = float(np.max(xy[:, 0]))
    y_min = float(np.min(xy[:, 1]))
    y_max = float(np.max(xy[:, 1]))

    if abs(x_max - x_min) <= 1e-9:
        x_min -= 1.0
        x_max += 1.0
    if abs(y_max - y_min) <= 1e-9:
        y_min -= 1.0
        y_max += 1.0

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def _draw_tapered_curve(ax, xy: np.ndarray, start_width: float, end_width: float, alpha: float = 0.96):
    if len(xy) < 2:
        return

    segments = np.stack([xy[:-1], xy[1:]], axis=1)
    widths = np.linspace(start_width, end_width, len(segments))
    ax.add_collection(LineCollection(
        segments,
        colors=VESSEL_BLUE,
        linewidths=widths,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
    ))
    ax.add_collection(LineCollection(
        segments,
        colors=VESSEL_BLUE_LIGHT,
        linewidths=np.maximum(widths * 0.35, 0.5),
        alpha=0.45,
        capstyle="round",
        joinstyle="round",
    ))


def _project_tree(centerlines: dict, projection: np.ndarray) -> dict:
    projected = {
        branch_name: _project_points(points, projection)
        for branch_name, points in centerlines.items()
    }

    # Use the projected LMCA endpoint as the shared bifurcation in the drawing.
    bifurcation = projected["LMCA"][-1].copy()
    for branch_name in ["LAD", "LCX"]:
        projected[branch_name] = projected[branch_name] - projected[branch_name][0] + bifurcation

    return projected


def _draw_tree(ax, centerlines: dict, projection: np.ndarray):
    widths = {
        "LMCA": (8.0, 5.0),
        "LAD": (5.2, 1.7),
        "LCX": (4.5, 1.5),
    }
    projected = _project_tree(centerlines, projection)

    for branch_name in ["LCX", "LAD", "LMCA"]:
        _draw_tapered_curve(ax, projected[branch_name], *widths[branch_name])

    all_xy = np.vstack(list(projected.values()))
    _set_bounds_from_xy(ax, all_xy)
    _equal_2d(ax, margin=0.22)
    ax.axis("off")


def _save_tree_plot(path: Path, sample: dict, projection: np.ndarray):
    fig, ax = plt.subplots(figsize=(3.0, 2.5))
    _draw_tree(ax, sample["centerlines"], projection)
    metrics = sample["metrics"]
    ax.set_title(
        f"Patient {sample['tree_index']:03d} | "
        f"LAD {metrics['LAD']['tortuosity']:.2f}, "
        f"LCX {metrics['LCX']['tortuosity']:.2f}",
        fontsize=8,
        pad=2,
    )
    plt.tight_layout(pad=0.15)
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_dataset_tree_gallery(path: Path, samples: list, projection: np.ndarray):
    cols = 5
    rows = int(math.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.35, rows * 1.9), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, sample in zip(axes.flat, samples):
        _draw_tree(ax, sample["centerlines"], projection)
        ax.set_title(f"Patient {sample['tree_index']:03d}", fontsize=8, pad=2)

    plt.tight_layout(pad=0.35)
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_tortuosity_label_gallery(path: Path, samples: list, projection: np.ndarray):
    cols = 5
    rows = int(math.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.35, rows * 1.9), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    for ax, sample in zip(axes.flat, samples):
        _draw_tree(ax, sample["centerlines"], projection)
        metrics = sample["metrics"]
        ax.set_title(
            f"LAD {metrics['LAD']['tortuosity']:.2f} | LCX {metrics['LCX']['tortuosity']:.2f}",
            fontsize=8,
            pad=2,
        )

    plt.tight_layout(pad=0.35)
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _make_tree_tortuosity_variants(sample: dict):
    target_steps = {
        "original": 0.0,
        "low": 0.12,
        "medium": 0.32,
        "high": 0.58,
        "very_high": 0.9,
    }
    variants = {}

    for label, increment in target_steps.items():
        variant_centerlines = {}
        variant_metadata = {}
        for branch_name, points in sample["centerlines"].items():
            base_tortuosity = sample["metrics"][branch_name]["tortuosity"]
            if label == "original" or base_tortuosity is None:
                variant_centerlines[branch_name] = points.copy()
                variant_metadata[branch_name] = {
                    "target_tortuosity": None,
                    "actual_tortuosity": base_tortuosity,
                    "amplitude_mm": 0.0,
                    "status": "original" if label == "original" else "undefined_base_tortuosity",
                }
                continue

            target = base_tortuosity + increment
            variant_points, metadata = make_target_tortuosity_variant(
                points,
                target_tortuosity=target,
                cycles=1.25 + increment * 1.2,
                phase=0.35 * len(branch_name),
                sample_points=len(points),
            )
            variant_centerlines[branch_name] = variant_points
            variant_metadata[branch_name] = metadata

        variants[label] = {
            "centerlines": variant_centerlines,
            "metrics": calculate_lca_tortuosity(variant_centerlines),
            "metadata": variant_metadata,
        }

    return variants


def _save_controlled_tortuosity_tree_gallery(path: Path, sample: dict, projection: np.ndarray):
    variants = _make_tree_tortuosity_variants(sample)
    labels = ["original", "low", "medium", "high", "very_high"]

    fig, axes = plt.subplots(1, len(labels), figsize=(len(labels) * 2.45, 2.35), squeeze=False)
    for ax, label in zip(axes.flat, labels):
        variant = variants[label]
        _draw_tree(ax, variant["centerlines"], projection)
        metrics = variant["metrics"]
        ax.set_title(
            f"{label.replace('_', ' ')}\n"
            f"LAD {metrics['LAD']['tortuosity']:.2f} | LCX {metrics['LCX']['tortuosity']:.2f}",
            fontsize=8,
            pad=2,
        )

    plt.tight_layout(pad=0.35)
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return variants


def _save_controlled_branch_tortuosity_gallery(path: Path, sample: dict):
    labels = ["original", "low", "medium", "high", "very_high"]
    branch_names = ["LMCA", "LAD", "LCX"]
    fig, axes = plt.subplots(len(branch_names), len(labels), figsize=(12.0, 7.4), squeeze=False)

    for row, branch_name in enumerate(branch_names):
        base_points = sample["centerlines"][branch_name]
        base_metric = sample["metrics"][branch_name]
        base_tortuosity = base_metric["tortuosity"]
        variants = {}
        increments = {
            "original": 0.0,
            "low": 0.12,
            "medium": 0.32,
            "high": 0.58,
            "very_high": 0.9,
        }
        for label in labels:
            if label == "original" or base_tortuosity is None:
                points = base_points.copy()
            else:
                points, _ = make_target_tortuosity_variant(
                    base_points,
                    target_tortuosity=base_tortuosity + increments[label],
                    cycles=1.25 + increments[label] * 1.2,
                    phase=0.35 * row,
                    sample_points=len(base_points),
                )
            variants[label] = {
                "points": points,
                "metric": calculate_tortuosity(points),
            }

        for col, label in enumerate(labels):
            points = variants[label]["points"]
            metric = variants[label]["metric"]
            xy = points[:, :2]
            xy = xy - np.mean(xy, axis=0)
            _draw_tapered_curve(axes[row, col], xy, start_width=7.0, end_width=1.8)
            _set_bounds_from_xy(axes[row, col], xy)
            _equal_2d(axes[row, col], margin=0.25)
            axes[row, col].set_title(
                f"{branch_name} {label.replace('_', ' ')}\nT={metric['tortuosity']:.2f}",
                fontsize=8,
            )
            axes[row, col].axis("off")

    plt.tight_layout(pad=0.5)
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_3d_tree_matrix(ax, sample: dict):
    control_tree = sample["control_tree"]
    centerlines = sample["centerlines"]

    for branch_name in ["LMCA", "LAD", "LCX"]:
        color = BRANCH_COLORS[branch_name]
        centerline = centerlines[branch_name]
        controls = control_tree[BRANCH_SLICES[branch_name]]
        ax.plot(
            centerline[:, 0],
            centerline[:, 1],
            centerline[:, 2],
            color=color,
            linewidth=2.2,
            alpha=0.85,
            label=f"{branch_name} spline",
        )
        ax.plot(
            controls[:, 0],
            controls[:, 1],
            controls[:, 2],
            color=color,
            linewidth=1.0,
            linestyle="--",
            alpha=0.45,
        )
        ax.scatter(
            controls[:, 0],
            controls[:, 1],
            controls[:, 2],
            color=color,
            edgecolor="white",
            linewidth=0.6,
            s=28,
            depthshade=False,
            label=f"{branch_name} ctrl pts",
        )

    bifurcation = control_tree[4]
    ax.scatter(
        [bifurcation[0]],
        [bifurcation[1]],
        [bifurcation[2]],
        color="purple",
        edgecolor="white",
        linewidth=0.8,
        s=48,
        depthshade=False,
        label="bifurcation",
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)


def _save_3d_tree_plot(path: Path, sample: dict):
    fig = plt.figure(figsize=(6.5, 5.2))
    ax = fig.add_subplot(projection="3d")
    _draw_3d_tree_matrix(ax, sample)
    metrics = sample["metrics"]
    ax.set_title(
        f"Patient {sample['tree_index']:03d} 27x3 Control Matrix\n"
        f"LMCA {_fmt_tortuosity(metrics['LMCA']['tortuosity'])}, "
        f"LAD {_fmt_tortuosity(metrics['LAD']['tortuosity'])}, "
        f"LCX {_fmt_tortuosity(metrics['LCX']['tortuosity'])}",
        fontsize=9,
    )
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_3d_tree_gallery(path: Path, samples: list):
    cols = 3
    rows = int(math.ceil(len(samples) / cols))
    fig = plt.figure(figsize=(cols * 5.0, rows * 4.3))

    for idx, sample in enumerate(samples):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        _draw_3d_tree_matrix(ax, sample)
        metrics = sample["metrics"]
        ax.set_title(
            f"Patient {sample['tree_index']:03d}\n"
            f"LAD {metrics['LAD']['tortuosity']:.2f} | LCX {metrics['LCX']['tortuosity']:.2f}",
            fontsize=9,
        )
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=6, loc="upper left")

    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_branch_gallery(path: Path, samples: list, projection: np.ndarray):
    branch_names = ["LMCA", "LAD", "LCX"]
    labels = ["low", "medium", "high"]
    fig, axes = plt.subplots(len(branch_names), len(labels), figsize=(10.0, 7.6), squeeze=False)

    for row, branch_name in enumerate(branch_names):
        valid = [
            sample
            for sample in samples
            if sample["metrics"][branch_name]["tortuosity"] is not None
        ]
        ranked = sorted(valid, key=lambda sample: sample["metrics"][branch_name]["tortuosity"])
        picks = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
        for col, sample in enumerate(picks):
            points = sample["centerlines"][branch_name]
            xy = _project_points(points, projection)
            xy = xy - np.mean(xy, axis=0)
            _draw_tapered_curve(axes[row, col], xy, start_width=7.0, end_width=1.8)
            _set_bounds_from_xy(axes[row, col], xy)
            _equal_2d(axes[row, col], margin=0.25)
            tortuosity = sample["metrics"][branch_name]["tortuosity"]
            axes[row, col].set_title(f"{branch_name} {labels[col]} T={tortuosity:.2f}", fontsize=9)
            axes[row, col].axis("off")

    plt.tight_layout(pad=0.8)
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_spline(ax, points: np.ndarray, title: str):
    xy = points[:, :2]
    xy = xy - np.mean(xy, axis=0)
    _draw_tapered_curve(ax, xy, start_width=7.0, end_width=1.8)
    _set_bounds_from_xy(ax, xy)
    _equal_2d(ax, margin=0.25)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _make_tortuous_spline(strength: float, n: int = 220) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    y = 100.0 * t
    x = strength * (
        9.0 * np.sin(2.0 * np.pi * t + 0.4)
        + 4.0 * np.sin(5.0 * np.pi * t + 1.1)
        + 1.8 * np.sin(9.0 * np.pi * t)
    )
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)


def _save_controlled_spline_gallery(path: Path):
    strengths = [0.0, 0.25, 0.45, 0.7, 1.0]
    fig, axes = plt.subplots(1, len(strengths), figsize=(11.0, 2.25), squeeze=False)
    for ax, strength in zip(axes.flat, strengths):
        points = _make_tortuous_spline(strength)
        metric = calculate_tortuosity(points)
        _draw_spline(ax, points, f"T={metric['tortuosity']:.2f}")

    plt.tight_layout(pad=0.35)
    plt.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _radius_model_from_args(args) -> dict:
    defaults = DEFAULT_STATIC_RADIUS_MODEL["branches"]
    branch_types = DEFAULT_STATIC_RADIUS_MODEL["branch_types"]
    return {
        "units": DEFAULT_STATIC_RADIUS_MODEL["units"],
        "description": DEFAULT_STATIC_RADIUS_MODEL["description"],
        "model": DEFAULT_STATIC_RADIUS_MODEL["model"],
        "branch_types": {
            "parent_trunk": {
                **branch_types["parent_trunk"],
                "taper_rate_per_mm": args.parent_trunk_taper_rate,
            },
            "distributing": {
                **branch_types["distributing"],
                "taper_rate_per_mm": args.distributing_taper_rate,
            },
            "delivering": {
                **branch_types["delivering"],
                "taper_rate_per_mm": args.delivering_taper_rate,
            },
        },
        "lmca_bifurcation": DEFAULT_STATIC_RADIUS_MODEL["lmca_bifurcation"].copy(),
        "branches": {
            "LMCA": {
                "proximal_radius_mm": args.lmca_radius_proximal,
                "branch_type": defaults["LMCA"]["branch_type"],
            },
            "LAD": {
                "proximal_radius_mm": args.lad_radius_proximal,
                "branch_type": defaults["LAD"]["branch_type"],
            },
            "LCX": {
                "proximal_radius_mm": args.lcx_radius_proximal,
                "branch_type": defaults["LCX"]["branch_type"],
            },
        },
    }


def _load_patient_radius_metadata(dataset_dir: Path, tree_index: int) -> dict:
    metadata_path = dataset_dir / f"patient_{tree_index:04d}" / "patient_info.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as json_file:
        metadata = json.load(json_file)
    return {
        "path": str(metadata_path),
        "radius_mm": {
            "LMCA": metadata.get("lmca_radius_mm"),
            "LAD": metadata.get("lad_radius_mm"),
            "LCX": metadata.get("lcx_radius_mm"),
        },
        "diameter_mm": {
            "LMCA": metadata.get("lmca_diameter_mm"),
            "LAD": metadata.get("lad_diameter_mm"),
            "LCX": metadata.get("lcx_diameter_mm"),
        },
    }


def _radius_model_for_patient(args, tree_index: int) -> dict:
    model = _radius_model_from_args(args)
    metadata = _load_patient_radius_metadata(Path(args.dataset_dir), tree_index)
    model["metadata_source"] = metadata.get("path")
    model["branch_radius_source"] = {}

    for branch_name in ["LMCA", "LAD", "LCX"]:
        metadata_radius = metadata.get("radius_mm", {}).get(branch_name)
        metadata_diameter = metadata.get("diameter_mm", {}).get(branch_name)
        if metadata_radius is not None:
            model["branches"][branch_name]["proximal_radius_mm"] = float(metadata_radius)
            model["branch_radius_source"][branch_name] = "patient_metadata_radius"
        elif metadata_diameter is not None:
            model["branches"][branch_name]["proximal_radius_mm"] = float(metadata_diameter) / 2.0
            model["branch_radius_source"][branch_name] = "patient_metadata_diameter"
        else:
            model["branch_radius_source"][branch_name] = "mvp_default"

    return model


def _radius_axis(points_with_radius: np.ndarray) -> np.ndarray:
    if len(points_with_radius) == 0:
        return np.zeros(0, dtype=float)
    if len(points_with_radius) == 1:
        return np.zeros(1, dtype=float)
    distances = np.zeros(len(points_with_radius), dtype=float)
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points_with_radius[:, :3], axis=0), axis=1))
    total = distances[-1]
    if total <= 1e-9:
        return np.zeros(len(points_with_radius), dtype=float)
    return distances / total


def _save_radius_profile_plot(path: Path, centerlines_with_radius: dict):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    for branch_name in ["LMCA", "LAD", "LCX"]:
        points = centerlines_with_radius[branch_name]
        ax.plot(
            _radius_axis(points),
            points[:, 3],
            color=BRANCH_COLORS[branch_name],
            linewidth=2.0,
            label=branch_name,
        )

    ax.set_xlabel("Normalized branch arc length")
    ax.set_ylabel("Radius (mm)")
    ax.set_title("Static radius taper")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_static_radius_gallery(path: Path, samples: list):
    cols = 3
    rows = int(math.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.85), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for ax, sample in zip(axes.flat, samples):
        ax.axis("on")
        for branch_name in ["LMCA", "LAD", "LCX"]:
            points = sample["centerlines_with_radius"][branch_name]
            ax.plot(
                _radius_axis(points),
                points[:, 3],
                color=BRANCH_COLORS[branch_name],
                linewidth=2.0,
                label=branch_name,
            )
        ax.set_title(f"Patient {sample['tree_index']:03d}", fontsize=9)
        ax.set_xlabel("Arc length", fontsize=8)
        ax.set_ylabel("Radius (mm)", fontsize=8)
        ax.grid(True, alpha=0.22)
        ax.tick_params(labelsize=7)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), pad=0.8)
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_3d_radius_tree_plot(path: Path, sample: dict):
    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_subplot(projection="3d")

    for branch_name in ["LMCA", "LAD", "LCX"]:
        points = sample["centerlines_with_radius"][branch_name]
        radius = points[:, 3]
        linewidth = 0.9 + float(np.mean(radius)) * 1.25
        ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color=BRANCH_COLORS[branch_name],
            linewidth=linewidth,
            alpha=0.86,
            label=(
                f"{branch_name} "
                f"{radius[0]:.2f}->{radius[-1]:.2f} mm"
            ),
        )
        ax.scatter(
            [points[0, 0], points[-1, 0]],
            [points[0, 1], points[-1, 1]],
            [points[0, 2], points[-1, 2]],
            color=BRANCH_COLORS[branch_name],
            edgecolor="white",
            linewidth=0.6,
            s=[40, 20],
            depthshade=False,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Patient {sample['tree_index']:03d} static radius")
    set_axes_equal(ax)
    ax.legend(fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_tube_surface_plot(path: Path, sample: dict):
    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_subplot(projection="3d")
    surface_colors = {
        "LMCA": "#4a4a4a",
        "LAD": "#d62728",
        "LCX": "#1f77b4",
    }

    for branch_name in ["LMCA", "LAD", "LCX"]:
        surface = sample["tube_surfaces"][branch_name]
        ax.plot_surface(
            surface[:, :, 0],
            surface[:, :, 1],
            surface[:, :, 2],
            color=surface_colors[branch_name],
            alpha=0.72,
            linewidth=0,
            antialiased=True,
            shade=True,
        )
        centerline = sample["centerlines_with_radius"][branch_name]
        ax.plot(
            centerline[:, 0],
            centerline[:, 1],
            centerline[:, 2],
            color="white",
            linewidth=0.8,
            alpha=0.85,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Patient {sample['tree_index']:03d} simple tube surface")
    set_axes_equal(ax)
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_tube_surface_gallery(path: Path, samples: list):
    cols = 3
    rows = int(math.ceil(len(samples) / cols))
    fig = plt.figure(figsize=(cols * 4.2, rows * 3.7))

    for idx, sample in enumerate(samples):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        for branch_name, color in [("LMCA", "#4a4a4a"), ("LAD", "#d62728"), ("LCX", "#1f77b4")]:
            surface = sample["tube_surfaces"][branch_name]
            ax.plot_surface(
                surface[:, :, 0],
                surface[:, :, 1],
                surface[:, :, 2],
                color=color,
                alpha=0.72,
                linewidth=0,
                antialiased=True,
                shade=True,
            )
        ax.set_title(f"Patient {sample['tree_index']:03d}", fontsize=9)
        ax.set_axis_off()
        set_axes_equal(ax)

    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _build_control_points_with_radius(control_tree: np.ndarray, radius_model: dict) -> np.ndarray:
    control_centerlines = {
        branch_name: control_tree[BRANCH_SLICES[branch_name]]
        for branch_name in ["LMCA", "LAD", "LCX"]
    }
    control_points_with_radius, _ = build_lca_radius_tree(control_centerlines, radius_model)
    return np.vstack([
        control_points_with_radius["LMCA"],
        control_points_with_radius["LAD"],
        control_points_with_radius["LCX"],
    ])


def _build_dataset_samples(tree_ctrl_points: np.ndarray, args) -> list:
    samples = []
    for tree_index, tree in enumerate(tree_ctrl_points):
        radius_model = _radius_model_for_patient(args, tree_index)
        centerlines = interpolate_lca_tree(
            tree,
            lmca_points=args.lmca_points,
            lad_points=args.lad_points,
            lcx_points=args.lcx_points,
        )
        metrics = calculate_lca_tortuosity(centerlines)
        validation = validate_lca_tree(tree, centerlines, metrics)
        centerlines_with_radius, radius_metadata = build_lca_radius_tree(centerlines, radius_model)
        radius_validation = validate_lca_radius_tree(
            centerlines_with_radius,
            adjacent_jump_threshold_mm=args.radius_adjacent_jump_threshold,
        )
        control_points_with_radius = _build_control_points_with_radius(tree, radius_model)
        tube_surfaces, tube_metadata = build_lca_tube_surfaces(
            centerlines_with_radius,
            num_circle_points=args.tube_circle_points,
        )
        tube_validation = validate_lca_tube_surfaces(tube_surfaces, centerlines_with_radius)
        tight_mesh_vertices, tight_mesh_faces, tight_mesh_metadata = build_lca_tight_mesh(tube_surfaces)
        tight_mesh_validation = validate_tight_mesh(
            tight_mesh_vertices,
            tight_mesh_faces,
            tight_mesh_metadata,
        )
        samples.append({
            "tree_index": tree_index,
            "control_tree": tree,
            "control_tree_with_radius": control_points_with_radius,
            "centerlines": centerlines,
            "centerlines_with_radius": centerlines_with_radius,
            "tube_surfaces": tube_surfaces,
            "tight_mesh_vertices": tight_mesh_vertices,
            "tight_mesh_faces": tight_mesh_faces,
            "metrics": metrics,
            "validation": validation,
            "radius_metadata": radius_metadata,
            "radius_validation": radius_validation,
            "tube_metadata": tube_metadata,
            "tube_validation": tube_validation,
            "tight_mesh_metadata": tight_mesh_metadata,
            "tight_mesh_validation": tight_mesh_validation,
        })
    return samples


def main():
    base_dir = Path(__file__).resolve().parent.parent
    default_dataset_dir = base_dir / "LCA_branch_control_points" / "generated"
    default_output_dir = base_dir / "outputs" / "dataset_lca"

    parser = argparse.ArgumentParser(description="Generate LCA centerlines and visuals directly from patient-derived control trees.")
    parser.add_argument("--dataset-dir", type=str, default=str(default_dataset_dir), help="Directory containing LCA_tree_ctrl_points.npy")
    parser.add_argument("--output", type=str, default=str(default_output_dir), help="Output directory for dataset-derived centerlines and plots")
    parser.add_argument("--num-trees", type=int, default=None, help="Limit number of patient trees to export")
    parser.add_argument("--include-invalid", action="store_true", help="Export trees that fail anatomical validation")
    parser.add_argument("--no-clean", action="store_true", help="Do not clear previous dataset output before writing")
    parser.add_argument("--include-controlled-splines", action="store_true", help="Also write a standalone illustrative tortuosity spline gallery")
    parser.add_argument("--lmca-points", type=int, default=150)
    parser.add_argument("--lad-points", type=int, default=300)
    parser.add_argument("--lcx-points", type=int, default=250)
    parser.add_argument("--lmca-radius-proximal", type=float, default=DEFAULT_STATIC_RADIUS_MODEL["branches"]["LMCA"]["proximal_radius_mm"])
    parser.add_argument("--lad-radius-proximal", type=float, default=DEFAULT_STATIC_RADIUS_MODEL["branches"]["LAD"]["proximal_radius_mm"])
    parser.add_argument("--lcx-radius-proximal", type=float, default=DEFAULT_STATIC_RADIUS_MODEL["branches"]["LCX"]["proximal_radius_mm"])
    parser.add_argument("--parent-trunk-taper-rate", type=float, default=DEFAULT_STATIC_RADIUS_MODEL["branch_types"]["parent_trunk"]["taper_rate_per_mm"], help="Distance taper rate k for parent trunk branches in radius = proximal * exp(-k * distance)")
    parser.add_argument("--distributing-taper-rate", type=float, default=DEFAULT_STATIC_RADIUS_MODEL["branch_types"]["distributing"]["taper_rate_per_mm"], help="Distance taper rate k for distributing branches in radius = proximal * exp(-k * distance)")
    parser.add_argument("--delivering-taper-rate", type=float, default=DEFAULT_STATIC_RADIUS_MODEL["branch_types"]["delivering"]["taper_rate_per_mm"], help="Distance taper rate k for future delivering branches")
    parser.add_argument("--radius-adjacent-jump-threshold", type=float, default=0.2, help="Maximum allowed radius change between adjacent centerline points in mm")
    parser.add_argument("--tube-circle-points", type=int, default=24, help="Number of radial samples per centerline point for simple tube surfaces")
    args = parser.parse_args()

    output_dir = Path(args.output)
    tree_dir = output_dir / "trees"
    if output_dir.exists() and not args.no_clean:
        shutil.rmtree(output_dir)
    tree_dir.mkdir(parents=True, exist_ok=True)

    tree_ctrl_path = Path(args.dataset_dir) / "LCA_tree_ctrl_points.npy"
    if not tree_ctrl_path.exists():
        raise FileNotFoundError(f"Patient LCA tree control points not found: {tree_ctrl_path}")

    tree_ctrl_points = np.load(tree_ctrl_path)
    samples = _build_dataset_samples(tree_ctrl_points, args)
    if args.num_trees is not None:
        samples = samples[:args.num_trees]

    validation_report = {
        "validation_config": DEFAULT_VALIDATION_CONFIG,
        "total_trees": len(samples),
        "valid_trees": int(sum(1 for sample in samples if sample["validation"]["is_valid"])),
        "invalid_trees": int(sum(1 for sample in samples if not sample["validation"]["is_valid"])),
        "trees": [
            {
                "tree_index": sample["tree_index"],
                **sample["validation"],
            }
            for sample in samples
        ],
    }

    export_samples = samples if args.include_invalid else [
        sample for sample in samples if sample["validation"]["is_valid"]
    ]
    if len(export_samples) == 0:
        _write_json(output_dir / "validation_report.json", validation_report)
        raise ValueError("No valid LCA trees passed validation.")

    all_points = np.vstack([
        points
        for sample in export_samples
        for points in sample["centerlines"].values()
    ])
    projection = _global_projection(all_points)

    for sample in export_samples:
        sample_dir = tree_dir / f"patient_{sample['tree_index']:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir / "control_points_27x3.npy", sample["control_tree"])
        np.save(sample_dir / "control_points_27x4.npy", sample["control_tree_with_radius"])
        for branch_name, points in sample["centerlines"].items():
            np.save(sample_dir / f"{branch_name.lower()}_centerline.npy", points)
        for branch_name, points in sample["centerlines_with_radius"].items():
            np.save(sample_dir / f"{branch_name.lower()}_centerline_radius.npy", points)
        np.savez(
            sample_dir / "tree_centerline_radius.npz",
            LMCA=sample["centerlines_with_radius"]["LMCA"],
            LAD=sample["centerlines_with_radius"]["LAD"],
            LCX=sample["centerlines_with_radius"]["LCX"],
        )
        np.savez(
            sample_dir / "tree_tube_surface.npz",
            LMCA=sample["tube_surfaces"]["LMCA"],
            LAD=sample["tube_surfaces"]["LAD"],
            LCX=sample["tube_surfaces"]["LCX"],
        )
        np.savez(
            sample_dir / "tree_tight_mesh.npz",
            vertices=sample["tight_mesh_vertices"],
            faces=sample["tight_mesh_faces"],
        )
        _write_json(
            sample_dir / "radius_validation.json",
            sample["radius_validation"],
        )
        _write_json(
            sample_dir / "tube_surface_validation.json",
            sample["tube_validation"],
        )
        _write_json(
            sample_dir / "tight_mesh_validation.json",
            sample["tight_mesh_validation"],
        )
        _write_json(
            sample_dir / "radius_summary.json",
            {
                "units": "mm",
                "columns": ["x", "y", "z", "radius_mm"],
                "radius_model": sample["radius_metadata"],
                "branch_radius_source": sample["radius_metadata"].get("branch_radius_source", {}),
                "output_shapes": {
                    "control_points_27x4": list(sample["control_tree_with_radius"].shape),
                    "LMCA": list(sample["centerlines_with_radius"]["LMCA"].shape),
                    "LAD": list(sample["centerlines_with_radius"]["LAD"].shape),
                    "LCX": list(sample["centerlines_with_radius"]["LCX"].shape),
                    "tube_surface_LMCA": list(sample["tube_surfaces"]["LMCA"].shape),
                    "tube_surface_LAD": list(sample["tube_surfaces"]["LAD"].shape),
                    "tube_surface_LCX": list(sample["tube_surfaces"]["LCX"].shape),
                    "tight_mesh_vertices": list(sample["tight_mesh_vertices"].shape),
                    "tight_mesh_faces": list(sample["tight_mesh_faces"].shape),
                },
                "validation": sample["radius_validation"],
                "tube_surface": sample["tube_metadata"],
                "tube_validation": sample["tube_validation"],
                "tight_mesh": sample["tight_mesh_metadata"],
                "tight_mesh_validation": sample["tight_mesh_validation"],
            },
        )
        _save_radius_profile_plot(sample_dir / "radius_profile.png", sample["centerlines_with_radius"])
        _save_3d_radius_tree_plot(sample_dir / "tree_with_radius.png", sample)
        _save_tube_surface_plot(sample_dir / "tree_tube_surface.png", sample)
        save_tight_mesh_preview(
            sample_dir / "tree_tight_mesh.png",
            sample["tight_mesh_vertices"],
            sample["tight_mesh_faces"],
        )
        save_tight_mesh_ply(
            sample_dir / "tree_tight_mesh.ply",
            sample["tight_mesh_vertices"],
            sample["tight_mesh_faces"],
        )
        save_tight_mesh_stl(
            sample_dir / "tree_tight_mesh.stl",
            sample["tight_mesh_vertices"],
            sample["tight_mesh_faces"],
        )
        _write_json(
            sample_dir / "tortuosity_metrics.json",
            {
                "units": "mm",
                "definition": "distance_ratio = centerline_path_length / endpoint_chord_length",
                "final_centerlines": sample["metrics"],
                "validation": sample["validation"],
            },
        )
        _save_tree_plot(sample_dir / "tree_visualization.png", sample, projection)
        _save_3d_tree_plot(sample_dir / "tree_3d_matrix.png", sample)
        variants = _save_controlled_tortuosity_tree_gallery(
            sample_dir / "controlled_tortuosity_variants.png",
            sample,
            projection,
        )
        variants_dir = sample_dir / "controlled_tortuosity_variants"
        variants_dir.mkdir(parents=True, exist_ok=True)
        variant_metadata = {}
        for variant_label, variant in variants.items():
            variant_metadata[variant_label] = {
                "metrics": variant["metrics"],
                "branches": variant["metadata"],
            }
            for branch_name, points in variant["centerlines"].items():
                np.save(variants_dir / f"{variant_label}_{branch_name.lower()}_centerline.npy", points)
        _write_json(variants_dir / "metadata.json", variant_metadata)

    summary = {
        "mode": "dataset_driven",
        "num_trees": len(export_samples),
        "total_input_trees": len(samples),
        "invalid_tree_indices": [
            sample["tree_index"]
            for sample in samples
            if not sample["validation"]["is_valid"]
        ],
        "include_invalid": args.include_invalid,
        "source": str(tree_ctrl_path),
        "control_point_matrix_shape": [27, 3],
        "control_point_indexing": {
            "LMCA": "0:5",
            "LAD": "5:17",
            "LCX": "17:27",
            "shared_bifurcation_index": 4,
        },
        "visual_style": "2D projected tapered vessel centerlines",
        "controlled_tortuosity": {
            "description": "Endpoint-preserving sinusoidal branch variants generated from dataset-derived centerlines",
            "variant_order": ["original", "low", "medium", "high", "very_high"],
        },
        "radius_taper_model_mvp": {
            "defaults": DEFAULT_STATIC_RADIUS_MODEL,
            "metadata_override_keys": [
                "lmca_radius_mm",
                "lad_radius_mm",
                "lcx_radius_mm",
                "lmca_diameter_mm",
                "lad_diameter_mm",
                "lcx_diameter_mm",
            ],
            "metadata_priority": "radius_mm, then diameter_mm / 2, then MVP default",
            "taper_formula": "LAD/LCX radius = proximal_radius * exp(-branch_type_taper_rate * cumulative_distance_mm)",
            "lmca_distal_rule": "LMCA distal radius = (LAD proximal^3 + LCX proximal^3)^(1/3)",
            "branch_type_taper_rates_per_mm": {
                "parent_trunk": args.parent_trunk_taper_rate,
                "distributing": args.distributing_taper_rate,
                "delivering": args.delivering_taper_rate,
            },
            "radius_validation": [
                "positive finite radius values",
                "proximal radius >= distal radius",
                "adjacent radius jumps below configured threshold",
                "LMCA proximal radius > LAD/LCX proximal radius",
                "LMCA distal radius compatible with LAD/LCX proximal radii by cube-law tolerance",
            ],
            "no_disease_no_motion_no_pulsatility": True,
        },
        "trees": [
            {
                "tree_index": sample["tree_index"],
                "validation_status": "valid" if sample["validation"]["is_valid"] else "invalid",
                "radius_validation_status": "valid" if sample["radius_validation"]["is_valid"] else "invalid",
                "tube_surface_validation_status": "valid" if sample["tube_validation"]["is_valid"] else "invalid",
                "tight_mesh_validation_status": "valid" if sample["tight_mesh_validation"]["is_valid"] else "invalid",
                "radius_source": sample["radius_metadata"].get("branch_radius_source", {}),
                "radius_output_shapes": {
                    "control_points_27x4": list(sample["control_tree_with_radius"].shape),
                    "LMCA": list(sample["centerlines_with_radius"]["LMCA"].shape),
                    "LAD": list(sample["centerlines_with_radius"]["LAD"].shape),
                    "LCX": list(sample["centerlines_with_radius"]["LCX"].shape),
                    "tube_surface_LMCA": list(sample["tube_surfaces"]["LMCA"].shape),
                    "tube_surface_LAD": list(sample["tube_surfaces"]["LAD"].shape),
                    "tube_surface_LCX": list(sample["tube_surfaces"]["LCX"].shape),
                    "tight_mesh_vertices": list(sample["tight_mesh_vertices"].shape),
                    "tight_mesh_faces": list(sample["tight_mesh_faces"].shape),
                },
                "tortuosity": {
                    branch_name: sample["metrics"][branch_name]["tortuosity"]
                    for branch_name in ["LMCA", "LAD", "LCX"]
                },
                "radius": {
                    branch_name: {
                        "proximal_radius_mm": sample["radius_validation"]["branches"][branch_name]["proximal_radius_mm"],
                        "distal_radius_mm": sample["radius_validation"]["branches"][branch_name]["distal_radius_mm"],
                        "branch_type": sample["radius_metadata"]["branches"][branch_name]["branch_type"],
                        "taper_rate_per_mm": sample["radius_metadata"]["branches"][branch_name]["taper_rate_per_mm"],
                        "arc_length_mm": sample["radius_metadata"]["branches"][branch_name]["arc_length_mm"],
                    }
                    for branch_name in ["LMCA", "LAD", "LCX"]
                },
                "tight_mesh": {
                    "vertex_count": sample["tight_mesh_validation"]["vertex_count"],
                    "face_count": sample["tight_mesh_validation"]["face_count"],
                    "boundary_edge_count": sample["tight_mesh_validation"]["boundary_edge_count"],
                    "connected_component_count": sample["tight_mesh_validation"]["connected_component_count"],
                },
            }
            for sample in export_samples
        ],
    }

    radius_validation_report = {
        "units": "mm",
        "total_exported_trees": len(export_samples),
        "valid_radius_trees": int(sum(1 for sample in export_samples if sample["radius_validation"]["is_valid"])),
        "invalid_radius_trees": int(sum(1 for sample in export_samples if not sample["radius_validation"]["is_valid"])),
        "trees": [
            {
                "tree_index": sample["tree_index"],
                **sample["radius_validation"],
            }
            for sample in export_samples
        ],
    }

    radius_summary = {
        "mode": "distance_based_radius_taper_mvp",
        "units": "mm",
        "num_valid_dataset_trees_processed": len(export_samples),
        "output_point_format": ["x", "y", "z", "radius_mm"],
        "branch_files": [
            "lmca_centerline_radius.npy",
            "lad_centerline_radius.npy",
            "lcx_centerline_radius.npy",
            "tree_centerline_radius.npz",
            "control_points_27x4.npy",
            "tree_tube_surface.npz",
            "tree_tight_mesh.npz",
            "tree_tight_mesh.ply",
            "tree_tight_mesh.stl",
        ],
        "default_proximal_radius_mm": {
            "LMCA": DEFAULT_STATIC_RADIUS_MODEL["branches"]["LMCA"]["proximal_radius_mm"],
            "LAD": DEFAULT_STATIC_RADIUS_MODEL["branches"]["LAD"]["proximal_radius_mm"],
            "LCX": DEFAULT_STATIC_RADIUS_MODEL["branches"]["LCX"]["proximal_radius_mm"],
        },
        "branch_types": DEFAULT_STATIC_RADIUS_MODEL["branch_types"],
        "branch_type_assignments": {
            branch_name: DEFAULT_STATIC_RADIUS_MODEL["branches"][branch_name]["branch_type"]
            for branch_name in ["LMCA", "LAD", "LCX"]
        },
        "branch_type_taper_rates_per_mm": {
            "parent_trunk": args.parent_trunk_taper_rate,
            "distributing": args.distributing_taper_rate,
            "delivering": args.delivering_taper_rate,
        },
        "taper_formula": "radius(distance) = proximal_radius * exp(-k * cumulative_distance_mm)",
        "lmca_distal_formula": "lmca_distal = (lad_proximal^3 + lcx_proximal^3)^(1/3)",
        "metadata_priority": "Use patient radius metadata first, patient diameter / 2 second, MVP default third.",
        "adjacent_jump_threshold_mm": args.radius_adjacent_jump_threshold,
        "tube_generation_compatibility": {
            "compatible_shape": True,
            "lca_format": "Each branch is N x 4 with columns x,y,z,radius in mm.",
            "rca_format_observed": "RCA tube_generator saves each branch as N x 4 with columns X,Y,Z,R.",
            "unit_note": "RCA tube code internally uses meters; convert LCA mm arrays to meters before direct surface generation.",
        },
        "simple_tube_surface": {
            "surface_file": "tree_tube_surface.npz",
            "surface_shape": "N x num_circle_points x 3",
            "num_circle_points": args.tube_circle_points,
            "method": "Circular cross-section sweep along each radius-bearing centerline",
            "limitations": "Simple branch surfaces are generated independently and are not boolean-unioned at the bifurcation.",
        },
        "tight_mesh": {
            "mesh_file": "tree_tight_mesh.npz",
            "mesh_arrays": {
                "vertices": "V x 3",
                "faces": "F x 3 triangle indices",
            },
            "exports": ["tree_tight_mesh.ply", "tree_tight_mesh.stl"],
            "method": "MVP branch-preserving mesh with capped free ends and simple bifurcation bridge faces",
            "limitations": "Not CFD-grade, not clinical-grade, and not an advanced boolean union.",
        },
        "trees": [
            {
                "tree_index": sample["tree_index"],
                "radius_validation_status": "valid" if sample["radius_validation"]["is_valid"] else "invalid",
                "tube_surface_validation_status": "valid" if sample["tube_validation"]["is_valid"] else "invalid",
                "tight_mesh_validation_status": "valid" if sample["tight_mesh_validation"]["is_valid"] else "invalid",
                "radius_source": sample["radius_metadata"].get("branch_radius_source", {}),
                "shapes": {
                    "control_points_27x4": list(sample["control_tree_with_radius"].shape),
                    "LMCA": list(sample["centerlines_with_radius"]["LMCA"].shape),
                    "LAD": list(sample["centerlines_with_radius"]["LAD"].shape),
                    "LCX": list(sample["centerlines_with_radius"]["LCX"].shape),
                    "tube_surface_LMCA": list(sample["tube_surfaces"]["LMCA"].shape),
                    "tube_surface_LAD": list(sample["tube_surfaces"]["LAD"].shape),
                    "tube_surface_LCX": list(sample["tube_surfaces"]["LCX"].shape),
                    "tight_mesh_vertices": list(sample["tight_mesh_vertices"].shape),
                    "tight_mesh_faces": list(sample["tight_mesh_faces"].shape),
                },
                "tight_mesh_boundary_edge_count": sample["tight_mesh_validation"]["boundary_edge_count"],
                "tight_mesh_connected_component_count": sample["tight_mesh_validation"]["connected_component_count"],
            }
            for sample in export_samples
        ],
    }

    tight_mesh_validation_report = {
        "total_exported_trees": len(export_samples),
        "valid_tight_meshes": int(sum(1 for sample in export_samples if sample["tight_mesh_validation"]["is_valid"])),
        "invalid_tight_meshes": int(sum(1 for sample in export_samples if not sample["tight_mesh_validation"]["is_valid"])),
        "trees": [
            {
                "tree_index": sample["tree_index"],
                **sample["tight_mesh_validation"],
            }
            for sample in export_samples
        ],
    }

    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "validation_report.json", validation_report)
    _write_json(output_dir / "radius_validation.json", radius_validation_report)
    _write_json(output_dir / "radius_summary.json", radius_summary)
    _write_json(output_dir / "tight_mesh_validation.json", tight_mesh_validation_report)
    _save_dataset_tree_gallery(output_dir / "01_dataset_trees.png", export_samples, projection)
    _save_3d_tree_gallery(output_dir / "01_dataset_trees_3d_matrix.png", export_samples)
    controlled_sample = export_samples[0]
    _save_controlled_tortuosity_tree_gallery(
        output_dir / "02_dataset_tree_with_controlled_tortuosity.png",
        controlled_sample,
        projection,
    )
    _save_controlled_branch_tortuosity_gallery(
        output_dir / "03_branch_controlled_tortuosity_examples.png",
        controlled_sample,
    )
    _save_tortuosity_label_gallery(output_dir / "04_dataset_trees_with_measured_tortuosity.png", export_samples, projection)
    _save_branch_gallery(output_dir / "05_dataset_branch_tortuosity_ranking.png", export_samples, projection)
    _save_static_radius_gallery(output_dir / "06_static_radius_profiles.png", export_samples)
    _save_tube_surface_gallery(output_dir / "07_lca_tube_surfaces.png", export_samples)
    if args.include_controlled_splines:
        _save_controlled_spline_gallery(output_dir / "controlled_tortuosity_splines.png")

    print(f"Generated {len(export_samples)} valid dataset-derived LCA trees in: {output_dir}")
    print(f"Skipped invalid tree indices: {summary['invalid_tree_indices']}")
    print(f"Dataset tree gallery: {output_dir / '01_dataset_trees.png'}")
    print(f"3D matrix gallery: {output_dir / '01_dataset_trees_3d_matrix.png'}")
    print(f"Dataset + controlled tortuosity gallery: {output_dir / '02_dataset_tree_with_controlled_tortuosity.png'}")
    print(f"Controlled branch tortuosity examples: {output_dir / '03_branch_controlled_tortuosity_examples.png'}")
    print(f"Measured tortuosity labels: {output_dir / '04_dataset_trees_with_measured_tortuosity.png'}")
    print(f"Dataset branch tortuosity ranking: {output_dir / '05_dataset_branch_tortuosity_ranking.png'}")
    print(f"Static radius profiles: {output_dir / '06_static_radius_profiles.png'}")
    print(f"Simple tube surface gallery: {output_dir / '07_lca_tube_surfaces.png'}")
    print(f"Validation report: {output_dir / 'validation_report.json'}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
