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
from .tortuosity import calculate_lca_tortuosity, calculate_tortuosity
from .tortuosity_augmentation import make_target_tortuosity_variant
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


def _build_dataset_samples(tree_ctrl_points: np.ndarray, args) -> list:
    samples = []
    for tree_index, tree in enumerate(tree_ctrl_points):
        centerlines = interpolate_lca_tree(
            tree,
            lmca_points=args.lmca_points,
            lad_points=args.lad_points,
            lcx_points=args.lcx_points,
        )
        metrics = calculate_lca_tortuosity(centerlines)
        validation = validate_lca_tree(tree, centerlines, metrics)
        samples.append({
            "tree_index": tree_index,
            "control_tree": tree,
            "centerlines": centerlines,
            "metrics": metrics,
            "validation": validation,
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
        for branch_name, points in sample["centerlines"].items():
            np.save(sample_dir / f"{branch_name.lower()}_centerline.npy", points)
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
        "trees": [
            {
                "tree_index": sample["tree_index"],
                "validation_status": "valid" if sample["validation"]["is_valid"] else "invalid",
                "tortuosity": {
                    branch_name: sample["metrics"][branch_name]["tortuosity"]
                    for branch_name in ["LMCA", "LAD", "LCX"]
                },
            }
            for sample in export_samples
        ],
    }

    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "validation_report.json", validation_report)
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
    print(f"Validation report: {output_dir / 'validation_report.json'}")
    print(f"Summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
