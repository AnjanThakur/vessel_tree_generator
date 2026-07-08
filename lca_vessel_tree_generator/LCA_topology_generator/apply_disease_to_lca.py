import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from matplotlib.lines import Line2D

from .disease_model import (
    BRANCH_NAMES,
    DEFAULT_DISEASE_SETTINGS,
    apply_disease_to_lca_tree,
    example_disease_configs,
    validate_disease_config,
    validate_diseased_lca_tree,
)
from .paths import output_path
from .plaque_model import (
    build_lca_plaque_surfaces,
    has_eccentric_plaque,
    validate_plaque_surfaces,
)
from .tight_mesh import (
    build_lca_tight_mesh,
    save_tight_mesh_ply,
    save_tight_mesh_preview,
    save_tight_mesh_stl,
    validate_tight_mesh,
)
from .tube_surface import build_lca_tube_surfaces, validate_lca_tube_surfaces
from .visualize import set_axes_equal


BRANCH_COLORS = {
    "LMCA": "#222222",
    "LAD": "#d62728",
    "LCX": "#1f77b4",
}
STENOSIS_COLOR = "#ff9f1c"
STENOSIS_EDGE_COLOR = "#7a1f00"
STENOSIS_REDUCTION_THRESHOLD = 0.03


def _write_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, indent=2)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)


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


def _load_radius_tree(patient_dir: Path) -> dict:
    tree_path = patient_dir / "tree_centerline_radius.npz"
    if tree_path.exists():
        data = np.load(tree_path)
        return {branch_name: data[branch_name] for branch_name in BRANCH_NAMES}

    branch_tree = {}
    for branch_name in BRANCH_NAMES:
        branch_path = patient_dir / f"{branch_name.lower()}_centerline_radius.npy"
        if not branch_path.exists():
            raise FileNotFoundError(f"Missing radius centerline file: {branch_path}")
        branch_tree[branch_name] = np.load(branch_path)
    return branch_tree


def _save_radius_tree(case_dir: Path, diseased_tree: dict):
    for branch_name in BRANCH_NAMES:
        np.save(case_dir / f"{branch_name.lower()}_centerline_radius_diseased.npy", diseased_tree[branch_name])
    np.savez(
        case_dir / "tree_centerline_radius_diseased.npz",
        LMCA=diseased_tree["LMCA"],
        LAD=diseased_tree["LAD"],
        LCX=diseased_tree["LCX"],
    )


def _save_diseased_radius_profile(path: Path, baseline_tree: dict, diseased_tree: dict, config: dict):
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    lesion_text_by_branch = {branch_name: [] for branch_name in BRANCH_NAMES}
    for lesion in config["lesions"]:
        if lesion["type"] == "focal":
            lesion_text_by_branch[lesion["branch"]].append(
                f"focal {lesion['severity']:.0%} @ {lesion['center']:.2f}"
            )
        elif lesion["type"] == "diffuse":
            lesion_text_by_branch[lesion["branch"]].append(
                f"diffuse {lesion['severity']:.0%} {lesion['start']:.2f}-{lesion['end']:.2f}"
            )

    for ax, branch_name in zip(axes, BRANCH_NAMES):
        baseline = baseline_tree[branch_name]
        diseased = diseased_tree[branch_name]
        s = _radius_axis(diseased)
        ax.plot(
            s,
            baseline[:, 3],
            color="#777777",
            linewidth=1.5,
            linestyle="--",
            label="baseline",
        )
        ax.plot(
            s,
            diseased[:, 3],
            color=BRANCH_COLORS[branch_name],
            linewidth=2.4,
            label="diseased",
        )
        ax.fill_between(s, diseased[:, 3], baseline[:, 3], color=BRANCH_COLORS[branch_name], alpha=0.15)
        subtitle = "; ".join(lesion_text_by_branch[branch_name]) or "no lesion"
        ax.set_title(f"{branch_name}\n{subtitle}", fontsize=9)
        ax.set_xlabel("Normalized arc length")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("Radius (mm)")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(config["case_id"], fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_diseased_tree_plot(path: Path, baseline_tree: dict, diseased_tree: dict, config: dict):
    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_subplot(projection="3d")

    for branch_name in BRANCH_NAMES:
        baseline = baseline_tree[branch_name]
        points = diseased_tree[branch_name]
        segments = np.stack([points[:-1, :3], points[1:, :3]], axis=1)
        radius_mid = 0.5 * (points[:-1, 3] + points[1:, 3])
        baseline_radius_mid = 0.5 * (baseline[:-1, 3] + baseline[1:, 3])
        reduction_mid = 1.0 - radius_mid / np.maximum(baseline_radius_mid, 1e-9)
        stenosis_mask = reduction_mid > STENOSIS_REDUCTION_THRESHOLD
        widths = 1.0 + 2.3 * radius_mid / max(float(np.max(points[:, 3])), 1e-9)

        normal_segments = segments[~stenosis_mask]
        if len(normal_segments) > 0:
            collection = Line3DCollection(
                normal_segments,
                colors=BRANCH_COLORS[branch_name],
                linewidths=widths[~stenosis_mask],
                alpha=0.78,
            )
            ax.add_collection3d(collection)

        stenosis_segments = segments[stenosis_mask]
        if len(stenosis_segments) > 0:
            outline = Line3DCollection(
                stenosis_segments,
                colors=STENOSIS_EDGE_COLOR,
                linewidths=widths[stenosis_mask] + 2.2,
                alpha=0.96,
            )
            highlight = Line3DCollection(
                stenosis_segments,
                colors=STENOSIS_COLOR,
                linewidths=widths[stenosis_mask] + 1.1,
                alpha=0.98,
            )
            ax.add_collection3d(outline)
            ax.add_collection3d(highlight)

        lesion_point_mask = np.zeros(len(points), dtype=bool)
        lesion_point_mask[:-1] |= stenosis_mask
        lesion_point_mask[1:] |= stenosis_mask
        lesion_points = points[lesion_point_mask]
        if len(lesion_points) > 0:
            ax.scatter(
                lesion_points[:, 0],
                lesion_points[:, 1],
                lesion_points[:, 2],
                color=STENOSIS_COLOR,
                edgecolor=STENOSIS_EDGE_COLOR,
                linewidth=0.45,
                s=16,
                depthshade=False,
                alpha=0.95,
            )
        ax.scatter(
            [points[0, 0], points[-1, 0]],
            [points[0, 1], points[-1, 1]],
            [points[0, 2], points[-1, 2]],
            color=BRANCH_COLORS[branch_name],
            edgecolor="white",
            linewidth=0.6,
            s=[38, 22],
            depthshade=False,
        )

    all_points = np.vstack([diseased_tree[branch_name][:, :3] for branch_name in BRANCH_NAMES])
    ax.auto_scale_xyz(all_points[:, 0], all_points[:, 1], all_points[:, 2])
    set_axes_equal(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Diseased LCA radius tree\n{config['case_id']}", fontsize=10)
    legend_handles = [
        Line2D([0], [0], color=BRANCH_COLORS["LMCA"], linewidth=3.0, label="LMCA normal radius"),
        Line2D([0], [0], color=BRANCH_COLORS["LAD"], linewidth=3.0, label="LAD normal radius"),
        Line2D([0], [0], color=BRANCH_COLORS["LCX"], linewidth=3.0, label="LCX normal radius"),
        Line2D([0], [0], color=STENOSIS_COLOR, linewidth=4.0, label="stenosis radius reduction"),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_plaque_surfaces_and_mesh(
    case_dir: Path,
    baseline_tree: dict,
    config: dict,
    circle_points: int,
    minimum_radius_mm: float,
) -> dict:
    plaque_surfaces, plaque_metadata = build_lca_plaque_surfaces(
        baseline_tree,
        config,
        num_circle_points=circle_points,
        minimum_radius_mm=minimum_radius_mm,
    )
    plaque_validation = validate_plaque_surfaces(plaque_surfaces, baseline_tree, plaque_metadata)
    np.savez(
        case_dir / "plaque_tube_surface.npz",
        LMCA=plaque_surfaces["LMCA"],
        LAD=plaque_surfaces["LAD"],
        LCX=plaque_surfaces["LCX"],
    )

    vertices, faces, mesh_metadata = build_lca_tight_mesh(plaque_surfaces)
    mesh_validation = validate_tight_mesh(vertices, faces, mesh_metadata)
    np.savez(case_dir / "plaque_tight_mesh.npz", vertices=vertices, faces=faces)
    save_tight_mesh_preview(case_dir / "plaque_tight_mesh.png", vertices, faces)
    save_tight_mesh_ply(case_dir / "plaque_tight_mesh.ply", vertices, faces)
    save_tight_mesh_stl(case_dir / "plaque_tight_mesh.stl", vertices, faces)
    _write_json(case_dir / "plaque_surface_validation.json", plaque_validation)
    _write_json(case_dir / "plaque_tight_mesh_validation.json", mesh_validation)

    summary = {
        "mode": "optional_eccentric_plaque_surface_deformation",
        "plaque_surface": plaque_metadata,
        "plaque_surface_validation": plaque_validation,
        "plaque_tight_mesh_validation": mesh_validation,
        "output_shapes": {
            "plaque_tube_surface_LMCA": list(plaque_surfaces["LMCA"].shape),
            "plaque_tube_surface_LAD": list(plaque_surfaces["LAD"].shape),
            "plaque_tube_surface_LCX": list(plaque_surfaces["LCX"].shape),
            "plaque_tight_mesh_vertices": list(vertices.shape),
            "plaque_tight_mesh_faces": list(faces.shape),
        },
    }
    _write_json(case_dir / "plaque_summary.json", summary)
    return summary


def _save_diseased_surfaces_and_mesh(
    case_dir: Path,
    baseline_tree: dict,
    diseased_tree: dict,
    config: dict,
    circle_points: int,
    minimum_radius_mm: float,
) -> dict:
    surfaces, tube_metadata = build_lca_tube_surfaces(diseased_tree, num_circle_points=circle_points)
    tube_validation = validate_lca_tube_surfaces(surfaces, diseased_tree)
    np.savez(
        case_dir / "diseased_tube_surface.npz",
        LMCA=surfaces["LMCA"],
        LAD=surfaces["LAD"],
        LCX=surfaces["LCX"],
    )

    vertices, faces, mesh_metadata = build_lca_tight_mesh(surfaces)
    mesh_validation = validate_tight_mesh(vertices, faces, mesh_metadata)
    np.savez(case_dir / "diseased_tight_mesh.npz", vertices=vertices, faces=faces)
    save_tight_mesh_preview(case_dir / "diseased_tight_mesh.png", vertices, faces)
    save_tight_mesh_ply(case_dir / "diseased_tight_mesh.ply", vertices, faces)
    save_tight_mesh_stl(case_dir / "diseased_tight_mesh.stl", vertices, faces)
    _write_json(case_dir / "diseased_tight_mesh_validation.json", mesh_validation)

    outputs = {
        "tube_surface": tube_metadata,
        "tube_surface_validation": tube_validation,
        "tight_mesh_validation": mesh_validation,
        "plaque_surface_generated": False,
        "output_shapes": {
            "tube_surface_LMCA": list(surfaces["LMCA"].shape),
            "tube_surface_LAD": list(surfaces["LAD"].shape),
            "tube_surface_LCX": list(surfaces["LCX"].shape),
            "tight_mesh_vertices": list(vertices.shape),
            "tight_mesh_faces": list(faces.shape),
        },
    }
    if has_eccentric_plaque(config):
        plaque_outputs = _save_plaque_surfaces_and_mesh(
            case_dir,
            baseline_tree,
            config,
            circle_points=circle_points,
            minimum_radius_mm=minimum_radius_mm,
        )
        outputs["plaque_surface_generated"] = True
        outputs["plaque_surface"] = plaque_outputs["plaque_surface"]
        outputs["plaque_surface_validation"] = plaque_outputs["plaque_surface_validation"]
        outputs["plaque_tight_mesh_validation"] = plaque_outputs["plaque_tight_mesh_validation"]
        outputs["output_shapes"].update(plaque_outputs["output_shapes"])
    return outputs


def _case_summary(patient_id: str, config: dict, metadata: dict, validation: dict, extra_outputs: dict) -> dict:
    branch_shapes = {
        branch_name: validation["branches"].get(branch_name, {}).get("shape")
        for branch_name in BRANCH_NAMES
    }
    return {
        "patient_id": patient_id,
        "case_id": config["case_id"],
        "input_format": ["x", "y", "z", "radius_mm"],
        "disease_operation": "radius-only stenosis; x, y, z unchanged",
        "minimum_radius_mm": metadata["minimum_radius_mm"],
        "lesions": config["lesions"],
        "branch_shapes": branch_shapes,
        "branch_metadata": metadata["branches"],
        "validation_status": "valid" if validation["is_valid"] else "invalid",
        "validation": validation,
        **extra_outputs,
    }


def _patient_dirs(input_dir: Path) -> list:
    return sorted(path for path in input_dir.glob("patient_*") if path.is_dir())


def _configs_for_patient(patient_id: str, config_path: Path | None) -> list:
    if config_path is None:
        return example_disease_configs(patient_id)

    config = _load_json(config_path)
    config.setdefault("case_id", f"{patient_id}_{config_path.stem}")
    if not config["case_id"].startswith(patient_id):
        config["case_id"] = f"{patient_id}_{config['case_id']}"
    return [config]


def process_patient(patient_dir: Path, output_patient_dir: Path, args) -> list:
    patient_id = patient_dir.name
    baseline_tree = _load_radius_tree(patient_dir)
    processed_cases = []

    for raw_config in _configs_for_patient(patient_id, args.config):
        config_validation = validate_disease_config(raw_config)
        if not config_validation["is_valid"]:
            raise ValueError(f"{patient_id} {raw_config.get('case_id', 'case')}: {config_validation['errors']}")
        config = config_validation["normalized_config"]
        case_dir = output_patient_dir / config["case_id"]
        case_dir.mkdir(parents=True, exist_ok=True)

        diseased_tree, metadata = apply_disease_to_lca_tree(
            baseline_tree,
            config,
            minimum_radius_mm=args.minimum_radius_mm,
        )
        validation = validate_diseased_lca_tree(
            baseline_tree,
            diseased_tree,
            config,
            minimum_radius_mm=args.minimum_radius_mm,
            adjacent_jump_threshold_mm=args.adjacent_jump_threshold_mm,
        )

        _save_radius_tree(case_dir, diseased_tree)
        _write_json(case_dir / "disease_config.json", config)
        _write_json(case_dir / "disease_validation.json", validation)
        _save_diseased_radius_profile(case_dir / "diseased_radius_profile.png", baseline_tree, diseased_tree, config)
        _save_diseased_tree_plot(case_dir / "diseased_tree_with_radius.png", baseline_tree, diseased_tree, config)

        extra_outputs = {
            "surface_and_mesh_generated": False,
            "output_shapes": {
                "LMCA": list(diseased_tree["LMCA"].shape),
                "LAD": list(diseased_tree["LAD"].shape),
                "LCX": list(diseased_tree["LCX"].shape),
            },
        }
        if not args.skip_mesh:
            mesh_outputs = _save_diseased_surfaces_and_mesh(
                case_dir,
                baseline_tree,
                diseased_tree,
                config,
                args.tube_circle_points,
                args.minimum_radius_mm,
            )
            extra_outputs["surface_and_mesh_generated"] = True
            extra_outputs["output_shapes"].update(mesh_outputs["output_shapes"])
            extra_outputs["tube_surface"] = mesh_outputs["tube_surface"]
            extra_outputs["tube_surface_validation"] = mesh_outputs["tube_surface_validation"]
            extra_outputs["tight_mesh_validation"] = mesh_outputs["tight_mesh_validation"]
            extra_outputs["plaque_surface_generated"] = mesh_outputs["plaque_surface_generated"]
            if mesh_outputs["plaque_surface_generated"]:
                extra_outputs["plaque_surface"] = mesh_outputs["plaque_surface"]
                extra_outputs["plaque_surface_validation"] = mesh_outputs["plaque_surface_validation"]
                extra_outputs["plaque_tight_mesh_validation"] = mesh_outputs["plaque_tight_mesh_validation"]

        summary = _case_summary(patient_id, config, metadata, validation, extra_outputs)
        _write_json(case_dir / "disease_summary.json", summary)
        processed_cases.append({
            "patient_id": patient_id,
            "case_id": config["case_id"],
            "case_dir": str(case_dir),
            "validation_status": summary["validation_status"],
            "output_shapes": summary["output_shapes"],
            "warnings": validation["warnings"],
            "errors": validation["errors"],
            "surface_and_mesh_generated": extra_outputs["surface_and_mesh_generated"],
            "plaque_surface_generated": extra_outputs.get("plaque_surface_generated", False),
            "tight_mesh_validation_status": (
                "valid"
                if extra_outputs.get("tight_mesh_validation", {}).get("is_valid")
                else ("skipped" if args.skip_mesh else "invalid")
            ),
            "plaque_tight_mesh_validation_status": (
                "valid"
                if extra_outputs.get("plaque_tight_mesh_validation", {}).get("is_valid")
                else ("not_requested" if not extra_outputs.get("plaque_surface_generated") else "invalid")
            ),
        })

    return processed_cases


def main():
    parser = argparse.ArgumentParser(
        description="Apply MVP stenosis/disease radius changes to dataset-driven LCA radius centerlines."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=output_path("dataset_lca", "trees"),
        help="Folder containing normal patient_* radius-enabled LCA tree outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_path("dataset_lca_disease"),
        help="Folder where diseased outputs will be written.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional disease config JSON. If omitted, the three MVP example cases are generated per patient.",
    )
    parser.add_argument(
        "--minimum-radius-mm",
        type=float,
        default=DEFAULT_DISEASE_SETTINGS["minimum_radius_mm"],
        help="Minimum radius clamp after applying stenosis.",
    )
    parser.add_argument(
        "--adjacent-jump-threshold-mm",
        type=float,
        default=DEFAULT_DISEASE_SETTINGS["adjacent_jump_threshold_mm"],
        help="Warning threshold for adjacent diseased radius jumps.",
    )
    parser.add_argument(
        "--tube-circle-points",
        type=int,
        default=24,
        help="Number of ring samples for optional diseased tube surface generation.",
    )
    parser.add_argument(
        "--skip-mesh",
        action="store_true",
        help="Only write diseased centerlines/plots/JSON; skip tube and tight mesh regeneration.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    tree_output_dir = output_dir / "trees"
    tree_output_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = _patient_dirs(input_dir)
    if len(patient_dirs) == 0:
        raise FileNotFoundError(f"No patient_* folders found in {input_dir}")

    all_cases = []
    for patient_dir in patient_dirs:
        output_patient_dir = tree_output_dir / patient_dir.name
        output_patient_dir.mkdir(parents=True, exist_ok=True)
        all_cases.extend(process_patient(patient_dir, output_patient_dir, args))

    summary = {
        "mode": "dataset_lca_disease_mvp",
        "description": "Geometry-independent radius-only stenosis module for static dataset-driven LCA trees.",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "patients_processed": len(patient_dirs),
        "cases_generated": len(all_cases),
        "valid_cases": int(sum(1 for case in all_cases if case["validation_status"] == "valid")),
        "invalid_cases": int(sum(1 for case in all_cases if case["validation_status"] != "valid")),
        "minimum_radius_mm": args.minimum_radius_mm,
        "adjacent_jump_threshold_mm": args.adjacent_jump_threshold_mm,
        "surface_and_mesh_generated": not args.skip_mesh,
        "optional_eccentric_plaque_supported": True,
        "plaque_surface_cases": int(sum(1 for case in all_cases if case.get("plaque_surface_generated"))),
        "example_cases_per_patient": 1 if args.config else 3,
        "cases": all_cases,
        "limitations": [
            "Disease modifies radius only.",
            "Optional eccentric plaque modifies tube surface rings only and leaves radius centerline files unchanged.",
            "No plaque material, flow, motion, or pulsatility is modeled.",
            "No clinical stenosis validation is performed.",
            "The module is geometry-independent and can be rerun after anatomical orientation is corrected.",
        ],
    }
    _write_json(output_dir / "disease_summary.json", summary)
    _write_json(
        output_dir / "disease_validation.json",
        {
            "total_cases": len(all_cases),
            "valid_cases": summary["valid_cases"],
            "invalid_cases": summary["invalid_cases"],
            "cases": all_cases,
        },
    )

    print(f"Processed {len(patient_dirs)} patients from: {input_dir}")
    print(f"Generated {len(all_cases)} disease cases in: {output_dir}")
    print(f"Valid disease cases: {summary['valid_cases']}")
    print(f"Invalid disease cases: {summary['invalid_cases']}")
    if not args.skip_mesh:
        print("Regenerated diseased tube surfaces and tight meshes.")
    print(f"Summary: {output_dir / 'disease_summary.json'}")


if __name__ == "__main__":
    main()
