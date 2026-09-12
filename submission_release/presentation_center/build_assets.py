"""Build the copied, offline-safe visual evidence package."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .presentation_data import CENTER, DEMO_ROOT, RELEASE, ROOT, presentation_summary
from .verification import frozen_hash_snapshot


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    figures = CENTER / "figures"
    offline_assets = CENTER / "offline" / "assets"
    records = CENTER / "verification_records"
    figures.mkdir(parents=True, exist_ok=True)
    offline_assets.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)

    before = frozen_hash_snapshot()
    (records / "pre_presentation_frozen_hashes.json").write_text(
        json.dumps(before, indent=2) + "\n", encoding="utf-8"
    )

    validation_figures = RELEASE / "final_validation"
    for index, stem in (
        (1, "cohort_funnel"),
        (2, "ppt_two_plane_two_ellipse_summary"),
        (3, "pca_variance"),
        (4, "real_vs_generated_lengths"),
        (5, "real_vs_generated_angles"),
        (6, "real_vs_generated_tortuosity"),
        (7, "landmark_distributions"),
        (8, "scaffold_distributions"),
        (9, "generation_novelty"),
        (10, "holdout_results"),
        (11, "static_population_montage"),
        (12, "disease_comparison"),
        (13, "motion_qc"),
        (14, "pulsatility_qc"),
        (15, "vtk_export_qc"),
    ):
        name = f"{index:02d}_{stem}.png"
        copy(validation_figures / name, figures / name)

    anatomy = (
        ROOT
        / "outputs/lca_ssm/ppt_priority_completion/vtk_visualization/91.label/preview_06_complete_model.png"
    )
    copy(anatomy, figures / "anatomical_two_plane_model.png")
    copy(DEMO_ROOT / "disease_mode_comparison.png", figures / "disease_mode_comparison.png")
    for name in ("healthy", "focal_lad", "diffuse_lcx", "tandem_lad"):
        case = DEMO_ROOT / name
        copy(case / "preview.png", figures / f"{name}_preview.png")
        copy(case / "visualizations/validation_dashboard.png", figures / f"{name}_validation_dashboard.png")
        copy(case / "visualizations/cardiac_cycle.gif", figures / f"{name}_cardiac_cycle.gif")
    offline_names = (
        "01_cohort_funnel.png",
        "02_ppt_two_plane_two_ellipse_summary.png",
        "03_pca_variance.png",
        "09_generation_novelty.png",
        "10_holdout_results.png",
        "11_static_population_montage.png",
        "12_disease_comparison.png",
        "13_motion_qc.png",
        "14_pulsatility_qc.png",
        "15_vtk_export_qc.png",
        "disease_mode_comparison.png",
        "focal_lad_validation_dashboard.png",
        "focal_lad_cardiac_cycle.gif",
    )
    for name in offline_names:
        copy(figures / name, offline_assets / name)

    summary = presentation_summary()
    (CENTER / "offline" / "RESULT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Built {len(list(figures.iterdir()))} presentation figures")
    print(f"Captured {before['file_count']} frozen files before presentation QA")


if __name__ == "__main__":
    main()
