"""3D Static Rendering and Global Pipeline QC Exporter for Batch 7."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np

try:
    from motion.contraction_curve import contraction_curve
except ImportError:
    from pca_ssm_vessel_tree_generator.motion.contraction_curve import contraction_curve

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


BRANCH_COLORS = {
    "RCA": "darkorange",
    "LMCA": "#222222",
    "LAD": "#d62728",
    "LCX": "#087e8b",
}


def prepare_branch_plot_series(
    geom_static: np.ndarray, branch_names: list[str] | None
) -> list[dict[str, Any]]:
    """Return labeled plot series so coordinate/color identity is testable."""
    result = []
    for branch_index in range(geom_static.shape[0]):
        name = branch_names[branch_index] if branch_names and branch_index < len(branch_names) else f"branch_{branch_index}"
        result.append({
            "name": name,
            "points": geom_static[branch_index, :, :3],
            "radii": geom_static[branch_index, :, 3],
            "color": BRANCH_COLORS.get(name, "forestgreen"),
        })
    return result


def relative_ellipsoid_volume_percent(
    phases: np.ndarray,
    *,
    radial_amplitude: float,
    longitudinal_amplitude: float,
    peak_phase: float,
) -> np.ndarray:
    """Return true V/V0 for the documented triaxial deformation."""
    response = np.asarray([contraction_curve(float(phase), peak_phase=peak_phase) for phase in phases])
    radial_factor = 1.0 - radial_amplitude * response
    longitudinal_factor = 1.0 - longitudinal_amplitude * response
    return 100.0 * radial_factor**2 * longitudinal_factor


def save_tree_visualization(
    tree_dir: Path, geom_static: np.ndarray, branch_names: list[str] | None = None
) -> bool:
    """Save 3D static centerline rendering plot visualization.png for a single synthetic tree."""
    if not HAS_MATPLOTLIB:
        return False

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    for series in prepare_branch_plot_series(geom_static, branch_names):
        name, pts, radii = series["name"], series["points"], series["radii"]
        if name in BRANCH_COLORS:
            lw = float(np.mean(radii) * 2.0)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], label=name, color=series["color"], linewidth=lw)
        else:
            lw = float(np.mean(radii) * 2.0)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="forestgreen", alpha=0.7, linewidth=lw)

    ax.set_title(f"3D Synthetic Tree Geometry — {tree_dir.name}")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend(loc="upper right")
    plt.tight_layout()

    out_file = tree_dir / "visualization.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    return True


def save_pipeline_qc_report(output_dir: Path, pipeline_report: dict[str, Any]) -> bool:
    """Save global 4-panel pipeline quality control report plot pipeline_qc_report.png."""
    if not HAS_MATPLOTLIB:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Patient Pipeline Funnel Accounting
    p_acc = pipeline_report.get("patient_accounting", {})
    stages = ["Source labels", "Protected LCA", "Assignments resolved", "Core anatomy pass", "PCA eligible"]
    counts = [
        p_acc.get("source_label_inventory_count", 0),
        p_acc.get("source_cases_audited", 0),
        p_acc.get("assignments_confidence_resolved", 0),
        p_acc.get("core_anatomy_gate_pass_count", 0),
        p_acc.get("statistics_eligible_lca_cases", 0),
    ]

    axes[0, 0].bar(stages, counts, color=["#23395d", "navy", "steelblue", "teal", "darkgreen"])
    axes[0, 0].set_title("1. End-to-End Patient Cohort Funnel")
    axes[0, 0].set_ylabel("Patient Count")
    axes[0, 0].tick_params(axis="x", labelrotation=15, labelsize=8)
    for i, value in enumerate(counts):
        axes[0, 0].text(i, value + 2, str(value), ha="center", fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Statistical Shape Model Retained Modes
    ssm = pipeline_report.get("statistical_shape_model", {})
    k_modes = int(ssm.get("retained_pca_modes_k", 0))
    cum_var = float(ssm.get("retained_cumulative_variance_percent", 0.0))
    explained = np.asarray(ssm.get("explained_variance_ratio", []), dtype=float)

    if k_modes and len(explained) >= k_modes:
        cumulative = 100.0 * np.cumsum(explained[:k_modes])
        axes[0, 1].plot(range(1, k_modes + 1), cumulative, "o-", color="purple")
    axes[0, 1].axhline(95.0, color="crimson", linestyle="--", label="95% Target Variance")
    axes[0, 1].set_title(f"2. SSM Retained Modes ({k_modes} modes = {cum_var:.2f}%)")
    axes[0, 1].set_xlabel("PCA Mode Index j")
    axes[0, 1].set_ylabel("Cumulative Variance (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Synthetic Tree Generation Acceptance
    g_sum = pipeline_report.get("synthetic_generation", {})
    acc = int(g_sum.get("num_trees_accepted", 0))
    rej = max(int(g_sum.get("total_sampling_attempts", 0)) - acc, 0)
    axes[1, 0].pie([acc, rej], labels=[f"Accepted ({acc})", f"Rejected ({rej})"], colors=["limegreen", "tomato"], autopct="%1.1f%%", startangle=90)
    axes[1, 0].set_title(f"3. Synthetic Tree Sampling ({g_sum.get('acceptance_rate_percent', 0.0):.1f}% Acceptance)")

    # Panel 4: Cardiac Phase Snapshots Generated
    c_mot = pipeline_report.get("cardiac_motion", {})
    n_trees = int(c_mot.get("num_4d_trees_processed", 0))
    n_phases = int(c_mot.get("num_phases_per_tree", 0))
    tot_snaps = int(c_mot.get("total_3d_snapshots_generated", 0))

    phases = np.asarray(c_mot.get("phase_values", []), dtype=float)
    if n_phases and len(phases) != n_phases:
        raise ValueError("cardiac-motion phase_values do not match num_phases_per_tree")
    radial = float(c_mot.get("motion_parameters", {}).get("radial_amplitude", 0.0))
    longitudinal = float(c_mot.get("motion_parameters", {}).get("longitudinal_amplitude", 0.0))
    peak_phase = float(c_mot.get("peak_systole_phase", 0.35))
    volumes = relative_ellipsoid_volume_percent(
        phases,
        radial_amplitude=radial,
        longitudinal_amplitude=longitudinal,
        peak_phase=peak_phase,
    )
    axes[1, 1].plot(phases, volumes, "s-", color="crimson", linewidth=2)
    axes[1, 1].set_title(f"4. 4D Motion Snapshots ({n_trees} trees × {n_phases} phases = {tot_snaps})")
    axes[1, 1].set_xlabel("Cardiac Phase φ")
    axes[1, 1].set_ylabel("Relative Ellipsoid Volume (%)")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_file = output_dir / "pipeline_qc_report.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    return True
