"""3D Visual QC Plotter for Batch 2 Alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_batch2_qc_plot(
    centerlines_scanner: dict[str, np.ndarray],
    landmarks_scanner: dict[str, np.ndarray],
    centerlines_cardiac: dict[str, np.ndarray],
    landmarks_cardiac: dict[str, np.ndarray],
    output_path: Path,
    patient_id: str,
    status_str: str,
) -> None:
    """Generate side-by-side 3D visual QC plot: Scanner RAS (Left) vs. Canonical Cardiac Frame (Right)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 6))

    colors = {
        "LMCA": "#1A1A1A",   # Black
        "LAD": "#D62828",    # Red
        "LCX": "#1679B8",    # Blue
        "RCA": "#77589A",    # Purple
    }

    # Left Subplot: Scanner RAS Input
    ax1 = fig.add_subplot(121, projection="3d")
    for name, pts in centerlines_scanner.items():
        if pts is not None and len(pts) >= 2:
            ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors.get(name, "#555555"), linewidth=2.0, label=name)

    lca_ost = landmarks_scanner.get("lca_ostium")
    bif = landmarks_scanner.get("bifurcation")
    if lca_ost is not None:
        ax1.scatter(*lca_ost, color="#FFBE0B", s=80, marker="*", label="LCA Ostium", edgecolor="black")
    if bif is not None:
        ax1.scatter(*bif, color="#2CA02C", s=60, marker="o", label="Bifurcation", edgecolor="black")

    ax1.set_title(f"{patient_id} — Scanner RAS Frame", fontsize=10, fontweight="bold")
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.legend(loc="upper right", fontsize=7)

    # Right Subplot: Canonical Cardiac Frame Output
    ax2 = fig.add_subplot(122, projection="3d")
    for name, pts in centerlines_cardiac.items():
        if pts is not None and len(pts) >= 2:
            ax2.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors.get(name, "#555555"), linewidth=2.5, label=name)

    lca_c = landmarks_cardiac.get("lca_ostium")
    rca_c = landmarks_cardiac.get("rca_ostium")
    bif_c = landmarks_cardiac.get("bifurcation")
    lad_e = landmarks_cardiac.get("lad_endpoint")
    lcx_e = landmarks_cardiac.get("lcx_endpoint")

    if lca_c is not None:
        ax2.scatter(*lca_c, color="#FFBE0B", s=100, marker="*", label="LCA Ostium", edgecolor="black", zorder=10)
    if rca_c is not None:
        ax2.scatter(*rca_c, color="#FF00FF", s=100, marker="*", label="RCA Ostium", edgecolor="black", zorder=10)
    if bif_c is not None:
        ax2.scatter(*bif_c, color="#2CA02C", s=80, marker="o", label="Bifurcation", edgecolor="black", zorder=10)
    if lad_e is not None:
        ax2.scatter(*lad_e, color="#D62828", s=60, marker="s", label="LAD End", edgecolor="black", zorder=10)
    if lcx_e is not None:
        ax2.scatter(*lcx_e, color="#1679B8", s=60, marker="s", label="LCX End", edgecolor="black", zorder=10)

    # Draw Z=0 coronary plane grid
    xlim = ax2.get_xlim()
    ylim = ax2.get_ylim()
    gx, gy = np.meshgrid(np.linspace(xlim[0], xlim[1], 10), np.linspace(ylim[0], ylim[1], 10))
    gz = np.zeros_like(gx)
    ax2.plot_surface(gx, gy, gz, color="cyan", alpha=0.1, shade=False)

    # Draw coordinate axes lines from origin (0,0,0)
    ax2.quiver(0, 0, 0, 20, 0, 0, color="red", arrow_length_ratio=0.1, label="+X (LCA Ostium)")
    ax2.quiver(0, 0, 0, 0, 20, 0, color="green", arrow_length_ratio=0.1, label="+Y")
    ax2.quiver(0, 0, 0, 0, 0, 20, color="blue", arrow_length_ratio=0.1, label="+Z (Base->Apex)")

    ax2.set_title(f"{patient_id} — Canonical Cardiac Frame ({status_str})", fontsize=10, fontweight="bold")
    ax2.set_xlabel("X (mm)")
    ax2.set_ylabel("Y (mm)")
    ax2.set_zlabel("Z (mm)")
    ax2.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
