"""3D visual QC plot generator for Batch 1 extractions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_batch1_qc_plot(
    centerlines: dict[str, np.ndarray | None],
    landmarks: dict[str, np.ndarray | None],
    output_path: Path,
    patient_id: str,
    validation_res: dict[str, Any],
) -> None:
    """Generate 3D Matplotlib visual QC plot showing 4 centerlines and 6 landmarks in physical RAS frame."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = {
        "LMCA": "#1A1A1A",   # Black
        "LAD": "#D62828",    # Red
        "LCX": "#1679B8",    # Blue
        "RCA": "#77589A",    # Purple
    }

    # Plot centerlines
    for name in ("LMCA", "LAD", "LCX", "RCA"):
        pts = centerlines.get(name)
        if pts is not None and len(pts) >= 2:
            ax.plot(
                pts[:, 0], pts[:, 1], pts[:, 2],
                color=colors.get(name, "#555555"),
                linewidth=2.5,
                label=f"{name} centerline"
            )
            ax.scatter(
                pts[0, 0], pts[0, 1], pts[0, 2],
                color=colors.get(name, "#555555"),
                s=15
            )

    # Plot 6 landmarks
    lca_ost = landmarks.get("lca_ostium")
    rca_ost = landmarks.get("rca_ostium")
    bif = landmarks.get("bifurcation")
    lad_end = landmarks.get("lad_endpoint")
    lcx_end = landmarks.get("lcx_endpoint")
    rca_end = landmarks.get("rca_endpoint")

    if lca_ost is not None:
        ax.scatter(*lca_ost, color="#FFBE0B", s=100, marker="*", label="LCA Ostium", edgecolor="black", zorder=10)
    if rca_ost is not None:
        ax.scatter(*rca_ost, color="#FF00FF", s=100, marker="*", label="RCA Ostium", edgecolor="black", zorder=10)
    if bif is not None:
        ax.scatter(*bif, color="#2CA02C", s=80, marker="o", label="LMCA Bifurcation", edgecolor="black", zorder=10)

    if lad_end is not None:
        ax.scatter(*lad_end, color="#D62828", s=60, marker="s", label="LAD Endpoint", edgecolor="black", zorder=10)
    if lcx_end is not None:
        ax.scatter(*lcx_end, color="#1679B8", s=60, marker="s", label="LCX Endpoint", edgecolor="black", zorder=10)
    if rca_end is not None and validation_res.get("rca_resolved", False):
        ax.scatter(*rca_end, color="#77589A", s=60, marker="s", label="RCA Endpoint", edgecolor="black", zorder=10)

    status_str = validation_res.get("overall_status", "unknown")
    ax.set_title(f"Patient {patient_id} — Batch 1 Extraction ({status_str})", fontsize=11, fontweight="bold")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
