# LCA_topology_generator/visualize.py

from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_lca_tree(
    branches: Dict[str, np.ndarray],
    output_path: str,
    title: str = "Topology-Driven LCA Control Points",
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for name, points in branches.items():
        ax.plot(points[:, 0], points[:, 1], points[:, 2], marker="o", label=name)

        # Label branch end
        end = points[-1]
        ax.text(end[0], end[1], end[2], name)

    # Mark bifurcation
    bif = branches["LMCA"][-1]
    ax.scatter([bif[0]], [bif[1]], [bif[2]], s=80)
    ax.text(bif[0], bif[1], bif[2], "BIF")

    ax.set_title(title)
    ax.set_xlabel("X / forward (mm)")
    ax.set_ylabel("Y / lateral (mm)")
    ax.set_zlabel("Z / inferior-down (mm)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close(fig)