# LCA_topology_generator/visualize.py

from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth_polyline(points: np.ndarray, n: int = 200) -> np.ndarray:
    """
    Lightweight smooth-looking interpolation without needing scipy.
    This is for visualization only.
    """
    points = np.asarray(points, dtype=float)

    distances = np.zeros(len(points))
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))

    total = distances[-1]
    if total <= 1e-9:
        return points

    target = np.linspace(0.0, total, n)

    smooth = np.zeros((n, 3))
    for dim in range(3):
        smooth[:, dim] = np.interp(target, distances, points[:, dim])

    return smooth


def set_axes_equal(ax) -> None:
    """
    Make 3D plot axes scale similar so anatomy is not visually distorted.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range) / 2.0

    x_middle = sum(x_limits) / 2.0
    y_middle = sum(y_limits) / 2.0
    z_middle = sum(z_limits) / 2.0

    ax.set_xlim3d([x_middle - max_range, x_middle + max_range])
    ax.set_ylim3d([y_middle - max_range, y_middle + max_range])
    ax.set_zlim3d([z_middle - max_range, z_middle + max_range])


def plot_lca_tree(
    branches: Dict[str, np.ndarray],
    output_path: str,
    title: str = "Topology-Driven LCA Control Points",
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for name, points in branches.items():
        smooth = smooth_polyline(points, n=200)

        # Smooth path
        ax.plot(
            smooth[:, 0],
            smooth[:, 1],
            smooth[:, 2],
            linewidth=2.5,
            label=f"{name} path",
        )

        # Raw control points
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=18,
            alpha=0.75,
        )

        end = points[-1]
        ax.text(end[0], end[1], end[2], name)

    bif = branches["LMCA"][-1]
    ax.scatter([bif[0]], [bif[1]], [bif[2]], s=90)
    ax.text(bif[0], bif[1], bif[2], "BIF")

    ax.set_title(title)
    ax.set_xlabel("X / forward (mm)")
    ax.set_ylabel("Y / lateral (mm)")
    ax.set_zlabel("Z / inferior-down (mm)")
    ax.legend()
    ax.grid(True)

    set_axes_equal(ax)

    plt.tight_layout()
    plt.savefig(output, dpi=220)
    plt.close(fig)
