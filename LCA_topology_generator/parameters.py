# LCA_topology_generator/parameters.py

from dataclasses import dataclass
from typing import Dict
import numpy as np


@dataclass(frozen=True)
class StatRange:
    mean: float
    std: float
    min_value: float
    max_value: float
    unit: str
    note: str


# Literature-guided anatomical values.
# Units are in millimeters and degrees.
BRANCH_STATS: Dict[str, StatRange] = {
    "LMCA_LENGTH": StatRange(9.13, 3.23, 2.0, 19.5, "mm", "LMCA length"),
    "LAD_LENGTH": StatRange(109.46, 14.49, 72.46, 144.78, "mm", "LAD length"),
    "LCX_LENGTH": StatRange(66.27, 11.56, 40.7, 107.66, "mm", "LCX length"),

    "LMCA_DIAMETER": StatRange(4.38, 0.58, 2.25, 5.72, "mm", "LMCA luminal diameter"),
    "LAD_DIAMETER": StatRange(2.62, 0.50, 1.69, 4.06, "mm", "LAD luminal diameter"),
    "LCX_DIAMETER": StatRange(2.41, 0.43, 1.50, 3.67, "mm", "LCX luminal diameter"),

    "LAD_LCX_ANGLE": StatRange(77.88, 19.83, 11.77, 126.0, "degree", "Angle between LAD and LCX"),
}


# Branching probabilities for later extension.
BRANCHING_PROBABILITIES = {
    "bifurcation": 0.6877,
    "trifurcation_ri": 0.2677,
    "tetrafurcation": 0.0101,
}


# Current generator settings
DEFAULT_K_EXPORT = 8
DEFAULT_CANDIDATES = 100
CONTROL_POINTS_PER_BRANCH = 20
CENTERLINE_POINTS_PER_BRANCH = 200

# Validation tolerances
CONNECTION_TOLERANCE_MM = 1e-6
ANGLE_TOLERANCE_DEG = 12.0


def sample_truncated_normal(rng: np.random.Generator, stat: StatRange) -> float:
    """
    Sample value from normal distribution but keep it inside literature-supported range.
    """
    for _ in range(1000):
        value = rng.normal(stat.mean, stat.std)
        if stat.min_value <= value <= stat.max_value:
            return float(value)

    # Fallback if repeated sampling fails
    return float(np.clip(stat.mean, stat.min_value, stat.max_value))