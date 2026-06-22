# LCA_topology_generator/parameters.py

from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass(frozen=True)
class StatRange:
    mean: float
    std: float
    min_value: float
    max_value: float
    unit: str
    note: str


# A. Literature-derived anatomical values.
# These are measured/statistical anatomy inputs. Units are millimeters
# and degrees, depending on the field.
BRANCH_STATS: Dict[str, StatRange] = {
    "LMCA_LENGTH": StatRange(9.13, 3.23, 2.0, 19.5, "mm", "LMCA length"),
    "LAD_LENGTH": StatRange(109.46, 14.49, 72.46, 144.78, "mm", "LAD length"),
    "LCX_LENGTH": StatRange(66.27, 11.56, 40.7, 107.66, "mm", "LCX length"),

    "LMCA_DIAMETER": StatRange(4.38, 0.58, 2.25, 5.72, "mm", "LMCA luminal diameter"),
    "LAD_DIAMETER": StatRange(2.62, 0.50, 1.69, 4.06, "mm", "LAD luminal diameter"),
    "LCX_DIAMETER": StatRange(2.41, 0.43, 1.50, 3.67, "mm", "LCX luminal diameter"),

    "LAD_LCX_ANGLE": StatRange(77.88, 19.83, 11.77, 126.0, "degree", "Angle between LAD and LCX"),
}


# Literature-derived branching probabilities for later topology expansion.
BRANCHING_PROBABILITIES = {
    "bifurcation": 0.6877,
    "trifurcation_ri": 0.2677,
    "tetrafurcation": 0.0101,
}


# B. Engineering/model-derived shaping parameters.
# These coefficients make the sampled control points anatomically plausible
# while preserving the topology. They are modelling choices, not values
# directly measured from the cited papers.
ENGINEERING_SHAPE_PARAMETERS: Dict[str, Dict[str, Any]] = {
    "LMCA": {
        "bifurcation_y_std_mm": 0.35,
        "bifurcation_z_std_mm": 0.20,
        "side_vector": (0.0, 1.0, 0.0),
        "vertical_vector": (0.0, 0.0, -1.0),
        "side_strength_range": (0.005, 0.015),
        "vertical_strength_range": (-0.005, 0.005),
        "anchor_count": 50,
    },
    "LAD": {
        "direction_mean": (0.30, -0.12, -0.94),
        "direction_std": (0.04, 0.05, 0.04),
        "side_vector_x_mean": 0.10,
        "side_vector_x_std": 0.05,
        "side_vector_y": -1.0,
        "side_vector_z_mean": 0.05,
        "side_vector_z_std": 0.03,
        "vertical_curve_vector": (0.25, 0.05, -1.0),
        "side_strength_range": (0.015, 0.045),
        "vertical_strength_range": (0.015, 0.035),
        "anchor_count": 100,
        "wraparound_probability": 0.86,
        "wraparound_start_s": 0.70,
        "wraparound_strength_range_mm": (4.0, 8.0),
        "wraparound_vector_x_range": (0.20, 0.45),
        "wraparound_vector_y_range": (0.15, 0.35),
        "wraparound_vector_z_range": (-0.20, 0.05),
    },
    "LCX": {
        "preferred_lateral_side": (0.0, 1.0, 0.0),
        "lateral_vector": (0.0, 1.0, -0.05),
        "posterior_curve_vector": (-0.35, 0.35, -0.12),
        "lateral_strength_range": (0.06, 0.12),
        "posterior_strength_range": (0.06, 0.14),
        "downward_strength_range": (0.04, 0.09),
        "wave_strength_range": (0.003, 0.010),
        "sweep_start_s": 0.18,
        "lateral_exponent": 1.8,
        "posterior_exponent": 2.1,
        "downward_exponent": 1.5,
        "anchor_count": 120,
    },
}


SIDE_BRANCH_PARAMETRIC_POSITIONS: Dict[str, Dict[str, Any]] = {
    "RI": {
        "parent": "LMCA",
        "attach": "bifurcation",
        "status": "future_optional",
        "reason": "trifurcation branch",
    },
    "D1": {
        "parent": "LAD",
        "s_range": [0.25, 0.45],
        "status": "future_optional",
    },
    "D2": {
        "parent": "LAD",
        "s_range": [0.45, 0.70],
        "status": "future_optional",
    },
    "D3": {
        "parent": "LAD",
        "s_range": [0.65, 0.85],
        "status": "optional_if_multiple_diagonals",
    },
    "Septal": {
        "parent": "LAD",
        "s_range": [0.15, 0.75],
        "status": "future_optional",
    },
    "OM1": {
        "parent": "LCX",
        "s_range": [0.25, 0.50],
        "status": "future_optional",
    },
    "OM2": {
        "parent": "LCX",
        "s_range": [0.50, 0.75],
        "status": "future_optional",
    },
    "OM3": {
        "parent": "LCX",
        "s_range": [0.70, 0.90],
        "status": "optional_if_multiple_marginals",
    },
    "Left_PLV": {
        "parent": "LCX",
        "s_range": [0.75, 0.95],
        "status": "dominance_based_future_optional",
    },
    "Left_PDA": {
        "parent": "LCX",
        "s_range": [0.85, 1.00],
        "status": "left_dominance_future_optional",
    },
}


# Current generator settings.
DEFAULT_K_EXPORT = 50
DEFAULT_CANDIDATES = 1000
CONTROL_POINTS_PER_BRANCH = 20
CENTERLINE_POINTS_PER_BRANCH = 200

# Validation tolerances.
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
