"""Joint empirical sampling of fixed surface-coordinate branch trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TrajectorySample:
    branches: dict[str, np.ndarray]
    local_deviations: dict[str, np.ndarray]
    cardiac_points: dict[str, np.ndarray]
    source_case_id: str
    exact_local_deviations_available: bool
    exact_cardiac_points_available: bool


class TrajectorySampler:
    """Consume Person 1 fixed ``(u,v,offset)`` paths without pointwise noise."""

    def __init__(
        self,
        branches: dict[str, np.ndarray],
        case_ids: dict[str, np.ndarray],
        local_deviations: dict[str, np.ndarray] | None = None,
        cardiac_points: dict[str, np.ndarray] | None = None,
    ):
        self.branches = branches
        self.case_ids = {name: np.asarray(values).astype(str) for name, values in case_ids.items()}
        self.exact_local_deviations_available = local_deviations is not None
        self.exact_cardiac_points_available = cardiac_points is not None
        self.local_deviations = {} if local_deviations is None else local_deviations
        self.cardiac_points = {} if cardiac_points is None else cardiac_points
        for name, values in branches.items():
            array = np.asarray(values, dtype=float)
            if array.ndim != 3 or array.shape[2] != 3:
                raise ValueError(f"{name} trajectories must have shape (cases, points, 3); got {array.shape}")
            if len(array) != len(self.case_ids[name]):
                raise ValueError(f"{name} trajectory and case-id counts differ")
            self.branches[name] = array
            if name not in self.local_deviations:
                self.local_deviations[name] = np.zeros_like(array)
            deviation = np.asarray(self.local_deviations[name], dtype=float)
            if deviation.shape != array.shape or not np.all(np.isfinite(deviation)):
                raise ValueError(
                    f"{name} local deviations must have shape {array.shape}; got {deviation.shape}"
                )
            self.local_deviations[name] = deviation
            if name not in self.cardiac_points:
                self.cardiac_points[name] = np.zeros_like(array)
            cardiac = np.asarray(self.cardiac_points[name], dtype=float)
            if cardiac.shape != array.shape or not np.all(np.isfinite(cardiac)):
                raise ValueError(
                    f"{name} cardiac controls must have shape {array.shape}; got {cardiac.shape}"
                )
            self.cardiac_points[name] = cardiac

    @classmethod
    def from_npz(cls, path: Path) -> "TrajectorySampler":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"missing Person 1 fixed trajectory artifact: {path}")
        branches: dict[str, np.ndarray] = {}
        case_ids: dict[str, np.ndarray] = {}
        local_deviations: dict[str, np.ndarray] = {}
        cardiac_points: dict[str, np.ndarray] = {}
        with np.load(path, allow_pickle=False) as data:
            for name in ("LMCA", "LAD", "LCX", "RCA"):
                trajectory_key = f"{name}_uvo"
                cases_key = f"{name}_case_ids"
                if trajectory_key in data.files and cases_key in data.files:
                    branches[name] = np.asarray(data[trajectory_key]).copy()
                    case_ids[name] = np.asarray(data[cases_key]).copy()
                    deviation_key = f"{name}_local_deviation"
                    if deviation_key in data.files:
                        local_deviations[name] = np.asarray(data[deviation_key]).copy()
                    cardiac_key = f"{name}_cardiac_points"
                    if cardiac_key in data.files:
                        cardiac_points[name] = np.asarray(data[cardiac_key]).copy()
        if not {"LMCA", "LAD", "LCX"}.issubset(branches):
            raise ValueError(f"fixed trajectory artifact is missing mandatory LCA arrays: {path}")
        return cls(
            branches,
            case_ids,
            local_deviations if len(local_deviations) == len(branches) else None,
            cardiac_points if len(cardiac_points) == len(branches) else None,
        )

    def sample(
        self,
        rng: np.random.Generator,
        *,
        source_case_id: str | None,
        include_rca: bool,
    ) -> TrajectorySample:
        required = ["LMCA", "LAD", "LCX"] + (["RCA"] if include_rca else [])
        common = set(self.case_ids[required[0]].tolist())
        for name in required[1:]:
            common &= set(self.case_ids[name].tolist())
        if not common:
            raise ValueError(f"no common empirical trajectory case covers {required}")
        if source_case_id is not None and str(source_case_id) in common:
            selected = str(source_case_id)
        else:
            choices = sorted(common, key=lambda value: int(value.split(".", 1)[0]))
            selected = choices[int(rng.integers(0, len(choices)))]
        result = {}
        deviations = {}
        cardiac = {}
        for name in required:
            index = int(np.flatnonzero(self.case_ids[name] == selected)[0])
            result[name] = self.branches[name][index].copy()
            deviations[name] = self.local_deviations[name][index].copy()
            cardiac[name] = self.cardiac_points[name][index].copy()
        return TrajectorySample(
            result,
            deviations,
            cardiac,
            selected,
            self.exact_local_deviations_available,
            self.exact_cardiac_points_available,
        )
