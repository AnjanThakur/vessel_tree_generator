"""Ellipsoid parameter sampling with a strict Person 1 handoff adapter."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EllipsoidParameters:
    """Positive triaxial ellipsoid semi-axes in the common cardiac frame."""

    a: float
    b: float
    c: float
    sampling_method: str
    source_case_id: str | None = None

    def __post_init__(self) -> None:
        axes = np.asarray([self.a, self.b, self.c], dtype=float)
        if not np.all(np.isfinite(axes)) or np.any(axes <= 0.0):
            raise ValueError(f"ellipsoid axes must be finite and positive; got {axes.tolist()}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ParameterSampler:
    """Sample ``(a,b,c)`` without silently discarding cross-axis dependence.

    When valid per-patient rows are available, empirical joint sampling is the
    default.  The independent truncated-normal fallback is intended for the
    controlled Week-1 demo or an explicitly requested provisional mode only.
    """

    def __init__(self, scaffold_statistics: dict[str, Any], empirical_rows: list[dict[str, Any]] | None = None):
        self.scaffold_statistics = scaffold_statistics
        self.empirical_rows = empirical_rows or []
        for axis in ("a", "b", "c"):
            if axis not in scaffold_statistics:
                raise ValueError(f"missing scaffold statistics for axis {axis!r}")

    @classmethod
    def controlled(cls, a: float = 60.0, b: float = 50.0, c: float = 70.0) -> "ParameterSampler":
        """Build a deterministic, explicitly non-population demo sampler."""
        stats = {
            axis: {"mean": float(value), "std": 0.0, "p2_5": float(value), "p97_5": float(value)}
            for axis, value in (("a", a), ("b", b), ("c", c))
        }
        return cls(stats)

    @classmethod
    def from_person1_output(cls, directory: Path) -> "ParameterSampler":
        """Load Person 1 scaffold statistics and valid joint patient rows."""
        directory = Path(directory)
        stats_path = directory / "population_surface_statistics.json"
        rows_path = directory / "population_ellipsoid_parameters.csv"
        missing = [str(path) for path in (stats_path, rows_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Person 1 generator handoff is incomplete; missing: " + ", ".join(missing)
            )
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        scaffold = payload.get("scaffold")
        if not isinstance(scaffold, dict):
            raise ValueError(f"missing 'scaffold' mapping in {stats_path}")
        rows: list[dict[str, Any]] = []
        with rows_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                valid_text = str(row.get("is_valid", "true")).strip().lower()
                if valid_text not in {"true", "1", "yes"}:
                    continue
                try:
                    axes = [float(row[name]) for name in ("a", "b", "c")]
                except (KeyError, TypeError, ValueError):
                    continue
                if np.all(np.isfinite(axes)) and min(axes) > 0.0:
                    rows.append({"case_id": row.get("case_id"), "a": axes[0], "b": axes[1], "c": axes[2]})
        if not rows:
            raise ValueError(f"no valid joint ellipsoid rows found in {rows_path}")
        return cls(scaffold, rows)

    @staticmethod
    def _sample_axis(stats: dict[str, Any], rng: np.random.Generator) -> float:
        mean = float(stats["mean"])
        std = max(float(stats.get("std", 0.0)), 0.0)
        lower = float(stats.get("p2_5", mean - 2.0 * std))
        upper = float(stats.get("p97_5", mean + 2.0 * std))
        if not np.isfinite(lower):
            lower = mean - 2.0 * std
        if not np.isfinite(upper):
            upper = mean + 2.0 * std
        lower = max(lower, np.finfo(float).eps)
        if upper < lower:
            raise ValueError(f"invalid scaffold bounds [{lower}, {upper}]")
        value = mean if std == 0.0 else float(rng.normal(mean, std))
        return float(np.clip(value, lower, upper))

    def sample(
        self,
        rng: np.random.Generator,
        method: str = "auto",
        source_case_id: str | None = None,
    ) -> EllipsoidParameters:
        """Sample joint empirical axes when possible, otherwise a declared fallback."""
        if method not in {"auto", "empirical", "truncated_normal"}:
            raise ValueError(f"unsupported parameter sampling method: {method}")
        if method in {"auto", "empirical"} and self.empirical_rows:
            if source_case_id is None:
                row = self.empirical_rows[int(rng.integers(0, len(self.empirical_rows)))]
            else:
                matches = [
                    value for value in self.empirical_rows
                    if str(value.get("case_id")) == str(source_case_id)
                ]
                if not matches:
                    raise ValueError(f"ellipsoid rows do not contain requested case {source_case_id!r}")
                row = matches[0]
            return EllipsoidParameters(
                a=float(row["a"]), b=float(row["b"]), c=float(row["c"]),
                sampling_method="empirical_joint_patient_bootstrap",
                source_case_id=None if row.get("case_id") is None else str(row["case_id"]),
            )
        if method == "empirical":
            raise ValueError("empirical sampling requested but no per-patient ellipsoid rows were loaded")
        axes = [self._sample_axis(self.scaffold_statistics[name], rng) for name in ("a", "b", "c")]
        return EllipsoidParameters(*axes, sampling_method="controlled_or_provisional_truncated_normal")
