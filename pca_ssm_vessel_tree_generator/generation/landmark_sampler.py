"""Surface-relative landmark sampling with circular treatment of ``u``."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_LCA_LANDMARKS = ("lca_ostium", "bifurcation", "lad_endpoint", "lcx_endpoint")


def wrap_angle(angle: float) -> float:
    """Wrap radians to the half-open interval ``[-pi, pi)``."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


@dataclass(frozen=True)
class SurfaceLandmark:
    name: str
    u: float
    v: float
    offset: float
    sampling_method: str

    def __post_init__(self) -> None:
        values = np.asarray([self.u, self.v, self.offset], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"landmark {self.name!r} contains non-finite values")
        if not 0.0 < self.v < math.pi:
            raise ValueError(f"landmark {self.name!r} v must be inside (0, pi); got {self.v}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LandmarkDistribution:
    u_mean: float
    u_std: float
    v_mean: float
    v_std: float
    offset_mean: float
    offset_std: float

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "LandmarkDistribution":
        aliases = {
            "u_mean": ("u_mean", "circular_mean_rad"),
            "u_std": ("u_std", "circular_std_rad"),
            "v_mean": ("v_mean",),
            "v_std": ("v_std",),
            "offset_mean": ("offset_mean",),
            "offset_std": ("offset_std",),
        }
        values: dict[str, float] = {}
        for target, candidates in aliases.items():
            for candidate in candidates:
                if candidate in mapping:
                    values[target] = float(mapping[candidate])
                    break
            else:
                raise ValueError(f"landmark distribution is missing {target!r}")
        return cls(**values)


class LandmarkSampler:
    """Sample anatomically named ``(u,v,offset)`` landmarks."""

    def __init__(
        self,
        distributions: dict[str, LandmarkDistribution],
        source: str,
        empirical_cases: list[dict[str, Any]] | None = None,
    ):
        missing = sorted(set(REQUIRED_LCA_LANDMARKS) - set(distributions))
        if missing:
            raise ValueError(f"missing required LCA landmark distributions: {missing}")
        self.distributions = distributions
        self.source = source
        self.empirical_cases = empirical_cases or []

    @classmethod
    def controlled(cls) -> "LandmarkSampler":
        """Deterministic landmarks for architecture testing, not learned anatomy."""
        values = {
            "lca_ostium": (0.00, 1.20, 8.0),
            "bifurcation": (0.20, 1.43, 1.5),
            "lad_endpoint": (0.28, 2.72, 1.0),
            "lcx_endpoint": (1.82, 1.67, 1.0),
            "rca_ostium": (-0.18, 1.18, 8.0),
            "rca_endpoint": (-2.15, 1.72, 1.0),
        }
        distributions = {
            name: LandmarkDistribution(u, 0.0, v, 0.0, offset, 0.0)
            for name, (u, v, offset) in values.items()
        }
        return cls(distributions, source="controlled_week1_demo_not_population_derived")

    @classmethod
    def from_json(cls, path: Path) -> "LandmarkSampler":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        mappings = payload.get("landmarks", payload)
        if not isinstance(mappings, dict):
            raise ValueError(f"expected landmark mapping in {path}")
        distributions = {
            str(name): LandmarkDistribution.from_mapping(value)
            for name, value in mappings.items()
            if isinstance(value, dict)
        }
        empirical_cases = payload.get("empirical_cases", [])
        if not isinstance(empirical_cases, list):
            raise ValueError(f"'empirical_cases' must be a list in {path}")
        return cls(distributions, source=str(path), empirical_cases=empirical_cases)

    def sample(
        self,
        rng: np.random.Generator,
        include_rca: bool = False,
        source_case_id: str | None = None,
    ) -> dict[str, SurfaceLandmark]:
        names = list(REQUIRED_LCA_LANDMARKS)
        if include_rca:
            names.extend(("rca_ostium", "rca_endpoint"))
        eligible_empirical = [
            row for row in self.empirical_cases
            if isinstance(row, dict)
            and isinstance(row.get("landmarks"), dict)
            and all(name in row["landmarks"] for name in names)
        ]
        if eligible_empirical:
            matching = [row for row in eligible_empirical if str(row.get("case_id")) == str(source_case_id)]
            row = matching[0] if matching else eligible_empirical[int(rng.integers(0, len(eligible_empirical)))]
            return {
                name: SurfaceLandmark(
                    name=name,
                    u=wrap_angle(float(row["landmarks"][name]["u"])),
                    v=float(np.clip(float(row["landmarks"][name]["v"]), 0.01, math.pi - 0.01)),
                    offset=float(row["landmarks"][name]["offset"]),
                    sampling_method=f"empirical_joint_case_bootstrap:{row.get('case_id')}",
                )
                for name in names
            }
        missing = [name for name in names if name not in self.distributions]
        if missing:
            raise ValueError(f"requested landmark distributions are unavailable: {missing}")
        result: dict[str, SurfaceLandmark] = {}
        for name in names:
            stats = self.distributions[name]
            u = stats.u_mean if stats.u_std == 0.0 else float(rng.normal(stats.u_mean, max(stats.u_std, 0.0)))
            v = stats.v_mean if stats.v_std == 0.0 else float(rng.normal(stats.v_mean, max(stats.v_std, 0.0)))
            offset = (
                stats.offset_mean
                if stats.offset_std == 0.0
                else float(rng.normal(stats.offset_mean, max(stats.offset_std, 0.0)))
            )
            result[name] = SurfaceLandmark(
                name=name,
                u=wrap_angle(u),
                v=float(np.clip(v, 0.01, math.pi - 0.01)),
                offset=offset,
                sampling_method=self.source,
            )
        return result
