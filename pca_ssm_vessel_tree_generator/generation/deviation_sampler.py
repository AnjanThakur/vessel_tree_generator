"""Joint PCA sampling of coordinated coronary shape deviations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BRANCH_COUNTS = {"RCA": 15, "LMCA": 5, "LAD": 12, "LCX": 10}


@dataclass
class DeviationSample:
    branches: dict[str, np.ndarray]
    coefficients: np.ndarray
    coefficient_standard_norms: np.ndarray
    pca_applied: bool
    pca_mean_included: bool = True
    variation_scale: float = 1.0
    baseline_case_id: str | None = None
    baseline_coefficients: np.ndarray = field(default_factory=lambda: np.empty(0))
    innovation_coefficients: np.ndarray = field(default_factory=lambda: np.empty(0))
    innovation_branches: dict[str, np.ndarray] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "pca_applied": self.pca_applied,
            "pca_mean_included": self.pca_mean_included,
            "variation_scale": self.variation_scale,
            "coefficients": self.coefficients.tolist(),
            "coefficient_standard_norms": self.coefficient_standard_norms.tolist(),
            "baseline_case_id": self.baseline_case_id,
            "baseline_coefficients": self.baseline_coefficients.tolist(),
            "innovation_coefficients": self.innovation_coefficients.tolist(),
            "innovation_branch_shapes": {
                name: list(values.shape) for name, values in self.innovation_branches.items()
            },
            "branch_shapes": {name: list(values.shape) for name, values in self.branches.items()},
        }


class DeviationSampler:
    """Sample one joint vector so inter-branch PCA correlations are retained."""

    def __init__(
        self,
        mean_vector: np.ndarray,
        components: np.ndarray,
        eigenvalues: np.ndarray,
        branch_counts: dict[str, int] | None = None,
        *,
        include_mean: bool = True,
        variation_scale: float = 1.0,
        training_scores: np.ndarray | None = None,
        case_ids: np.ndarray | None = None,
    ):
        self.branch_counts = branch_counts or DEFAULT_BRANCH_COUNTS.copy()
        self.mean_vector = np.asarray(mean_vector, dtype=float).reshape(-1)
        self.components = np.asarray(components, dtype=float)
        self.eigenvalues = np.asarray(eigenvalues, dtype=float).reshape(-1)
        self.include_mean = bool(include_mean)
        self.variation_scale = float(variation_scale)
        self.training_scores = None if training_scores is None else np.asarray(training_scores, dtype=float)
        self.case_ids = None if case_ids is None else np.asarray(case_ids).astype(str)
        if not 0.0 <= self.variation_scale <= 1.0:
            raise ValueError("variation_scale must be in [0, 1]")
        expected = 3 * sum(self.branch_counts.values())
        if self.mean_vector.shape != (expected,):
            raise ValueError(f"PCA mean vector must have {expected} values; got {self.mean_vector.shape}")
        if self.components.ndim != 2 or self.components.shape[1] != expected:
            raise ValueError(f"PCA components must have shape (k, {expected}); got {self.components.shape}")
        if len(self.eigenvalues) < len(self.components):
            raise ValueError("PCA eigenvalues do not cover all retained components")
        if not np.all(np.isfinite(self.mean_vector)) or not np.all(np.isfinite(self.components)):
            raise ValueError("PCA arrays contain non-finite values")
        if (self.training_scores is None) != (self.case_ids is None):
            raise ValueError("training_scores and case_ids must be supplied together")
        if self.training_scores is not None:
            expected_scores = (len(self.case_ids), len(self.components))
            if self.training_scores.shape != expected_scores:
                raise ValueError(
                    f"training_scores must have shape {expected_scores}; got {self.training_scores.shape}"
                )
            if not np.all(np.isfinite(self.training_scores)):
                raise ValueError("PCA training scores contain non-finite values")

    @classmethod
    def from_npz(
        cls,
        path: Path,
        *,
        include_mean: bool = True,
        variation_scale: float = 1.0,
    ) -> "DeviationSampler":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"missing Person 1 PCA artifact: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"mean_vector", "components", "eigenvalues"}
            missing = sorted(required - set(data.files))
            if missing:
                raise ValueError(f"PCA artifact {path} is missing arrays: {missing}")
            has_training_scores = {"training_scores", "complete_case_ids"}.issubset(data.files)
            return cls(
                data["mean_vector"], data["components"], data["eigenvalues"],
                include_mean=include_mean, variation_scale=variation_scale,
                training_scores=data["training_scores"] if has_training_scores else None,
                case_ids=data["complete_case_ids"] if has_training_scores else None,
            )

    def sample(
        self,
        rng: np.random.Generator,
        coefficient_limit_sigma: float = 3.0,
        source_case_id: str | None = None,
    ) -> DeviationSample:
        if coefficient_limit_sigma <= 0.0:
            raise ValueError("coefficient_limit_sigma must be positive")
        standard = np.clip(rng.normal(size=len(self.components)), -coefficient_limit_sigma, coefficient_limit_sigma)
        scales = np.sqrt(np.maximum(self.eigenvalues[: len(self.components)], 0.0))
        innovation_coefficients = standard * scales * self.variation_scale
        baseline_case_id = None
        baseline_coefficients = np.zeros(len(self.components), dtype=float)
        if self.training_scores is not None and self.case_ids is not None:
            if source_case_id is None:
                index = int(rng.integers(0, len(self.case_ids)))
            else:
                matches = np.flatnonzero(self.case_ids == str(source_case_id))
                if not len(matches):
                    raise ValueError(f"PCA training scores do not contain matched case {source_case_id!r}")
                index = int(matches[0])
            baseline_case_id = str(self.case_ids[index])
            baseline_coefficients = self.training_scores[index].copy()
        coefficients = baseline_coefficients + innovation_coefficients
        baseline = self.mean_vector if self.include_mean else np.zeros_like(self.mean_vector)
        vector = baseline + coefficients @ self.components
        innovation_vector = innovation_coefficients @ self.components
        branches: dict[str, np.ndarray] = {}
        innovation_branches: dict[str, np.ndarray] = {}
        cursor = 0
        for name, count in self.branch_counts.items():
            width = count * 3
            branches[name] = vector[cursor : cursor + width].reshape(count, 3)
            innovation_branches[name] = innovation_vector[cursor : cursor + width].reshape(count, 3)
            cursor += width
        return DeviationSample(
            branches, coefficients, standard, pca_applied=True,
            pca_mean_included=self.include_mean, variation_scale=self.variation_scale,
            baseline_case_id=baseline_case_id,
            baseline_coefficients=baseline_coefficients,
            innovation_coefficients=innovation_coefficients,
            innovation_branches=innovation_branches,
        )


class ZeroDeviationSampler:
    """Explicit Week-1 baseline; it never adds pointwise random noise."""

    def sample(
        self,
        rng: np.random.Generator,
        coefficient_limit_sigma: float = 3.0,
        source_case_id: str | None = None,
    ) -> DeviationSample:
        del rng, coefficient_limit_sigma, source_case_id
        branches = {name: np.zeros((count, 3), dtype=float) for name, count in DEFAULT_BRANCH_COUNTS.items()}
        return DeviationSample(
            branches, np.empty(0), np.empty(0), pca_applied=False,
            pca_mean_included=False, variation_scale=0.0,
            innovation_branches={name: values.copy() for name, values in branches.items()},
        )


def interpolate_branch_deviation(values: np.ndarray, sample_count: int) -> np.ndarray:
    """Smoothly interpolate fixed PCA deviations along normalized arc position."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError(f"branch deviations must have shape (n>=2, 3); got {values.shape}")
    source_s = np.linspace(0.0, 1.0, len(values))
    target_s = np.linspace(0.0, 1.0, sample_count)
    result = np.column_stack([np.interp(target_s, source_s, values[:, dimension]) for dimension in range(3)])
    result[0] = values[0]
    result[-1] = values[-1]
    return result
