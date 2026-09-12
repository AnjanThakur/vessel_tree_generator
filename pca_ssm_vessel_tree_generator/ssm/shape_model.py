"""Statistical Shape Model (SSM) class encapsulating x = mu + P_k b generative representation for Batch 4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

EPS = 1.0e-12

DIMENSION_MAPPING = {
    "RCA": {"start_idx": 0, "end_idx": 45, "points": 15, "dims": 45},
    "LMCA": {"start_idx": 45, "end_idx": 60, "points": 5, "dims": 15},
    "LAD": {"start_idx": 60, "end_idx": 96, "points": 12, "dims": 36},
    "LCX": {"start_idx": 96, "end_idx": 126, "points": 10, "dims": 30},
}


class StatisticalShapeModel:
    """Statistical Shape Model for 126-D vessel local deviation vectors.

    Generative Representation (Design Doc §5.3.3):
    x = mu + P_k * b

    x: 126-D shape vector [dev_x, dev_y, dev_z] for RCA, LMCA, LAD, LCX (42 control points)
    mu: 126-D population mean shape vector
    P_k: (126, k) retained principal component eigenvectors
    b: (k,) vector of PCA mode coefficients
    """

    def __init__(
        self,
        mean_vector: np.ndarray,
        components_all: np.ndarray,
        components_retained: np.ndarray,
        singular_values: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvalues_retained: np.ndarray,
        explained_variance_ratio: np.ndarray,
        cumulative_explained_variance: np.ndarray,
        k_retained: int,
        variance_cutoff: float = 0.95,
        n_samples: int = 174,
        n_features: int = 126,
    ):
        self.mean_vector = np.asarray(mean_vector, dtype=float).reshape(126)
        self.components_all = np.asarray(components_all, dtype=float)  # (126, 126) or (N, 126)
        self.components_retained = np.asarray(components_retained, dtype=float)  # (k, 126)
        self.singular_values = np.asarray(singular_values, dtype=float)
        self.eigenvalues = np.asarray(eigenvalues, dtype=float)
        self.eigenvalues_retained = np.asarray(eigenvalues_retained, dtype=float)
        self.explained_variance_ratio = np.asarray(explained_variance_ratio, dtype=float)
        self.cumulative_explained_variance = np.asarray(cumulative_explained_variance, dtype=float)
        self.k_retained = int(k_retained)
        self.variance_cutoff = float(variance_cutoff)
        self.n_samples = int(n_samples)
        self.n_features = int(n_features)

        # Retained mode eigenvectors matrix P_k: (126, k)
        self.P_k = self.components_retained.T  # (126, k)
        self.mode_stds = np.sqrt(np.maximum(self.eigenvalues_retained, 0.0))  # (k,)

    def encode(self, shape_vector: np.ndarray) -> np.ndarray:
        """Encode a 126-D shape vector x into PCA mode coefficients b = P_k^T * (x - mu).

        Returns:
            b: (k,) vector of PCA mode coefficients
        """
        x = np.asarray(shape_vector, dtype=float).reshape(126)
        centered = x - self.mean_vector
        return self.P_k.T @ centered  # (k,)

    def decode(self, mode_weights: np.ndarray) -> np.ndarray:
        """Decode PCA mode coefficients b into 126-D reconstructed shape vector x_hat = mu + P_k * b.

        Returns:
            x_hat: (126,) reconstructed shape vector
        """
        b = np.asarray(mode_weights, dtype=float).reshape(self.k_retained)
        return self.mean_vector + self.P_k @ b  # (126,)

    def reconstruct(self, shape_vector: np.ndarray) -> tuple[np.ndarray, float]:
        """Reconstruct a 126-D shape vector and compute Root Mean Square Error (RMSE).

        Returns:
            x_hat: (126,) reconstructed shape vector
            rmse: float RMSE = sqrt(mean((x - x_hat)^2))
        """
        x = np.asarray(shape_vector, dtype=float).reshape(126)
        b = self.encode(x)
        x_hat = self.decode(b)
        residual = x - x_hat
        rmse = float(np.sqrt(np.mean(residual**2)))
        return x_hat, rmse

    def to_dict(self) -> dict[str, Any]:
        """Serialize shape model summary parameters to a JSON-compatible dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "mean_shape_vector": self.mean_vector.tolist(),
            "singular_values": self.singular_values.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "cumulative_explained_variance": self.cumulative_explained_variance.tolist(),
            "k_retained": self.k_retained,
            "variance_cutoff": self.variance_cutoff,
            "retained_mode_eigenvalues": self.eigenvalues_retained.tolist(),
            "retained_mode_std": self.mode_stds.tolist(),
            "dimension_mapping": DIMENSION_MAPPING,
        }

    def save(self, json_path: Path, npz_path: Path) -> None:
        """Save Statistical Shape Model to JSON metadata and NPZ binary payload."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path.parent.mkdir(parents=True, exist_ok=True)

        json_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

        np.savez_compressed(
            npz_path,
            mean_vector=self.mean_vector,
            components_all=self.components_all,
            components_retained=self.components_retained,
            P_k=self.P_k,
            singular_values=self.singular_values,
            eigenvalues=self.eigenvalues,
            eigenvalues_retained=self.eigenvalues_retained,
            explained_variance_ratio=self.explained_variance_ratio,
            cumulative_explained_variance=self.cumulative_explained_variance,
            mode_stds=self.mode_stds,
        )

    @classmethod
    def load(cls, json_path: Path, npz_path: Path) -> StatisticalShapeModel:
        """Load Statistical Shape Model from JSON metadata and NPZ binary payload."""
        with open(json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        data = np.load(npz_path)

        return cls(
            mean_vector=data["mean_vector"],
            components_all=data["components_all"],
            components_retained=data["components_retained"],
            singular_values=data["singular_values"],
            eigenvalues=data["eigenvalues"],
            eigenvalues_retained=data["eigenvalues_retained"],
            explained_variance_ratio=data["explained_variance_ratio"],
            cumulative_explained_variance=data["cumulative_explained_variance"],
            k_retained=meta["k_retained"],
            variance_cutoff=meta["variance_cutoff"],
            n_samples=meta["n_samples"],
            n_features=meta["n_features"],
        )
