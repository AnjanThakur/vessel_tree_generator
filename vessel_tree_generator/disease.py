"""Public, user-controlled stenosis API for the 4D LCA generator."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from ._disease_model import (
    DEFAULT_DISEASE_SETTINGS,
    apply_disease_to_lca_tree,
    validate_disease_config,
    validate_diseased_lca_tree,
)


BRANCHES = ("LMCA", "LAD", "LCX")
LESION_TYPES = ("focal", "diffuse", "tandem")


def healthy_config(case_id: str = "healthy") -> dict[str, Any]:
    """Return an explicit no-lesion configuration."""
    return {"case_id": case_id, "lesions": []}


def stenosis_config(
    branch_id: str,
    position: float,
    length: float,
    severity: float,
    lesion_type: str = "focal",
    *,
    case_id: str | None = None,
    tandem_positions: Iterable[float] | None = None,
) -> dict[str, Any]:
    """Build a validated normalized lesion configuration.

    Parameters are normalized by branch arc length. ``severity`` is fractional
    radius reduction, for example ``0.70`` for a 70% radius stenosis.
    """
    branch = str(branch_id).upper()
    lesion_kind = str(lesion_type).lower()
    position = float(position)
    length = float(length)
    severity = float(severity)
    if branch not in BRANCHES:
        raise ValueError(f"branch_id must be one of {BRANCHES}")
    if lesion_kind not in LESION_TYPES:
        raise ValueError(f"lesion_type must be one of {LESION_TYPES}")
    if not 0.0 <= position <= 1.0:
        raise ValueError("position must lie in [0, 1]")
    if not 0.0 < length <= 1.0:
        raise ValueError("length must lie in (0, 1]")
    if not 0.0 <= severity < 1.0:
        raise ValueError("severity must lie in [0, 1)")

    if lesion_kind == "focal":
        lesion: dict[str, Any] = {
            "branch": branch,
            "type": "focal",
            "center": position,
            "length": length,
            "severity": severity,
        }
    elif lesion_kind == "diffuse":
        start = position - 0.5 * length
        end = position + 0.5 * length
        lesion = {
            "branch": branch,
            "type": "diffuse",
            "start": start,
            "end": end,
            "severity": severity,
        }
    else:
        positions = list(tandem_positions) if tandem_positions is not None else [
            position - 0.75 * length,
            position + 0.75 * length,
        ]
        sub_length = min(length, 0.45)
        lesion = {
            "branch": branch,
            "type": "tandem",
            "lesions": [
                {
                    "type": "focal",
                    "center": float(center),
                    "length": sub_length,
                    "severity": severity,
                }
                for center in positions
            ],
        }

    config = {
        "case_id": case_id or f"{branch.lower()}_{lesion_kind}_{int(round(100 * severity))}",
        "lesions": [lesion],
    }
    checked = validate_disease_config(config)
    if not checked["is_valid"]:
        raise ValueError("; ".join(checked["errors"]))
    # Return the user-facing structure so tandem identity is preserved in the
    # public API. The application layer validates and flattens tandem lesions
    # internally for numerical composition.
    return config


def apply_stenosis(
    radius_tree: dict[str, np.ndarray],
    branch_id: str,
    position: float,
    length: float,
    severity: float,
    lesion_type: str = "focal",
    *,
    tandem_positions: Iterable[float] | None = None,
    minimum_radius_mm: float = DEFAULT_DISEASE_SETTINGS["minimum_radius_mm"],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply one focal, diffuse, or tandem stenosis to an LCA radius tree."""
    config = stenosis_config(
        branch_id,
        position,
        length,
        severity,
        lesion_type,
        tandem_positions=tandem_positions,
    )
    diseased, metadata = apply_disease_to_lca_tree(
        radius_tree,
        config,
        minimum_radius_mm=minimum_radius_mm,
    )
    validation = validate_diseased_lca_tree(
        radius_tree,
        diseased,
        config,
        minimum_radius_mm=minimum_radius_mm,
    )
    if not validation["is_valid"]:
        raise RuntimeError("disease validation failed: " + "; ".join(validation["errors"]))
    metadata["validation"] = validation
    return diseased, metadata


def apply_disease_config(
    radius_tree: dict[str, np.ndarray],
    config: dict[str, Any] | None,
    *,
    minimum_radius_mm: float = DEFAULT_DISEASE_SETTINGS["minimum_radius_mm"],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Apply an arbitrary validated configuration, or return a healthy copy."""
    if config is None or not config.get("lesions"):
        copied = {name: np.asarray(values, dtype=float).copy() for name, values in radius_tree.items()}
        return copied, {
            "case_id": "healthy" if config is None else config.get("case_id", "healthy"),
            "config": healthy_config() if config is None else config,
            "radius_only": True,
            "healthy": True,
            "requested_config": healthy_config() if config is None else config,
            "lesion_modes": [],
            "validation": {"is_valid": True, "errors": [], "warnings": []},
        }

    checked = validate_disease_config(config)
    if not checked["is_valid"]:
        raise ValueError("; ".join(checked["errors"]))
    normalized = checked["normalized_config"]
    diseased, metadata = apply_disease_to_lca_tree(
        radius_tree,
        normalized,
        minimum_radius_mm=minimum_radius_mm,
    )
    validation = validate_diseased_lca_tree(
        radius_tree,
        diseased,
        normalized,
        minimum_radius_mm=minimum_radius_mm,
    )
    if not validation["is_valid"]:
        raise RuntimeError("disease validation failed: " + "; ".join(validation["errors"]))
    metadata.update({
        "healthy": False,
        "requested_config": config,
        "lesion_modes": sorted({str(item.get("type", "")).lower() for item in config["lesions"]}),
        "validation": validation,
    })
    return diseased, metadata
