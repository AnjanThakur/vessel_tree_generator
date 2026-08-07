"""Read-only hook for consuming authoritative derived Stage-1 branch roles.

Future SSM code may call :func:`load_resolved_assignment_map` and
:func:`resolve_raw_daughters`.  This helper never rewrites extraction outputs
and deliberately rejects unresolved cases.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


RESOLVED = {"resolved_existing_assignment", "resolved_swapped_assignment"}


def load_resolved_assignment_map(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("resolved assignment map does not contain a cases object")
    return cases


def resolve_raw_daughters(
    patient_id: str,
    branch_a: np.ndarray,
    branch_b: np.ndarray,
    assignment_map: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    record = assignment_map.get(patient_id)
    if record is None:
        raise KeyError(f"{patient_id}: absent from resolved Stage-1 assignment map")
    if record.get("resolution_status") not in RESOLVED:
        raise ValueError(f"{patient_id}: branch roles are not resolved ({record.get('resolution_status')})")
    assignment = record.get("new_resolved_assignment", {})
    neutral = {"branch_a": branch_a, "branch_b": branch_b}
    lad_source, lcx_source = assignment.get("lad_source"), assignment.get("lcx_source")
    if {lad_source, lcx_source} != set(neutral):
        raise ValueError(f"{patient_id}: invalid resolved daughter mapping")
    return {"lad": neutral[lad_source], "lcx": neutral[lcx_source]}
