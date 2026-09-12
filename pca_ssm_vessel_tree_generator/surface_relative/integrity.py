"""Upstream protected data integrity verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Compute SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_protected_assets(ppt_output_dir: Path) -> dict[str, Any]:
    """Verify that protected PPT output files exist and match integrity json if available."""
    protected_files = [
        "population_two_plane_two_ellipse_parameters.csv",
        "population_pointwise_ellipse_residuals.csv",
        "population_case_status.csv",
        "population_statistics.json",
        "protected_stage1_integrity.json",
    ]
    status = {}
    all_valid = True
    for name in protected_files:
        file_path = ppt_output_dir / name
        exists = file_path.is_file()
        digest = file_sha256(file_path) if exists else None
        status[name] = {
            "exists": exists,
            "sha256": digest,
        }
        if not exists:
            all_valid = False

    # Check against protected_stage1_integrity.json if available
    integrity_file = ppt_output_dir / "protected_stage1_integrity.json"
    hashes_matched = True
    if integrity_file.is_file():
        try:
            stored_data = json.loads(integrity_file.read_text(encoding="utf-8"))
            stored_hashes = stored_data.get("file_hashes", {})
            for rel_path, expected_hash in stored_hashes.items():
                target = ppt_output_dir / rel_path
                if target.is_file():
                    actual_hash = file_sha256(target)
                    if actual_hash != expected_hash:
                        hashes_matched = False
                        break
        except Exception:
            hashes_matched = False

    return {
        "all_files_exist": all_valid,
        "hashes_matched": hashes_matched,
        "file_status": status,
    }
