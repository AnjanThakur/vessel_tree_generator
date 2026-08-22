#!/usr/bin/env python
"""Build and verify the self-contained ParaView mentor presentation pack."""

from __future__ import annotations

import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyvista as pv


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "submission_release"
SOURCE_COHORT = ROOT / "outputs/lca_ssm/lca_population_cohort"
SOURCE_DEMOS = RELEASE / "demo_cases"
PACK = RELEASE / "mentor_visualization_pack"
STATIC = PACK / "01_static_population"
CINE = PACK / "02_4d_disease_cases"
ASSETS = PACK / "03_presentation_assets"
CASES = ("healthy", "focal_lad", "diffuse_lcx", "tandem_lad")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, target: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def copy_static_population() -> None:
    copy(SOURCE_COHORT / "synthetic_cohort.vtm", STATIC / "OPEN_ALL_52_TREES.vtm")
    tree_directories = sorted(SOURCE_COHORT.glob("tree_[0-9][0-9][0-9][0-9]"))
    if len(tree_directories) != 52:
        raise RuntimeError(f"expected 52 static trees; found {len(tree_directories)}")
    required = (
        "synthetic_tree.vtm",
        "synthetic_ellipsoid.vtp",
        "LMCA.vtp",
        "LAD.vtp",
        "LCX.vtp",
        "landmarks.vtp",
    )
    for directory in tree_directories:
        source = directory / "vtk"
        target = STATIC / directory.name / "vtk"
        for name in required:
            copy(source / name, target / name)


def copy_cine_cases() -> None:
    for case in CASES:
        source = SOURCE_DEMOS / case / "vtk"
        target = CINE / case
        copy(source / "cine.pvd", target / "OPEN_4D_CINE.pvd")
        phase_directories = sorted(source.glob("phase_[0-9][0-9][0-9]"))
        if len(phase_directories) != 10:
            raise RuntimeError(f"expected 10 phases for {case}; found {len(phase_directories)}")
        for phase in phase_directories:
            phase_target = target / phase.name
            copy(phase / "tree.vtm", phase_target / "tree.vtm")
            for name in ("tree_0.vtp", "tree_1.vtp", "tree_2.vtp"):
                copy(phase / "tree" / name, phase_target / "tree" / name)


def copy_presentation_assets() -> None:
    copy(SOURCE_COHORT / "cohort_preview_montage.png", ASSETS / "52_tree_population_montage.png")
    copy(SOURCE_DEMOS / "disease_mode_comparison.png", ASSETS / "disease_mode_comparison.png")
    copy(
        RELEASE / "design_spec_alignment/00_full_surface_design_concept.png",
        ASSETS / "ellipsoid_surface_design_concept.png",
    )
    for case in CASES:
        copy(SOURCE_DEMOS / case / "preview.png", ASSETS / f"{case}_preview.png")
        copy(
            SOURCE_DEMOS / case / "visualizations/cardiac_cycle.gif",
            ASSETS / f"{case}_cardiac_cycle.gif",
        )
        copy(
            SOURCE_DEMOS / case / "visualizations/validation_dashboard.png",
            ASSETS / f"{case}_validation_dashboard.png",
        )


def verify_xml_references() -> dict[str, Any]:
    xml_files = sorted([*PACK.rglob("*.pvd"), *PACK.rglob("*.vtm")])
    references = 0
    missing: list[str] = []
    for path in xml_files:
        root = ET.parse(path).getroot()
        for element in root.iter():
            referenced = element.attrib.get("file")
            if not referenced:
                continue
            references += 1
            target = (path.parent / referenced).resolve()
            if not target.is_file():
                missing.append(f"{path.relative_to(PACK).as_posix()} -> {referenced}")
    return {
        "xml_index_file_count": len(xml_files),
        "xml_reference_count": references,
        "missing_reference_count": len(missing),
        "missing_references": missing,
        "status": "PASS" if not missing else "FAIL",
    }


def verify_readback() -> dict[str, Any]:
    targets = [STATIC / "OPEN_ALL_52_TREES.vtm"]
    targets.extend(CINE / case / "OPEN_4D_CINE.pvd" for case in CASES)
    results = []
    for path in targets:
        dataset = pv.read(path)
        results.append({
            "path": path.relative_to(PACK).as_posix(),
            "type": type(dataset).__name__,
            "block_count": int(dataset.n_blocks) if isinstance(dataset, pv.MultiBlock) else 1,
            "readback": "PASS",
        })
    return {"status": "PASS", "targets": results}


def write_readme() -> None:
    text = """# Mentor Visualization Pack

This folder is a self-contained ParaView presentation package for the verified
major-vessel LCA research generator. Keep the directory structure unchanged:
the `.pvd` and `.vtm` index files use relative paths to their `.vtp` data.

## Start here in ParaView

1. Open `01_static_population/OPEN_ALL_52_TREES.vtm` to show the full generated
   population. In the Pipeline Browser, expand individual tree blocks. Hide the
   ellipsoid blocks when you want an uncluttered coronary-only comparison.
2. Open `02_4d_disease_cases/healthy/OPEN_4D_CINE.pvd`, click **Apply**, then
   press **Play** to demonstrate the closed cardiac cycle.
3. Repeat with `focal_lad`, `diffuse_lcx`, and `tandem_lad`. These four cases use
   the same representative anatomy, so differences isolate the disease model.
4. For disease visualization, select the vessel blocks and color by
   `disease_reduction_fraction`. For calibre and pulsatility, color by
   `radius_mm` and play the time series.

Recommended display settings: use **Tube** representation with radius about
`0.5-0.8 mm`, keep the scalar bar visible, use a white background, and use
**Reset Camera** after opening each dataset.

## What each entry demonstrates

| Entry | Main point |
|---|---|
| `OPEN_ALL_52_TREES.vtm` | Population diversity, ellipsoid scaffold, LMCA/LAD/LCX identity, landmarks and surface-relative arrays |
| `healthy/OPEN_4D_CINE.pvd` | Cardiac motion and global/local pulsatility without disease |
| `focal_lad/OPEN_4D_CINE.pvd` | One localized LAD stenosis with reduced lesion compliance |
| `diffuse_lcx/OPEN_4D_CINE.pvd` | Extended LCX disease region |
| `tandem_lad/OPEN_4D_CINE.pvd` | Multiple separated LAD stenoses |

Static population branches expose `surface_u_rad`, `surface_v_rad`,
`surface_normal_offset_mm` and local deviation coefficients. The 4D case
branches expose `radius_mm`, `normalized_arc_length` and
`disease_reduction_fraction`.

## Suggested mentor explanation

- The two fitted ellipses measure the triaxial ellipsoid scaffold; they are not
  LAD or LCX vessel molds.
- LAD mainly progresses base-to-apex in surface `v`; LCX mainly travels
  circumferentially in surface `u`.
- The 81-D PCA models 27 surface-relative points with tangent-u, tangent-v and
  normal coefficients across LMCA, LAD and LCX.
- Disease changes radius, not centerline geometry.
- Cardiac deformation moves the ellipsoid-associated vessels coherently, while
  radius pulsatility and reduced lesion compliance remain visible over time.
- Scope remains LCA-only and research/engineering; it is not clinically
  validated.

The files in `03_presentation_assets` are quick-reference images and GIFs for a
meeting where ParaView is unavailable. `VISUALIZATION_PACK_MANIFEST.json`
contains hashes and machine verification results.
"""
    (PACK / "README.md").write_text(text, encoding="utf-8")


def write_manifest(xml_check: dict[str, Any], readback: dict[str, Any]) -> None:
    files = sorted(path for path in PACK.rglob("*") if path.is_file() and path.name != "VISUALIZATION_PACK_MANIFEST.json")
    extensions = Counter(path.suffix.lower() for path in files)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "verified LCA major-vessel research visualization pack",
        "static_population_tree_count": 52,
        "cine_case_count": 4,
        "stored_frames_per_cine_case": 10,
        "independent_positions_plus_closure": "9 independent positions + repeated closure frame",
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "extension_counts": dict(sorted(extensions.items())),
        "xml_reference_verification": xml_check,
        "pyvista_readback_verification": readback,
        "status": "PASS" if xml_check["status"] == readback["status"] == "PASS" else "FAIL",
        "files": {
            path.relative_to(PACK).as_posix(): {
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    (PACK / "VISUALIZATION_PACK_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    resolved = PACK.resolve()
    if resolved.parent != RELEASE.resolve() or resolved.name != "mentor_visualization_pack":
        raise RuntimeError(f"refusing to rebuild unexpected path: {resolved}")
    if PACK.exists():
        shutil.rmtree(PACK)
    copy_static_population()
    copy_cine_cases()
    copy_presentation_assets()
    write_readme()
    xml_check = verify_xml_references()
    if xml_check["status"] != "PASS":
        raise RuntimeError(json.dumps(xml_check, indent=2))
    readback = verify_readback()
    write_manifest(xml_check, readback)
    manifest = json.loads((PACK / "VISUALIZATION_PACK_MANIFEST.json").read_text(encoding="utf-8"))
    print(json.dumps({key: manifest[key] for key in (
        "status", "static_population_tree_count", "cine_case_count",
        "stored_frames_per_cine_case", "file_count", "size_bytes", "extension_counts",
    )}, indent=2))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
