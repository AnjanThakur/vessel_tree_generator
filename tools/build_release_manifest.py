"""Build the final, machine-readable submission release manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "submission_release"
OUTPUT = RELEASE_DIR / "RELEASE_MANIFEST.json"
STATISTICS_DIR = (
    ROOT
    / "outputs"
    / "lca_ssm"
    / "lca_population_model"
    / "generator_statistics"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    ).stdout.rstrip()


def artifact(path: Path, status: str = "included") -> dict[str, str | int | None]:
    return {
        "status": status,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path) if path.is_file() else None,
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def main() -> None:
    validation = json.loads(
        (RELEASE_DIR / "VALIDATION_SUMMARY.json").read_text(encoding="utf-8")
    )
    tests = json.loads(
        (RELEASE_DIR / "final_validation" / "test_validation.json").read_text(
            encoding="utf-8"
        )
    )
    integrity = json.loads(
        (
            RELEASE_DIR
            / "final_audit"
            / "protected_integrity_comparison.json"
        ).read_text(encoding="utf-8")
    )
    statistics = {
        path.relative_to(ROOT).as_posix(): {
            "sha256": sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(STATISTICS_DIR.glob("*"))
        if path.is_file()
    }
    report_path = RELEASE_DIR / "FINAL_PROJECT_REPORT.docx"
    ppt_path = RELEASE_DIR / "CORONARY4D_FINAL_PRESENTATION.pptx"

    manifest = {
        "schema_version": 1,
        "release": {
            "name": "4D Coronary LCA Generator",
            "version": "1.0.0",
            "release_date": date.today().isoformat(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "branch": git("branch", "--show-current"),
            "git_head_at_audit": git("rev-parse", "HEAD"),
            "committed_by_this_audit": False,
            "pushed_by_this_audit": False,
        },
        "scope": {
            "anatomy": "LCA major-vessel tree: LMCA, LAD, LCX",
            "source_labels_discovered": validation["original_ppt_measurement"][
                "nifti_discovered"
            ],
            "statistics_eligible_source_cases": validation["source_data_audit"][
                "statistics_eligible_cases"
            ],
            "generated_population_size": validation["final_population_generation"][
                "accepted_trees"
            ],
            "stored_motion_frames": validation["population_motion"][
                "stored_frames_per_tree"
            ],
            "independent_positions_per_closed_cycle": validation["population_motion"][
                "independent_geometric_positions_per_closed_cycle"
            ],
            "clinical_validation": validation["clinical_validation"],
        },
        "statistical_model": validation["statistical_model"],
        "generation": validation["final_population_generation"],
        "novelty": validation["novelty"],
        "internal_holdout": validation["internal_holdout"],
        "motion_and_disease_demo": validation["submission_demo"],
        "verification": {
            "status": validation["status"],
            "tests": {
                "status": tests["status"],
                "passed": tests["total_tests_passed"],
                "failed": tests["tests_failed"],
                "compileall_pass": tests["compileall_pass"],
                "pip_check_pass": tests["pip_check_pass"],
            },
            "immutable_evidence_preserved": integrity[
                "immutable_evidence_preserved"
            ],
            "vtk_readback_pass": validation["submission_demo"][
                "vtk_readback_pass"
            ],
        },
        "frozen_generator_statistics": statistics,
        "primary_artifacts": {
            "project_report": artifact(report_path),
            "validation_summary": artifact(RELEASE_DIR / "VALIDATION_SUMMARY.json"),
            "final_truth_audit": artifact(
                RELEASE_DIR / "final_audit" / "FINAL_IMPLEMENTATION_TRUTH_AUDIT.md"
            ),
            "integrity_comparison": artifact(
                RELEASE_DIR / "final_audit" / "protected_integrity_comparison.json"
            ),
            "presentation": {
                "status": "deferred_by_user",
                "path": ppt_path.relative_to(ROOT).as_posix(),
                "sha256": None,
                "size_bytes": None,
                "note": "Presentation authoring was explicitly deferred for a later task.",
            },
        },
        "limitations": validation["limitations"],
        "release_readiness": {
            "technical_status": "PASS",
            "submission_status": "TECHNICAL_PACKAGE_READY; PRESENTATION_DEFERRED",
            "clinical_claim": "None. Outputs are research/engineering artifacts and are not clinically validated.",
        },
    }
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
