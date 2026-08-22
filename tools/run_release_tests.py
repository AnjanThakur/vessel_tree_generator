#!/usr/bin/env python
"""Run every release test family and write one machine-readable result."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "submission_release/final_validation/test_validation.json"


def execute(arguments: list[str], *, pythonpath: str | None = None) -> dict[str, object]:
    environment = os.environ.copy()
    if pythonpath is not None:
        environment["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        arguments, cwd=ROOT, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return {
        "command": arguments,
        "exit_code": result.returncode,
        "output": result.stdout.strip(),
    }


def run() -> dict[str, object]:
    python = str(ROOT / ".venv/Scripts/python.exe")
    pytest = execute([python, "-m", "pytest", "-q"], pythonpath="pca_ssm_vessel_tree_generator")
    person2 = execute(
        [python, "-m", "unittest", "pca_ssm_vessel_tree_generator.tests.test_person2_generation", "-q"],
        pythonpath="pca_ssm_vessel_tree_generator",
    )
    design_alignment = execute(
        [python, "-m", "unittest", "pca_ssm_vessel_tree_generator.tests.test_design_spec_alignment", "-q"],
        pythonpath="pca_ssm_vessel_tree_generator",
    )
    compileall = execute([
        python, "-m", "compileall", "-q", "vessel_tree_generator",
        "pca_ssm_vessel_tree_generator", "lca_vessel_tree_generator",
        "rca_vessel_tree_generator", "tests", "tools",
    ])
    pip_check = execute([python, "-m", "pip", "check"])
    pytest_match = re.search(r"(\d+) passed", str(pytest["output"]))
    person2_match = re.search(r"Ran (\d+) tests?", str(person2["output"]))
    design_alignment_match = re.search(r"Ran (\d+) tests?", str(design_alignment["output"]))
    pytest_count = int(pytest_match.group(1)) if pytest_match else 0
    person2_count = int(person2_match.group(1)) if person2_match else 0
    design_alignment_count = int(design_alignment_match.group(1)) if design_alignment_match else 0
    commands = {
        "pytest": pytest,
        "person2_unittest": person2,
        "design_alignment_unittest": design_alignment,
        "compileall": compileall,
        "pip_check": pip_check,
    }
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(item["exit_code"] == 0 for item in commands.values()) else "FAIL",
        "pytest_passed": pytest_count,
        "person2_unittest_passed": person2_count,
        "design_alignment_unittest_passed": design_alignment_count,
        "total_tests_passed": pytest_count + person2_count + design_alignment_count,
        "tests_failed": 0 if all(item["exit_code"] == 0 for item in (pytest, person2, design_alignment)) else None,
        "compileall_pass": compileall["exit_code"] == 0,
        "pip_check_pass": pip_check["exit_code"] == 0,
        "commands": commands,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in (
        "status", "pytest_passed", "person2_unittest_passed", "design_alignment_unittest_passed", "total_tests_passed",
        "compileall_pass", "pip_check_pass",
    )}, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
