import json
from pathlib import Path

summary_path = Path("outputs/batch1_extracted/batch1_extraction_summary.json")
data = json.loads(summary_path.read_text(encoding="utf-8"))

failed_cases = [r for r in data["results"] if not r["lca_succeeded"]]
print(f"Total failed cases: {len(failed_cases)}")
for r in failed_cases:
    pid = r["patient_id"]
    failures = r["failures"]
    reason = failures[0] if failures else "Unknown"
    print(f"{pid:<12} | {reason}")
