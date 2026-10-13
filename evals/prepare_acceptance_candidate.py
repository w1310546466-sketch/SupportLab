"""Record exposure and freeze the repaired candidate before new cases are authored."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def save(name, value):
    with (ROOT / name).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


base = read("evals/acceptance_baseline_v1.json")
first_path = "reports/holdout-v1-first-2026-10-06.json"
first = read(first_path)
affected = sorted({row["family"] for row in first["failures"]})
disposition = {"created_at": datetime.now(timezone.utc).isoformat(), "source_suite": "composition-holdout-v1",
               "first_report": first_path, "first_report_sha256": sha256((ROOT / first_path).read_bytes()).hexdigest(),
               "first_score": {"passed": first["passed_count"], "cases": first["case_count"]},
               "failure_kind": "RAG usage omitted explicit tool_calls=0; outcome, diagnosis, state and citation checks passed",
               "affected_families": affected,
               "retired_case_count": sum(row["family"] in affected for row in first["cases"]),
               "suite_replay_classification": "regression_after_holdout_exposure",
               "policy": "All v1 replays are regression; do not replace first report or claim unseen accuracy."}
save("evals/holdout_v1.disposition.json", disposition)
base.update(version="acceptance-candidate-v2", created_at=datetime.now(timezone.utc).isoformat(),
            parent_application_sha256=base["frozen_application_sha256"])
base["application_files"] = {name: sha256((ROOT / name).read_bytes()).hexdigest() for name in base["application_files"]}
digest = sha256()
for name in sorted(base["application_files"]):
    digest.update(name.encode() + b"\0")
    digest.update((ROOT / name).read_bytes() + b"\0")
base["frozen_application_sha256"] = digest.hexdigest()
base["change"] = {"file": "supportlab/rag.py", "reason": "Emit explicit zero tool calls on every RAG exit",
                  "version": "extractive-rag-v2.1"}
save("evals/acceptance_candidate_v2.json", base)
print(json.dumps({"retired_cases": disposition["retired_case_count"], "retired_families": len(affected),
                  "candidate_sha256": base["frozen_application_sha256"]}))
