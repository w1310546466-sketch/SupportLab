"""Candidate verification: exposed v1 regression and separately sealed v2 holdout."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from .acceptance import file_hash, report_reservation, run_case, verify_map
from .evaluation import fingerprint
from .storage import PROJECT_ROOT

CANDIDATE = PROJECT_ROOT / "evals/acceptance_candidate_v2.json"
SEAL = PROJECT_ROOT / "evals/holdout_v2.seal.json"
EXTRA_MODULES = {"supportlab/acceptance.py", "supportlab/acceptance_fixtures.py", "supportlab/acceptance_release.py"}


def read(name):
    return json.loads((PROJECT_ROOT / name).read_text(encoding="utf-8"))


def validate_delta(original, candidate):
    if set(original) != set(candidate):
        raise ValueError("Candidate file set changed")
    changed = {name for name in original if original[name] != candidate[name]}
    if changed != {"supportlab/rag.py"}:
        raise ValueError("Only the recorded RAG usage fix is accepted by this candidate manifest")
    return sorted(changed)


def classification(suite_name):
    if suite_name == "regression":
        return "regression_after_holdout_exposure"
    if suite_name == "holdout":
        return "prospective_composition_holdout"
    raise ValueError("Unknown suite")


def verify_candidate():
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    required = {"evals/holdout_v2.json", "evals/build_holdout_v2.py", "evals/acceptance_candidate_v2.json",
                "evals/holdout_v1.disposition.json", "evals/prepare_acceptance_candidate.py",
                "evals/holdout_v1.seal.json", "supportlab/acceptance_release.py", "tests/test_acceptance_release.py",
                "tests/test_rag_usage.py", "reports/acceptance-v1-frozen-source.zip"}
    if seal.get("version") != "holdout-seal-v2" or set(seal.get("files", {})) != required:
        raise ValueError("Incomplete candidate seal")
    verify_map(seal["files"])
    old_seal = read("evals/holdout_v1.seal.json")
    verify_map(old_seal["files"])
    original, candidate = read("evals/acceptance_baseline_v1.json"), read("evals/acceptance_candidate_v2.json")
    validate_delta(original["application_files"], candidate["application_files"])
    for field in ("application_files", "source_files", "development_files"):
        verify_map(candidate[field])
    if fingerprint([PROJECT_ROOT / name for name in candidate["application_files"]]) != candidate["frozen_application_sha256"]:
        raise ValueError("Candidate aggregate fingerprint mismatch")
    actual = {path.relative_to(PROJECT_ROOT).as_posix() for path in (PROJECT_ROOT / "supportlab").glob("*.py")}
    if actual != set(candidate["application_files"]) | EXTRA_MODULES:
        raise ValueError("Unexpected application modules")
    disposition = read("evals/holdout_v1.disposition.json")
    verify_map({disposition["first_report"]: disposition["first_report_sha256"]})
    old, new = read("evals/holdout_v1.json"), read("evals/holdout_v2.json")
    if len(old["cases"]) != 40 or len(new["cases"]) != 12 or len({row["id"] for row in new["cases"]}) != 12:
        raise ValueError("Wrong sealed case counts")
    previous_families = set(old["families"])
    for path in candidate["development_files"]:
        data = read(path)
        for key in ("cases", "rag_cases", "conversation_cases"):
            previous_families.update(row["family"] for row in data.get(key, []))
    new_families = {row["family"] for row in new["cases"]}
    if new_families & previous_families or new_families != set(new["families"]) or len(new_families) != 3:
        raise ValueError("Review supplemental family allocation")
    if new.get("split") != "prospective_composition_holdout" or new.get("simulation") is not True:
        raise ValueError("Invalid supplemental provenance")
    return old, new, candidate, disposition


def summarize(rows):
    failures = [{"case": row["id"], "step": step["step"],
                 "failed_checks": [key for key, good in step["grade"]["checks"].items() if not good]}
                for row in rows for step in row["steps"] if not step["grade"]["passed"]]
    replies = [step["output"]["result"] for row in rows for step in row["steps"] if "result" in step["output"]]
    return {"passed_count": sum(row["passed"] for row in rows), "case_count": len(rows), "failures": failures,
            "step_checks": {"numerator": sum(step["grade"]["passed"] for row in rows for step in row["steps"]),
                            "denominator": sum(len(row["steps"]) for row in rows)},
            "model_api_calls": sum(result["usage"]["model_api_calls"] for result in replies),
            "tool_calls": sum(result["usage"].get("tool_calls", 0) for result in replies),
            "families": {family: {"passed": sum(row["passed"] for row in rows if row["family"] == family),
                                    "cases": sum(row["family"] == family for row in rows)}
                         for family in sorted({row["family"] for row in rows})}}


def run_release(suite_name, *, progress=None):
    old, new, candidate, disposition = verify_candidate()
    dataset = old if suite_name == "regression" else new
    split = classification(suite_name)
    started = perf_counter()
    rows = []
    for case in dataset["cases"]:
        row = run_case(case)
        rows.append(row)
        if progress:
            progress(row)
    verify_candidate()
    return {"created_at": datetime.now(timezone.utc).isoformat(), "suite_version": dataset["dataset_version"],
            "split": split, "simulation": True, "execution_mode": "offline_fixed_application", **summarize(rows),
            "elapsed_ms": round((perf_counter() - started) * 1000, 3), "token_usage": None, "cost": None,
            "human_review": "pending", "independent_authorship": False,
            "first_v1_result": disposition["first_score"], "v1_retired_families": disposition["affected_families"],
            "provenance": {"candidate_application_sha256": candidate["frozen_application_sha256"],
                           "original_application_sha256": candidate["parent_application_sha256"],
                           "changed_application_files": ["supportlab/rag.py"],
                           "candidate_manifest_sha256": file_hash(CANDIDATE), "seal_sha256": file_hash(SEAL),
                           "suite_sha256": file_hash(PROJECT_ROOT / ("evals/holdout_v1.json" if suite_name == "regression" else "evals/holdout_v2.json")),
                           "first_report_preserved": True, "frozen_files_verified_before_and_after": True},
            "reproduce": f".\\.venv\\Scripts\\python.exe -m supportlab.acceptance_release --suite {suite_name} --output reports/{suite_name}-new-run.json",
            "limitations": ["v1 已暴露失败，所有重跑标为回归，不能替代首次 25/40 记录。",
                            "v2 为修复后冻结代码再编写的 12 个新组合场景；基础业务原语相同，不是新根因测试。",
                            "AI 构造与预期均需人工复核，不是第三方盲测。", "本次只测离线应用，不能说明真实模型稳定性或提示注入防护效果。"], "cases": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description="修复后候选验收：旧集回归与新集保留验证分开；零 API")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--suite", choices=("regression", "holdout"), default="holdout")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        verify_candidate()
        if args.verify_only:
            print("候选封存校验通过：40 个旧集回归场景、12 个新保留场景；首次报告未改。")
            return 0
        if args.output is None:
            raise ValueError("A new report destination is required")
        with report_reservation(args.output) as stream:
            result = run_release(args.suite, progress=lambda row: print(f"{'PASS' if row['passed'] else 'FAIL'} | {row['id']}", flush=True))
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"{result['split']}: {result['passed_count']}/{result['case_count']}; API={result['model_api_calls']}")
        return 0 if result["passed_count"] == result["case_count"] else 1
    except (OSError, ValueError, KeyError) as error:
        print(f"Candidate verification not completed: {type(error).__name__}. Check seal and destination; no overwrite or automatic retry.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
