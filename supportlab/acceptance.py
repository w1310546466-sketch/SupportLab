"""Sealed, prospective offline composition acceptance. Not a real-model benchmark.

Run separately so adding the harness does not alter the frozen application CLI:
    python -m supportlab.acceptance --output reports/holdout-v1-first.json
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter

from .acceptance_fixtures import Sandbox
from .storage import PROJECT_ROOT

BASELINE = PROJECT_ROOT / "evals" / "acceptance_baseline_v1.json"
SUITE = PROJECT_ROOT / "evals" / "holdout_v1.json"
SEAL = PROJECT_ROOT / "evals" / "holdout_v1.seal.json"
HARNESS_VERSION = "composition-acceptance-v1"
ALLOWED_ADDITIONS = {"supportlab/acceptance.py", "supportlab/acceptance_fixtures.py"}
MISSING = object()


def value_at(value, path):
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return MISSING
        value = value[key]
    return value


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def verify_map(mapping, *, root=PROJECT_ROOT):
    for name, expected in mapping.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Frozen file changed or unavailable: {name}")


def load_sealed_suite():
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal.get("version") != "holdout-seal-v1" or seal.get("case_count") != 40:
        raise ValueError("Invalid acceptance seal")
    required = {"evals/acceptance_baseline_v1.json", "evals/holdout_v1.json", *ALLOWED_ADDITIONS,
                "evals/build_holdout_v1.py", "tests/test_acceptance.py"}
    if set(seal.get("files", {})) != required:
        raise ValueError("Seal must cover the complete harness, dataset and baseline")
    verify_map(seal["files"])
    for field in ("application_files", "source_files", "development_files"):
        verify_map(baseline[field])
    actual = {path.relative_to(PROJECT_ROOT).as_posix() for path in (PROJECT_ROOT / "supportlab").glob("*.py")}
    if actual != set(baseline["application_files"]) | ALLOWED_ADDITIONS:
        raise ValueError("Unexpected application modules; baseline must be reviewed")
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    if suite.get("simulation") is not True or suite.get("split") != "prospective_composition_holdout":
        raise ValueError("Expected explicitly simulated prospective composition cases")
    cases = suite["cases"]
    if len(cases) != 40 or len({row["id"] for row in cases}) != 40:
        raise ValueError("Expected exactly 40 unique locked cases")
    families = {row["family"] for row in cases}
    if families != set(suite["families"]) or len(families) != 8:
        raise ValueError("Missing composition-family definitions")
    development_families = set()
    for name in baseline["development_files"]:
        data = json.loads((PROJECT_ROOT / name).read_text(encoding="utf-8"))
        for key in ("cases", "rag_cases", "conversation_cases"):
            development_families.update(row["family"] for row in data.get(key, []))
    if families & development_families:
        raise ValueError("Family label overlap: review split allocation")
    # This mechanical check is necessary but cannot prove semantic independence.
    return suite, baseline, seal


def grade_step(step, output, *, docs_root=None):
    expected = step.get("expect", {})
    checks = {}
    for path, value in expected.get("equals", {}).items():
        actual = value_at(output, path)
        checks["equals:" + path] = actual is not MISSING and type(actual) is type(value) and actual == value
    for path in expected.get("absent", []):
        checks["absent:" + path] = value_at(output, path) is MISSING
    for path, floor in expected.get("minimum", {}).items():
        value = value_at(output, path)
        checks["minimum:" + path] = type(value) is int and value >= floor
    result = output.get("result", {})
    answer = result.get("answer", "")
    for phrase in expected.get("contains", []):
        checks["contains:" + phrase] = phrase in answer
    for phrase in expected.get("not_contains", []):
        checks["not_contains:" + phrase] = phrase not in answer
    if "error" in output:
        checks["expected_error"] = expected.get("error") == output["error"]
    elif "error" in expected:
        checks["expected_error"] = False
    elif step["op"] == "chat":
        checks["zero_api"] = result.get("usage", {}).get("model_api_calls") == 0
        checks["fixed_session"] = result.get("session_mode") == "fixed"
        checks["bounded_tools"] = 0 <= result.get("usage", {}).get("tool_calls", 0) <= 4
        checks["business_sources_unchanged"] = output.get("business_sources_unchanged") is True
        context = result.get("context_report", {})
        if "bytes" in context:
            checks["context_budget"] = context["bytes"] <= context["max_bytes"]
        if result.get("outcome") == "diagnosis":
            checks["grounded_diagnosis"] = result.get("answer_check", {}).get("passed") is True
            checks["has_citation"] = bool(result.get("citations"))
            selected = value_at(result, "structured_answer.facts.request.values.request_id")
            queries = [row["validated_arguments"].get("request_id") for row in result.get("trace", [])
                       if row["event"] == "tool" and row.get("tool") in ("get_request_record", "get_account_state")
                       and row.get("validated_arguments")]
            checks["fresh_target_evidence"] = len(queries) >= 2 and all(rid == selected for rid in queries)
        if result.get("structured_stop"):
            checks["grounded_stop"] = result.get("stop_check", {}).get("passed") is True
            checks["no_external_handoff"] = (result["structured_stop"]["ticket_created"] is False
                                               and result["structured_stop"]["human_contacted"] is False)
        if result.get("outcome") == "docs":
            chosen = result.get("selected_evidence", [])
            checks["selected_sources"] = bool(chosen) and result.get("rag_check", {}).get("passed") is True
            checks["source_bytes_match"] = docs_root is not None and all(_source_matches(hit, docs_root) for hit in chosen)
            for phrase in expected.get("evidence_contains", []):
                checks["evidence:" + phrase] = any(phrase in hit["excerpt"] for hit in chosen)
    elif step["op"] == "state" and "context" in output:
        packet = output["context"]
        checks["context_budget"] = packet["bytes"] <= packet["max_bytes"]
        checks["at_most_four_turns"] = len(packet["payload"]["recent_turns"]) <= 4
        checks["history_not_authorization"] = packet["payload"]["trust"] == "reference_only_not_tool_evidence"
        checks["zero_api"] = output["usage"]["model_api_calls"] == 0
    if step["op"] in ("chat", "state") and not expected:
        checks["declared_expected_behavior"] = False
    return {"passed": all(checks.values()), "checks": checks}


def _source_matches(hit, root):
    try:
        path = (Path(root) / Path(hit["source"]).name).resolve()
        if not path.is_relative_to(Path(root).resolve()):
            return False
        raw = path.read_bytes()
        excerpt = "\n".join(raw.decode().splitlines()[hit["line_start"] - 1:hit["line_end"]])
        return (file_hash(path) == hit["document_sha256"] and excerpt == hit["excerpt"]
                and hit["citation"] == f"{hit['doc_id']}@{hit['document_sha256'][:12]}:L{hit['line_start']}-L{hit['line_end']}")
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_case(case):
    started = perf_counter()
    rows = []
    with TemporaryDirectory(prefix="supportlab_holdout_") as directory:
        sandbox = Sandbox(Path(directory))
        for number, step in enumerate(case["steps"], 1):
            try:
                output = sandbox.run(step)
            except Exception as error:
                # No raw exception text or credentials are included in reports.
                output = {"error": type(error).__name__}
            grade = grade_step(step, output, docs_root=sandbox.docs)
            rows.append({"step": number, "operation": step["op"], "grade": grade, "output": output})
            # Setup faults make later observations meaningless. Expected application
            # errors (for example invalid preferences) may continue normally.
            if "error" in output and not grade["passed"]:
                break
    return {"id": case["id"], "family": case["family"], "title": case["title"],
            "passed": len(rows) == len(case["steps"]) and all(row["grade"]["passed"] for row in rows),
            "elapsed_ms": round((perf_counter() - started) * 1000, 3), "steps": rows}


def run_acceptance(*, case_id=None, progress=None):
    suite, baseline, seal = load_sealed_suite()
    cases = [row for row in suite["cases"] if case_id is None or row["id"] == case_id]
    if not cases:
        raise ValueError("Unknown locked case ID")
    started = perf_counter()
    rows = []
    for case in cases:
        rows.append(run_case(case))
        if progress:
            progress(rows[-1])
    # Detect source changes that happened during a run, too.
    load_sealed_suite()
    chat_outputs = [step["output"]["result"] for row in rows for step in row["steps"] if "result" in step["output"]]
    families = {family: {"passed": sum(row["passed"] for row in rows if row["family"] == family),
                          "cases": sum(row["family"] == family for row in rows)} for family in sorted({row["family"] for row in rows})}
    failures = [{"id": row["id"], "family": row["family"], "step": step["step"],
                 "failed_checks": [name for name, passed in step["grade"]["checks"].items() if not passed]}
                for row in rows for step in row["steps"] if not step["grade"]["passed"]]
    return {"created_at": datetime.now(timezone.utc).isoformat(), "suite_version": suite["dataset_version"],
            "split": suite["split"], "simulation": True, "execution_mode": "offline_fixed_application",
            "case_count": len(rows), "passed_count": sum(row["passed"] for row in rows), "families": families,
            "step_checks": {"numerator": sum(step["grade"]["passed"] for row in rows for step in row["steps"]),
                            "denominator": sum(len(row["steps"]) for row in rows)},
            "model_api_calls": sum(row["usage"]["model_api_calls"] for row in chat_outputs),
            "tool_calls": sum(row["usage"].get("tool_calls", 0) for row in chat_outputs),
            "token_usage": None, "cost": None, "human_review": "pending", "independent_authorship": False,
            "full_suite": case_id is None, "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            "provenance": {"frozen_application_sha256": baseline["frozen_application_sha256"],
                           "baseline_sha256": file_hash(BASELINE), "suite_sha256": file_hash(SUITE),
                           "seal_sha256": file_hash(SEAL), "harness_version": HARNESS_VERSION,
                           "frozen_files_verified_before_and_after": True,
                           "harness_files": {name: seal["files"][name] for name in sorted(ALLOWED_ADDITIONS)},
                           "python": platform.python_version(), "platform": platform.system(),
                           "dependencies": {name: version(name) for name in ("openai", "langgraph", "langgraph-checkpoint-sqlite")}},
            "reproduce": ".\\.venv\\Scripts\\python.exe -m supportlab.acceptance --output reports/holdout-v1-new-run.json" + (f" --case {case_id}" if case_id else ""),
            "limitations": ["保留的是完整组合/会话轨迹模板；基本业务规则与开发集相同，并非未见根因。",
                            "AI 在冻结实现后构造题目与真值，不是第三方盲测；人工复核仍 pending。",
                            "家族名不重合只能排除标签重叠，语义隔离需参照家族说明人工复核。",
                            "只运行离线固定流程，未测模型选工具、真实模型鲁棒性或架构优势。",
                            "若据失败修改实现或提示，相关家族应转为开发数据，并补新保留家族；后续通过不替代首次结果。"],
            "failures": failures, "cases": rows}


@contextmanager
def report_reservation(output):
    """Exclusive create BEFORE execution: never overwrite an earlier acceptance report."""
    path = output.resolve()
    if path.parent != (PROJECT_ROOT / "reports").resolve() or path.suffix != ".json":
        raise ValueError("报告须为项目 reports 目录下的新 .json 文件。")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        try:
            yield stream
        except BaseException as error:
            stream.seek(0)
            stream.truncate()
            json.dump({"status": "incomplete", "error_type": type(error).__name__,
                       "created_at": datetime.now(timezone.utc).isoformat(), "do_not_count_as_pass": True}, stream)
            stream.write("\n")
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description="冻结版本的组合场景验收；仅离线，不加载模型配置")
    parser.add_argument("--verify-only", action="store_true", help="只校验代码、资料、题目及验收程序的封存指纹")
    parser.add_argument("--case", dest="case_id", help="只回放一个案例；子集不算完整验收")
    parser.add_argument("--output", type=Path, help="保存到新的 reports/*.json；已有报告拒绝覆盖")
    args = parser.parse_args(argv)
    try:
        suite, baseline, _ = load_sealed_suite()
        if args.verify_only:
            print(f"封存校验通过：{len(suite['cases'])} 个案例；冻结实现 {baseline['frozen_application_sha256']}。")
            return 0
        if args.output is None:
            raise ValueError("请用 --output 指定新的报告路径，以保留首次与后续结果。")
        with report_reservation(args.output) as stream:
            report = run_acceptance(case_id=args.case_id,
                                    progress=lambda row: print(f"{'通过' if row['passed'] else '失败'} | {row['id']} | {row['title']}", flush=True))
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"离线组合验收：{report['passed_count']}/{report['case_count']}；实际 API {report['model_api_calls']} 次。")
        print("不代表真实模型准确率；人工复核 pending。首次结果应永久保留。")
        return 0 if report["passed_count"] == report["case_count"] else 1
    except (OSError, ValueError, KeyError) as error:
        print(f"验收未完成：{type(error).__name__}。请检查封存文件及报告路径；没有自动修订封存记录。")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
