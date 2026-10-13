"""Local development replay and narrow rule grading. Never a real-model score."""

from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import sqlite3
from tempfile import TemporaryDirectory

from .agent import LOOP_VERSION, SYSTEM_MESSAGE, LoopLimits, run_loop
from .execution import ToolExecutor
from .mock_model import DemoMockModel
from .retrieval import DEFAULT_DOCS_ROOT, DOCUMENTS, RETRIEVAL_VERSION
from .storage import DEFAULT_ACCOUNT_FIXTURE, DEFAULT_FIXTURE, PROJECT_ROOT, init_demo_db
from .tools import AppContext


CASE_FILE = PROJECT_ROOT / "evals" / "dev_cases.json"
GRADER_VERSION = "mock-rules-v1"
TOOL_ORDER = ("get_request_record", "get_account_state", "search_docs")
SETUPS = {
    "baseline", "missing_snapshot", "stale_snapshot", "snapshot_version",
    "quota_conflict", "rate_conflict", "both_exceeded", "unsupported_503",
    "unsupported_401", "docs_version", "docs_missing", "database_missing",
}


def load_cases(path: Path = CASE_FILE) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("simulation") is not True or data.get("split") != "development"
            or data.get("dataset_version") != "agent-dev-v1" or not data.get("cases")):
        raise ValueError("Expected simulated development cases")
    identifiers = [case["id"] for case in data["cases"]]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate case IDs")
    for case in data["cases"]:
        if case["setup"] not in SETUPS or case["expected"]["outcome"] not in {"diagnosis", "clarify", "handoff"}:
            raise ValueError("Unsupported case")
        if len(case["expected"]["tool_statuses"]) > len(TOOL_ORDER):
            raise ValueError("Unsupported mock path")
        LoopLimits(**case.get("limits", {}))
    return data


class RecordingExecutor(ToolExecutor):
    """Keep simulated feedback for grading; nothing here is given to the model."""

    def __init__(self, **kwargs):
        super().__init__(AppContext("acc_demo"), **kwargs)
        self.calls = []

    def execute(self, name, arguments):
        execution = super().execute(name, arguments)
        self.calls.append({"name": name, "arguments": deepcopy(execution.validated_arguments),
                           "result": deepcopy(execution.result)})
        return execution


class WrongCauseProbe(DemoMockModel):
    """Deliberately faulty local template: genuine citation, incorrect conclusion."""

    def next_action(self, messages, tools):
        action = super().next_action(messages, tools)
        if action["type"] == "final" and action["outcome"] == "diagnosis":
            cause = "频率限制" if action["message"].startswith("模拟诊断：月度配额耗尽。") else "月度配额耗尽"
            _, rest = action["message"].split("。", 1)
            action["message"] = f"模拟诊断：{cause}。{rest}"
        return action


def prepare_case(case: dict, root: Path) -> tuple[Path, Path]:
    """Changes affect only a new temporary sandbox for this case."""
    db, docs = root / "demo.sqlite3", root / "docs"
    docs.mkdir()
    for document in DOCUMENTS:
        if case["setup"] == "docs_missing" and document.doc_id == "demo-errors-v1":
            continue
        (docs / document.filename).write_bytes((DEFAULT_DOCS_ROOT / document.filename).read_bytes())
    if case["setup"] == "database_missing":
        return db, docs
    init_demo_db(db)
    # These statements are app-owned fixture changes, never model-supplied SQL.
    statements = {
        "missing_snapshot": "DELETE FROM account_states WHERE request_id = ?",
        "stale_snapshot": "UPDATE account_states SET captured_at = '2026-09-26T00:00:00Z' WHERE request_id = ?",
        "snapshot_version": "UPDATE account_states SET service_version = 'demo-api-v2' WHERE request_id = ?",
        "quota_conflict": "UPDATE account_states SET monthly_quota_used = 999 WHERE request_id = ?",
        "rate_conflict": "UPDATE account_states SET requests_in_window = 2 WHERE request_id = ?",
        "both_exceeded": "UPDATE account_states SET requests_in_window = 61 WHERE request_id = ?",
        "unsupported_503": "UPDATE request_records SET http_status = 503, error_code = 'SERVICE_UNAVAILABLE' WHERE request_id = ?",
        "unsupported_401": "UPDATE request_records SET http_status = 401, error_code = 'INVALID_API_KEY' WHERE request_id = ?",
    }
    with closing(sqlite3.connect(db)) as connection, connection:
        if case["setup"] in statements:
            connection.execute(statements[case["setup"]], (case["request_id"],))
        elif case["setup"] == "docs_version":
            connection.execute("UPDATE request_records SET service_version = 'demo-api-v2' WHERE request_id = ?", (case["request_id"],))
            connection.execute("UPDATE account_states SET service_version = 'demo-api-v2' WHERE request_id = ?", (case["request_id"],))
    return db, docs


def grade_case(case: dict, result: dict, calls: list[dict]) -> dict:
    """Exact mock regression rules; no LLM judge or general semantic entailment."""
    expected = case["expected"]
    failures = []

    def check(name, condition):
        if not condition:
            failures.append(name)

    check("outcome", result["outcome"] == expected["outcome"])
    check("status", result["status"] == {"diagnosis": "completed", "clarify": "needs_input", "handoff": "handoff"}[expected["outcome"]])
    check("stop_reason", result["reason"] == expected.get("reason", "final_reply"))
    check("mock_tool_path", [call["name"] for call in calls] == list(TOOL_ORDER[:len(expected["tool_statuses"])]))
    check("tool_statuses", [call["result"]["status"] for call in calls] == expected["tool_statuses"])
    check("call_accounting", len(calls) == result["usage"]["tool_calls"] <= result["limits"]["max_tool_calls"])
    check("turn_budget", result["usage"]["decision_turns"] <= result["limits"]["max_turns"])
    check("mock_label", result["execution_mode"] == "mock" and result["model_called"] is False
          and result["usage"]["model_api_calls"] == 0 and result["usage"]["token_usage"] is None)
    by_name = {call["name"]: call for call in calls}
    for call in calls:
        if call["name"] in TOOL_ORDER[:2] and call["result"]["status"] == "ok":
            record = call["result"].get("record", call["result"].get("account_state", {}))
            check("request_scope", record.get("request_id") == case["request_id"])
        if call["result"]["status"] != "ok":
            check("no_evidence_on_error", not any(key in call["result"] for key in ("record", "account_state")))

    docs = by_name.get("search_docs", {}).get("result", {}).get("matches", [])
    citations = result["citations"]
    check("citation_origin", set(citations) <= {hit["citation"] for hit in docs})
    if expected["outcome"] == "diagnosis":
        check("diagnosis_text", f"模拟诊断：{expected['cause']}。" in result["answer"]
              and expected["forbidden_cause"] not in result["answer"])
        request = by_name.get("get_request_record", {}).get("result", {}).get("record", {})
        state = by_name.get("get_account_state", {}).get("result", {}).get("account_state", {})
        check("request_error_code", request.get("error_code") == expected["error_code"])
        check("account_evidence", all(state.get(key) == value for key, value in expected["evidence"].items()))
        check("evidence_alignment", bool(request) and bool(state)
              and state.get("captured_at") == request.get("occurred_at")
              and state.get("service_version") == request.get("service_version"))
        search_args = by_name.get("search_docs", {}).get("arguments") or {}
        check("feedback_used_for_search", search_args.get("query") == expected["error_code"]
              and search_args.get("service_version") == request.get("service_version"))
        check("expected_rule_cited", bool(citations) and any(
            hit["citation"] in citations and hit["doc_id"] == "demo-errors-v1"
            and hit["line_start"] == expected["rule_line"]
            and expected["error_code"] in hit["excerpt"]
            and hit["service_version"] == request.get("service_version") for hit in docs))
    else:
        check("no_diagnosis_citation", not citations)
        check("action_text", expected["answer_contains"] in result["answer"])
    return {"automated_pass": not failures, "failed_checks": failures,
            "semantic_support_review": "pending" if expected["outcome"] == "diagnosis" else "not_applicable"}


def fingerprint(paths: list[Path], *, root: Path = PROJECT_ROOT) -> str:
    digest = sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def run_evaluation(*, case_id: str | None = None, wrong_cause: bool = False) -> dict:
    dataset = load_cases()
    cases = [case for case in dataset["cases"] if case_id is None or case["id"] == case_id]
    if not cases:
        raise ValueError("Unknown case ID")
    if wrong_cause and (len(cases) != 1 or cases[0]["expected"]["outcome"] != "diagnosis"):
        raise ValueError("Wrong-cause demo requires one diagnosis case")
    rows = []
    for case in cases:
        with TemporaryDirectory(prefix="supportlab_eval_") as temporary:
            db, docs = prepare_case(case, Path(temporary))
            executor = RecordingExecutor(db_path=db, docs_root=docs)
            model = (WrongCauseProbe if wrong_cause else DemoMockModel)(case["request_id"])
            # Expected labels, fixture setup and rationale are deliberately excluded.
            result = run_loop(model, executor, user_message=f"请排查请求 {case['request_id'] or '（尚未提供）'}。",
                              limits=LoopLimits(**case.get("limits", {})))
            grade = grade_case(case, result, executor.calls)
            rows.append({"case_id": case["id"], "family": case["family"], "title": case["title"],
                         "setup": case["setup"], "expected": case["expected"], "rationale": case["rationale"],
                         **grade, "result": result, "tool_evidence": executor.calls})
    return {
        "created_at": datetime.now(timezone.utc).isoformat(), "simulation": True,
        "split": dataset["split"], "dataset_version": dataset["dataset_version"],
        "dataset_sha256": sha256(CASE_FILE.read_bytes()).hexdigest(),
        "code_sha256": fingerprint(list((PROJECT_ROOT / "supportlab").glob("*.py"))),
        "source_sha256": fingerprint([DEFAULT_FIXTURE, DEFAULT_ACCOUNT_FIXTURE,
                                       *(DEFAULT_DOCS_ROOT / document.filename for document in DOCUMENTS)]),
        "prompt_sha256": sha256(SYSTEM_MESSAGE.encode("utf-8")).hexdigest(),
        "loop_version": LOOP_VERSION, "retrieval_version": RETRIEVAL_VERSION, "grader_version": GRADER_VERSION,
        "model": "deliberately-wrong-cause-mock-v1" if wrong_cause else "demo-rule-mock-v1",
        "model_api_calls": 0, "token_usage": None, "cost": None,
        "python_version": platform.python_version(), "sqlite_version": sqlite3.sqlite_version,
        "human_review_status": dataset["human_review_status"], "task_success_rate": None,
        "automated_pass_count": sum(row["automated_pass"] for row in rows), "case_count": len(rows),
        "tool_calls_total": sum(row["result"]["usage"]["tool_calls"] for row in rows),
        "elapsed_ms_total": round(sum(row["result"]["usage"]["elapsed_ms"] for row in rows), 3),
        "failed_case_ids": [row["case_id"] for row in rows if not row["automated_pass"]],
        "reproduce": ".\\.venv\\Scripts\\python.exe -m supportlab eval-mock"
                     + (f" --case {case_id}" if case_id else "") + (" --demo-wrong-answer" if wrong_cause else ""),
        "limitations": ["构造开发集，已参与开发；不是保留测试集或真实模型评测。",
                        "当前精确路径和答案词面规则适用于 mock；接入真实模型后需审核等价行为与语义。",
                        "AI 核对不能代替用户人工核验；引用来源存在不等于支持结论。",
                        "工具时限沿用 2 秒；运行耗时包含本地进程开销，不代表模型延迟。"],
        "cases": rows,
    }


def format_report(report: dict) -> str:
    labels = {"diagnosis": "诊断", "clarify": "追问", "handoff": "转人工"}
    lines = [f"SupportLab 构造开发集 {report['dataset_version']} | {report['model']}",
             f"自动规则通过：{report['automated_pass_count']}/{report['case_count']}；真实模型调用：0；人工核验：待完成。"]
    for row in report["cases"]:
        status = "通过" if row["automated_pass"] else "未通过"
        result = row["result"]
        lines.append(f"[{status}] {row['case_id']}：{row['title']} → {labels[result['outcome']]}，工具 {result['usage']['tool_calls']} 次")
        if row["failed_checks"]:
            lines.append("  未满足：" + ", ".join(row["failed_checks"]))
    if len(report["cases"]) == 1:
        lines.append("实际答复：" + report["cases"][0]["result"]["answer"])
        lines.append("验收依据：" + report["cases"][0]["rationale"])
    lines.append("这是 mock 程序行为检查，不能报告为真实 Agent 成功率。")
    return "\n".join(lines)
