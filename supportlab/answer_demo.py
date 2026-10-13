"""Four scripted development probes of the answer gate; no network or API key."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from .agent import run_loop
from .answer_check import ANSWER_CONTRACT
from .execution import ToolExecutor
from .storage import PROJECT_ROOT, init_demo_db
from .tools import AppContext


DEMO_CASES = {
    "supported": ("正确配额诊断，建议标为待核实", None),
    "wrong-cause": ("拿着真实引用，却断言频率超限", "cause_not_supported"),
    "unsupported-advice": ("诊断正确，但承诺立即增加配额", "unsupported_follow_up"),
    "missing-account": ("缺少账号查询结果，仍尝试诊断", "missing_evidence"),
}


class AnswerDemoModel:
    execution_mode = "scripted"
    provider = None
    model_name = "answer-demo-script-v1"

    def __init__(self, case):
        self.case = case

    def next_action(self, messages, tools):
        results = {item["name"]: item for item in messages if item["role"] == "tool"}
        if "get_request_record" not in results:
            return {"type": "tool_call", "call_id": "request", "name": "get_request_record",
                    "arguments": {"request_id": "req_demo_018"}}
        if self.case != "missing-account" and "get_account_state" not in results:
            return {"type": "tool_call", "call_id": "account", "name": "get_account_state",
                    "arguments": {"request_id": "req_demo_018"}}
        if "search_docs" not in results:
            return {"type": "tool_call", "call_id": "docs", "name": "search_docs",
                    "arguments": {"query": "MONTHLY_QUOTA_EXHAUSTED", "service_version": "demo-api-v1", "top_k": 1}}
        citations = results["search_docs"]["content"].get("matches", [])
        if not citations:
            return {"type": "final", "outcome": "handoff", "message": "演示资料未准备好。", "citations": []}
        assessment = {"cause": "monthly_quota_exhausted",
                      "request_evidence_id": results["get_request_record"]["content"]["evidence_id"],
                      "account_evidence_id": results.get("get_account_state", {}).get("content", {}).get("evidence_id", "E9"),
                      "rule_citation": citations[0]["citation"],
                      "follow_up": ["check_current_status", "verify_quota_options"]}
        if self.case == "wrong-cause":
            assessment["cause"] = "rate_limit_exceeded"
            assessment["follow_up"] = ["check_current_status"]
        if self.case == "unsupported-advice":
            assessment["follow_up"] = ["check_current_status", "increase_quota_now"]
        return {"type": "final", "outcome": "diagnosis", "assessment": assessment}


def run_answer_demo(case_id=None):
    from .evaluation import fingerprint
    from .retrieval import DEFAULT_DOCS_ROOT, DOCUMENTS
    from .storage import DEFAULT_FIXTURE, DEFAULT_ACCOUNT_FIXTURE

    names = list(DEMO_CASES) if case_id is None else [case_id]
    if any(name not in DEMO_CASES for name in names):
        raise ValueError("Unknown answer demonstration")
    cases = []
    for name in names:
        with TemporaryDirectory(prefix="supportlab_answer_") as directory:
            db = Path(directory) / "demo.sqlite3"
            init_demo_db(db)
            result = run_loop(AnswerDemoModel(name), ToolExecutor(AppContext("acc_demo"), db_path=db),
                              user_message="排查 req_demo_018", target_request_id="req_demo_018")
        title, expected_issue = DEMO_CASES[name]
        check = result.get("answer_check", {})
        matched = (result["status"] == "completed" and check.get("passed") is True) if expected_issue is None else (
            result["status"] == "handoff" and check.get("issues") == [expected_issue])
        cases.append({"id": name, "title": title, "expected_issue": expected_issue,
                      "behavior_as_expected": matched, "result": result})
    return {"created_at": datetime.now(timezone.utc).isoformat(), "simulation": True,
            "execution_mode": "scripted", "split": "development", "dataset_version": "answer-demo-v1",
            "answer_contract": ANSWER_CONTRACT, "model_api_calls": 0, "token_usage": None,
            "human_review": "pending", "case_count": len(cases),
            "expected_behavior_count": sum(case["behavior_as_expected"] for case in cases),
            "code_sha256": fingerprint(list((PROJECT_ROOT / "supportlab").glob("*.py"))),
            "source_sha256": fingerprint([DEFAULT_FIXTURE, DEFAULT_ACCOUNT_FIXTURE,
                                           *(DEFAULT_DOCS_ROOT / doc.filename for doc in DOCUMENTS)]),
            "reproduce": ".\\.venv\\Scripts\\python.exe -m supportlab answer-demo",
            "limitations": ["脚本故意构造的开发反例，不是观测到的真实模型错误或成功率。",
                            "仅核验固定版本两种 429 和有限建议选项，不是通用语义评审。"], "cases": cases}
