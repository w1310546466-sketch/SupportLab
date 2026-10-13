"""Isolated development examples of clarification and handoff, offline by default."""

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
from pathlib import Path
from tempfile import TemporaryDirectory

from .agent import SYSTEM_MESSAGE, run_loop
from .evaluation import fingerprint, prepare_case
from .execution import ToolExecutor
from .storage import PROJECT_ROOT
from .stop_check import STOP_CONTRACT
from .tools import AppContext


STOP_CASES = {
    "missing-id": {"title": "没有请求编号，应追问", "setup": "baseline", "request_id": None,
                   "outcome": "clarify", "reason": "missing_request_id", "tools": []},
    "missing-account": {"title": "请求查到了，账号快照没查到", "setup": "missing_snapshot", "request_id": "req_demo_018",
                        "outcome": "handoff", "reason": "account_unavailable", "tools": ["get_request_record", "get_account_state"]},
    "conflicting-records": {"title": "错误码称配额耗尽，但快照为 999/1000", "setup": "quota_conflict", "request_id": "req_demo_018",
                            "outcome": "handoff", "reason": "evidence_conflict", "tools": ["get_request_record", "get_account_state"]},
    "tool-failure": {"title": "数据库不可用，不能当成业务根因", "setup": "database_missing", "request_id": "req_demo_018",
                     "outcome": "handoff", "reason": "tool_failure", "tools": ["get_request_record"]},
    "premature-stop": {"title": "尚未查账号，却声称快照找不到，应拦下", "setup": "baseline", "request_id": "req_demo_018",
                       "outcome": "handoff", "reason": "account_unavailable", "tools": ["get_request_record"]},
}


class StopDemoModel:
    execution_mode = "scripted"
    structured_stops = True
    provider = None
    model_name = "stop-demo-script-v1"

    def __init__(self, case):
        self.case = case

    def next_action(self, messages, tools):
        count = sum(item["role"] == "tool" for item in messages)
        if count < len(self.case["tools"]):
            return {"type": "tool_call", "call_id": f"lookup_{count}", "name": self.case["tools"][count],
                    "arguments": {"request_id": self.case["request_id"]}}
        return {"type": "final", "outcome": self.case["outcome"], "resolution": {"reason": self.case["reason"]}}


def run_stop_demo(case_id=None, *, live=False):
    if live and case_id != "missing-account":
        raise ValueError("Live demonstration is limited to missing-account")
    names = list(STOP_CASES) if case_id is None else [case_id]
    if any(name not in STOP_CASES for name in names):
        raise ValueError("Unknown stopping demonstration")
    reproducibility = {"stop_contract": STOP_CONTRACT,
                       "code_sha256": fingerprint(list((PROJECT_ROOT / "supportlab").glob("*.py")))}
    if live:
        from .configuration import load_deepseek_config
        from .deepseek import ADAPTER_PROMPT, ADAPTER_VERSION, DeepSeekModel
        from .format_recovery import FORMAT_RECOVERY_VERSION, REPAIR_MESSAGE
        model = DeepSeekModel(load_deepseek_config())
        reproducibility.update(adapter_version=ADAPTER_VERSION, sdk_version=importlib.metadata.version("openai"),
                               prompt_sha256=sha256((SYSTEM_MESSAGE + "\n" + ADAPTER_PROMPT).encode("utf-8")).hexdigest(),
                               format_recovery_version=FORMAT_RECOVERY_VERSION,
                               repair_prompt_sha256=sha256(REPAIR_MESSAGE.encode("utf-8")).hexdigest())
    cases = []
    for name in names:
        case = STOP_CASES[name]
        with TemporaryDirectory(prefix="supportlab_stop_") as directory:
            db, docs = prepare_case(case, Path(directory))
            fixture_hash = fingerprint([path for path in [db, *docs.glob("*.md")] if path.exists()], root=Path(directory))
            result = run_loop(model if live else StopDemoModel(case),
                              ToolExecutor(AppContext("acc_demo"), db_path=db, docs_root=docs),
                              user_message=f"请排查请求 {case['request_id'] or '（尚未提供）'}。",
                              target_request_id=case["request_id"])
        check = result.get("stop_check", {})
        if name == "premature-stop":
            matched = result["reason"] == "unsupported_resolution" and check.get("issues") == ["stop_reason_not_supported"]
        else:
            matched = (result["outcome"] == case["outcome"] and result["reason"] == "final_reply"
                       and check.get("passed") is True and check.get("proposed_reason") == case["reason"])
        cases.append({"id": name, "title": case["title"], "setup": case["setup"], "request_id": case["request_id"],
                      "prepared_fixture_sha256": fixture_hash, "behavior_as_expected": matched, "result": result})
    return {"created_at": datetime.now(timezone.utc).isoformat(), "simulation": True,
            "execution_mode": "live" if live else "scripted", "split": "development", "dataset_version": "stop-demo-v1",
            "case_count": len(cases), "expected_behavior_count": sum(case["behavior_as_expected"] for case in cases),
            "model_api_calls": sum(case["result"]["usage"]["model_api_calls"] for case in cases),
            "human_review": "pending", "cost": None, "reproducibility": reproducibility,
            "reproduce": ".\\.venv\\Scripts\\python.exe -m supportlab stop-demo" +
                         (f" --case {case_id}" if case_id else "") + (" --live" if live else ""),
            "limitations": ["构造开发场景；手写脚本与真实模型分别标记，不代表测试集成功率。",
                            "只检查现有有限停止理由；没有联系人工、创建工单或续接任务。"], "cases": cases}
