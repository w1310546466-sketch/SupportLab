"""Offline adapter replay: real local tools, deliberately constructed responses."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .agent import SYSTEM_MESSAGE, run_loop
from .configuration import DeepSeekConfig
from .deepseek import ADAPTER_PROMPT, ADAPTER_VERSION, DeepSeekModel, ModelLimits
from .evaluation import fingerprint, prepare_case
from .execution import ToolExecutor
from .format_recovery import FORMAT_RECOVERY_VERSION, REPAIR_MESSAGE
from .storage import PROJECT_ROOT
from .tools import AppContext


RECOVERY_CASES = {
    "recovered": ("格式错一次，纠正后通过停止理由检查", "final_reply", 1),
    "still-invalid": ("纠正后仍格式错误，不再重试", "model_invalid_json", 1),
    "budget-used": ("总调用预算耗尽，不追加纠正请求", "model_budget_exhausted", 0),
    "correction-disabled": ("关闭格式纠正，首次错误即停止", "model_invalid_json", 0),
    "unsupported-reason": ("格式改对但停止理由无依据，仍被拦下", "unsupported_resolution", 1),
    "tool-during-repair": ("纠正时又提出工具调用，阻止执行", "model_repair_action_not_allowed", 1),
}


def _call(identifier, name):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps({"request_id": "req_demo_018"})}}


def _response(content=None, calls=None):
    return {"status": "ok", "finish_reason": "tool_calls" if calls else "stop", "usage": None,
            "message": {"content": content, "tool_calls": calls or []}}


class OfflineAdapter(DeepSeekModel):
    """Exercise the production adapter while explicitly reporting zero real usage."""
    execution_mode = "offline_adapter"
    provider = None

    def __init__(self, responses, *, limits=ModelLimits()):
        responses = iter(responses)
        super().__init__(DeepSeekConfig("offline-placeholder-not-a-real-key"), limits=limits,
                         transport=lambda *_: next(responses))
        self.model_name = "constructed-response-replay-v1"

    def usage_snapshot(self):
        usage = super().usage_snapshot()
        usage.update(simulated_model_requests=self.api_calls, simulated_responses_received=self.responses,
                     model_api_calls=0, model_responses_received=0, token_usage=None, usage_complete=False)
        return usage


def run_recovery_demo(case_id=None):
    names = list(RECOVERY_CASES) if case_id is None else [case_id]
    if any(name not in RECOVERY_CASES for name in names):
        raise ValueError("Unknown recovery case")
    correct = json.dumps({"type": "final", "outcome": "handoff", "resolution": {"reason": "account_unavailable"}})
    cases = []
    for name in names:
        responses = [_response(calls=[_call("request", "get_request_record"), _call("account", "get_account_state")]),
                     _response("```json\n" + correct + "\n```")]
        if name == "still-invalid":
            responses.append(_response('{"type":'))
        elif name == "unsupported-reason":
            responses.append(_response(json.dumps({"type": "final", "outcome": "clarify",
                                                   "resolution": {"reason": "missing_request_id"}})))
        elif name == "tool-during-repair":
            responses.append(_response(calls=[_call("extra", "get_account_state")]))
        elif name == "recovered":
            responses.append(_response(correct))
        limits = ModelLimits(max_api_calls=2 if name == "budget-used" else 4,
                             max_format_corrections=0 if name == "correction-disabled" else 1)
        with TemporaryDirectory(prefix="supportlab_recovery_") as directory:
            root = Path(directory)
            db, docs = prepare_case({"setup": "missing_snapshot", "request_id": "req_demo_018"}, root)
            fixture_hash = fingerprint([db, *docs.glob("*.md")], root=root)
            result = run_loop(OfflineAdapter(responses, limits=limits),
                              ToolExecutor(AppContext("acc_demo"), db_path=db, docs_root=docs),
                              user_message="排查 req_demo_018", target_request_id="req_demo_018")
        title, reason, corrections = RECOVERY_CASES[name]
        matched = (result["reason"] == reason and result["usage"]["tool_calls"] == 2
                   and result["usage"]["format_recovery"]["correction_requests"] == corrections
                   and result["usage"]["simulated_model_requests"] == 2 + corrections
                   and result["structured_stop"]["observations"][1]["status"] == "not_found")
        if name == "recovered":
            matched = matched and result.get("stop_check", {}).get("passed") is True
        cases.append({"id": name, "title": title, "behavior_as_expected": matched,
                      "prepared_fixture_sha256": fixture_hash, "result": result})
    return {"created_at": datetime.now(timezone.utc).isoformat(), "simulation": True,
            "execution_mode": "offline_adapter", "split": "development", "dataset_version": "recovery-demo-v1",
            "model_api_calls": 0, "token_usage": None, "cost": None, "human_review": "pending",
            "case_count": len(cases), "expected_behavior_count": sum(case["behavior_as_expected"] for case in cases),
            "reproducibility": {"adapter_version": ADAPTER_VERSION, "format_recovery_version": FORMAT_RECOVERY_VERSION,
                                "code_sha256": fingerprint(list((PROJECT_ROOT / "supportlab").glob("*.py"))),
                                "prompt_sha256": sha256((SYSTEM_MESSAGE + "\n" + ADAPTER_PROMPT).encode("utf-8")).hexdigest(),
                                "repair_prompt_sha256": sha256(REPAIR_MESSAGE.encode("utf-8")).hexdigest()},
            "reproduce": ".\\.venv\\Scripts\\python.exe -m supportlab recovery-demo" + (f" --case {case_id}" if case_id else ""),
            "limitations": ["模型回复由程序故意构造；实际运行适配器和本地工具，但不联网、不加载真实密钥。",
                            "代码围栏只是本次构造例子，不能据此确定上次真实解析失败的具体原因。",
                            "格式恢复不等于证据成立；不是模型成功率或真实恢复成功率。"], "cases": cases}
