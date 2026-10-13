"""Controlled fixed-workflow vs dynamic-agent comparison and offline harness."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from .agent import SYSTEM_MESSAGE, LoopLimits, run_loop
from .configuration import DeepSeekConfig, load_deepseek_config
from .contracts import model_tool_schemas
from .deepseek import ADAPTER_PROMPT, ADAPTER_VERSION, DeepSeekModel, ModelLimits
from .evaluation import fingerprint, load_cases, prepare_case
from .execution import ToolExecutor
from .fixed_flow import FixedFlowModel
from .storage import PROJECT_ROOT
from .tools import AppContext
from .workflow import ModelWorkflow, WORKFLOW_PROMPT, WORKFLOW_VERSION

COMPARISON_VERSION = "controlled-comparison-v1"
CASE_IDS = ("rate-ok", "quota-ok", "missing-snapshot", "quota-conflict", "unsupported-401", "database-missing")
STOP_EXPECTATIONS = {"missing-snapshot": "account_unavailable", "quota-conflict": "evidence_conflict",
                     "unsupported-401": "unsupported_case", "database-missing": "tool_failure"}


class RuleReplayAdapter(DeepSeekModel):
    """Constructed responses exercise the harness only, never model quality."""
    execution_mode = "offline_adapter"
    provider = None

    def __init__(self, request_id, *, system_prompt=ADAPTER_PROMPT, limits=ModelLimits()):
        self.planner = FixedFlowModel(request_id)
        super().__init__(DeepSeekConfig("offline-placeholder-not-real"), system_prompt=system_prompt,
                         limits=limits, transport=self._respond)
        self.model_name = "rule-replay-harness-v1"

    def _respond(self, payload, timeout):
        messages = payload["messages"][:2]
        names = {}
        try:
            envelope = json.loads(messages[1]["content"])
        except ValueError:
            envelope = {}
        for row in envelope.get("application_tool_results", []) if isinstance(envelope, dict) else []:
            messages.append({"role": "assistant", "tool_call": {"type": "tool_call", "call_id": row["call_id"],
                             "name": row["name"], "arguments": row["arguments"]}})
            messages.append({"role": "tool", "tool_call_id": row["call_id"], "name": row["name"], "content": row["result"]})
        for item in payload["messages"][2:]:
            if item["role"] == "assistant":
                for call in item.get("tool_calls", []):
                    names[call["id"]] = call["function"]["name"]
                    messages.append({"role": "assistant", "tool_call": {"type": "tool_call", "call_id": call["id"],
                                     "name": call["function"]["name"], "arguments": json.loads(call["function"]["arguments"])}})
            elif item["role"] == "tool":
                messages.append({"role": "tool", "tool_call_id": item["tool_call_id"], "name": names[item["tool_call_id"]],
                                 "content": json.loads(item["content"])})
        action = self.planner.next_action(messages, [])
        call = None
        if action["type"] == "tool_call":
            call = {"id": action["call_id"], "type": "function", "function": {"name": action["name"], "arguments": json.dumps(action["arguments"])}}
        return {"status": "ok", "finish_reason": "tool_calls" if call else "stop", "usage": None,
                "message": {"tool_calls": [call] if call else [], "content": None if call else json.dumps(action)}}

    def usage_snapshot(self):
        result = super().usage_snapshot()
        result.update(simulated_model_requests=self.api_calls, model_api_calls=0, model_responses_received=0,
                      token_usage=None, usage_complete=False)
        return result


def grade_comparison(case, result):
    expected = case["expected"]
    checks = {"expected_outcome": result["outcome"] == expected["outcome"],
              "within_tool_budget": result["usage"]["tool_calls"] <= 4,
              "within_model_budget": result["usage"]["model_api_calls"] <= 4,
              "structured_final": result["reason"] == "final_reply"}
    if expected["outcome"] == "diagnosis":
        checks["evidence_check"] = result.get("answer_check", {}).get("passed") is True
        checks["cause"] = result.get("structured_answer", {}).get("diagnosis", {}).get("cause") == expected["error_code"].lower()
    else:
        checks["stop_check"] = result.get("stop_check", {}).get("passed") is True
        checks["supported_stop_reason"] = result.get("stop_check", {}).get("proposed_reason") == STOP_EXPECTATIONS[case["id"]]
    return {"passed": all(checks.values()), "checks": checks}


def run_comparison(*, case_id=None, live=False):
    if case_id not in (None, *CASE_IDS):
        raise ValueError("未知对照场景。")
    if live and case_id not in ("rate-ok", "quota-ok", "missing-snapshot"):
        raise ValueError("真实对照须指定 rate-ok、quota-ok 或 missing-snapshot，只运行一个案例，两组总上限 8 次模型请求。")
    cases = [case for case in load_cases()["cases"] if case["id"] in CASE_IDS and (case_id is None or case["id"] == case_id)]
    config = load_deepseek_config() if live else None
    model_limits, loop_limits = ModelLimits(), LoopLimits()
    rows = []
    for case in cases:
        with TemporaryDirectory(prefix="supportlab_compare_") as directory:
            root = Path(directory)
            db, docs = prepare_case(case, root)
            sources = [path for path in [db, *docs.glob("*.md")] if path.exists()]
            source_hash = fingerprint(sources, root=root)
            for mode, prompt in (("workflow", WORKFLOW_PROMPT), ("agent", ADAPTER_PROMPT)):
                adapter = (DeepSeekModel(config, limits=model_limits, system_prompt=prompt) if live else
                           RuleReplayAdapter(case["request_id"], limits=model_limits, system_prompt=prompt))
                model = ModelWorkflow(case["request_id"], adapter) if mode == "workflow" else adapter
                result = run_loop(model, ToolExecutor(AppContext("acc_demo"), db_path=db, docs_root=docs),
                                  user_message=f"请排查请求 {case['request_id']}。", target_request_id=case["request_id"], limits=loop_limits)
                unchanged = source_hash == fingerprint(sources, root=root)
                grade = grade_comparison(case, result)
                grade["checks"]["unchanged_shared_sources"] = unchanged
                grade["passed"] = all(grade["checks"].values())
                rows.append({"case_id": case["id"], "family": case["family"], "configuration": mode,
                             "source_sha256": source_hash, "grade": grade, "result": result})
    summaries = {}
    for mode in ("workflow", "agent"):
        group = [row for row in rows if row["configuration"] == mode]
        usages = [row["result"]["usage"] for row in group]
        tokens = [usage["token_usage"] for usage in usages if usage["token_usage"] is not None]
        summaries[mode] = {"passed": sum(row["grade"]["passed"] for row in group), "cases": len(group),
                           "model_api_calls": sum(usage["model_api_calls"] for usage in usages),
                           "tool_calls": sum(usage["tool_calls"] for usage in usages),
                           "elapsed_ms_sum": round(sum(usage["elapsed_ms"] for usage in usages), 3),
                           "token_usage": {key: sum(row[key] for row in tokens) for key in ("input_tokens", "output_tokens", "total_tokens")} if tokens else None,
                           "usage_complete": bool(live) and all(usage["usage_complete"] for usage in usages)}
    return {"created_at": datetime.now(timezone.utc).isoformat(), "simulation": True, "split": "development",
            "execution_mode": "live" if live else "offline_harness", "comparison_version": COMPARISON_VERSION,
            "dataset_version": "agent-dev-v1-subset", "summaries": summaries, "human_review": "pending", "cost": None,
            "model_api_calls": sum(row["result"]["usage"]["model_api_calls"] for row in rows),
            "model_api_cap": 8 if live else 0,
            "controlled": {"same_model": config.model if config else "rule-replay-harness-v1", "same_tool_schemas": model_tool_schemas(),
                           "same_account": "acc_demo", "same_case_and_sources": True, "max_model_requests_per_arm": 4,
                           "max_tools_per_arm": 4, "max_turns_per_arm": 8, "max_output_tokens_per_request": 768,
                           "request_timeout_seconds": 25, "adapter_seconds_per_arm": 110,
                           "order": ["workflow", "agent"], "repetitions": 1},
            "reproducibility": {"workflow_version": WORKFLOW_VERSION, "adapter_version": ADAPTER_VERSION,
                                "sdk_version": version("openai"), "code_sha256": fingerprint(list((PROJECT_ROOT / "supportlab").glob("*.py"))),
                                "dataset_sha256": fingerprint([PROJECT_ROOT / "evals" / "dev_cases.json"]),
                                "prompts": {name: sha256((SYSTEM_MESSAGE + "\n" + prompt).encode()).hexdigest()
                                            for name, prompt in (("workflow", WORKFLOW_PROMPT), ("agent", ADAPTER_PROMPT))}},
            "reproduce": ".\\.venv\\Scripts\\python.exe -m supportlab compare" + (f" --case {case_id}" if case_id else "") + (" --live" if live else ""),
            "limitations": ["固定流程由程序决定取证路径、同一个模型形成最终结论；Agent 自主选择取证。两者提示词因控制方式不同而不同。",
                            "Chat Completions 不支持插入模型未生成的工具调用：固定流程以应用证据包传入反馈，Agent 使用原生工具消息；信息载体也不同。",
                            "同权限和可用资料，不强制两组实际收集完全相同的证据；消耗分别测量。",
                            "离线响应来自规则，仅验证对照程序，不证明模型表现。单次真实开发案例不能证明哪种架构更好。",
                            "运行顺序固定、没有重复稳定性测量；缓存和服务波动可能影响延迟与 token；保留原始报告。"], "rows": rows}
