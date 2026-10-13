"""Reproducible development checks for the added business evidence; no hidden truths."""

from contextlib import closing
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from .agent import SYSTEM_MESSAGE, run_loop
from .comparison import RuleReplayAdapter
from .configuration import load_deepseek_config
from .deepseek import ADAPTER_PROMPT, ADAPTER_VERSION, DeepSeekModel
from .diagnostic_check import DIAGNOSTIC_CONTRACT, RULE_SHA256
from .evaluation import fingerprint
from .execution import ToolExecutor
from .retrieval import DEFAULT_DOCS_ROOT, DOCUMENTS
from .scenarios import DEFAULT_SCENARIOS, init_scenarios
from .storage import PROJECT_ROOT, init_demo_db
from .tools import AppContext
from .workflow import ModelWorkflow, WORKFLOW_PROMPT

DATASET = PROJECT_ROOT / "evals" / "diagnostic_dev.json"


def prepare_diagnostic_case(case, root):
    db, docs = root / "demo.sqlite3", root / "docs"
    docs.mkdir()
    for doc in DOCUMENTS:
        (docs / doc.filename).write_bytes((DEFAULT_DOCS_ROOT / doc.filename).read_bytes())
    init_demo_db(db)
    init_scenarios(db)
    rid, setup = case["request_id"], case["setup"]
    with closing(sqlite3.connect(db)) as connection, connection:
        if setup == "facts":
            payload = json.loads(connection.execute("SELECT facts_json FROM diagnostic_snapshots WHERE account_id='acc_demo' AND request_id=?", (rid,)).fetchone()[0])
            payload.update(case["changes"])
            connection.execute("UPDATE diagnostic_snapshots SET facts_json=? WHERE account_id='acc_demo' AND request_id=?", (json.dumps(payload), rid))
        elif setup == "missing":
            connection.execute("DELETE FROM diagnostic_snapshots WHERE request_id=?", (rid,))
        elif setup == "stale":
            connection.execute("UPDATE diagnostic_snapshots SET captured_at='2026-09-26T00:00:00Z' WHERE request_id=?", (rid,))
        elif setup == "version":
            connection.execute("UPDATE diagnostic_snapshots SET service_version='demo-api-v2' WHERE request_id=?", (rid,))
        elif setup == "foreign":
            connection.execute("UPDATE request_records SET account_id='acc_other' WHERE request_id=?", (rid,))
            connection.execute("UPDATE diagnostic_snapshots SET account_id='acc_other' WHERE request_id=?", (rid,))
        elif setup == "docs_missing":
            (docs / "demo-diagnostics-v1.md").unlink()
        elif setup == "docs_changed":
            path = docs / "demo-diagnostics-v1.md"
            path.write_bytes(path.read_bytes() + "\n未经审核的修订。\n".encode())
        elif setup != "baseline":
            raise ValueError("Unknown diagnostic setup")
    return db, docs


def grade_diagnostic(case, result):
    checks = {"outcome": result["outcome"] == case["outcome"], "structured_final": result["reason"] == "final_reply",
              "tool_budget": result["usage"]["tool_calls"] <= 4, "api_budget": result["usage"]["model_api_calls"] <= 4}
    if case["outcome"] == "diagnosis":
        checks.update(evidence_check=result.get("answer_check", {}).get("passed") is True,
                      cause=result.get("structured_answer", {}).get("diagnosis", {}).get("cause") == case["cause"])
    else:
        checks.update(stop_check=result.get("stop_check", {}).get("passed") is True,
                      stop_reason=result.get("structured_stop", {}).get("reason") == case["reason"])
        if case["reason"] == "execution_unknown":
            uncertainty = result.get("structured_stop", {}).get("uncertainty_evidence")
            checks["documented_uncertainty"] = bool(uncertainty and uncertainty["diagnosis"]["rule_citation"])
            checks["no_retry_promise"] = "不自动重试" in result["answer"]
    return {"passed": all(checks.values()), "checks": checks}


def run_diagnostic_evaluation(*, case_id=None, live=False):
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = [row for row in dataset["cases"] if case_id is None or row["id"] == case_id]
    if not cases:
        raise ValueError("Unknown diagnostic case")
    if live and (len(cases) != 1 or case_id not in ("revoked", "expired", "permission", "incident", "timeout")):
        raise ValueError("真实验证须指定一个正常场景；只运行动态 Agent，上限 4 次模型请求。")
    config = load_deepseek_config() if live else None
    rows = []
    for case in cases:
        with TemporaryDirectory(prefix="supportlab_diagnostics_") as directory:
            root = Path(directory)
            db, docs = prepare_diagnostic_case(case, root)
            paths = [db, *docs.glob("*.md")]
            digest = fingerprint(paths, root=root)
            for mode in (("agent",) if live else ("workflow", "agent")):
                prompt = WORKFLOW_PROMPT if mode == "workflow" else ADAPTER_PROMPT
                adapter = DeepSeekModel(config, system_prompt=prompt) if live else RuleReplayAdapter(case["request_id"], system_prompt=prompt)
                model = ModelWorkflow(case["request_id"], adapter) if mode == "workflow" else adapter
                result = run_loop(model, ToolExecutor(AppContext("acc_demo"), db_path=db, docs_root=docs),
                                  user_message=f"请排查 {case['request_id']}，依据证据给出结论。", target_request_id=case["request_id"])
                grade = grade_diagnostic(case, result)
                grade["checks"]["sources_unchanged"] = digest == fingerprint(paths, root=root)
                grade["passed"] = all(grade["checks"].values())
                rows.append({"id": case["id"], "family": case["family"], "configuration": mode,
                             "source_sha256": digest, "grade": grade, "result": result})
    usages = [row["result"]["usage"] for row in rows]
    tokens = [usage["token_usage"] for usage in usages if usage["token_usage"] is not None]
    return {"created_at": datetime.now(timezone.utc).isoformat(), "simulation": True, "split": "development",
            "dataset_version": dataset["dataset_version"], "case_count": len(cases), "run_count": len(rows),
            "passed_runs": sum(row["grade"]["passed"] for row in rows), "human_review": "pending",
            "execution_mode": "live" if live else "offline_harness", "model": config.model if live else "rule-replay-harness-v1",
            "model_api_calls": sum(usage["model_api_calls"] for usage in usages), "model_api_cap": 4 if live else 0,
            "token_usage": {key: sum(token[key] for token in tokens) for key in ("input_tokens", "output_tokens", "total_tokens")} if tokens else None,
            "usage_complete": bool(live) and all(usage["usage_complete"] for usage in usages), "cost": None,
            "reproducibility": {"code_sha256": fingerprint(list((PROJECT_ROOT / "supportlab").glob("*.py"))),
                                "dataset_sha256": fingerprint([DATASET, DEFAULT_SCENARIOS]), "rule_sha256": RULE_SHA256,
                                "contract": DIAGNOSTIC_CONTRACT, "adapter_version": ADAPTER_VERSION,
                                "prompts": {key: sha256((SYSTEM_MESSAGE + "\n" + prompt).encode()).hexdigest()
                                            for key, prompt in (("workflow", WORKFLOW_PROMPT), ("agent", ADAPTER_PROMPT))}},
            "reproduce": ".\\.venv\\Scripts\\python.exe -m supportlab eval-diagnostics" + (f" --case {case_id}" if case_id else "") + (" --live" if live else ""),
            "limitations": ["构造开发案例已参与实现与调优，不是独立保留测试。", "离线回复由规则构造，只验证执行、约束与报告；不证明真实模型质量。",
                            "实际模型验证每次仅一个案例一次运行；缺少广泛评估、重复稳定性与独立人工复核。"], "rows": rows}
