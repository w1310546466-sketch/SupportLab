"""Isolated feature demonstration; live variant has at most six model requests."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from .agent import SYSTEM_MESSAGE
from .conversation import CONVERSATION_VERSION, ConversationService
from .deepseek import ADAPTER_PROMPT, ADAPTER_VERSION
from .evaluation import fingerprint
from .format_recovery import REPAIR_MESSAGE
from .knowledge import KNOWLEDGE_DOCUMENTS, INDEX_VERSION, build_index
from .memory import MEMORY_VERSION, MemoryStore
from .rag import RAG_PROMPT, RAG_REPAIR_MESSAGE, RAG_VERSION
from .retrieval import DEFAULT_DOCS_ROOT
from .storage import DEFAULT_FIXTURE, DEFAULT_ACCOUNT_FIXTURE, PROJECT_ROOT, init_demo_db


def run_application_demo(*, live=False):
    started = perf_counter()
    turns, checks = [], {}
    with TemporaryDirectory(prefix="supportlab_app_") as directory:
        root = Path(directory)
        db_path = root / "business.sqlite3"
        init_demo_db(db_path)
        store_path, index_path = root / "memory.sqlite3", root / "index.sqlite3"
        indexed = build_index(index_path=index_path)

        def service():
            return ConversationService(store=MemoryStore(store_path), db_path=db_path, index_path=index_path)

        app = service()
        app.new_session("demo", mode="agent" if live else "fixed")

        def send(message, session_id="demo"):
            result = app.chat(session_id, message)
            turns.append({"message": message, "session_id": session_id, "result": result})
            return result

        send("以后回答简洁一些")
        paused = send("帮我排查")
        checks["pause_without_model"] = paused["status"] == "needs_input" and not paused["model_called"]
        app = service()  # Reopen both independent memory and graph checkpoint connections.
        diagnosed = send("req_demo_018")
        checks["resume_with_evidence"] = diagnosed.get("answer_check", {}).get("passed") is True
        checks["brief_preference_applied"] = diagnosed.get("presentation", {}).get("detail") == "brief" and "full_answer" in diagnosed
        docs = send("/docs 429 是什么意思")
        checks["document_answer_with_sources"] = docs.get("rag_check", {}).get("passed") is True
        history = send("/history")
        checks["history_survives_reopen"] = "req_demo_018" in history["answer"] and app.store.history(app.actor, "demo")["total_turns"] == 3
        context = send("/context")["context_report"]
        checks["bounded_context"] = context["bytes"] <= context["max_bytes"]
        checks["historical_reference_label"] = context["payload"]["trust"] == "reference_only_not_tool_evidence"
        app.new_session("second", mode="agent" if live else "fixed")
        prefs = send("/prefs", "second")
        checks["preferences_cross_session"] = '"detail": "brief"' in prefs["answer"]
        checks["new_session_history_is_empty"] = app.store.history(app.actor, "second")["total_turns"] == 0
        checks["four_versioned_documents"] = indexed["document_count"] == 4
        if not live:
            followup = send("再查一下刚才那个请求")
            checks["followup_queries_fresh_evidence"] = followup["usage"]["tool_calls"] == 3 and followup.get("answer_check", {}).get("passed") is True
            unavailable = send("排查 req_other_001")
            checks["other_account_hidden"] = unavailable.get("stop_check", {}).get("proposed_reason") == "request_unavailable"
        send("/cancel")
        checks["cancel_clears_task_keeps_history"] = app.store.get_session(app.actor, "demo")["target"] is None and app.store.history(app.actor, "demo")["total_turns"] > 0
    calls = sum(turn["result"]["usage"]["model_api_calls"] for turn in turns)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(), "simulation": True,
        "execution_mode": "live" if live else "fixed_and_extractive_offline", "dataset_version": "app-demo-v1",
        "split": "development", "checks": checks, "passed_checks": sum(checks.values()), "check_count": len(checks),
        "model_api_calls": calls, "model_api_call_cap": 6 if live else 0,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3), "human_review": "pending", "cost": None,
        "reproducibility": {
            "conversation_version": CONVERSATION_VERSION, "memory_version": MEMORY_VERSION,
            "rag_version": RAG_VERSION, "index_version": INDEX_VERSION, "adapter_version": ADAPTER_VERSION,
            "dependencies": {name: version(name) for name in ("langgraph", "langgraph-checkpoint-sqlite", "openai")},
            "code_sha256": fingerprint(list((PROJECT_ROOT / "supportlab").glob("*.py"))),
            "source_sha256": fingerprint([DEFAULT_FIXTURE, DEFAULT_ACCOUNT_FIXTURE,
                                          *(DEFAULT_DOCS_ROOT / doc.filename for doc in KNOWLEDGE_DOCUMENTS)]),
            "diagnosis_prompt_sha256": sha256((SYSTEM_MESSAGE + "\n" + ADAPTER_PROMPT).encode()).hexdigest(),
            "rag_prompt_sha256": sha256(RAG_PROMPT.encode()).hexdigest(),
            "repair_prompt_sha256": sha256((REPAIR_MESSAGE + "\n" + RAG_REPAIR_MESSAGE).encode()).hexdigest(),
        },
        "reproduce": ".\\.venv\\Scripts\\python.exe -m supportlab app-demo" + (" --live" if live else ""),
        "limitations": ["功能开发演示；不是保留测试集准确率，也不是固定流程与 Agent 的公平比较。",
                        "文档回答输出已检索原文，选择是否完整相关仍需人工审查。",
                        "本演示选取两种 429；另有认证/权限/上游/超时扩展演示；全部业务数据模拟，没有实际联系人工或创建工单。"],
        "turns": turns,
    }
