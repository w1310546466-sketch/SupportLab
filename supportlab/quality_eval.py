"""Reproducible offline RAG and multi-turn development checks, not held-out scores."""

from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from .conversation import ConversationService, CONVERSATION_VERSION
from .evaluation import fingerprint
from .knowledge import build_index, KNOWLEDGE_DOCUMENTS, INDEX_VERSION
from .memory import MemoryStore
from .rag import document_answer, RAG_VERSION
from .retrieval import DEFAULT_DOCS_ROOT
from .storage import PROJECT_ROOT, init_demo_db

CASE_FILE = PROJECT_ROOT / "evals" / "conversation_dev.json"


def run_quality_evaluation():
    dataset = json.loads(CASE_FILE.read_text(encoding="utf-8"))
    if dataset.get("split") != "development" or dataset.get("simulation") is not True:
        raise ValueError("Expected simulated development cases")
    rows = []
    started = perf_counter()
    with TemporaryDirectory(prefix="supportlab_quality_") as directory:
        root = Path(directory)
        index, db = root / "index.sqlite3", root / "business.sqlite3"
        init_demo_db(db)
        build_index(index_path=index)
        for case in dataset["rag_cases"]:
            result = document_answer(case["query"], index_path=index)
            excerpts = "\n".join(hit["excerpt"] for hit in result["retrieval"]["matches"])
            if case["required"]:
                passed = all(phrase in excerpts for phrase in case["required"])
                checks = {"required_evidence_in_top3": passed}
            else:
                passed = not result["retrieval"]["matches"] and result["outcome"] == "clarify"
                checks = {"irrelevant_evidence_rejected": passed}
            checks["zero_model_calls"] = result["usage"]["model_api_calls"] == 0
            rows.append({"id": case["id"], "kind": "rag", "family": case["family"], "checks": checks,
                         "passed": all(checks.values()), "result": result})
        for case in dataset["conversation_cases"]:
            store = MemoryStore(root / (case["id"] + ".sqlite3"))
            service = ConversationService(store=store, db_path=db, index_path=index)
            service.new_session("eval")
            turns = [service.chat("eval", message) for message in case["messages"]]
            result, expected = turns[-1], case["expected"]
            checks = {"outcome": result["outcome"] == expected["outcome"]}
            if "cause" in expected:
                checks["cause"] = (result.get("answer_check", {}).get("passed") is True and
                                    result.get("structured_answer", {}).get("diagnosis", {}).get("cause") == expected["cause"])
            if "tool_calls" in expected:
                checks["tool_calls"] = result["usage"].get("tool_calls", 0) == expected["tool_calls"]
            if "answer_contains" in expected:
                checks["answer_contains"] = expected["answer_contains"] in result["answer"]
            if "intermediate_outcome" in expected:
                checks["intermediate_outcome"] = turns[1]["outcome"] == expected["intermediate_outcome"]
            checks["zero_model_calls"] = all(turn["usage"]["model_api_calls"] == 0 for turn in turns)
            rows.append({"id": case["id"], "kind": "conversation", "family": case["family"], "checks": checks,
                         "passed": all(checks.values()), "messages": case["messages"], "turns": turns})
    positive = [row for row in rows if "required_evidence_in_top3" in row["checks"]]
    negative = [row for row in rows if "irrelevant_evidence_rejected" in row["checks"]]
    return {"dataset_version": dataset["dataset_version"], "split": "development", "simulation": True,
            "execution_mode": "offline", "model_api_calls": 0, "human_review": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(), "case_count": len(rows),
            "passed_count": sum(row["passed"] for row in rows),
            "metrics": {"required_evidence_hits": {"numerator": sum(row["passed"] for row in positive), "denominator": len(positive)},
                        "correct_abstentions": {"numerator": sum(row["passed"] for row in negative), "denominator": len(negative)}},
            "elapsed_ms": round((perf_counter() - started) * 1000, 3),
            "versions": {"conversation": CONVERSATION_VERSION, "rag": RAG_VERSION, "index": INDEX_VERSION},
            "code_sha256": fingerprint(list((PROJECT_ROOT / "supportlab").glob("*.py"))),
            "dataset_sha256": fingerprint([CASE_FILE]),
            "source_sha256": fingerprint([DEFAULT_DOCS_ROOT / doc.filename for doc in KNOWLEDGE_DOCUMENTS]),
            "reproduce": ".\\.venv\\Scripts\\python.exe -m supportlab eval-app",
            "limitations": ["所有题目参与开发优化，不是锁定测试。", "关键词与行为检查不证明任意语义正确性；没有模型评审替代人工。"],
            "cases": rows}
