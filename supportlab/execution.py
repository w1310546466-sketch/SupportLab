"""Whitelist execution with app-owned scope and a killable tool process."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from .contracts import validate_arguments
from .retrieval import DEFAULT_DOCS_ROOT, search_docs
from .storage import DEFAULT_DB, PROJECT_ROOT
from .tools import AppContext, get_account_state, get_request_record


def error_result(status: str, message: str) -> dict:
    return {"simulation": True, "status": status, "message": message}


def invoke_tool(name: str, arguments: dict, *, context: AppContext, db_path: Path, docs_root: Path) -> dict:
    """Called inside the worker; the argument validator also rejects internal fields."""
    validated = validate_arguments(name, arguments)
    handlers = {
        "get_request_record": lambda: get_request_record(**validated, context=context, db_path=db_path),
        "get_account_state": lambda: get_account_state(**validated, context=context, db_path=db_path),
        "search_docs": lambda: search_docs(**validated, docs_root=docs_root),
    }
    return handlers[name]()


@dataclass
class Execution:
    result: dict
    validated_arguments: dict | None
    elapsed_ms: float


class ToolExecutor:
    def __init__(self, context: AppContext, *, db_path: Path = DEFAULT_DB,
                 docs_root: Path = DEFAULT_DOCS_ROOT, timeout_seconds: float = 2.0):
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or not 0 < timeout_seconds <= 30):
            raise ValueError("tool timeout must be finite and within (0, 30] seconds")
        self.context = context
        self.db_path = db_path.resolve()
        self.docs_root = docs_root.resolve()
        self.timeout_seconds = timeout_seconds

    def execute(self, name: str, arguments: object) -> Execution:
        started = perf_counter()
        validated = None
        try:
            validated = validate_arguments(name, arguments)
        except ValueError as error:
            result = error_result(str(error), "工具名或参数不符合应用白名单与契约。")
        else:
            # Scope and paths come only from this application instance, not the model.
            payload = {"name": name, "arguments": validated,
                       "account_id": self.context.account_id,
                       "db_path": str(self.db_path), "docs_root": str(self.docs_root)}
            try:
                worker = subprocess.run(
                    [sys.executable, "-m", "supportlab.tool_worker"],
                    input=json.dumps(payload), capture_output=True, text=True, encoding="utf-8",
                    cwd=PROJECT_ROOT, timeout=self.timeout_seconds,
                )
                if worker.returncode != 0:
                    raise ValueError("worker failed")
                result = json.loads(worker.stdout)
                if (not isinstance(result, dict) or result.get("simulation") is not True
                        or result.get("status") not in {"ok", "not_found", "invalid_arguments", "storage_error", "tool_error"}):
                    raise ValueError("invalid tool result")
            except subprocess.TimeoutExpired:
                # subprocess.run kills and waits for its direct worker on timeout.
                result = error_result("timeout", "工具超过时间上限，已终止本次工具进程。")
            except (OSError, ValueError):
                result = error_result("tool_error", "工具执行失败，未返回可用证据。")
        return Execution(result, validated, round((perf_counter() - started) * 1000, 3))
