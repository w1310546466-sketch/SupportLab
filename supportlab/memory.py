"""Local, scoped conversation history and explicitly chosen presentation preferences."""

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4

from .storage import PROJECT_ROOT
from .tools import AppContext


MEMORY_VERSION = "memory-v1.1"
DEFAULT_MEMORY_DB = PROJECT_ROOT / "data" / "memory.sqlite3"
PREFERENCES = {"detail": {"brief", "detailed"}, "layout": {"paragraphs", "bullets"}}
DEFAULT_PREFERENCES = {"detail": "detailed", "layout": "paragraphs"}


def now():
    return datetime.now(timezone.utc).isoformat()


def redact(text):
    # Best-effort filtering of credential-shaped input before storage or prompting.
    text = re.sub(r"(?i)\bsk-[a-z0-9_-]{8,}\b", "[已隐藏密钥]", text)
    text = re.sub(r"(?i)\bBearer\s+\S+", "Bearer [已隐藏]", text)
    return re.sub(r"(?i)((?:api[_ -]?key|password|密钥|密码)\s*[:=：]\s*)[^\s,，;；]+", r"\1[已隐藏]", text)


def redact_value(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_value(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class Actor:
    account_id: str = "acc_demo"
    user_id: str = "learner"

    def __post_init__(self):
        AppContext(self.account_id)
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", self.user_id):
            raise ValueError("Invalid trusted user context")

    @property
    def key(self):
        return sha256(f"{self.account_id}\0{self.user_id}".encode()).hexdigest()


class MemoryStore:
    def __init__(self, path=DEFAULT_MEMORY_DB):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    actor TEXT NOT NULL, id TEXT NOT NULL, thread_key TEXT NOT NULL UNIQUE,
                    mode TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ready', target TEXT, summary TEXT NOT NULL DEFAULT '{}',
                    lease_until REAL NOT NULL DEFAULT 0, lease_owner TEXT,
                    PRIMARY KEY(actor,id));
                CREATE TABLE IF NOT EXISTS turns (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, thread_key TEXT NOT NULL,
                    turn_id TEXT NOT NULL UNIQUE, user_text TEXT NOT NULL, answer TEXT NOT NULL,
                    result_json TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS turns_thread ON turns(thread_key, seq);
                CREATE TABLE IF NOT EXISTS preferences (
                    actor TEXT NOT NULL, name TEXT NOT NULL, value TEXT NOT NULL,
                    updated_at TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'explicit_user',
                    PRIMARY KEY(actor,name));
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=2)
        db.row_factory = sqlite3.Row
        return db

    def open_session(self, actor, session_id=None, *, mode=None):
        session_id = session_id or uuid4().hex[:12]
        if not isinstance(session_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", session_id):
            raise ValueError("会话编号只能包含字母、数字、下划线和短横线。")
        if mode not in (None, "fixed", "agent", "workflow"):
            raise ValueError("Unknown session mode")
        with closing(self.connect()) as db, db:
            row = db.execute("SELECT * FROM sessions WHERE actor=? AND id=?", (actor.key, session_id)).fetchone()
            if row:
                if mode is not None and row["mode"] != mode:
                    raise ValueError("会话运行方式已固定；切换方式请新建会话。")
                return dict(row)
            stamp = now()
            thread = sha256(f"{actor.key}\0{session_id}".encode()).hexdigest()
            db.execute("INSERT INTO sessions(actor,id,thread_key,mode,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                       (actor.key, session_id, thread, mode or "fixed", stamp, stamp))
        return self.get_session(actor, session_id)

    def get_session(self, actor, session_id):
        with closing(self.connect()) as db:
            row = db.execute("SELECT * FROM sessions WHERE actor=? AND id=?", (actor.key, session_id)).fetchone()
        if row is None:
            raise ValueError("当前用户范围内没有此会话。")
        return dict(row)

    def sessions(self, actor):
        with closing(self.connect()) as db:
            return [dict(row) for row in db.execute(
                "SELECT id,mode,status,target,updated_at FROM sessions WHERE actor=? ORDER BY updated_at DESC LIMIT 50", (actor.key,))]

    def acquire(self, actor, session_id):
        owner = uuid4().hex
        with closing(self.connect()) as db, db:
            changed = db.execute("UPDATE sessions SET lease_owner=?,lease_until=? WHERE actor=? AND id=? AND lease_until<?",
                                 (owner, time.time() + 180, actor.key, session_id, time.time())).rowcount
            if changed != 1:
                raise ValueError("此会话正在处理另一条消息；请等当前请求结束。")
        return owner

    def release(self, actor, session_id, owner):
        with closing(self.connect()) as db, db:
            db.execute("UPDATE sessions SET lease_until=0,lease_owner=NULL WHERE actor=? AND id=? AND lease_owner=?",
                       (actor.key, session_id, owner))

    def save_turn(self, actor, session_id, *, turn_id, user_text, result, target):
        session = self.get_session(actor, session_id)
        summary = {"last_status": result["status"], "last_request_id": target,
                   "last_outcome": result.get("outcome"), "historical_only": True}
        if result.get("outcome") == "docs" and result.get("document_topic"):
            summary["last_document_topic"] = result["document_topic"][:128]
        checked = result.get("answer_check", {}).get("passed") is True
        if checked:
            summary["last_verified_cause"] = result["structured_answer"]["diagnosis"]["cause"]
        elif target is not None and target == session["target"] and result.get("outcome") == "docs":
            previous = json.loads(session["summary"])
            if "last_verified_cause" in previous:
                summary["last_verified_cause"] = previous["last_verified_cause"]
        with closing(self.connect()) as db, db:
            inserted = db.execute("INSERT OR IGNORE INTO turns(thread_key,turn_id,user_text,answer,result_json,created_at) VALUES(?,?,?,?,?,?)",
                       (session["thread_key"], turn_id, redact(user_text), redact(result["answer"]),
                        json.dumps(redact_value(result), ensure_ascii=False), now())).rowcount
            if not inserted:
                return
            db.execute("UPDATE sessions SET status=?,target=?,summary=?,updated_at=? WHERE actor=? AND id=?",
                       ("needs_input" if result.get("pending_request_id") else result["status"], target,
                        json.dumps(summary, ensure_ascii=False), now(), actor.key, session_id))

    def history(self, actor, session_id, *, limit=20):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("History limit must be 1–100")
        session = self.get_session(actor, session_id)
        with closing(self.connect()) as db:
            rows = db.execute("SELECT * FROM turns WHERE thread_key=? ORDER BY seq DESC LIMIT ?", (session["thread_key"], limit)).fetchall()
            total = db.execute("SELECT COUNT(*) FROM turns WHERE thread_key=?", (session["thread_key"],)).fetchone()[0]
        return {"total_turns": total, "turns": [dict(row) for row in reversed(rows)]}

    def preferences(self, actor):
        with closing(self.connect()) as db:
            rows = db.execute("SELECT name,value FROM preferences WHERE actor=?", (actor.key,)).fetchall()
        return {**DEFAULT_PREFERENCES, **{row["name"]: row["value"] for row in rows}}

    def usage(self, actor, session_id):
        session = self.get_session(actor, session_id)
        summary = {"saved_turns": 0, "model_api_calls": 0, "tool_calls": 0,
                   "known_tokens": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                   "turns_with_incomplete_usage": 0, "scope": "saved_turns_only_not_billing"}
        with closing(self.connect()) as db:
            for row in db.execute("SELECT result_json FROM turns WHERE thread_key=?", (session["thread_key"],)):
                usage = json.loads(row["result_json"]).get("usage", {})
                summary["saved_turns"] += 1
                summary["model_api_calls"] += usage.get("model_api_calls", 0)
                summary["tool_calls"] += usage.get("tool_calls", 0)
                if usage.get("model_api_calls", 0) and not usage.get("usage_complete", False):
                    summary["turns_with_incomplete_usage"] += 1
                tokens = usage.get("token_usage")
                if tokens:
                    for key in summary["known_tokens"]:
                        summary["known_tokens"][key] += tokens[key]
        return summary

    def set_preference(self, actor, name, value):
        if not isinstance(name, str) or not isinstance(value, str) or name not in PREFERENCES or value not in PREFERENCES[name]:
            raise ValueError("只支持 detail=brief/detailed 和 layout=paragraphs/bullets。")
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO preferences(actor,name,value,updated_at) VALUES(?,?,?,?) "
                       "ON CONFLICT(actor,name) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at",
                       (actor.key, name, value, now()))
        return self.preferences(actor)

    def forget_preferences(self, actor):
        with closing(self.connect()) as db, db:
            db.execute("DELETE FROM preferences WHERE actor=?", (actor.key,))


def build_context(store, actor, session_id, *, max_bytes=3600):
    """History is reference material, not authorization or current tool evidence."""
    session = store.get_session(actor, session_id)
    history = store.history(actor, session_id, limit=4)
    payload = {"trust": "reference_only_not_tool_evidence", "preferences": store.preferences(actor),
               "task": json.loads(session["summary"]), "recent_turns": [],
               "omitted_turns": history["total_turns"], "version": "context-v1"}
    if type(max_bytes) is not int or max_bytes < 800:
        raise ValueError("Context budget is too small")
    while len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > max_bytes and payload["task"].get("last_document_topic"):
        payload["task"]["last_document_topic"] = payload["task"]["last_document_topic"][:-1]
    for row in reversed(history["turns"]):
        item = {"user": redact(row["user_text"])[:300], "assistant": redact(row["answer"])[:450]}
        payload["recent_turns"].insert(0, item)
        payload["omitted_turns"] -= 1
        if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) > max_bytes:
            payload["recent_turns"].pop(0)
            payload["omitted_turns"] += 1
            break
    packed = json.dumps(payload, ensure_ascii=False)
    return {"payload": payload, "serialized": packed, "bytes": len(packed.encode("utf-8")), "max_bytes": max_bytes}
