"""Temporary integration-scenario operations. Never model-visible or production data."""

from contextlib import closing
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

from .conversation import ConversationService
from .knowledge import KNOWLEDGE_DOCUMENTS, build_index
from .memory import Actor, MemoryStore, build_context
from .retrieval import DEFAULT_DOCS_ROOT
from .scenarios import init_scenarios
from .storage import init_demo_db
from .tools import _valid_request_id

TEMPLATES = {"rate": "req_demo_017", "quota": "req_demo_018", "revoked": "req_auth_revoked",
             "expired": "req_auth_expired", "permission": "req_permission_missing",
             "incident": "req_upstream_incident", "timeout": "req_timeout_unknown"}
ACTORS = {"owner": Actor(), "peer": Actor("acc_other", "learner"),
          "colleague": Actor("acc_demo", "another_learner")}


def no_model(_):
    raise RuntimeError("Acceptance is offline; model construction is prohibited")


class Sandbox:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.db = self.root / "business.sqlite3"
        self.docs, self.index = self.root / "docs", self.root / "index.sqlite3"
        self.docs.mkdir()
        self.original_docs = {doc.filename: (DEFAULT_DOCS_ROOT / doc.filename).read_bytes() for doc in KNOWLEDGE_DOCUMENTS}
        for name, content in self.original_docs.items():
            (self.docs / name).write_bytes(content)
        init_demo_db(self.db)
        init_scenarios(self.db)
        build_index(index_path=self.index, docs_root=self.docs)
        self.store = MemoryStore(self.root / "memory.sqlite3")
        self.templates = {}
        with closing(sqlite3.connect(self.db)) as db:
            db.row_factory = sqlite3.Row
            for name, rid in TEMPLATES.items():
                request = dict(db.execute("SELECT * FROM request_records WHERE account_id='acc_demo' AND request_id=?", (rid,)).fetchone())
                table = "account_states" if name in ("rate", "quota") else "diagnostic_snapshots"
                snapshot = dict(db.execute(f"SELECT * FROM {table} WHERE account_id='acc_demo' AND request_id=?", (rid,)).fetchone())
                self.templates[name] = (request, table, snapshot)

    def service(self, step):
        actor = ACTORS[step.get("actor", "owner")]
        # A fresh service instance on every step exercises persisted state.
        service = ConversationService(store=self.store, actor=actor, db_path=self.db,
                                      docs_root=self.docs, index_path=self.index, model_factory=no_model)
        session = step.get("session", "main")
        service.new_session(session, mode="fixed")
        return service, actor, session

    def seed(self, step):
        request, table, snapshot = deepcopy(self.templates[step["template"]])
        rid = step["request_id"]
        if not _valid_request_id(rid) or not rid.startswith("req_hold_"):
            raise ValueError("Acceptance fixtures must use their own request namespace")
        account = ACTORS[step.get("actor", "owner")].account_id
        request.update(account_id=account, request_id=rid)
        request.update(step.get("request_changes", {}))
        snapshot.update(account_id=account, request_id=rid, captured_at=request["occurred_at"],
                        service_version=request["service_version"])
        snapshot.update(step.get("snapshot_changes", {}))
        if "facts" in step:
            facts = json.loads(snapshot["facts_json"])
            facts.update(step["facts"])
            snapshot["facts_json"] = json.dumps(facts, ensure_ascii=False)
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("PRAGMA foreign_keys=ON")
            for kind in ("account_states", "diagnostic_snapshots"):
                db.execute(f"DELETE FROM {kind} WHERE account_id=? AND request_id=?", (account, rid))
            db.execute("DELETE FROM request_records WHERE account_id=? AND request_id=?", (account, rid))
            columns = tuple(request)
            db.execute(f"INSERT INTO request_records ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", tuple(request.values()))
            if not step.get("omit_snapshot", False):
                columns = tuple(snapshot)
                db.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", tuple(snapshot.values()))
        return {"template": step["template"], "request_id": rid, "actor": step.get("actor", "owner"),
                "snapshot_present": not step.get("omit_snapshot", False)}

    def sources(self):
        from .evaluation import fingerprint
        return fingerprint([self.db, *self.docs.glob("*.md")], root=self.root)

    def run(self, step):
        op = step["op"]
        if op == "seed":
            return self.seed(step)
        if op == "doc_append":
            if step["file"] not in self.original_docs:
                raise ValueError("Unregistered fixture document")
            path = self.docs / step["file"]
            path.write_bytes(path.read_bytes() + ("\n" + step["text"] + "\n").encode())
            return {"document": step["file"], "changed": True}
        if op == "doc_remove":
            if step["file"] not in self.original_docs:
                raise ValueError("Unregistered fixture document")
            (self.docs / step["file"]).unlink()
            return {"document": step["file"], "removed": True}
        if op == "doc_restore":
            (self.docs / step["file"]).write_bytes(self.original_docs[step["file"]])
            return {"document": step["file"], "restored": True}
        if op == "index":
            return build_index(index_path=self.index, docs_root=self.docs)
        service, actor, session = self.service(step)
        if op == "open":
            return {"mode": service.new_session(session)["mode"]}
        if op == "chat":
            before = self.sources()
            result = service.chat(session, step["text"])
            return {"result": result, "business_sources_unchanged": self.sources() == before}
        if op == "state":
            current = self.store.get_session(actor, session)
            packet = build_context(self.store, actor, session)
            history = self.store.history(actor, session)
            return {"target": current["target"], "status": current["status"], "mode": current["mode"],
                    "summary": json.loads(current["summary"]), "preferences": self.store.preferences(actor),
                    "history_count": history["total_turns"], "context": packet,
                    "usage": self.store.usage(actor, session)}
        raise ValueError("Unknown acceptance operation")
