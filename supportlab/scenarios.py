"""Explicit, additive import of simulated diagnostic snapshots (never an Agent tool)."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3

from .storage import DEFAULT_DB, PROJECT_ROOT, _load_fixture, open_readonly_db

SCENARIO_VERSION = "diagnostic-scenarios-v1"
SNAPSHOT_VERSION = "diagnostic-snapshots-v1"
DEFAULT_SCENARIOS = PROJECT_ROOT / "data" / "diagnostic_scenarios.json"
SNAPSHOT_FIELDS = {
    "CREDENTIAL_REVOKED": {"credential_status"},
    "TOKEN_EXPIRED": {"credential_status", "credential_expires_at"},
    "SCOPE_MISSING": {"credential_status", "required_scope", "granted_scopes"},
    "UPSTREAM_INCIDENT_CONFIRMED": {"upstream_service", "incident_service", "incident_id",
                                     "incident_started_at", "incident_ended_at", "execution_state"},
    "UPSTREAM_TIMEOUT_UNKNOWN": {"execution_state", "idempotency_supported"},
}


def init_scenarios(db_path: Path = DEFAULT_DB, fixture_path: Path = DEFAULT_SCENARIOS):
    from .tools import AppContext, _valid_request_id
    from .answer_check import _time
    records = _load_fixture(fixture_path, SCENARIO_VERSION)
    keys = set()
    for row in records:
        if set(row) != {"account_id", "request_id", "service_version", "occurred_at", "http_status",
                        "error_code", "retry_after_seconds", "snapshot"}:
            raise ValueError("Invalid scenario fields")
        AppContext(row["account_id"])
        key = (row["account_id"], row["request_id"])
        if key in keys or not _valid_request_id(row["request_id"]) or row["service_version"] != "demo-api-v1":
            raise ValueError("Invalid or duplicate scenario identity")
        keys.add(key)
        _time(row["occurred_at"])
        if (type(row["http_status"]) is not int or not 100 <= row["http_status"] <= 599
                or row["error_code"] not in SNAPSHOT_FIELDS or row["retry_after_seconds"] is not None
                or type(row["snapshot"]) is not dict or set(row["snapshot"]) != SNAPSHOT_FIELDS[row["error_code"]]):
            raise ValueError("Invalid scenario facts")
        if len(json.dumps(row["snapshot"], ensure_ascii=False)) > 4000:
            raise ValueError("Snapshot too large")
    # Require the base database first; opening a missing path must not create it.
    with closing(open_readonly_db(db_path)) as connection:
        connection.execute("SELECT 1 FROM request_records LIMIT 1")
    with closing(sqlite3.connect(db_path, timeout=1.0)) as connection, connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN")
        connection.execute("""CREATE TABLE IF NOT EXISTS diagnostic_snapshots (
            account_id TEXT NOT NULL, request_id TEXT NOT NULL,
            service_version TEXT NOT NULL, captured_at TEXT NOT NULL, facts_json TEXT NOT NULL,
            PRIMARY KEY(account_id, request_id),
            FOREIGN KEY(account_id, request_id) REFERENCES request_records(account_id, request_id))""")
        for row in records:
            # Refuse to replace any unrelated request even if its ID collides.
            existing = connection.execute("SELECT 1 FROM request_records WHERE account_id=? AND request_id=?",
                                          (row["account_id"], row["request_id"])).fetchone()
            owned = connection.execute("SELECT 1 FROM diagnostic_snapshots WHERE account_id=? AND request_id=?",
                                       (row["account_id"], row["request_id"])).fetchone()
            if existing and not owned:
                raise ValueError("Scenario would overwrite an unrelated request")
            connection.execute("""INSERT INTO request_records VALUES
                (:account_id,:request_id,:service_version,:occurred_at,:http_status,:error_code,:retry_after_seconds)
                ON CONFLICT(account_id,request_id) DO UPDATE SET service_version=excluded.service_version,
                occurred_at=excluded.occurred_at,http_status=excluded.http_status,error_code=excluded.error_code,
                retry_after_seconds=excluded.retry_after_seconds""", row)
            connection.execute("""INSERT INTO diagnostic_snapshots VALUES(?,?,?,?,?)
                ON CONFLICT(account_id,request_id) DO UPDATE SET service_version=excluded.service_version,
                captured_at=excluded.captured_at,facts_json=excluded.facts_json""",
                (row["account_id"], row["request_id"], row["service_version"], row["occurred_at"],
                 json.dumps(row["snapshot"], ensure_ascii=False, allow_nan=False)))
    return {"extended_requests": len(records), "diagnostic_snapshots": len(records)}


def read_snapshot(connection, account_id, request_id):
    """None means not an extended scenario; missing/misaligned is explicit, no fallback."""
    if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='diagnostic_snapshots'").fetchone():
        return None
    row = connection.execute("""SELECT s.request_id,s.service_version,s.captured_at,s.facts_json,
        r.occurred_at AS request_time,r.service_version AS request_version
        FROM diagnostic_snapshots s JOIN request_records r
        ON s.account_id=r.account_id AND s.request_id=r.request_id
        WHERE s.account_id=? AND s.request_id=?""", (account_id, request_id)).fetchone()
    if row is None:
        return None
    base = {"simulation": True, "dataset_version": SNAPSHOT_VERSION}
    if row["captured_at"] != row["request_time"] or row["service_version"] != row["request_version"]:
        return {**base, "status": "not_found", "message": "没有与请求时间和版本一致的诊断快照。"}
    try:
        from .format_recovery import parse_final_object
        facts = parse_final_object(row["facts_json"])
        request = connection.execute("SELECT error_code FROM request_records WHERE account_id=? AND request_id=?",
                                     (account_id, request_id)).fetchone()
        if set(facts) != SNAPSHOT_FIELDS.get(request["error_code"]):
            raise ValueError("Unrecognized snapshot fields")
    except (ValueError, TypeError):
        return {**base, "status": "storage_error", "message": "诊断快照格式不可用。"}
    return {**base, "status": "ok", "account_state": {
        "request_id": row["request_id"], "service_version": row["service_version"],
        "captured_at": row["captured_at"], "diagnostic_facts": facts}}
