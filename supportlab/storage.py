"""Load editable JSON fixtures into SQLite; query through read-only connections."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "supportlab.sqlite3"
DEFAULT_FIXTURE = PROJECT_ROOT / "data" / "requests.json"
DATASET_VERSION = "requests-v1"
DEFAULT_ACCOUNT_FIXTURE = PROJECT_ROOT / "data" / "account_states.json"
ACCOUNT_DATASET_VERSION = "account-states-v1"


def _load_fixture(path: Path, version: str) -> list[dict]:
    fixture = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixture, dict) or fixture.get("simulation") is not True or fixture.get("dataset_version") != version:
        raise ValueError(f"Expected simulated {version} fixture data.")
    records = fixture.get("records")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("Fixture records must be a list of objects.")
    return records


def init_demo_db(
    db_path: Path = DEFAULT_DB,
    fixture_path: Path = DEFAULT_FIXTURE,
    account_fixture_path: Path = DEFAULT_ACCOUNT_FIXTURE,
) -> dict[str, int]:
    """Import both fixtures together. This local setup command is not an Agent tool."""
    records = _load_fixture(fixture_path, DATASET_VERSION)
    snapshots = _load_fixture(account_fixture_path, ACCOUNT_DATASET_VERSION)
    requests_by_key = {(row["account_id"], row["request_id"]): row for row in records}
    for snapshot in snapshots:
        request = requests_by_key.get((snapshot["account_id"], snapshot["request_id"]))
        if request is None or snapshot["captured_at"] != request["occurred_at"] or snapshot["service_version"] != request["service_version"]:
            raise ValueError("Account snapshot must match its request's account, time and service version.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path, timeout=1.0)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            # Either both datasets update, or neither does (including initial DDL).
            connection.execute("BEGIN")
            connection.execute("""
                CREATE TABLE IF NOT EXISTS request_records (
                    account_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    service_version TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    http_status INTEGER NOT NULL CHECK (http_status BETWEEN 100 AND 599),
                    error_code TEXT NOT NULL,
                    retry_after_seconds INTEGER CHECK (retry_after_seconds >= 0),
                    PRIMARY KEY (account_id, request_id)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS account_states (
                    account_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    service_version TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    rate_window_seconds INTEGER NOT NULL CHECK (rate_window_seconds > 0),
                    rate_limit_per_window INTEGER NOT NULL CHECK (rate_limit_per_window >= 0),
                    requests_in_window INTEGER NOT NULL CHECK (requests_in_window >= 0),
                    monthly_quota_limit INTEGER NOT NULL CHECK (monthly_quota_limit >= 0),
                    monthly_quota_used INTEGER NOT NULL CHECK (monthly_quota_used >= 0),
                    quota_resets_at TEXT NOT NULL,
                    PRIMARY KEY (account_id, request_id),
                    FOREIGN KEY (account_id, request_id)
                        REFERENCES request_records (account_id, request_id)
                )
            """)
            connection.executemany("""
                INSERT INTO request_records VALUES (
                    :account_id, :request_id, :service_version, :occurred_at,
                    :http_status, :error_code, :retry_after_seconds
                )
                ON CONFLICT(account_id, request_id) DO UPDATE SET
                    service_version = excluded.service_version,
                    occurred_at = excluded.occurred_at,
                    http_status = excluded.http_status,
                    error_code = excluded.error_code,
                    retry_after_seconds = excluded.retry_after_seconds
            """, records)
            connection.executemany("""
                INSERT INTO account_states VALUES (
                    :account_id, :request_id, :service_version, :captured_at,
                    :rate_window_seconds, :rate_limit_per_window, :requests_in_window,
                    :monthly_quota_limit, :monthly_quota_used, :quota_resets_at
                )
                ON CONFLICT(account_id, request_id) DO UPDATE SET
                    service_version = excluded.service_version,
                    captured_at = excluded.captured_at,
                    rate_window_seconds = excluded.rate_window_seconds,
                    rate_limit_per_window = excluded.rate_limit_per_window,
                    requests_in_window = excluded.requests_in_window,
                    monthly_quota_limit = excluded.monthly_quota_limit,
                    monthly_quota_used = excluded.monthly_quota_used,
                    quota_resets_at = excluded.quota_resets_at
            """, snapshots)
    return {"request_records": len(records), "account_states": len(snapshots)}


def open_readonly_db(db_path: Path) -> sqlite3.Connection:
    # mode=ro also prevents a missing database from becoming a new empty file.
    connection = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0)
    connection.row_factory = sqlite3.Row
    return connection
