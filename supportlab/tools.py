"""Read simulated request evidence within the application's trusted account scope."""

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3

from .storage import ACCOUNT_DATASET_VERSION, DATASET_VERSION, DEFAULT_DB, open_readonly_db


@dataclass(frozen=True)
class AppContext:
    """Supplied by the application. Never take account_id from model arguments."""

    account_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.account_id, str) or not re.fullmatch(r"acc_[a-z0-9_]{1,60}", self.account_id):
            raise ValueError("Invalid application account context.")


def _valid_request_id(request_id: str) -> bool:
    return isinstance(request_id, str) and re.fullmatch(r"req_[a-z0-9_]{1,60}", request_id) is not None


def get_request_record(
    request_id: str, *, context: AppContext, db_path: Path = DEFAULT_DB
) -> dict:
    """Return a simulated record or a structured error; never modify the database.

    Only request_id is an external tool argument. context and db_path are
    application dependencies and must not be populated from a model's output.
    """
    result = {"simulation": True, "dataset_version": DATASET_VERSION}
    if not _valid_request_id(request_id):
        return {
            **result,
            "status": "invalid_arguments",
            "message": "request_id 须为 req_ 加 1–60 个小写字母、数字或下划线。",
        }

    try:
        with closing(open_readonly_db(db_path)) as connection:
            # Both conditions belong in the query. A request ID is not permission.
            row = connection.execute("""
                SELECT request_id, service_version, occurred_at, http_status,
                       error_code, retry_after_seconds
                FROM request_records
                WHERE account_id = ? AND request_id = ?
            """, (context.account_id, request_id)).fetchone()
    except sqlite3.Error:
        return {
            **result,
            "status": "storage_error",
            "message": "模拟数据库不可用，请确认已运行 init-demo 且文件可读。",
        }

    if row is None:
        # Same response for absent and out-of-scope records: do not disclose existence.
        return {
            **result,
            "status": "not_found",
            "message": "当前账号范围内未找到该请求，请核对请求编号。",
        }
    return {**result, "status": "ok", "record": dict(row)}


def get_account_state(
    request_id: str, *, context: AppContext, db_path: Path = DEFAULT_DB
) -> dict:
    """Read the account snapshot captured for this request, not today's live state.

    The only external argument is request_id; account scope comes from context.
    Missing snapshots are missing evidence: never substitute a different time.
    """
    result = {"simulation": True, "dataset_version": ACCOUNT_DATASET_VERSION}
    if not _valid_request_id(request_id):
        return {
            **result,
            "status": "invalid_arguments",
            "message": "request_id 须为 req_ 加 1–60 个小写字母、数字或下划线。",
        }

    try:
        with closing(open_readonly_db(db_path)) as connection:
            from .scenarios import read_snapshot
            extended = read_snapshot(connection, context.account_id, request_id)
            if extended is not None:
                return extended
            row = connection.execute("""
                SELECT s.request_id, s.service_version, s.captured_at,
                       s.rate_window_seconds, s.rate_limit_per_window, s.requests_in_window,
                       s.monthly_quota_limit, s.monthly_quota_used, s.quota_resets_at
                FROM account_states AS s
                JOIN request_records AS r
                  ON s.account_id = r.account_id AND s.request_id = r.request_id
                 AND s.captured_at = r.occurred_at AND s.service_version = r.service_version
                WHERE s.account_id = ? AND s.request_id = ?
            """, (context.account_id, request_id)).fetchone()
    except sqlite3.Error:
        return {
            **result,
            "status": "storage_error",
            "message": "模拟账号数据库不可用，请确认已运行新版 init-demo 且文件可读。",
        }

    if row is None:
        return {
            **result,
            "status": "not_found",
            "message": "当前账号范围内未找到与该请求时间和版本匹配的账号快照。",
        }
    return {**result, "status": "ok", "account_state": dict(row)}
