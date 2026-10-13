from contextlib import closing, redirect_stderr, redirect_stdout
from functools import partial
from io import StringIO
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from supportlab.__main__ import main
from supportlab.storage import DEFAULT_ACCOUNT_FIXTURE, DEFAULT_FIXTURE, init_demo_db, open_readonly_db
from supportlab.tools import AppContext, get_account_state, get_request_record


class AccountStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="supportlab_account_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.db = self.root / "模拟 data #2" / "accounts.sqlite3"
        init_demo_db(self.db)
        self.context = AppContext("acc_demo")

    def query(self, request_id, context=None):
        return get_account_state(request_id, context=context or self.context, db_path=self.db)

    def seed_copies(self):
        request_path = self.root / "requests.json"
        account_path = self.root / "account_states.json"
        request_path.write_text(DEFAULT_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
        account_path.write_text(DEFAULT_ACCOUNT_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
        return request_path, account_path

    def run_cli(self, *arguments):
        stdout, stderr = StringIO(), StringIO()
        with patch("supportlab.__main__.get_account_state",
                   side_effect=partial(get_account_state, db_path=self.db)):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(list(arguments))
        return code, json.loads(stdout.getvalue()), json.loads(stderr.getvalue())

    def test_rate_evidence_matches_request_time_and_version(self):
        result = self.query("req_demo_017")
        self.assertEqual(result["status"], "ok")
        self.assertIs(result["simulation"], True)
        self.assertEqual(result["dataset_version"], "account-states-v1")
        state = result["account_state"]
        request = get_request_record("req_demo_017", context=self.context, db_path=self.db)["record"]
        self.assertEqual(state["captured_at"], request["occurred_at"])
        self.assertEqual(state["service_version"], request["service_version"])
        self.assertGreater(state["requests_in_window"], state["rate_limit_per_window"])
        self.assertLess(state["monthly_quota_used"], state["monthly_quota_limit"])
        self.assertNotIn("account_id", state)

    def test_quota_snapshot_is_distinct_from_earlier_rate_snapshot(self):
        earlier = self.query("req_demo_017")["account_state"]
        later = self.query("req_demo_018")["account_state"]
        self.assertLess(later["requests_in_window"], later["rate_limit_per_window"])
        self.assertGreaterEqual(later["monthly_quota_used"], later["monthly_quota_limit"])
        self.assertNotEqual(earlier["captured_at"], later["captured_at"])
        self.assertNotEqual(earlier["monthly_quota_used"], later["monthly_quota_used"])

    def test_cross_account_unknown_and_missing_context_return_no_evidence(self):
        result = self.query("req_other_001")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result, self.query("req_missing_001"))
        self.assertEqual(result, self.query("req_demo_017", AppContext("acc_missing")))
        self.assertNotIn("account_state", result)

    def test_only_application_context_can_select_other_account(self):
        other = AppContext("acc_other")
        self.assertEqual(self.query("req_other_001", other)["status"], "ok")
        self.assertEqual(self.query("req_demo_017", other)["status"], "not_found")

    def test_same_request_id_in_two_accounts_cannot_mix_snapshots(self):
        request_path, account_path = self.seed_copies()
        for path in (request_path, account_path):
            fixture = json.loads(path.read_text(encoding="utf-8"))
            fixture["records"][2]["request_id"] = "req_demo_017"
            path.write_text(json.dumps(fixture), encoding="utf-8")
        init_demo_db(self.db, request_path, account_path)
        own = self.query("req_demo_017")["account_state"]
        other = self.query("req_demo_017", AppContext("acc_other"))["account_state"]
        self.assertEqual(own["rate_limit_per_window"], 60)
        self.assertEqual(other["rate_limit_per_window"], 10)

    def test_invalid_request_is_rejected_before_database_access(self):
        with patch("supportlab.tools.open_readonly_db") as connection:
            for value in (None, 7, {}, "", "req_", "req_DEMO", "req_x\n",
                          "req_" + "a" * 61, "req_x' OR 1=1 --"):
                with self.subTest(value=value):
                    self.assertEqual(self.query(value)["status"], "invalid_arguments")
            connection.assert_not_called()

    def test_missing_snapshot_does_not_fall_back_to_another_time(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DELETE FROM account_states WHERE account_id = ? AND request_id = ?",
                               ("acc_demo", "req_demo_017"))
        self.assertEqual(self.query("req_demo_017")["status"], "not_found")
        self.assertEqual(self.query("req_demo_018")["status"], "ok")
        self.assertEqual(get_request_record("req_demo_017", context=self.context, db_path=self.db)["status"], "ok")

    def test_stale_time_or_version_is_not_returned_as_matching_evidence(self):
        for field, value in (("captured_at", "2026-09-30T00:00:00Z"), ("service_version", "demo-api-v2")):
            with self.subTest(field=field):
                init_demo_db(self.db)
                with closing(sqlite3.connect(self.db)) as connection, connection:
                    # field is a test-owned constant; values are still bound parameters.
                    connection.execute(f"UPDATE account_states SET {field} = ? WHERE request_id = ?",
                                       (value, "req_demo_017"))
                self.assertEqual(self.query("req_demo_017")["status"], "not_found")

    def test_queries_are_readonly(self):
        before = self.db.read_bytes()
        for request_id in ("req_demo_017", "req_demo_018", "req_other_001", "req_missing_001"):
            self.query(request_id)
        self.assertEqual(self.db.read_bytes(), before)
        with closing(open_readonly_db(self.db)) as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM account_states")

    def test_missing_corrupt_and_uninitialized_databases_return_storage_error(self):
        missing = self.root / "missing.sqlite3"
        corrupt = self.root / "corrupt.sqlite3"
        corrupt.write_text("invalid database", encoding="utf-8")
        uninitialized = self.root / "uninitialized.sqlite3"
        sqlite3.connect(uninitialized).close()
        for path in (missing, corrupt, uninitialized):
            with self.subTest(path=path):
                result = get_account_state("req_demo_017", context=self.context, db_path=path)
                self.assertEqual(result["status"], "storage_error")
                self.assertNotIn(str(path), json.dumps(result))
        self.assertFalse(missing.exists())

    def test_reimport_updates_matching_snapshot_without_duplicates(self):
        request_path, account_path = self.seed_copies()
        fixture = json.loads(account_path.read_text(encoding="utf-8"))
        new_used = fixture["records"][0]["monthly_quota_used"] - 1
        fixture["records"][0]["monthly_quota_used"] = new_used
        account_path.write_text(json.dumps(fixture), encoding="utf-8")
        self.assertNotEqual(self.query("req_demo_017")["account_state"]["monthly_quota_used"], new_used)
        counts = init_demo_db(self.db, request_path, account_path)
        self.assertEqual(self.query("req_demo_017")["account_state"]["monthly_quota_used"], new_used)
        with closing(open_readonly_db(self.db)) as connection:
            actual = connection.execute("SELECT COUNT(*) FROM account_states").fetchone()[0]
        self.assertEqual(actual, counts["account_states"])

    def test_non_simulated_or_mismatched_snapshots_are_rejected(self):
        request_path, account_path = self.seed_copies()
        original = account_path.read_text(encoding="utf-8")
        for key, value in (("simulation", False), ("dataset_version", "wrong-version"),
                           ("captured_at", "2026-09-30T00:00:00Z"),
                           ("service_version", "demo-api-v2"), ("account_id", "acc_missing")):
            with self.subTest(key=key):
                fixture = json.loads(original)
                target = fixture if key in ("simulation", "dataset_version") else fixture["records"][0]
                target[key] = value
                account_path.write_text(json.dumps(fixture), encoding="utf-8")
                with self.assertRaises(ValueError):
                    init_demo_db(self.db, request_path, account_path)
        self.assertEqual(self.query("req_demo_017")["status"], "ok")

    def test_failed_snapshot_import_rolls_back_request_changes_too(self):
        request_path, account_path = self.seed_copies()
        requests = json.loads(request_path.read_text(encoding="utf-8"))
        requests["records"][0]["retry_after_seconds"] = 600
        request_path.write_text(json.dumps(requests), encoding="utf-8")
        snapshots = json.loads(account_path.read_text(encoding="utf-8"))
        snapshots["records"][0]["monthly_quota_used"] = 998
        snapshots["records"][1]["monthly_quota_used"] = -1
        account_path.write_text(json.dumps(snapshots), encoding="utf-8")
        before_request = get_request_record("req_demo_017", context=self.context, db_path=self.db)
        before_account = self.query("req_demo_017")
        with self.assertRaises(sqlite3.IntegrityError):
            init_demo_db(self.db, request_path, account_path)
        self.assertEqual(get_request_record("req_demo_017", context=self.context, db_path=self.db), before_request)
        self.assertEqual(self.query("req_demo_017"), before_account)

    def test_cli_returns_account_data_and_separate_trace(self):
        code, result, trace = self.run_cli("account", "req_demo_017")
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(trace["tool"], "get_account_state")
        self.assertEqual(trace["dataset_version"], "account-states-v1")
        self.assertEqual(trace["validated_arguments"], {"request_id": "req_demo_017"})
        self.assertEqual(trace["result_summary"]["captured_at"], result["account_state"]["captured_at"])
        self.assertIs(trace["model_called"], False)
        self.assertIsNone(trace["token_usage"])

    def test_cli_not_found_invalid_and_storage_error_exit_codes(self):
        for request_id, code in (("req_other_001", 1), ("req_missing_001", 1), ("INVALID_SECRET_EXAMPLE", 2)):
            with self.subTest(request_id=request_id):
                actual, result, trace = self.run_cli("account", request_id)
                self.assertEqual(actual, code)
                self.assertNotIn("account_state", result)
                if code == 2:
                    self.assertIsNone(trace["validated_arguments"])
                    self.assertNotIn(request_id, json.dumps(trace))
        self.db = self.root / "missing.sqlite3"
        code, result, trace = self.run_cli("account", "req_demo_017")
        self.assertEqual(code, 3)
        self.assertEqual(result["status"], "storage_error")

    def test_cli_requires_request_and_rejects_account_override(self):
        for arguments in (["account"], ["account", "req_other_001", "--account-id", "acc_other"]):
            with self.subTest(arguments=arguments):
                with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
                    main(arguments)
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
