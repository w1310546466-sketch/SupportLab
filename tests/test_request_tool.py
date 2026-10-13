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
from supportlab.storage import init_demo_db, open_readonly_db
from supportlab.tools import AppContext, get_request_record


class RequestToolTests(unittest.TestCase):
    def setUp(self):
        # Each test gets its own database; tests do not change the user's demo DB.
        self.temporary = TemporaryDirectory(prefix="supportlab_")
        self.addCleanup(self.temporary.cleanup)
        self.db = Path(self.temporary.name) / "模拟 data #1" / "requests.sqlite3"
        init_demo_db(self.db)
        self.context = AppContext("acc_demo")

    def query(self, request_id, context=None):
        return get_request_record(request_id, context=context or self.context, db_path=self.db)

    def run_cli(self, *arguments):
        stdout, stderr = StringIO(), StringIO()
        with patch("supportlab.__main__.get_request_record",
                   side_effect=partial(get_request_record, db_path=self.db)):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = main(list(arguments))
        return code, json.loads(stdout.getvalue()), json.loads(stderr.getvalue())

    def test_rate_limit_record_has_evidence_and_simulation_label(self):
        result = self.query("req_demo_017")
        self.assertEqual(result["status"], "ok")
        self.assertIs(result["simulation"], True)
        self.assertEqual(result["dataset_version"], "requests-v1")
        self.assertEqual(result["record"]["http_status"], 429)
        self.assertEqual(result["record"]["error_code"], "RATE_LIMIT_EXCEEDED")
        self.assertEqual(result["record"]["service_version"], "demo-api-v1")
        self.assertNotIn("account_id", result["record"])

    def test_same_http_status_can_have_a_different_error_code(self):
        rate = self.query("req_demo_017")["record"]
        quota = self.query("req_demo_018")["record"]
        self.assertEqual(rate["http_status"], quota["http_status"])
        self.assertEqual(quota["error_code"], "MONTHLY_QUOTA_EXHAUSTED")
        self.assertNotEqual(rate["error_code"], quota["error_code"])
        self.assertIsNone(quota["retry_after_seconds"])

    def test_unknown_request_has_no_record(self):
        result = self.query("req_missing_001")
        self.assertEqual(result["status"], "not_found")
        self.assertNotIn("record", result)

    def test_cross_account_and_unknown_are_indistinguishable(self):
        result = self.query("req_other_001")
        self.assertEqual(result, self.query("req_missing_001"))
        self.assertNotIn("PERMISSION_DENIED", json.dumps(result))

    def test_scope_comes_from_application_context(self):
        other = AppContext("acc_other")
        self.assertEqual(self.query("req_other_001", other)["record"]["http_status"], 403)
        self.assertEqual(self.query("req_demo_017", other)["status"], "not_found")

    def test_malformed_request_is_rejected_before_database_access(self):
        invalid = [None, 17, {}, "", "req_", "req_DEMO", "req_demo_017\n",
                   "req_" + "a" * 61, "req_x' OR 1=1 --", "../requests"]
        with patch("supportlab.tools.open_readonly_db") as connection:
            for value in invalid:
                with self.subTest(value=value):
                    self.assertEqual(self.query(value)["status"], "invalid_arguments")
            connection.assert_not_called()

    def test_invalid_application_context_is_rejected(self):
        for value in (None, "", "acc_", "acc_demo' OR 1=1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                AppContext(value)

    def test_missing_database_is_not_created(self):
        missing = self.db.parent / "missing.sqlite3"
        result = get_request_record("req_demo_017", context=self.context, db_path=missing)
        self.assertEqual(result["status"], "storage_error")
        self.assertFalse(missing.exists())

    def test_corrupt_database_returns_structured_error(self):
        broken = self.db.parent / "broken.sqlite3"
        broken.write_text("this is not a database", encoding="utf-8")
        result = get_request_record("req_demo_017", context=self.context, db_path=broken)
        self.assertEqual(result["status"], "storage_error")
        self.assertNotIn(str(broken), json.dumps(result))

    def test_queries_do_not_change_database_bytes(self):
        before = self.db.read_bytes()
        self.query("req_demo_017")
        self.query("req_missing_001")
        self.query("req_other_001")
        self.assertEqual(self.db.read_bytes(), before)

    def test_readonly_connection_refuses_writes(self):
        with closing(open_readonly_db(self.db)) as connection:
            with self.assertRaises(sqlite3.OperationalError):
                connection.execute("DELETE FROM request_records")

    def test_reimport_does_not_duplicate_records(self):
        count = init_demo_db(self.db)
        with closing(open_readonly_db(self.db)) as connection:
            actual = connection.execute("SELECT COUNT(*) FROM request_records").fetchone()[0]
        self.assertEqual(actual, count["request_records"])

    def test_cli_emits_separate_result_and_trace(self):
        code, result, trace = self.run_cli("request", "req_demo_017")
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(trace["tool"], "get_request_record")
        self.assertEqual(trace["validated_arguments"], {"request_id": "req_demo_017"})
        self.assertEqual(trace["result_summary"], "RATE_LIMIT_EXCEEDED")
        self.assertGreaterEqual(trace["elapsed_ms"], 0)
        self.assertIs(trace["model_called"], False)
        self.assertIsNone(trace["token_usage"])

    def test_cli_failure_exit_codes_and_unvalidated_input_not_logged(self):
        for request_id, expected_code, status in (
            ("req_missing_001", 1, "not_found"),
            ("req_other_001", 1, "not_found"),
            ("INVALID_SECRET_EXAMPLE", 2, "invalid_arguments"),
        ):
            with self.subTest(request_id=request_id):
                code, result, trace = self.run_cli("request", request_id)
                self.assertEqual(code, expected_code)
                self.assertEqual(result["status"], status)
                self.assertEqual(trace["error"], status)
                if status == "invalid_arguments":
                    self.assertIsNone(trace["validated_arguments"])
                    self.assertNotIn(request_id, json.dumps(trace))

    def test_cli_rejects_account_override(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as caught:
            main(["request", "req_other_001", "--account-id", "acc_other"])
        self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
