"""Harness integrity tests; no locked case is executed here."""

from contextlib import redirect_stdout
from copy import deepcopy
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from supportlab.acceptance import grade_step, report_reservation, value_at, verify_map, MISSING, main


class AcceptanceHarnessTests(unittest.TestCase):
    def test_explicit_expected_cause_rejects_wrong_cause_even_if_checker_claims_pass(self):
        step = {"op": "chat", "expect": {"equals": {"result.structured_answer.diagnosis.cause": "expected-cause"}}}
        output = {"business_sources_unchanged": True, "result": {
            "outcome": "diagnosis", "usage": {"model_api_calls": 0, "tool_calls": 2}, "session_mode": "fixed",
            "answer_check": {"passed": True}, "citations": ["a"],
            "structured_answer": {"diagnosis": {"cause": "wrong-cause"},
                                  "facts": {"request": {"values": {"request_id": "req_harness"}}}},
            "trace": [{"event": "tool", "tool": name, "validated_arguments": {"request_id": "req_harness"}}
                      for name in ("get_request_record", "get_account_state")]}}
        grade = grade_step(step, output)
        self.assertFalse(grade["passed"])
        self.assertFalse(grade["checks"]["equals:result.structured_answer.diagnosis.cause"])
        self.assertTrue(grade["checks"]["grounded_diagnosis"])

    def test_json_types_and_missing_keys_do_not_pass_as_zero_or_false(self):
        for output in ({}, {"value": None}, {"value": False}):
            self.assertFalse(grade_step({"op": "open", "expect": {"equals": {"value": 0}}}, output)["passed"])
        self.assertIs(value_at({}, "a.b"), MISSING)
        self.assertTrue(grade_step({"op": "open", "expect": {"equals": {"value": None}}}, {"value": None})["passed"])

    def test_only_declared_exception_is_accepted(self):
        self.assertFalse(grade_step({"op": "chat", "expect": {}}, {"error": "ValueError"})["passed"])
        self.assertTrue(grade_step({"op": "chat", "expect": {"error": "ValueError"}}, {"error": "ValueError"})["passed"])
        self.assertFalse(grade_step({"op": "chat", "expect": {"error": "ValueError"}}, {"result": {}})["passed"])

    def test_source_hash_map_detects_modification_missing_files_and_escape(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "source.txt"
            path.write_text("fixed", encoding="utf-8")
            mapping = {"source.txt": sha256(path.read_bytes()).hexdigest()}
            verify_map(mapping, root=root)
            path.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_map(mapping, root=root)
            with self.assertRaises(ValueError):
                verify_map({"../outside.txt": "0" * 64}, root=root)
            with self.assertRaises(ValueError):
                verify_map({"absent.txt": "0" * 64}, root=root)

    def test_report_reservation_never_overwrites_existing_file(self):
        with TemporaryDirectory() as temporary, patch("supportlab.acceptance.PROJECT_ROOT", Path(temporary)):
            path = Path(temporary) / "reports" / "first.json"
            with report_reservation(path) as stream:
                stream.write('{"original":true}')
            with self.assertRaises(FileExistsError):
                with report_reservation(path):
                    self.fail("Must not open a second report writer")
            self.assertEqual(path.read_text(), '{"original":true}')

    def test_interrupted_report_is_explicitly_incomplete(self):
        with TemporaryDirectory() as temporary, patch("supportlab.acceptance.PROJECT_ROOT", Path(temporary)):
            path = Path(temporary) / "reports" / "broken.json"
            with self.assertRaises(RuntimeError):
                with report_reservation(path) as stream:
                    stream.write("partial")
                    raise RuntimeError("synthetic interruption")
            result = json.loads(path.read_text())
            self.assertEqual(result["status"], "incomplete")
            self.assertTrue(result["do_not_count_as_pass"])

    def test_invalid_destination_does_not_execute_suite(self):
        with TemporaryDirectory() as temporary, patch("supportlab.acceptance.load_sealed_suite", return_value=({}, {}, {})), \
                patch("supportlab.acceptance.run_acceptance") as run, redirect_stdout(StringIO()):
            self.assertEqual(main(["--output", str(Path(temporary) / "elsewhere.json")]), 2)
            run.assert_not_called()

    def test_docs_cannot_pass_with_forged_excerpt_or_hash(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "evidence.md"
            path.write_text("source\n", encoding="utf-8")
            digest = sha256(path.read_bytes()).hexdigest()
            hit = {"doc_id": "test", "source": "docs/evidence.md", "document_sha256": digest,
                   "line_start": 1, "line_end": 1, "excerpt": "source", "citation": f"test@{digest[:12]}:L1-L1"}
            step = {"op": "chat", "expect": {"equals": {"result.outcome": "docs"}}}
            result = {"business_sources_unchanged": True, "result": {"outcome": "docs", "session_mode": "fixed",
                      "usage": {"model_api_calls": 0}, "rag_check": {"passed": True}, "selected_evidence": [hit]}}
            self.assertTrue(grade_step(step, result, docs_root=root)["passed"])
            for field, value in (("excerpt", "forged"), ("document_sha256", "0" * 64), ("citation", "invented")):
                corrupted = deepcopy(result)
                corrupted["result"]["selected_evidence"][0][field] = value
                self.assertFalse(grade_step(step, corrupted, docs_root=root)["passed"])

    def test_real_api_usage_fails_offline_acceptance(self):
        step = {"op": "chat", "expect": {"equals": {"result.outcome": "information"}}}
        result = {"business_sources_unchanged": True, "result": {"outcome": "information", "session_mode": "fixed",
                  "usage": {"model_api_calls": 1, "tool_calls": 0}}}
        grade = grade_step(step, result)
        self.assertFalse(grade["passed"])
        self.assertFalse(grade["checks"]["zero_api"])

    def test_mutated_business_data_during_chat_fails(self):
        step = {"op": "chat", "expect": {"equals": {"result.outcome": "information"}}}
        result = {"business_sources_unchanged": False, "result": {"outcome": "information", "session_mode": "fixed",
                  "usage": {"model_api_calls": 0, "tool_calls": 0}}}
        self.assertFalse(grade_step(step, result)["passed"])


if __name__ == "__main__":
    unittest.main()
