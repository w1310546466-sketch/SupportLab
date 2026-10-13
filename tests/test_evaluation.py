from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from supportlab.__main__ import main
from supportlab.evaluation import CASE_FILE, fingerprint, grade_case, load_cases, run_evaluation
from supportlab.retrieval import DEFAULT_DOCS_ROOT, DOCUMENTS
from supportlab.storage import DEFAULT_ACCOUNT_FIXTURE, DEFAULT_FIXTURE


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_cases()
        cls.case = next(case for case in cls.dataset["cases"] if case["id"] == "quota-ok")
        cls.report = run_evaluation(case_id="quota-ok")

    def setUp(self):
        row = deepcopy(self.report["cases"][0])
        self.result, self.calls = row["result"], row["tool_evidence"]

    def grade(self):
        return grade_case(self.case, self.result, self.calls)

    def test_dataset_and_report_do_not_claim_heldout_or_real_model_success(self):
        self.assertEqual(len(self.dataset["cases"]), 20)
        self.assertEqual(len({case["id"] for case in self.dataset["cases"]}), 20)
        self.assertGreaterEqual(len({case["family"] for case in self.dataset["cases"]}), 8)
        self.assertEqual(self.report["split"], "development")
        self.assertEqual(self.report["human_review_status"], "pending")
        self.assertIsNone(self.report["task_success_rate"])
        self.assertIsNone(self.report["token_usage"])
        self.assertIsNone(self.report["cost"])
        self.assertEqual(self.report["model_api_calls"], 0)
        self.assertEqual(self.report["automated_pass_count"], 1)
        for key in ("code_sha256", "source_sha256", "dataset_sha256", "prompt_sha256"):
            self.assertRegex(self.report[key], r"^[0-9a-f]{64}$")

    def test_real_citation_with_wrong_conclusion_fails(self):
        self.result["answer"] = self.result["answer"].replace("模拟诊断：月度配额耗尽。", "模拟诊断：频率限制。")
        grade = self.grade()
        self.assertEqual(self.result["status"], "completed")
        self.assertFalse(grade["automated_pass"])
        self.assertIn("diagnosis_text", grade["failed_checks"])
        self.assertNotIn("citation_origin", grade["failed_checks"])

    def test_wrong_or_missing_rule_cannot_pass(self):
        for citations in ([], ["fabricated"], [self.result["citations"][0].replace("L11-L11", "L10-L10")]):
            with self.subTest(citations=citations):
                self.result["citations"] = citations
                self.assertIn("expected_rule_cited", self.grade()["failed_checks"])

    def test_correct_answer_without_account_evidence_fails(self):
        self.calls.pop(1)
        grade = self.grade()
        self.assertFalse(grade["automated_pass"])
        self.assertIn("account_evidence", grade["failed_checks"])

    def test_wrong_request_and_stale_snapshot_fail(self):
        self.calls[0]["result"]["record"]["request_id"] = "req_other_001"
        self.calls[1]["result"]["account_state"]["captured_at"] = "2000-01-01T00:00:00Z"
        failures = self.grade()["failed_checks"]
        self.assertIn("request_scope", failures)
        self.assertIn("evidence_alignment", failures)

    def test_wrong_action_cannot_pass_just_by_containing_expected_words(self):
        self.result["outcome"] = "handoff"
        self.result["status"] = "handoff"
        self.assertIn("outcome", self.grade()["failed_checks"])

    def test_call_accounting_and_query_feedback_are_checked(self):
        self.result["usage"]["tool_calls"] = 1
        self.calls[2]["arguments"]["query"] = "429"
        failures = self.grade()["failed_checks"]
        self.assertIn("call_accounting", failures)
        self.assertIn("feedback_used_for_search", failures)

    def test_fault_demo_exercises_loop_but_is_rejected_by_grader(self):
        report = run_evaluation(case_id="quota-ok", wrong_cause=True)
        self.assertEqual(report["cases"][0]["result"]["status"], "completed")
        self.assertEqual(report["automated_pass_count"], 0)
        self.assertEqual(report["failed_case_ids"], ["quota-ok"])
        self.assertIn("deliberately-wrong", report["model"])

    def test_case_changes_are_isolated_and_handoff_can_pass(self):
        paths = [DEFAULT_FIXTURE, DEFAULT_ACCOUNT_FIXTURE,
                 *(DEFAULT_DOCS_ROOT / document.filename for document in DOCUMENTS)]
        before = fingerprint(paths)
        for case_id in ("missing-snapshot", "docs-version", "database-missing"):
            with self.subTest(case_id=case_id):
                report = run_evaluation(case_id=case_id)
                self.assertEqual(report["automated_pass_count"], 1)
                self.assertEqual(report["cases"][0]["result"]["outcome"], "handoff")
        self.assertEqual(fingerprint(paths), before)

    def test_unknown_case_invalid_probe_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            run_evaluation(case_id="not-a-case")
        with self.assertRaises(ValueError):
            run_evaluation(wrong_cause=True)
        with self.assertRaises(ValueError):
            run_evaluation(case_id="missing-id", wrong_cause=True)
        with TemporaryDirectory() as temporary:
            data = json.loads(CASE_FILE.read_text(encoding="utf-8"))
            data["cases"][1]["id"] = data["cases"][0]["id"]
            path = Path(temporary) / "cases.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_cases(path)

    def test_cli_json_and_file_report_agree(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "report.json"
            out = StringIO()
            with redirect_stdout(out):
                exit_code = main(["eval-mock", "--case", "missing-id", "--json", "--output", str(path)])
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(out.getvalue()), json.loads(path.read_text(encoding="utf-8")))
        out = StringIO()
        with redirect_stdout(out):
            self.assertEqual(main(["eval-mock", "--case", "invalid-case"]), 2)
        self.assertEqual(json.loads(out.getvalue())["status"], "evaluation_error")


if __name__ == "__main__":
    unittest.main()
