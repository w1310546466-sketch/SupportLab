from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from supportlab.__main__ import main
from supportlab.answer_check import check_assessment, reference_summary
from supportlab.answer_demo import run_answer_demo
from supportlab.retrieval import search_docs
from supportlab.storage import init_demo_db
from supportlab.tools import AppContext, get_account_state, get_request_record


class AnswerEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.db = Path(cls.temporary.name) / "demo.sqlite3"
        init_demo_db(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.load_request("req_demo_018")

    def load_request(self, request_id):
        context = AppContext("acc_demo")
        self.evidence = {
            "r": {"tool": "get_request_record", "result": get_request_record(request_id, context=context, db_path=self.db)},
            "a": {"tool": "get_account_state", "result": get_account_state(request_id, context=context, db_path=self.db)},
            "d": {"tool": "search_docs", "result": search_docs("429", "demo-api-v1", 2)},
        }
        code = self.evidence["r"]["result"]["record"]["error_code"]
        self.target = request_id
        hit = next(hit for hit in self.evidence["d"]["result"]["matches"] if code in hit["excerpt"])
        self.assessment = {"cause": code.lower(), "request_evidence_id": "r", "account_evidence_id": "a",
                           "rule_citation": hit["citation"], "follow_up": ["check_current_status"]}

    def check(self):
        return check_assessment(self.assessment, self.evidence, target_request_id=self.target)

    def assert_rejected(self, issue):
        result = self.check()
        self.assertFalse(result["passed"])
        self.assertEqual(result["issues"], [issue])
        self.assertNotIn("answer", result)

    def test_supported_quota_renders_facts_and_conditional_follow_up(self):
        self.assessment["follow_up"].append("verify_quota_options")
        original = deepcopy(self.evidence)
        result = self.check()
        self.assertTrue(result["check"]["passed"])
        structured = result["structured_answer"]
        self.assertEqual(structured["facts"]["account"]["values"]["monthly_quota_used"], 1000)
        self.assertEqual(structured["facts"]["account"]["evidence_id"], "a")
        self.assertEqual(structured["diagnosis"]["label"], "月度配额耗尽")
        self.assertTrue(all(row["status"] == "needs_verification" for row in structured["follow_up"]))
        self.assertIn("现有资料未说明是否能申请", result["answer"])
        self.assertIn("不能证明今天已经恢复", result["answer"])
        self.assertIn("没有提供建议等待秒数", result["answer"])
        self.assertEqual(self.evidence, original)

    def test_rate_case_is_also_supported_with_actual_wait_record(self):
        self.load_request("req_demo_017")
        result = self.check()
        self.assertTrue(result["check"]["passed"])
        self.assertIn("频率限制", result["answer"])
        self.assertIn("建议等待秒数为 30", result["answer"])
        self.assertIn("61/60", result["answer"])

    def test_genuine_citation_does_not_rescue_wrong_cause(self):
        self.assessment["cause"] = "rate_limit_exceeded"
        self.assert_rejected("cause_not_supported")

    def test_cause_must_agree_with_numbers_and_error_code(self):
        for change in ("quota_not_exhausted", "both_limits", "wrong_error", "wrong_http"):
            self.load_request("req_demo_018")
            if change == "quota_not_exhausted":
                self.evidence["a"]["result"]["account_state"]["monthly_quota_used"] = 999
            elif change == "both_limits":
                self.evidence["a"]["result"]["account_state"]["requests_in_window"] = 61
            elif change == "wrong_error":
                self.evidence["r"]["result"]["record"]["error_code"] = "RATE_LIMIT_EXCEEDED"
            else:
                self.evidence["r"]["result"]["record"]["http_status"] = 503
            with self.subTest(change=change):
                self.assert_rejected("cause_not_supported")

    def test_equal_rate_threshold_is_not_exceeded(self):
        self.evidence["a"]["result"]["account_state"]["requests_in_window"] = 60
        self.assertTrue(self.check()["check"]["passed"])

    def test_missing_failed_or_wrong_tool_evidence_is_rejected(self):
        for change in ("missing", "failed", "wrong_tool", "wrong_dataset"):
            self.load_request("req_demo_018")
            if change == "missing":
                del self.evidence["a"]
            elif change == "failed":
                self.evidence["a"]["result"]["status"] = "not_found"
            elif change == "wrong_tool":
                self.evidence["a"]["tool"] = "search_docs"
            else:
                self.evidence["a"]["result"]["dataset_version"] = "other"
            with self.subTest(change=change):
                self.assert_rejected("missing_evidence")

    def test_target_request_time_and_version_must_match(self):
        for field, value in (("request_id", "req_demo_017"), ("captured_at", "2026-09-26T02:01:00Z"),
                             ("service_version", "demo-api-v2")):
            self.load_request("req_demo_018")
            self.evidence["a"]["result"]["account_state"][field] = value
            with self.subTest(field=field):
                self.assert_rejected("evidence_mismatch")
        self.load_request("req_demo_018")
        self.target = "req_demo_017"
        self.assert_rejected("evidence_mismatch")

    def test_absent_user_target_cannot_be_invented_by_model(self):
        self.target = None
        self.assert_rejected("missing_target_request")

    def test_forged_or_unobserved_citation_is_rejected(self):
        self.assessment["rule_citation"] = "invented-document:L1"
        self.assert_rejected("citation_not_observed")
        self.load_request("req_demo_018")
        del self.evidence["d"]
        self.assert_rejected("citation_not_observed")

    def test_observed_but_wrong_rule_citation_is_rejected(self):
        hit = next(hit for hit in self.evidence["d"]["result"]["matches"] if "RATE_LIMIT_EXCEEDED" in hit["excerpt"])
        self.assessment["rule_citation"] = hit["citation"]
        self.assert_rejected("rule_not_supported")

    def test_changed_doc_revision_or_rule_text_requires_review(self):
        for field, value in (("document_sha256", "changed"), ("excerpt", "MONTHLY_QUOTA_EXHAUSTED 意味着立即加钱即可恢复"),
                             ("line_start", 99), ("service_version", "demo-api-v2")):
            self.load_request("req_demo_018")
            hit = next(hit for hit in self.evidence["d"]["result"]["matches"]
                       if hit["citation"] == self.assessment["rule_citation"])
            hit[field] = value
            with self.subTest(field=field):
                self.assert_rejected("rule_not_supported")

    def test_unapproved_action_or_text_advice_cannot_be_published(self):
        for proposal in ("increase_quota_now", "立即购买配额一定恢复", {"text": "增加配额"}):
            self.assessment["follow_up"] = ["check_current_status", proposal]
            with self.subTest(proposal=proposal):
                self.assert_rejected("unsupported_follow_up")

    def test_current_status_uncertainty_is_mandatory(self):
        self.assessment["follow_up"] = ["verify_quota_options"]
        self.assert_rejected("missing_current_status_caveat")

    def test_quota_option_is_not_attached_to_rate_diagnosis(self):
        self.load_request("req_demo_017")
        self.assessment["follow_up"].append("verify_quota_options")
        self.assert_rejected("irrelevant_follow_up")

    def test_model_cannot_override_displayed_numbers_or_add_free_text(self):
        for field in ("facts", "message"):
            self.load_request("req_demo_018")
            self.assessment[field] = "已经成功增加了配额"
            self.assert_rejected("invalid_assessment")

    def test_malformed_structure_is_rejected_without_echo(self):
        for value in ({}, [], {**self.assessment, "cause": ["monthly_quota_exhausted"]},
                      {**self.assessment, "request_evidence_id": ""}, {**self.assessment, "follow_up": []}):
            result = check_assessment(value, self.evidence, target_request_id=self.target)
            self.assertFalse(result["passed"])

    def test_unknown_cause_is_not_guessed(self):
        self.assessment["cause"] = "out_of_money"
        self.assert_rejected("unsupported_cause")

    def test_numeric_and_time_corruption_cannot_create_a_diagnosis(self):
        for value in (True, "1000", -1):
            self.load_request("req_demo_018")
            self.evidence["a"]["result"]["account_state"]["monthly_quota_used"] = value
            self.assert_rejected("invalid_account_facts")
        for value in (None, "2026-10-01", "2026-09-01T00:00:00Z", "not-a-date"):
            self.load_request("req_demo_018")
            self.evidence["a"]["result"]["account_state"]["quota_resets_at"] = value
            self.assert_rejected("invalid_time_evidence")

    def test_order_and_call_id_spelling_do_not_change_judgment(self):
        self.evidence = {"new_doc": self.evidence["d"], "new_account": self.evidence["a"], "new_request": self.evidence["r"]}
        self.assessment.update(request_evidence_id="new_request", account_evidence_id="new_account")
        self.assertTrue(self.check()["check"]["passed"])

    def test_reference_log_does_not_echo_free_text(self):
        self.assessment.update(request_evidence_id="arbitrary sensitive text", account_evidence_id="E2")
        summary = reference_summary(self.assessment, self.evidence)
        self.assertEqual(summary["references"]["request_evidence_id"], "<invalid>")
        self.assertNotIn("sensitive", str(summary))


class AnswerDemoTests(unittest.TestCase):
    def test_four_scripted_probes_use_real_local_tools_and_zero_api_calls(self):
        with patch("supportlab.configuration.load_deepseek_config", side_effect=AssertionError("No credentials needed")):
            report = run_answer_demo()
        self.assertEqual(report["case_count"], 4)
        self.assertEqual(report["expected_behavior_count"], 4)
        self.assertEqual(report["model_api_calls"], 0)
        self.assertEqual([case["result"]["status"] for case in report["cases"]],
                         ["completed", "handoff", "handoff", "handoff"])
        for case in report["cases"]:
            self.assertFalse(case["result"]["model_called"])
            self.assertEqual(case["result"]["execution_mode"], "scripted")
            self.assertGreater(case["result"]["usage"]["tool_calls"], 0)

    def test_demo_cli_explains_result_without_network(self):
        output = StringIO()
        with redirect_stdout(output):
            code = main(["answer-demo", "--case", "unsupported-advice"])
        self.assertEqual(code, 0)  # Intentional rejection was the expected behavior.
        self.assertIn("模型调用 0 次", output.getvalue())
        self.assertIn("unsupported_follow_up", output.getvalue())

    def test_demo_output_path_cannot_overwrite_configuration(self):
        with redirect_stdout(StringIO()), patch("supportlab.answer_demo.run_answer_demo") as run:
            self.assertEqual(main(["answer-demo", "--output", ".env.local"]), 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
