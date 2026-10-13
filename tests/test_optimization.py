from contextlib import closing, redirect_stdout, redirect_stderr
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from supportlab.__main__ import main
from supportlab.agent import ModelFailure, run_loop
from supportlab.comparison import RuleReplayAdapter, grade_comparison, run_comparison
from supportlab.conversation import ConversationService
from supportlab.contracts import model_tool_schemas
from supportlab.execution import ToolExecutor
from supportlab.knowledge import build_index
from supportlab.memory import MemoryStore, build_context
from supportlab.quality_eval import run_quality_evaluation
from supportlab.recovery_demo import OfflineAdapter, _response
from supportlab.routing import choose_route
from supportlab.storage import init_demo_db
from supportlab.tools import AppContext
from supportlab.workflow import ModelWorkflow, WORKFLOW_PROMPT


class OptimizedConversationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "business.sqlite3"
        init_demo_db(self.db)
        self.store = MemoryStore(self.root / "memory.sqlite3")
        self.index = self.root / "index.sqlite3"
        self.app = self.service()
        self.app.new_session("test")

    def service(self, **kwargs):
        return ConversationService(store=self.store, db_path=self.db, index_path=self.index, **kwargs)

    def test_knowledge_topic_change_after_diagnosis_avoids_requery(self):
        self.app.chat("test", "req_demo_018")
        answer = self.app.chat("test", "为什么会出现 401 错误")
        self.assertEqual(answer["outcome"], "docs")
        self.assertEqual(answer["routing"]["route_reason"], "information_question")
        self.assertIn("401", answer["answer"])
        self.assertEqual(answer["usage"].get("tool_calls", 0), 0)

    def test_side_question_keeps_checkpoint_waiting_across_restart(self):
        self.app.chat("test", "帮我排查")
        docs = self.app.chat("test", "认证失败是什么意思")
        self.assertEqual(docs["outcome"], "docs")
        self.assertTrue(docs["pending_request_id"])
        self.assertEqual(self.store.get_session(self.app.actor, "test")["status"], "needs_input")
        answer = self.service().chat("test", "req_demo_017")
        self.assertEqual(answer["routing"]["route_reason"], "resume_request")
        self.assertEqual(answer["structured_answer"]["diagnosis"]["cause"], "rate_limit_exceeded")

    def test_doc_followup_reuses_topic_but_explicit_new_topic_wins(self):
        self.app.chat("test", "/docs 401 是什么")
        for _ in range(3):
            result = self.service().chat("test", "详细解释一下")
            self.assertIn("401", result["retrieval_query"])
            self.assertEqual(result["retrieval_query"].count("详细解释一下"), 1)
        changed = self.app.chat("test", "429 是什么意思")
        self.assertNotIn("401", changed["retrieval_query"])
        packet = build_context(self.store, self.app.actor, "test", max_bytes=800)
        self.assertLessEqual(packet["bytes"], 800)

    def test_unknown_question_live_mode_does_not_even_load_configuration(self):
        self.app.new_session("live", mode="agent")
        app = self.service(model_factory=lambda _: self.fail("must not construct model"))
        for question in ("/docs API 支持退款吗", "/docs 这个服务如何申请发票", "/docs 请说明文档里的训练参数"):
            result = app.chat("live", question)
            self.assertEqual(result["outcome"], "clarify")
            self.assertEqual(result["usage"]["model_api_calls"], 0)

    def test_missing_source_does_not_leave_graph_in_crashed_state(self):
        app = self.service(docs_root=self.root / "missing_docs")
        self.assertEqual(app.chat("test", "/docs 429")["reason"], "index_unavailable")
        self.assertEqual(app.chat("test", "帮我排查")["reason"], "waiting_request_id")

    def test_long_document_question_is_not_silently_truncated(self):
        result = self.app.chat("test", "/docs " + "说" * 257)
        self.assertEqual(result["reason"], "document_query_too_long")
        self.assertEqual(result["usage"]["model_api_calls"], 0)

    def test_stale_index_does_not_construct_model(self):
        build_index(index_path=self.index)
        with closing(sqlite3.connect(self.index)) as db, db:
            db.execute("UPDATE knowledge_meta SET version='old-index'")
        self.app.new_session("live", mode="agent")
        app = self.service(model_factory=lambda _: self.fail("must not call model"))
        self.assertEqual(app.chat("live", "/docs 429")["reason"], "stale_index")

    def test_explicit_request_and_conflicting_request_ids_are_not_inferred(self):
        self.assertEqual(choose_route("解释 req_demo_017 为什么失败", target="req_demo_018")["target"], "req_demo_017")
        multiple = choose_route("排查 req_demo_017 req_demo_018", target="req_demo_018")
        self.assertEqual(multiple["route"], "wait_request")
        self.assertIsNone(multiple["target"])

    def test_usage_counts_saved_turns_and_does_not_claim_bill_completeness(self):
        self.store.save_turn(self.app.actor, "test", turn_id="usage", user_text="test", target=None,
                            result={"status": "handoff", "answer": "test", "usage": {"model_api_calls": 2,
                                    "tool_calls": 1, "token_usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                                    "usage_complete": False}})
        result = self.app.chat("test", "/usage")
        self.assertEqual(result["usage_summary"]["model_api_calls"], 2)
        self.assertEqual(result["usage_summary"]["known_tokens"]["total_tokens"], 15)
        self.assertEqual(result["usage_summary"]["turns_with_incomplete_usage"], 1)
        self.app.new_session("other")
        self.assertEqual(self.app.chat("other", "/usage")["usage_summary"]["model_api_calls"], 0)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name) / "db.sqlite3"
        init_demo_db(self.db)

    def run_model(self, adapter, request_id="req_demo_018"):
        return run_loop(ModelWorkflow(request_id, adapter), ToolExecutor(AppContext("acc_demo"), db_path=self.db),
                        user_message="请排查 " + request_id, target_request_id=request_id)

    def test_model_sees_actual_paired_feedback_and_no_expected_answers(self):
        class CapturingAdapter(RuleReplayAdapter):
            def _respond(self, payload, timeout):
                self.captured = deepcopy(payload)
                return super()._respond(payload, timeout)
        model = CapturingAdapter("req_demo_018", system_prompt=WORKFLOW_PROMPT)
        result = self.run_model(model)
        self.assertTrue(result["answer_check"]["passed"])
        self.assertEqual(result["usage"]["simulated_model_requests"], 1)
        self.assertEqual(result["usage"]["model_api_calls"], 0)
        self.assertEqual(result["control_mode"], "fixed_workflow")
        self.assertEqual([row["role"] for row in model.captured["messages"]], ["system", "user"])
        tools = json.loads(model.captured["messages"][1]["content"])["application_tool_results"]
        self.assertEqual([row["result"]["evidence_id"] for row in tools], ["E1", "E2", "E3"])
        self.assertNotIn('"assessment"', json.dumps(tools))
        self.assertEqual([row["decision_owner"] for row in result["trace"] if row["event"] == "decision"],
                         ["application_fixed_policy"] * 3 + ["model_final_synthesis"])

    def test_final_synthesizer_cannot_execute_extra_tool(self):
        response = _response(calls=[{"id": "extra", "type": "function", "function": {
            "name": "get_request_record", "arguments": '{"request_id":"req_other_001"}'}}])
        result = self.run_model(OfflineAdapter([response]))
        self.assertEqual(result["reason"], "model_protocol_error")
        self.assertEqual(result["usage"]["tool_calls"], 3)
        self.assertFalse(result["model_called"])

    def test_application_history_requires_whitelisted_valid_paired_calls(self):
        prefix = [{"role": "system", "content": "system"}, {"role": "user", "content": "request"}]
        call = {"role": "assistant", "tool_call": {"call_id": "one", "name": "get_request_record", "arguments": {"request_id": "req_demo_018"}}}
        feedback = {"role": "tool", "tool_call_id": "wrong", "name": "get_request_record", "content": {}}
        unknown = deepcopy(call)
        unknown["tool_call"]["name"] = "run_shell"
        for tail in ([call], [feedback], [call, feedback], [unknown, feedback]):
            with self.subTest(tail=tail), self.assertRaises(ModelFailure):
                OfflineAdapter([]).prime_from_application_history(prefix + tail)

    def test_offline_comparison_is_labeled_and_source_budgets_are_equal(self):
        with patch("supportlab.comparison.load_deepseek_config", side_effect=AssertionError("no real config")):
            report = run_comparison()
        self.assertEqual(report["execution_mode"], "offline_harness")
        self.assertEqual(report["model_api_calls"], 0)
        for mode in ("workflow", "agent"):
            self.assertEqual(report["summaries"][mode]["passed"], 6)
            self.assertIsNone(report["summaries"][mode]["token_usage"])
        self.assertEqual(report["controlled"]["same_tool_schemas"], model_tool_schemas())
        for i in range(0, len(report["rows"]), 2):
            self.assertEqual(report["rows"][i]["source_sha256"], report["rows"][i + 1]["source_sha256"])
        self.assertNotEqual(report["reproducibility"]["prompts"]["workflow"], report["reproducibility"]["prompts"]["agent"])

    def test_live_batch_and_bad_output_rejected_before_config_or_network(self):
        with patch("supportlab.comparison.load_deepseek_config", side_effect=AssertionError("must not load")):
            with self.assertRaises(ValueError):
                run_comparison(live=True)
            with redirect_stderr(StringIO()):
                self.assertEqual(main(["compare", "--case", "rate-ok", "--live", "--output", ".env.local"]), 2)

    def test_wrong_cause_cannot_pass_by_matching_outcome_only(self):
        result = self.run_model(RuleReplayAdapter("req_demo_018"))
        result["structured_answer"]["diagnosis"]["cause"] = "rate_limit_exceeded"
        case = {"id": "quota-ok", "expected": {"outcome": "diagnosis", "error_code": "MONTHLY_QUOTA_EXHAUSTED"}}
        self.assertFalse(grade_comparison(case, result)["passed"])

    def test_workflow_chat_mode_constructs_workflow_and_uses_same_loop(self):
        store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3")
        app = ConversationService(store=store, db_path=self.db,
                                  model_factory=lambda purpose: RuleReplayAdapter("req_demo_018", system_prompt=WORKFLOW_PROMPT))
        app.new_session("workflow", mode="workflow")
        self.assertEqual(app.chat("workflow", "req_demo_018")["control_mode"], "fixed_workflow")


class QualityRegressionTests(unittest.TestCase):
    def test_versioned_development_cases_keep_correct_hits_and_abstentions(self):
        with patch("supportlab.configuration.load_deepseek_config", side_effect=AssertionError("no API")):
            report = run_quality_evaluation()
        self.assertEqual(report["passed_count"], 20, [row["id"] for row in report["cases"] if not row["passed"]])
        self.assertEqual(report["metrics"]["correct_abstentions"], {"numerator": 4, "denominator": 4})
        self.assertEqual(report["human_review"], "pending")


if __name__ == "__main__":
    unittest.main()
