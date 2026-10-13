from contextlib import closing, redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from supportlab.__main__ import main
from supportlab.agent import LoopLimits, run_loop
from supportlab.contracts import model_tool_schemas
from supportlab.execution import ToolExecutor
from supportlab.mock_model import DemoMockModel
from supportlab.retrieval import DEFAULT_DOCS_ROOT, DOCUMENTS
from supportlab.storage import init_demo_db
from supportlab.tools import AppContext


class ScriptedModel:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.seen = []

    def next_action(self, messages, tools):
        self.seen.append(deepcopy(messages))
        return next(self.actions)


class ObservedDemo(DemoMockModel):
    def __init__(self, request_id):
        super().__init__(request_id)
        self.seen = []

    def next_action(self, messages, tools):
        self.seen.append(deepcopy(messages))
        return super().next_action(messages, tools)


class AgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory(prefix="supportlab_loop_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.db = self.root / "demo.sqlite3"
        self.docs = self.root / "docs"
        self.docs.mkdir()
        for document in DOCUMENTS:
            (self.docs / document.filename).write_bytes((DEFAULT_DOCS_ROOT / document.filename).read_bytes())
        init_demo_db(self.db)
        self.executor = ToolExecutor(AppContext("acc_demo"), db_path=self.db, docs_root=self.docs)

    def run_model(self, model, **limits):
        return run_loop(model, self.executor, user_message="模拟诊断", limits=LoopLimits(**limits))

    def test_schemas_expose_only_three_readonly_tools_and_external_arguments(self):
        schemas = model_tool_schemas()
        self.assertEqual({tool["name"] for tool in schemas},
                         {"get_request_record", "get_account_state", "search_docs"})
        for tool in schemas:
            self.assertIs(tool["parameters"]["additionalProperties"], False)
            self.assertTrue({"account_id", "context", "db_path", "docs_root"}.isdisjoint(tool["parameters"]["properties"]))
        schemas[0]["parameters"]["required"].clear()
        self.assertEqual(model_tool_schemas()[0]["parameters"]["required"], ["request_id"])

    def test_unknown_tools_and_internal_overrides_never_start_worker(self):
        cases = [("init_demo_db", {}), ("run_shell", {"command": "echo should_not_run"}),
                 ("get_request_record", {"request_id": "req_demo_018", "account_id": "acc_other"}),
                 ("get_account_state", {"request_id": "req_demo_018", "db_path": "elsewhere"}),
                 ("search_docs", {"query": "429", "service_version": "demo-api-v1", "docs_root": "elsewhere"})]
        with patch("supportlab.execution.subprocess.run") as worker:
            for name, arguments in cases:
                with self.subTest(name=name):
                    result = self.executor.execute(name, arguments)
                    self.assertIn(result.result["status"], ("unknown_tool", "invalid_arguments"))
                    self.assertIsNone(result.validated_arguments)
            worker.assert_not_called()

    def test_bad_types_missing_fields_and_bounds_are_rejected(self):
        cases = [("get_request_record", []), ("get_request_record", {}),
                 ("get_request_record", {"request_id": "req_demo_018\n"}),
                 ("search_docs", {"query": "429"}),
                 ("search_docs", {"query": " ", "service_version": "demo-api-v1"}),
                 ("search_docs", {"query": "429", "service_version": "demo-api-v1", "top_k": True}),
                 ("search_docs", {"query": "429", "service_version": "demo-api-v1", "top_k": 6})]
        for name, arguments in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.executor.execute(name, arguments).result["status"], "invalid_arguments")

    def test_optional_top_k_is_filled_by_application(self):
        execution = self.executor.execute("search_docs", {"query": "429", "service_version": "demo-api-v1"})
        self.assertEqual(execution.validated_arguments["top_k"], 3)
        self.assertEqual(execution.result["status"], "ok")

    def test_worker_enforces_trusted_account_scope(self):
        result = self.executor.execute("get_request_record", {"request_id": "req_other_001"}).result
        self.assertEqual(result["status"], "not_found")
        self.assertNotIn("record", result)

    def test_both_demo_paths_use_real_tool_feedback_and_finish_with_citations(self):
        for request_id, expected in (("req_demo_017", "频率限制"), ("req_demo_018", "月度配额耗尽")):
            with self.subTest(request_id=request_id):
                model = ObservedDemo(request_id)
                result = self.run_model(model)
                self.assertEqual(result["status"], "completed")
                self.assertIn(expected, result["answer"])
                self.assertEqual(result["execution_mode"], "mock")
                self.assertIs(result["model_called"], False)
                self.assertEqual(result["usage"]["model_api_calls"], 0)
                self.assertIsNone(result["usage"]["token_usage"])
                self.assertEqual(result["usage"]["tool_calls"], 3)
                self.assertEqual(result["usage"]["decision_turns"], 4)
                tool_messages = [message for message in model.seen[-1] if message["role"] == "tool"]
                self.assertEqual([message["tool_call_id"] for message in tool_messages], ["call_1", "call_2", "call_3"])
                self.assertEqual([message["name"] for message in tool_messages],
                                 ["get_request_record", "get_account_state", "search_docs"])
                self.assertEqual(tool_messages[0]["content"]["record"]["request_id"], request_id)
                self.assertIn(result["citations"][0], [hit["citation"] for hit in tool_messages[2]["content"]["matches"]])

    def test_final_answer_is_allowed_at_exact_tool_budget(self):
        result = self.run_model(DemoMockModel("req_demo_018"), max_tool_calls=3)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["usage"]["tool_calls"], 3)

    def test_budget_two_stops_before_retrieval_without_fabricating_diagnosis(self):
        result = self.run_model(DemoMockModel("req_demo_018"), max_tool_calls=2)
        self.assertEqual(result["reason"], "tool_budget_exhausted")
        self.assertEqual(result["status"], "handoff")
        self.assertEqual(result["citations"], [])
        tools = [entry["tool"] for entry in result["trace"] if entry["event"] == "tool"]
        self.assertEqual(tools, ["get_request_record", "get_account_state"])

    def test_zero_budget_and_turn_limit_stop_execution(self):
        result = self.run_model(DemoMockModel("req_demo_018"), max_tool_calls=0)
        self.assertEqual(result["usage"]["tool_calls"], 0)
        self.assertEqual(result["reason"], "tool_budget_exhausted")
        result = self.run_model(DemoMockModel("req_demo_018"), max_turns=1)
        self.assertEqual(result["usage"]["tool_calls"], 1)
        self.assertEqual(result["reason"], "turn_budget_exhausted")

    def test_invalid_tool_attempts_consume_budget(self):
        model = ScriptedModel([DemoMockModel.call(number, "run_shell") for number in range(1, 4)])
        with patch("supportlab.execution.subprocess.run") as worker:
            result = self.run_model(model, max_tool_calls=2)
            worker.assert_not_called()
        self.assertEqual(result["usage"]["tool_calls"], 2)
        self.assertEqual(result["reason"], "tool_budget_exhausted")
        self.assertEqual(model.seen[1][-1]["content"]["status"], "unknown_tool")

    def test_argument_error_is_returned_and_next_turn_can_correct_it(self):
        model = ScriptedModel([
            DemoMockModel.call(1, "get_request_record", request_id="BAD"),
            DemoMockModel.call(2, "get_request_record", request_id="req_demo_018"),
            DemoMockModel.final("clarify", "参数修正示范已取得请求记录。"),
        ])
        result = self.run_model(model)
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(model.seen[1][-1]["content"]["status"], "invalid_arguments")
        self.assertEqual(model.seen[2][-1]["content"]["record"]["error_code"], "MONTHLY_QUOTA_EXHAUSTED")

    def test_duplicate_call_id_is_rejected_before_second_execution(self):
        call = DemoMockModel.call(1, "get_request_record", request_id="req_demo_018")
        result = self.run_model(ScriptedModel([call, call]))
        self.assertEqual(result["reason"], "duplicate_call_id")
        self.assertEqual(result["usage"]["tool_calls"], 1)

    def test_missing_or_unavailable_request_produces_clarification(self):
        for request_id, calls in ((None, 0), ("req_missing_001", 1), ("req_other_001", 1)):
            with self.subTest(request_id=request_id):
                result = self.run_model(DemoMockModel(request_id))
                self.assertEqual(result["status"], "needs_input")
                self.assertEqual(result["usage"]["tool_calls"], calls)
                self.assertNotIn("PERMISSION_DENIED", result["answer"])

    def test_missing_snapshot_and_conflicting_numbers_produce_handoff(self):
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DELETE FROM account_states WHERE request_id = 'req_demo_018'")
            connection.execute("UPDATE account_states SET monthly_quota_used = monthly_quota_limit WHERE request_id = 'req_demo_017'")
        for request_id, calls in (("req_demo_018", 2), ("req_demo_017", 3)):
            with self.subTest(request_id=request_id):
                result = self.run_model(DemoMockModel(request_id))
                self.assertEqual(result["status"], "handoff")
                self.assertEqual(result["usage"]["tool_calls"], calls)
                self.assertEqual(result["citations"], [])

    def test_fabricated_or_empty_diagnosis_citations_are_rejected(self):
        for citations in ([], ["invented-document:L1"]):
            result = self.run_model(ScriptedModel([DemoMockModel.final("diagnosis", "无证据结论", citations)]))
            self.assertEqual(result["reason"], "invalid_citation")
            self.assertNotIn("无证据结论", result["answer"])

    def test_malformed_decision_and_model_exception_stop_cleanly(self):
        for action in ("not an action", {}, {"type": "tool_call", "name": "get_request_record"}):
            self.assertEqual(self.run_model(ScriptedModel([action]))["reason"], "invalid_action")
        self.assertEqual(self.run_model(ScriptedModel([]))["reason"], "model_error")

    def test_tool_process_timeout_is_returned_and_does_not_poison_next_call(self):
        fast = ToolExecutor(AppContext("acc_demo"), db_path=self.db, docs_root=self.docs, timeout_seconds=0.2)
        with closing(sqlite3.connect(self.db)) as connection:
            connection.execute("BEGIN EXCLUSIVE")
            try:
                result = run_loop(DemoMockModel("req_demo_018"), fast, user_message="测试超时")
            finally:
                connection.rollback()
        self.assertEqual(result["status"], "handoff")
        self.assertEqual(next(entry for entry in result["trace"] if entry["event"] == "tool")["error"], "timeout")
        self.assertEqual(self.executor.execute("get_request_record", {"request_id": "req_demo_018"}).result["status"], "ok")

    def test_tool_crash_is_structured_and_raw_exception_is_not_exposed(self):
        with patch("supportlab.execution.subprocess.run", side_effect=OSError("SECRET_EXCEPTION_TEXT")):
            result = self.run_model(DemoMockModel("req_demo_018"))
        self.assertEqual(result["status"], "handoff")
        self.assertNotIn("SECRET_EXCEPTION_TEXT", json.dumps(result))

    def test_model_cannot_mutate_history_or_tool_contracts_in_place(self):
        class MutatingModel:
            def next_action(inner, messages, tools):
                tools[0]["parameters"]["properties"]["account_id"] = {"type": "string"}
                messages.clear()
                return DemoMockModel.call(1, "get_request_record", request_id="req_demo_018", account_id="acc_other")
        result = self.run_model(MutatingModel(), max_turns=1)
        self.assertEqual(next(entry for entry in result["trace"] if entry["event"] == "tool")["error"], "invalid_arguments")
        self.assertNotIn("account_id", model_tool_schemas()[0]["parameters"]["properties"])

    def test_invalid_limits_are_rejected(self):
        for count in (-1, 21, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                LoopLimits(max_tool_calls=count)
        for timeout in (0, -1, 31, float("nan"), float("inf"), True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                ToolExecutor(AppContext("acc_demo"), timeout_seconds=timeout)

    def test_cli_schemas_and_invalid_limits_are_machine_readable(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["tool-schemas"]), 0)
        self.assertEqual(len(json.loads(stdout.getvalue())), 3)
        for arguments in (["agent-mock", "--max-tool-calls", "-1"], ["agent-mock", "--tool-timeout", "nan"]):
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(arguments), 2)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "invalid_arguments")


if __name__ == "__main__":
    unittest.main()
