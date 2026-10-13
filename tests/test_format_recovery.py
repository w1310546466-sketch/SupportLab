from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from supportlab.__main__ import main
from supportlab.agent import ModelFailure, run_loop
from supportlab.configuration import DeepSeekConfig
from supportlab.deepseek import ADAPTER_PROMPT, DeepSeekModel, ModelLimits
from supportlab.evaluation import prepare_case
from supportlab.execution import ToolExecutor
from supportlab.format_recovery import FinalFormatError, REPAIR_MESSAGE, parse_final_object
from supportlab.recovery_demo import run_recovery_demo
from supportlab.tools import AppContext


CONFIG = DeepSeekConfig("fake-key-not-for-network")
MESSAGES = [{"role": "system", "content": "system"}, {"role": "user", "content": "排查"}]
CLARIFY = {"type": "final", "outcome": "clarify", "resolution": {"reason": "missing_request_id"}}
HANDOFF = {"type": "final", "outcome": "handoff", "resolution": {"reason": "account_unavailable"}}


def response(content, *, finish="stop", calls=None, usage=True):
    return {"status": "ok", "finish_reason": finish, "message": {"content": content, "tool_calls": calls or []},
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120} if usage else None}


def native(identifier, name):
    return {"id": identifier, "type": "function", "function": {
        "name": name, "arguments": json.dumps({"request_id": "req_demo_018"})}}


class Responses:
    def __init__(self, items):
        self.items = iter(items)
        self.requests = []

    def __call__(self, payload, timeout):
        self.requests.append(deepcopy(payload))
        item = next(self.items)
        if isinstance(item, Exception):
            raise item
        return item


class FinalParserTests(unittest.TestCase):
    def test_valid_object_whitespace_and_unicode_are_preserved(self):
        obj = {"type": "final", "example": "中文"}
        self.assertEqual(parse_final_object(" \n" + json.dumps(obj) + "\t"), obj)

    def test_wrappers_and_multiple_objects_are_not_silently_extracted(self):
        for content in ("```json\n{}\n```", "说明：{}", "{} {}", ""):
            with self.subTest(content=content), self.assertRaises(FinalFormatError) as error:
                parse_final_object(content)
            self.assertEqual(error.exception.diagnostic["category"], "json_syntax")

    def test_ambiguous_and_non_object_values_are_rejected(self):
        for content, category in (("[]", "not_object"), ("null", "not_object"),
                                  ('{"a":1,"a":2}', "duplicate_key"), ('{"a":{"b":1,"b":2}}', "duplicate_key"),
                                  ('{"a":NaN}', "nonstandard_constant"), ('{"a":Infinity}', "nonstandard_constant")):
            with self.subTest(content=content), self.assertRaises(FinalFormatError) as error:
                parse_final_object(content)
            self.assertEqual(error.exception.diagnostic["category"], category)

    def test_diagnostics_contain_only_positions_counts_and_fixed_categories(self):
        bad = '```json\n{"private_key":"DO_NOT_PUBLISH"}\n```'
        with self.assertRaises(FinalFormatError) as error:
            parse_final_object(bad)
        diagnostic = error.exception.diagnostic
        self.assertTrue(diagnostic["starts_with_code_fence"])
        self.assertEqual(diagnostic["content_chars"], len(bad))
        self.assertEqual((diagnostic["line"], diagnostic["column"]), (1, 1))
        self.assertNotIn("DO_NOT_PUBLISH", str(error.exception) + json.dumps(diagnostic))
        self.assertNotIn("private_key", json.dumps(diagnostic))

    def test_excessive_nesting_and_size_stop_without_raw_output(self):
        for content, category in (("[" * 2000 + "]" * 2000, "excessive_nesting"), (" " * 32769, "response_too_large")):
            with self.subTest(category=category), self.assertRaises(FinalFormatError) as error:
                parse_final_object(content)
            self.assertEqual(error.exception.diagnostic["category"], category)


class RecoveryAdapterTests(unittest.TestCase):
    def run_responses(self, rows, *, limits=ModelLimits(), target=None):
        transport = Responses(rows)
        model = DeepSeekModel(CONFIG, transport=transport, limits=limits)
        result = run_loop(model, ToolExecutor(AppContext("acc_demo")), user_message="排查", target_request_id=target)
        return result, transport, model

    def test_correction_is_counted_and_raw_invalid_prose_is_not_replayed(self):
        result, transport, model = self.run_responses([
            response("```json\nRAW_SECRET_UNSUPPORTED_ADVICE\n```"), response(json.dumps(CLARIFY))])
        self.assertEqual(result["status"], "needs_input")
        self.assertTrue(result["stop_check"]["passed"])
        self.assertEqual(result["usage"]["model_api_calls"], 2)  # Fake responses, not an actual API measurement.
        self.assertEqual(result["usage"]["token_usage"]["total_tokens"], 240)
        self.assertEqual(result["usage"]["format_recovery"]["status"], "parsed")
        self.assertEqual(result["usage"]["format_recovery"]["correction_requests"], 1)
        self.assertEqual(transport.requests[1]["messages"][-1], {"role": "user", "content": REPAIR_MESSAGE})
        self.assertNotIn("RAW_SECRET", json.dumps(result) + json.dumps(transport.requests[1]["messages"]))
        snapshot = model.usage_snapshot()
        snapshot["format_recovery"]["diagnostics"].clear()
        self.assertEqual(len(model.format_diagnostics), 1)

    def test_second_format_failure_stops_after_one_correction(self):
        result, transport, _ = self.run_responses([response("bad"), response("still bad"), response(json.dumps(CLARIFY))])
        self.assertEqual(result["reason"], "model_invalid_json")
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(len(result["usage"]["format_recovery"]["diagnostics"]), 2)
        self.assertEqual(result["usage"]["format_recovery"]["status"], "stopped")

    def test_total_call_budget_blocks_correction_before_network(self):
        result, transport, _ = self.run_responses([response("bad")], limits=ModelLimits(max_api_calls=1))
        self.assertEqual(result["reason"], "model_budget_exhausted")
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(result["usage"]["format_recovery"]["correction_requests"], 0)

    def test_correction_can_be_disabled_without_losing_diagnostics(self):
        result, transport, _ = self.run_responses([response("bad")], limits=ModelLimits(max_format_corrections=0))
        self.assertEqual(result["reason"], "model_invalid_json")
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(len(result["usage"]["format_recovery"]["diagnostics"]), 1)

    def test_new_correction_prompt_rechecks_input_size_budget(self):
        initial = {"messages": [{"role": "system", "content": "system\n" + ADAPTER_PROMPT}, MESSAGES[1]],
                   "tools": [], "max_tokens": 768}
        size = len(json.dumps(initial, ensure_ascii=False).encode("utf-8"))
        transport = Responses([response("bad")])
        model = DeepSeekModel(CONFIG, limits=ModelLimits(max_input_bytes=size + 1), transport=transport)
        with self.assertRaises(ModelFailure) as error:
            model.next_action(MESSAGES, [])
        self.assertEqual(error.exception.reason, "model_input_budget")
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(model.format_corrections, 0)

    def test_correction_rechecks_total_time_before_network(self):
        clock = [0.0]
        calls = []
        def slow_response(*_):
            calls.append(True)
            clock[0] = 2.0
            return response("bad")
        model = DeepSeekModel(CONFIG, limits=ModelLimits(total_seconds=1), transport=slow_response)
        with patch("supportlab.deepseek.perf_counter", side_effect=lambda: clock[0]), self.assertRaises(ModelFailure) as error:
            model.next_action(MESSAGES, [])
        self.assertEqual(error.exception.reason, "model_deadline")
        self.assertEqual(len(calls), 1)

    def test_transport_errors_and_truncation_do_not_trigger_format_repair(self):
        for row, reason in ((ModelFailure("model_timeout"), "model_timeout"),
                            ({"status": "error", "error": "model_auth_error"}, "model_auth_error"),
                            (response("partial", finish="length"), "model_incomplete_output")):
            with self.subTest(reason=reason):
                result, transport, _ = self.run_responses([row])
                self.assertEqual(result["reason"], reason)
                self.assertEqual(len(transport.requests), 1)
                self.assertEqual(result["usage"]["format_recovery"]["status"], "not_needed")

    def test_transport_failure_during_correction_has_incomplete_usage(self):
        result, transport, _ = self.run_responses([response("bad"), ModelFailure("model_timeout")])
        self.assertEqual(result["reason"], "model_timeout")
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(result["usage"]["token_usage"]["total_tokens"], 120)
        self.assertFalse(result["usage"]["usage_complete"])
        self.assertEqual(result["usage"]["format_recovery"]["stop_reason"], "model_timeout")

    def test_corrected_prose_and_native_calls_cannot_execute_tools(self):
        embedded = {"type": "tool_call", "call_id": "smuggled", "name": "get_request_record",
                    "arguments": {"request_id": "req_demo_018"}}
        for corrected in (response(json.dumps(embedded)),
                          response(None, finish="tool_calls", calls=[native("extra", "get_account_state")])):
            with self.subTest(corrected=corrected):
                result, transport, _ = self.run_responses([response("bad"), corrected])
                self.assertEqual(result["reason"], "model_repair_action_not_allowed")
                self.assertEqual(result["usage"]["tool_calls"], 0)
                self.assertEqual(len(transport.requests), 2)

    def test_format_recovery_does_not_bypass_evidence_or_legacy_checks(self):
        for corrected, expected in ((CLARIFY, "unsupported_resolution"),
                                    ({"type": "final", "outcome": "diagnosis", "message": "立即恢复", "citations": []}, "unsupported_assessment")):
            result, transport, _ = self.run_responses([response("bad"), response(json.dumps(corrected))], target="req_demo_018")
            self.assertEqual(result["reason"], expected)
            self.assertEqual(result["usage"]["format_recovery"]["status"], "parsed")
            self.assertEqual(len(transport.requests), 2)
            self.assertNotIn("立即恢复", result["answer"])

    def test_valid_but_unsupported_answer_does_not_start_correction(self):
        result, transport, _ = self.run_responses([response(json.dumps(CLARIFY))], target="req_demo_018")
        self.assertEqual(result["reason"], "unsupported_resolution")
        self.assertEqual(len(transport.requests), 1)
        self.assertEqual(result["usage"]["format_recovery"]["status"], "not_needed")

    def test_collected_evidence_and_native_pairing_survive_correction(self):
        transport = Responses([
            response(None, finish="tool_calls", calls=[native("r", "get_request_record"), native("a", "get_account_state")]),
            response("bad"), response(json.dumps(HANDOFF))])
        with TemporaryDirectory() as directory:
            db, docs = prepare_case({"setup": "missing_snapshot", "request_id": "req_demo_018"}, Path(directory))
            result = run_loop(DeepSeekModel(CONFIG, transport=transport), ToolExecutor(AppContext("acc_demo"), db_path=db, docs_root=docs),
                              user_message="排查 req_demo_018", target_request_id="req_demo_018")
        self.assertTrue(result["stop_check"]["passed"])
        self.assertEqual(result["usage"]["tool_calls"], 2)
        self.assertEqual(result["usage"]["model_api_calls"], 3)
        self.assertEqual(result["usage"]["decision_turns"], 3)
        self.assertEqual([row for row in transport.requests[1]["messages"] if row["role"] == "tool"],
                         [row for row in transport.requests[2]["messages"] if row["role"] == "tool"])
        self.assertEqual([row["tool_call_id"] for row in transport.requests[2]["messages"] if row["role"] == "tool"], ["r", "a"])
        self.assertNotIn("1000/1000", result["answer"])


class RecoveryDemoTests(unittest.TestCase):
    def test_six_replays_are_offline_and_report_zero_real_usage(self):
        with patch("supportlab.configuration.load_deepseek_config") as config, patch("supportlab.deepseek.call_worker") as network:
            report = run_recovery_demo()
        config.assert_not_called()
        network.assert_not_called()
        self.assertEqual((report["expected_behavior_count"], report["case_count"]), (6, 6))
        for case in report["cases"]:
            result = case["result"]
            self.assertEqual(result["execution_mode"], "offline_adapter")
            self.assertFalse(result["model_called"])
            self.assertEqual(result["usage"]["model_api_calls"], 0)
            self.assertIsNone(result["usage"]["token_usage"])

    def test_cli_runs_offline_and_rejects_unsafe_report_path(self):
        for arguments, expected in ((["recovery-demo", "--case", "still-invalid"], 0),
                                    (["recovery-demo", "--output", ".env.local"], 2)):
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(arguments), expected)
            if expected == 0:
                self.assertIn("真实模型调用 0 次", output.getvalue())
                self.assertIn("模型请求 3 次", output.getvalue())


if __name__ == "__main__":
    unittest.main()
