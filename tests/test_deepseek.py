from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import httpx2
import openai

from supportlab.__main__ import main
from supportlab.agent import ModelFailure, run_loop
from supportlab.configuration import ConfigurationError, DeepSeekConfig, load_deepseek_config
from supportlab.deepseek import DeepSeekModel, ModelLimits, call_worker
from supportlab.execution import ToolExecutor
from supportlab.model_worker import request_once
from supportlab.retrieval import search_docs
from supportlab.storage import init_demo_db
from supportlab.tools import AppContext


CONFIG = DeepSeekConfig("test-key-never-real")
MESSAGES = [{"role": "system", "content": "test system"}, {"role": "user", "content": "req_demo_018"}]


def tool(identifier, name, arguments):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def reply(*, calls=None, content=None, finish=None, usage=True):
    return {"status": "ok", "message": {"content": content, "tool_calls": calls or []},
            "finish_reason": finish or ("tool_calls" if calls else "stop"),
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120} if usage else None}


def final(outcome="clarify", message="请提供编号。", citations=None):
    return json.dumps({"type": "final", "outcome": outcome, "message": message, "citations": citations or []}, ensure_ascii=False)


def clarify_missing_id():
    return json.dumps({"type": "final", "outcome": "clarify", "resolution": {"reason": "missing_request_id"}})


def diagnosis(citation, request_evidence="E1", account_evidence="E2"):
    return json.dumps({"type": "final", "outcome": "diagnosis", "assessment": {
        "cause": "monthly_quota_exhausted", "request_evidence_id": request_evidence, "account_evidence_id": account_evidence,
        "rule_citation": citation, "follow_up": ["check_current_status", "verify_quota_options"]}})


class SequenceTransport:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def __call__(self, payload, timeout):
        self.requests.append(deepcopy(payload))
        return next(self.responses)


class ConfigurationTests(unittest.TestCase):
    def test_quotes_environment_precedence_and_secret_not_in_repr(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env.local"
            path.write_text('# local\nDEEPSEEK_API_KEY="test-key-local"\nDEEPSEEK_MODEL=deepseek-flash\nUNRELATED_KEY=unused\n', encoding="utf-8")
            config = load_deepseek_config(path, environ={"DEEPSEEK_API_KEY": "test-key-environment"})
            self.assertEqual(config.api_key, "test-key-environment")
            self.assertNotIn("test-key", repr(config))
            self.assertNotIn("test-key", json.dumps(config.safe_summary()))

    def test_missing_key_external_endpoint_and_invalid_model_are_rejected(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing"
            for values in ({}, {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_BASE_URL": "https://example.com"},
                           {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_MODEL": "a secret invalid model"}):
                with self.subTest(values=list(values)):
                    with self.assertRaises(ConfigurationError):
                        load_deepseek_config(path, environ=values)

    def test_duplicate_fields_are_rejected_without_echoing_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config"
            path.write_text("DEEPSEEK_API_KEY=test-key-first\nDEEPSEEK_API_KEY=test-key-second\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError) as error:
                load_deepseek_config(path, environ={})
            self.assertNotIn("test-key", str(error.exception))


class DeepSeekAdapterTests(unittest.TestCase):
    def test_native_tools_feedback_and_real_usage_reach_shared_loop(self):
        citation = search_docs("MONTHLY_QUOTA_EXHAUSTED", "demo-api-v1", 1)["matches"][0]["citation"]
        transport = SequenceTransport([
            reply(calls=[tool("one", "get_request_record", {"request_id": "req_demo_018"}),
                         tool("two", "get_account_state", {"request_id": "req_demo_018"})]),
            reply(calls=[tool("three", "search_docs", {"query": "MONTHLY_QUOTA_EXHAUSTED", "service_version": "demo-api-v1", "top_k": 1})]),
            reply(content=diagnosis(citation)),
        ])
        with TemporaryDirectory() as directory:
            db = Path(directory) / "demo.sqlite3"
            init_demo_db(db)
            result = run_loop(DeepSeekModel(CONFIG, transport=transport), ToolExecutor(AppContext("acc_demo"), db_path=db),
                              user_message="排查 req_demo_018", target_request_id="req_demo_018")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["execution_mode"], "live")  # Fake transport only; not a real API test.
        self.assertEqual(result["usage"]["model_api_calls"], 3)
        self.assertEqual(result["usage"]["tool_calls"], 3)
        self.assertEqual(result["usage"]["decision_turns"], 4)
        self.assertEqual(result["usage"]["token_usage"]["total_tokens"], 360)
        self.assertTrue(result["answer_check"]["passed"])
        self.assertEqual(result["structured_answer"]["diagnosis"]["cause"], "monthly_quota_exhausted")
        messages = transport.requests[1]["messages"]
        self.assertEqual(len(messages[2]["tool_calls"]), 2)
        self.assertEqual([row["tool_call_id"] for row in messages if row["role"] == "tool"], ["one", "two"])
        self.assertEqual(json.loads(messages[-1]["content"])["account_state"]["monthly_quota_used"], 1000)
        self.assertEqual([json.loads(row["content"])["evidence_id"] for row in messages if row["role"] == "tool"], ["E1", "E2"])
        self.assertEqual(result["answer_check"]["references"], {"request_evidence_id": "E1", "account_evidence_id": "E2"})
        self.assertNotIn(CONFIG.api_key, json.dumps(result))

    def test_malformed_tool_arguments_are_feedback_not_executed(self):
        bad = tool("one", "get_request_record", {})
        bad["function"]["arguments"] = "{bad json"
        transport = SequenceTransport([reply(calls=[bad]), reply(content=clarify_missing_id())])
        with patch("supportlab.execution.subprocess.run") as worker:
            result = run_loop(DeepSeekModel(CONFIG, transport=transport), ToolExecutor(AppContext("acc_demo")), user_message="test")
        worker.assert_not_called()
        self.assertEqual(result["status"], "needs_input")
        self.assertEqual(json.loads(transport.requests[1]["messages"][-1]["content"])["status"], "invalid_arguments")

    def test_call_budget_blocks_next_network_request(self):
        transport = SequenceTransport([reply(calls=[tool("one", "get_request_record", {"request_id": "BAD"})])])
        model = DeepSeekModel(CONFIG, limits=ModelLimits(max_api_calls=1), transport=transport)
        result = run_loop(model, ToolExecutor(AppContext("acc_demo")), user_message="test")
        self.assertEqual(result["reason"], "model_budget_exhausted")
        self.assertEqual(len(transport.requests), 1)

    def test_input_size_and_deadline_block_network(self):
        transport = SequenceTransport([])
        model = DeepSeekModel(CONFIG, limits=ModelLimits(max_input_bytes=1024), transport=transport)
        with self.assertRaises(ModelFailure) as failure:
            model.next_action(MESSAGES, [])
        self.assertEqual(failure.exception.reason, "model_input_budget")
        model = DeepSeekModel(CONFIG, transport=transport)
        model.started = 0
        with self.assertRaises(ModelFailure) as failure:
            model.next_action(MESSAGES, [])
        self.assertEqual(failure.exception.reason, "model_deadline")
        self.assertEqual(len(transport.requests), 0)

    def test_truncated_or_disabled_correction_stops_on_first_error(self):
        for response, reason in ((reply(content=final(), finish="length"), "model_incomplete_output"),
                                 (reply(content=""), "model_invalid_json"),
                                 (reply(content="not json"), "model_invalid_json")):
            with self.subTest(reason=reason):
                result = run_loop(DeepSeekModel(CONFIG, limits=ModelLimits(max_format_corrections=0),
                                                transport=SequenceTransport([response])),
                                  ToolExecutor(AppContext("acc_demo")), user_message="test")
                self.assertEqual(result["reason"], reason)
                self.assertEqual(result["usage"]["model_api_calls"], 1)

    def test_error_labels_and_missing_usage_are_not_fabricated(self):
        for response in ({"status": "error", "error": "model_auth_error"},
                         {"status": "error", "error": "raw-secret-unrecognized"},
                         reply(content=final(), usage=False)):
            with self.subTest(status=response["status"]):
                result = run_loop(DeepSeekModel(CONFIG, transport=SequenceTransport([response])),
                                  ToolExecutor(AppContext("acc_demo")), user_message="test")
                self.assertIsNone(result["usage"]["token_usage"])
                self.assertFalse(result["usage"]["usage_complete"])
                self.assertNotIn("raw-secret", json.dumps(result))

    def test_diagnosis_with_only_document_evidence_is_rejected(self):
        citation = search_docs("MONTHLY_QUOTA_EXHAUSTED", "demo-api-v1", 1)["matches"][0]["citation"]
        transport = SequenceTransport([
            reply(calls=[tool("one", "search_docs", {"query": "MONTHLY_QUOTA_EXHAUSTED", "service_version": "demo-api-v1", "top_k": 1})]),
            reply(content=diagnosis(citation, request_evidence="E9")),
        ])
        result = run_loop(DeepSeekModel(CONFIG, transport=transport), ToolExecutor(AppContext("acc_demo")),
                          user_message="test", target_request_id="req_demo_018")
        self.assertEqual(result["reason"], "unsupported_assessment")
        self.assertEqual(result["answer_check"]["issues"], ["missing_evidence"])

    def test_legacy_free_text_diagnosis_cannot_bypass_new_guard(self):
        transport = SequenceTransport([reply(content=final("diagnosis", "申请额外配额即可恢复", []))])
        result = run_loop(DeepSeekModel(CONFIG, transport=transport), ToolExecutor(AppContext("acc_demo")),
                          user_message="排查 req_demo_018", target_request_id="req_demo_018")
        self.assertEqual(result["answer_check"]["issues"], ["unstructured_diagnosis"])
        self.assertNotIn("即可恢复", result["answer"])

    def test_duplicate_batch_ids_are_rejected(self):
        call = tool("same", "get_request_record", {"request_id": "req_demo_018"})
        result = run_loop(DeepSeekModel(CONFIG, transport=SequenceTransport([reply(calls=[call, call])])),
                          ToolExecutor(AppContext("acc_demo")), user_message="test")
        self.assertEqual(result["reason"], "model_protocol_error")
        self.assertEqual(result["usage"]["tool_calls"], 0)

    def test_worker_timeout_and_exception_are_sanitized(self):
        for exception, reason in ((subprocess.TimeoutExpired("worker", 1), "model_timeout"),
                                  (OSError("secret exception"), "model_worker_error")):
            with patch("supportlab.deepseek.subprocess.run", side_effect=exception):
                with self.assertRaises(ModelFailure) as failure:
                    call_worker({}, 1)
            self.assertEqual(failure.exception.reason, reason)
            self.assertNotIn("secret", str(failure.exception))

    def test_invalid_limits_rejected(self):
        for values in ({"max_api_calls": 0}, {"max_api_calls": True}, {"max_output_tokens": 100000},
                       {"request_timeout_seconds": float("nan")}, {"total_seconds": -1},
                       {"max_format_corrections": 2}, {"max_format_corrections": True}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ModelLimits(**values)


class SDKBoundaryTests(unittest.TestCase):
    def test_sdk_request_shape_non_thinking_and_no_reasoning_in_output(self):
        captured = []

        def handler(request):
            captured.append(json.loads(request.content))
            self.assertEqual(request.url.host, "api.deepseek.com")
            self.assertEqual(request.headers["authorization"], "Bearer test-key-never-real")
            return httpx2.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                "model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": final(), "reasoning_content": "DO_NOT_EXPORT"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}})

        def factory(**kwargs):
            self.assertEqual(kwargs["max_retries"], 0)
            self.assertEqual(kwargs["timeout"], 20)
            return openai.OpenAI(**kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))

        output = request_once({"config": {"api_key": CONFIG.api_key, "model": CONFIG.model, "base_url": CONFIG.base_url},
                               "messages": MESSAGES, "tools": [], "max_tokens": 768}, client_factory=factory)
        self.assertEqual(output["status"], "ok")
        self.assertEqual(captured[0]["thinking"], {"type": "disabled"})
        self.assertEqual(captured[0]["max_tokens"], 768)
        self.assertNotIn("tools", captured[0])
        self.assertNotIn("tool_choice", captured[0])
        self.assertNotIn("DO_NOT_EXPORT", json.dumps(output))

    def test_sdk_auth_failure_does_not_echo_server_body(self):
        def factory(**kwargs):
            transport = httpx2.MockTransport(lambda request: httpx2.Response(401, json={"error": {"message": "RAW_SECRET_ERROR"}}))
            return openai.OpenAI(**kwargs, http_client=httpx2.Client(transport=transport))
        output = request_once({"config": {"api_key": CONFIG.api_key}, "messages": MESSAGES,
                               "tools": [], "max_tokens": 768}, client_factory=factory)
        self.assertEqual(output, {"status": "error", "error": "model_auth_error"})

    def test_cli_config_and_bad_report_path_do_not_call_model(self):
        for arguments in (["check-model-config"], ["agent", "req_demo_018", "--output", ".env.local"]):
            output = StringIO()
            with patch("supportlab.configuration.load_deepseek_config", return_value=CONFIG), \
                    patch("supportlab.deepseek.DeepSeekModel") as model, redirect_stdout(output), redirect_stderr(StringIO()):
                code = main(arguments)
            model.assert_not_called()
            self.assertNotIn(CONFIG.api_key, output.getvalue())
            self.assertEqual(code, 0 if arguments[0] == "check-model-config" else 2)


if __name__ == "__main__":
    unittest.main()
