"""Regression derived from exposed holdout failures: consistent usage on all exits."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from supportlab.agent import ModelFailure
from supportlab.knowledge import build_index
from supportlab.rag import document_answer


class FakeSelectionModel:
    execution_mode = "offline_adapter"

    def __init__(self, failure=None):
        self.failure = failure

    def next_action(self, messages, tools):
        if tools != []:
            raise AssertionError("RAG must expose no executable tools")
        if self.failure:
            raise ModelFailure(self.failure)
        return {"type": "final", "outcome": "docs", "citations": []}

    def usage_snapshot(self):
        return {"model_api_calls": 0, "token_usage": None, "simulated_model_requests": 1}


class RagUsageRegressionTests(unittest.TestCase):
    def test_usage_includes_zero_tools_for_success_no_evidence_and_bad_index(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.sqlite3"
            build_index(index_path=path)
            with patch("supportlab.configuration.load_deepseek_config", side_effect=AssertionError("offline only")):
                success = document_answer("CREDENTIAL_REVOKED", index_path=path)
                absent = document_answer("量子纠缠", index_path=path)
                unavailable = document_answer("429", index_path=path.with_name("missing.sqlite3"))
            self.assertEqual([row["outcome"] for row in (success, absent, unavailable)], ["docs", "clarify", "clarify"])
            for result in (success, absent, unavailable):
                self.assertIs(type(result["usage"]["tool_calls"]), int)
                self.assertEqual(result["usage"]["tool_calls"], 0)
                self.assertEqual(result["usage"]["model_api_calls"], 0)

    def test_model_abstention_or_failure_keeps_explicit_zero_without_erasing_usage(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.sqlite3"
            build_index(index_path=path)
            for failure, expected in ((None, "clarify"), ("model_timeout", "handoff")):
                result = document_answer("429", model=FakeSelectionModel(failure), index_path=path)
                self.assertEqual(result["outcome"], expected)
                self.assertEqual(result["usage"]["tool_calls"], 0)
                self.assertEqual(result["usage"]["simulated_model_requests"], 1)
