from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evals.stability import Artifacts, RequestGate, main, make_plan, run_batch, summarize
from supportlab.agent import ModelFailure
from supportlab.configuration import DeepSeekConfig


def response():
    return {"status": "ok", "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}}


def row(case, repeat, *, passed=True):
    return {"id": case["id"], "repeat": repeat, "grade": {"passed": passed}, "source_sha256": "same",
            "result": {"outcome": "diagnosis", "reason": "final_reply", "usage": {"tool_calls": 3}}}


class RequestGateTests(unittest.TestCase):
    def test_reservation_is_durable_before_dispatch_and_cap_is_global(self):
        events, dispatches = [], []
        def transport(payload, timeout):
            self.assertEqual(events[-1][0], "request_reserved")
            dispatches.append(payload)
            return response()
        gate = RequestGate(2, 30, lambda kind, **fields: events.append((kind, deepcopy(fields))), transport=transport)
        gate({"config": {"api_key": "never-log-me"}}, 25)
        gate({}, 25)
        with self.assertRaises(ModelFailure):
            gate({}, 25)
        self.assertEqual(len(dispatches), 2)
        self.assertEqual(gate.usage()["token_usage"]["total_tokens"], 30)
        self.assertNotIn("never-log-me", json.dumps(events))

    def test_uncertain_transport_spends_one_request_and_stops(self):
        def transport(*args):
            raise RuntimeError("secret-error-body")
        events = []
        gate = RequestGate(8, 30, lambda kind, **fields: events.append(fields), transport=transport)
        with self.assertRaises(ModelFailure):
            gate({}, 25)
        self.assertEqual(gate.usage()["model_api_calls"], 1)
        self.assertFalse(gate.usage()["usage_complete"])
        self.assertFalse(gate.can_start_run())
        self.assertNotIn("secret-error-body", json.dumps(events))

    def test_missing_usage_stops_next_dispatch(self):
        calls = []
        def transport(*args):
            calls.append(1)
            return {"status": "ok"}
        gate = RequestGate(8, 30, lambda *args, **kwargs: None, transport=transport)
        gate({}, 25)
        with self.assertRaises(ModelFailure):
            gate({}, 25)
        self.assertEqual(len(calls), 1)
        self.assertEqual(gate.stop_reason, "usage_unknown")

    def test_auth_failure_stops_batch_without_retry(self):
        gate = RequestGate(8, 30, lambda *a, **k: None,
                           transport=lambda *a: {"status": "error", "error": "model_auth_error"})
        gate({}, 25)
        self.assertEqual(gate.stop_reason, "model_auth_error")
        self.assertFalse(gate.can_start_run())

    def test_batch_deadline_bounds_transport_timeout(self):
        clock_value, timeouts = [0], []
        gate = RequestGate(8, 10, lambda *a, **k: None, clock=lambda: clock_value[0],
                           transport=lambda payload, timeout: timeouts.append(timeout) or response())
        clock_value[0] = 9
        gate({}, 25)
        self.assertEqual(timeouts, [1])
        clock_value[0] = 11
        with self.assertRaises(ModelFailure):
            gate({}, 25)
        self.assertEqual(len(timeouts), 1)

    def test_reserves_full_per_run_capacity(self):
        gate = RequestGate(6, 30, lambda *a, **k: None, transport=lambda *a: response())
        self.assertTrue(gate.can_start_run())
        for _ in range(3):
            gate({}, 25)
        self.assertFalse(gate.can_start_run())
        self.assertEqual(gate.stop_reason, "remaining_budget_below_per_run_cap")


class StabilityRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch("evals.stability.Artifacts", side_effect=lambda output: Artifacts(output, root=self.root)))
        self.patches.enter_context(patch("evals.stability.verify_release"))
        self.patches.enter_context(patch("evals.stability.file_hash", return_value="frozen"))

    def test_plan_is_zero_cost_and_has_two_separate_rounds(self):
        plan = make_plan("all")
        self.assertEqual((plan["case_count"], plan["planned_runs"], plan["max_api_calls"]), (8, 16, 64))
        self.assertEqual([r["case"]["id"] for r in plan["schedule"][:8]],
                         list(reversed([r["case"]["id"] for r in plan["schedule"][8:]])))
        with patch("evals.stability.load_deepseek_config") as loader, redirect_stdout(StringIO()):
            self.assertEqual(main(["--plan", "--scope", "gaps"]), 0)
        loader.assert_not_called()
        self.assertEqual(make_plan("gaps")["max_api_calls"], 32)

    def test_live_without_explicit_cap_is_rejected_before_credentials(self):
        with patch("evals.stability.load_deepseek_config") as loader:
            for cap in (None, 0, True, 65):
                with self.assertRaises(ValueError):
                    run_batch(self.root / "never.json", live=True, max_api_calls=cap)
            loader.assert_not_called()
        self.assertFalse((self.root / "never.json").exists())

    def test_existing_report_or_journal_is_never_overwritten(self):
        for suffix in (".json", ".journal.jsonl"):
            output = self.root / ("old" + suffix)
            output.write_text("keep", encoding="utf-8")
            with patch("evals.stability.load_deepseek_config") as loader:
                with self.assertRaises(FileExistsError):
                    run_batch(self.root / "old.json", live=True, max_api_calls=64)
                loader.assert_not_called()
            self.assertEqual(output.read_text(), "keep")
            output.unlink()

    def test_offline_never_loads_credentials_or_dispatches_network(self):
        with patch("evals.stability.load_deepseek_config") as loader, patch("evals.stability.one_run", side_effect=
                    lambda case, repeat, **kw: row(case, repeat)):
            result = run_batch(self.root / "offline.json")
        loader.assert_not_called()
        self.assertEqual(result["usage"]["model_api_calls"], 0)
        self.assertEqual(result["summary"]["completed_runs"], 16)
        self.assertEqual(result["status"], "completed")

    def test_interruption_preserves_completed_rows_and_request_reservation(self):
        counter = []
        def once(case, repeat, **kwargs):
            counter.append(1)
            if len(counter) == 2:
                gate = kwargs["gate"]
                gate.transport = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
                gate({}, 25)
            return row(case, repeat)
        with patch("evals.stability.load_deepseek_config", return_value=DeepSeekConfig("fake-test-key")), patch("evals.stability.one_run", side_effect=once):
            report = run_batch(self.root / "partial.json", live=True, max_api_calls=64)
        saved = json.loads((self.root / "partial.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "interrupted")
        self.assertEqual(saved["summary"]["completed_runs"], 1)
        self.assertEqual(saved["usage"]["model_api_calls"], 1)
        self.assertFalse(saved["usage"]["usage_complete"])
        journal = (self.root / "partial.journal.jsonl").read_text(encoding="utf-8")
        self.assertIn("request_reserved", journal)
        self.assertIn("run_completed", journal)
        self.assertNotIn("fake-test-key", journal)
        self.assertEqual(report, saved)

    def test_budget_stop_counts_unrun_cases_separately(self):
        def once(case, repeat, **kwargs):
            gate = kwargs["gate"]
            gate.transport = lambda *args: response()
            for _ in range(4):
                gate({}, 25)
            return row(case, repeat)
        with patch("evals.stability.load_deepseek_config", return_value=DeepSeekConfig("fake-test-key")), patch("evals.stability.one_run", side_effect=once):
            report = run_batch(self.root / "cap.json", live=True, max_api_calls=4)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["summary"]["completed_runs"], 1)
        self.assertEqual(report["summary"]["planned_runs"], 16)
        self.assertFalse(report["summary"]["per_case"]["expired"]["all_repeats_passed"])
        self.assertEqual(report["usage"]["model_api_calls"], 4)

    def test_unexpected_failure_keeps_prior_results_and_redacts_exception(self):
        with patch("evals.stability.one_run", side_effect=RuntimeError("secret-never-show")):
            report = run_batch(self.root / "error.json")
        self.assertEqual(report["status"], "error")
        self.assertEqual(report["error_type"], "RuntimeError")
        self.assertNotIn("secret-never-show", (self.root / "error.json").read_text(encoding="utf-8"))

    def test_consistently_wrong_is_not_stable_success(self):
        plan = make_plan("gaps")
        case = plan["schedule"][0]["case"]
        summary = summarize([row(case, 1, passed=False), row(case, 2, passed=False)], plan)
        self.assertTrue(summary["per_case"]["expired"]["same_outcome_signature"])
        self.assertFalse(summary["per_case"]["expired"]["all_repeats_passed"])


if __name__ == "__main__":
    unittest.main()
