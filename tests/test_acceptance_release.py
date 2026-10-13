"""Candidate harness checks; never execute or tune on the supplemental cases."""

import unittest

from supportlab.acceptance_release import classification, summarize, validate_delta


class CandidateHarnessTests(unittest.TestCase):
    def test_original_suite_replay_is_always_regression(self):
        self.assertEqual(classification("regression"), "regression_after_holdout_exposure")
        self.assertEqual(classification("holdout"), "prospective_composition_holdout")
        with self.assertRaises(ValueError):
            classification("all-pass")

    def test_only_explicit_rag_delta_is_allowed(self):
        original = {"supportlab/rag.py": "original", "supportlab/agent.py": "fixed"}
        self.assertEqual(validate_delta(original, {**original, "supportlab/rag.py": "candidate"}), ["supportlab/rag.py"])
        for candidate in (original, {**original, "supportlab/agent.py": "changed"},
                          {**original, "supportlab/rag.py": "candidate", "supportlab/extra.py": "new"}):
            with self.assertRaises(ValueError):
                validate_delta(original, candidate)

    def test_summary_keeps_failures_and_denominators(self):
        steps = [{"step": 1, "grade": {"passed": True, "checks": {"a": True}}, "output": {}},
                 {"step": 2, "grade": {"passed": False, "checks": {"a": True, "b": False}}, "output": {}}]
        report = summarize([{"id": "synthetic", "family": "harness-only", "passed": False, "steps": steps}])
        self.assertEqual(report["passed_count"], 0)
        self.assertEqual(report["case_count"], 1)
        self.assertEqual(report["step_checks"], {"numerator": 1, "denominator": 2})
        self.assertEqual(report["failures"][0]["failed_checks"], ["b"])
