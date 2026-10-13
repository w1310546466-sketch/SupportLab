"""Bounded, journaled repeated development runs; frozen application stays unchanged."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from supportlab.acceptance import file_hash, verify_map
from supportlab.acceptance_release import verify_candidate
from supportlab.agent import ModelFailure, SYSTEM_MESSAGE, run_loop
from supportlab.comparison import RuleReplayAdapter
from supportlab.configuration import load_deepseek_config
from supportlab.deepseek import ADAPTER_PROMPT, DeepSeekModel, ModelLimits, call_worker
from supportlab.diagnostic_eval import grade_diagnostic, prepare_diagnostic_case
from supportlab.evaluation import RecordingExecutor, fingerprint
from supportlab.storage import PROJECT_ROOT

DATASET = PROJECT_ROOT / "evals/stability_cases_v1.json"
SEAL = PROJECT_ROOT / "evals/stability_v1.seal.json"
GAPS = {"expired", "permission", "incident", "timeout"}
LIMITS = ModelLimits()
NETWORK_FAILURES = {"model_auth_error", "model_rate_limit", "model_timeout", "model_connection_error",
                    "model_http_error", "model_worker_error", "model_configuration_error"}


def now():
    return datetime.now(timezone.utc).isoformat()


def verify_release():
    verify_candidate()
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    required = {"evals/stability.py", "evals/stability_cases_v1.json", "tests/test_stability.py"}
    if seal.get("version") != "stability-seal-v1" or set(seal.get("files", {})) != required:
        raise ValueError("Invalid stability seal")
    verify_map(seal["files"])


def make_plan(scope="all"):
    if scope not in ("all", "gaps"):
        raise ValueError("Unknown scope")
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    if data["repetitions"] != 2 or len(data["cases"]) != 8 or data["split"] != "development_repeated":
        raise ValueError("Invalid repeated development dataset")
    cases = [row for row in data["cases"] if scope == "all" or row["id"] in GAPS]
    # Reverse the second round; this is disclosed fixed order, not randomization.
    schedule = [{"case": case, "repeat": repeat} for repeat in (1, 2)
                for case in (cases if repeat == 1 else list(reversed(cases)))]
    return {"version": "stability-runner-v1", "dataset_version": data["dataset_version"], "scope": scope,
            "simulation": True, "split": data["split"], "case_count": len(cases), "repetitions": 2,
            "planned_runs": len(schedule), "max_api_calls": len(schedule) * LIMITS.max_api_calls,
            "max_output_tokens_if_all_requests": len(schedule) * LIMITS.max_api_calls * LIMITS.max_output_tokens,
            "per_run_limits": asdict(LIMITS), "schedule": schedule,
            "order": "forward_then_reverse; fresh state every run", "cost": None,
            "scope_excludes": "RAG, multi-turn conversation and workflow comparison"}


class Artifacts:
    """Own new files only. JSONL survives interruption between summary rewrites."""
    def __init__(self, output, *, root=PROJECT_ROOT / "reports"):
        self.path = Path(output).resolve()
        self.journal_path = self.path.with_suffix(".journal.jsonl")
        if self.path.parent != root.resolve() or self.path.suffix != ".json":
            raise ValueError("Use a new reports/*.json file")
        if self.path.exists() or self.journal_path.exists():
            raise FileExistsError("Report or journal already exists")

    def __enter__(self):
        self.journal = self.journal_path.open("x", encoding="utf-8")
        try:
            self.summary = self.path.open("x", encoding="utf-8")
        except BaseException:
            self.journal.close()
            raise
        return self

    def event(self, kind, **fields):
        self.journal.write(json.dumps({"event": kind, "time": now(), **fields}, ensure_ascii=False) + "\n")
        self.journal.flush()
        os.fsync(self.journal.fileno())

    def save(self, report):
        self.summary.seek(0)
        self.summary.truncate()
        json.dump(report, self.summary, ensure_ascii=False, indent=2)
        self.summary.write("\n")
        self.summary.flush()
        os.fsync(self.summary.fileno())

    def __exit__(self, *args):
        self.summary.close()
        self.journal.close()


class RequestGate:
    """Count and journal BEFORE dispatch; failed/uncertain requests spend budget."""
    def __init__(self, cap, seconds, event, *, transport=call_worker, clock=perf_counter):
        self.cap, self.event, self.transport, self.clock = cap, event, transport, clock
        self.deadline = clock() + seconds
        self.calls = []
        self.active = None
        self.stop_reason = None

    def can_start_run(self):
        if self.stop_reason:
            return False
        if self.clock() >= self.deadline:
            self.stop_reason = "batch_deadline"
            return False
        if len(self.calls) + LIMITS.max_api_calls > self.cap:
            self.stop_reason = "remaining_budget_below_per_run_cap"
            return False
        return True

    def __call__(self, payload, timeout):
        remaining = self.deadline - self.clock()
        if remaining <= 0:
            self.stop_reason = "batch_deadline"
            raise ModelFailure("model_deadline")
        if len(self.calls) >= self.cap or self.stop_reason:
            raise ModelFailure("model_budget_exhausted")
        record = {"request": len(self.calls) + 1, "run": self.active, "status": "reserved", "tokens": None}
        self.calls.append(record)
        # No prompts, response prose, credentials or provider error bodies enter this journal.
        self.event("request_reserved", **record)
        try:
            response = self.transport(payload, min(timeout, remaining))
        except BaseException as error:
            record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "transport_exception"
            self.stop_reason = record["status"]
            self.event("request_finished", **record)
            if not isinstance(error, Exception):
                raise
            raise ModelFailure("model_worker_error") from None
        record["status"] = "response_received"
        usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0
                                          for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            record["tokens"] = {"input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
                                "total_tokens": usage["total_tokens"]}
        error = response.get("error") if isinstance(response, dict) else None
        if error in NETWORK_FAILURES:
            self.stop_reason = error
        elif record["tokens"] is None:
            self.stop_reason = "usage_unknown"
        self.event("request_finished", **record)
        return response if isinstance(response, dict) else {"status": "error", "error": "model_protocol_error"}

    def usage(self):
        known = [row["tokens"] for row in self.calls if row["tokens"] is not None]
        return {"model_api_calls": len(self.calls), "token_usage":
                {key: sum(row[key] for row in known) for key in ("input_tokens", "output_tokens", "total_tokens")} if known else None,
                "usage_complete": bool(self.calls) and len(known) == len(self.calls)}


def one_run(case, repeat, *, live, config, gate):
    with TemporaryDirectory(prefix="supportlab_stability_") as directory:
        root = Path(directory)
        db, docs = prepare_diagnostic_case(case, root)
        sources = [db, *docs.glob("*.md")]
        source_hash = fingerprint(sources, root=root)
        model = DeepSeekModel(config, limits=LIMITS, transport=gate) if live else RuleReplayAdapter(case["request_id"])
        executor = RecordingExecutor(db_path=db, docs_root=docs)
        result = run_loop(model, executor, user_message=f"请排查 {case['request_id']}，依据证据给出结论。",
                          target_request_id=case["request_id"])
        grade = grade_diagnostic(case, result)
        grade["checks"]["sources_unchanged"] = source_hash == fingerprint(sources, root=root)
        grade["passed"] = all(grade["checks"].values())
        return {"id": case["id"], "repeat": repeat, "title": case["title"], "expected": case,
                "source_sha256": source_hash, "grade": grade, "result": result,
                "tool_evidence": executor.calls, "human_review": "pending"}


def summarize(rows, plan):
    groups = {}
    for item in plan["schedule"]:
        cid = item["case"]["id"]
        if cid in groups:
            continue
        group = [row for row in rows if row["id"] == cid]
        signatures = {json.dumps([row["result"]["outcome"], row["result"]["reason"],
                                  row["result"].get("structured_answer", {}).get("diagnosis", {}).get("cause"),
                                  row["result"].get("structured_stop", {}).get("reason")]) for row in group}
        passed = sum(row["grade"]["passed"] for row in group)
        groups[cid] = {"passed": passed, "completed": len(group), "planned": 2,
                       "all_repeats_passed": passed == len(group) == 2,
                       "same_sources_across_repeats": len({row["source_sha256"] for row in group}) == 1 if len(group) == 2 else None,
                       "same_outcome_signature": len(signatures) == 1 if len(group) == 2 else None}
    return {"passed_runs": sum(row["grade"]["passed"] for row in rows), "completed_runs": len(rows),
            "planned_runs": plan["planned_runs"], "per_case": groups,
            "tool_calls": sum(row["result"]["usage"]["tool_calls"] for row in rows),
            "format_correction_requests": sum(row["result"]["usage"].get("format_recovery", {}).get("correction_requests", 0) for row in rows)}


def run_batch(output, *, scope="all", live=False, max_api_calls=None, seconds=900, progress=None):
    plan = make_plan(scope)
    if type(seconds) not in (int, float) or not 0 < seconds <= 1800:
        raise ValueError("Batch seconds must be 1..1800")
    if live and (type(max_api_calls) is not int or not 4 <= max_api_calls <= plan["max_api_calls"]):
        raise ValueError("Live run requires an explicit API cap within the plan")
    if not live and max_api_calls is not None:
        raise ValueError("API budget applies only to live runs")
    artifacts = Artifacts(output)  # Reject collisions and bad paths before loading credentials.
    verify_release()
    started = perf_counter()
    report = {"created_at": now(), "status": "running", "execution_mode": "live" if live else "offline_harness",
              "plan": plan, "simulation": True, "split": "development_repeated", "human_review": "pending",
              "model_api_cap": max_api_calls if live else 0, "batch_seconds": seconds, "cost": None,
              "provenance": {"candidate_sha256": file_hash(PROJECT_ROOT / "evals/acceptance_candidate_v2.json"),
                             "runner_seal_sha256": file_hash(SEAL), "dataset_sha256": file_hash(DATASET),
                             "prompt_sha256": sha256((SYSTEM_MESSAGE + "\n" + ADAPTER_PROMPT).encode()).hexdigest(),
                             "sdk_version": version("openai"), "frozen_files_verified_after": False},
              "limitations": ["Existing development cases, not held-out accuracy; independent human review pending.",
                              "Two repeats per case only; consistency can also mean consistently wrong.",
                              "No RAG, conversation or fixed-workflow comparison in this run.",
                              "Provider model aliases may change; no seed or temperature override; cache not controlled.",
                              "Batch deadline checked between runs and before network requests; local operations are not forcibly preempted.",
                              "Unknown token usage stops subsequent requests; attempted requests can still incur cost.",
                              "No automatic resume or retry of interrupted batches. Journal retains reserved requests."], "rows": []}
    with artifacts:
        gate = RequestGate(max_api_calls or 0, seconds, artifacts.event)
        report["usage"] = gate.usage()
        report["summary"] = summarize([], plan)
        artifacts.event("batch_started", mode=report["execution_mode"], plan=plan, api_cap=report["model_api_cap"])
        artifacts.save(report)
        try:
            config = load_deepseek_config() if live else None
            report["model"] = config.model if live else "rule-replay-harness-v1"
            for item in plan["schedule"]:
                verify_release()
                if live and not gate.can_start_run():
                    report["status"] = "stopped"
                    break
                if perf_counter() - started >= seconds:
                    gate.stop_reason = "batch_deadline"
                    report["status"] = "stopped"
                    break
                gate.active = f"{item['case']['id']}:{item['repeat']}"
                artifacts.event("run_started", run=gate.active)
                attempts_before = len(gate.calls)
                row = one_run(item["case"], item["repeat"], live=live, config=config, gate=gate)
                row["network_request_attempts"] = len(gate.calls) - attempts_before
                report["rows"].append(row)
                report["usage"] = gate.usage()
                report["summary"] = summarize(report["rows"], plan)
                artifacts.event("run_completed", row=row)
                artifacts.save(report)
                if progress:
                    progress(row)
            else:
                report["status"] = "completed"
            verify_release()
            report["provenance"]["frozen_files_verified_after"] = True
        except KeyboardInterrupt:
            report["status"] = "interrupted"
        except Exception as error:
            report.update(status="error", error_type=type(error).__name__)
        finally:
            report["usage"] = gate.usage()
            report["summary"] = summarize(report["rows"], plan)
            report["stop_reason"] = gate.stop_reason
            report["elapsed_ms"] = round((perf_counter() - started) * 1000, 3)
            artifacts.event("batch_finished", status=report["status"], usage=report["usage"], summary=report["summary"])
            artifacts.save(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bounded repeated development evaluation; default offline")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--scope", choices=("all", "gaps"), default="all")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-api-calls", type=int)
    parser.add_argument("--seconds", type=int, default=900)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.plan:
            verify_release()
            print(json.dumps(make_plan(args.scope), ensure_ascii=True, indent=2))
            return 0
        if args.output is None:
            raise ValueError("New output path required")
        report = run_batch(args.output, scope=args.scope, live=args.live, max_api_calls=args.max_api_calls,
                           seconds=args.seconds, progress=lambda row: print(
                               f"{'PASS' if row['grade']['passed'] else 'FAIL'} | {row['id']} | repeat={row['repeat']}", flush=True))
        summary = report["summary"]
        print(f"{report['status']} | {summary['passed_runs']}/{summary['completed_runs']} completed; "
              f"planned={summary['planned_runs']}; API={report['usage']['model_api_calls']}")
        return 0 if report["status"] == "completed" and summary["passed_runs"] == summary["planned_runs"] else 1
    except (OSError, ValueError, KeyError) as error:
        print(f"Run not started: {type(error).__name__}. Check budget, seal and new output path.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
