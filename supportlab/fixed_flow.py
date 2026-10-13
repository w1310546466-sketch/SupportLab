"""Offline fixed tool workflow using the same structured checks as the live agent."""

from .answer_check import CAUSES
from .diagnostic_check import RULES
from .stop_check import inspect_evidence, REASONS


class FixedFlowModel:
    execution_mode = "fixed"
    structured_stops = True
    provider = None
    model_name = "structured-fixed-v1"

    def __init__(self, request_id):
        self.request_id = request_id

    def next_action(self, messages, tools):
        arguments = {item["tool_call"]["call_id"]: item["tool_call"]["arguments"]
                     for item in messages if "tool_call" in item}
        feedback = [item for item in messages if item["role"] == "tool"]
        evidence = {item["content"]["evidence_id"]: {"tool": item["name"], "result": item["content"],
                                                     "arguments": arguments[item["tool_call_id"]]} for item in feedback}
        state = inspect_evidence(evidence, self.request_id)
        for reason in ("missing_request_id", "tool_failure", "request_unavailable", "unsupported_case",
                       "account_unavailable", "evidence_conflict", "documents_unavailable", "execution_unknown"):
            if reason in state["allowed"]:
                return {"type": "final", "outcome": REASONS[reason][0], "resolution": {"reason": reason}}
        results = {item["name"]: item["content"] for item in feedback}
        for name in ("get_request_record", "get_account_state", "search_docs"):
            if name not in results:
                args = {"request_id": self.request_id}
                if name == "search_docs":
                    args = {"query": state["request"]["error_code"], "service_version": state["request"]["service_version"], "top_k": 1}
                return {"type": "tool_call", "call_id": f"fixed_{len(feedback) + 1}", "name": name, "arguments": args}
        cause = next(key for key, rule in {**CAUSES, **RULES}.items() if rule["error_code"] == state["request"]["error_code"])
        return {"type": "final", "outcome": "diagnosis", "assessment": {
            "cause": cause, "request_evidence_id": results["get_request_record"]["evidence_id"],
            "account_evidence_id": results["get_account_state"]["evidence_id"],
            "rule_citation": results["search_docs"]["matches"][0]["citation"], "follow_up": ["check_current_status"]}}
