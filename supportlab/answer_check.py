"""Narrow, application-owned evidence checks for the two simulated 429 causes.

This is not a general natural-language judge. The model selects a cause and
evidence references; numbers and public wording come from checked tool results.
"""

from copy import deepcopy
from datetime import datetime


ANSWER_CONTRACT = "grounded-429-v2"
RULE_DOCUMENT_SHA256 = "d849e3e33fcd81799140558984cbe75f104129fac2cc009a7e5709805b4d90ff"
CAUSES = {
    "monthly_quota_exhausted": {
        "error_code": "MONTHLY_QUOTA_EXHAUSTED", "label": "月度配额耗尽", "line": 11,
        "row": "| 429 | `MONTHLY_QUOTA_EXHAUSTED` | 月度用量配额耗尽 | 当前月度配额、已用量及重置时间 |",
    },
    "rate_limit_exceeded": {
        "error_code": "RATE_LIMIT_EXCEEDED", "label": "频率限制", "line": 10,
        "row": "| 429 | `RATE_LIMIT_EXCEEDED` | 请求触发频率限制 | 当前频率额度、请求窗口、`retry_after_seconds` |",
    },
}
FOLLOW_UPS = {
    "check_current_status": "若要判断现在能否继续请求，需另行取得当前状态；现有工具只提供历史快照。",
    "verify_quota_options": "现有资料未说明是否能申请额外配额，需向服务方核实可用方案及生效条件。",
}
ASSESSMENT_FIELDS = {"cause", "request_evidence_id", "account_evidence_id", "rule_citation", "follow_up"}


def reference_summary(assessment, evidence):
    """Log short typed references, never arbitrary model-written prose."""
    import re
    refs = {}
    for key in ("request_evidence_id", "account_evidence_id"):
        value = assessment.get(key)
        refs[key] = value if isinstance(value, str) and re.fullmatch(r"E[1-9][0-9]?", value) else "<invalid>"
    return {"references": refs,
            "available_evidence": [{"id": key, "tool": entry["tool"], "status": entry["result"]["status"]}
                                   for key, entry in evidence.items()]}


def rejected(issue: str) -> dict:
    return {"contract": ANSWER_CONTRACT, "passed": False, "issues": [issue],
            "scope": "two_simulated_429_causes", "human_review": "pending"}


def _time(value):
    if not isinstance(value, str):
        raise ValueError("invalid time")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone required")
    return result


def check_assessment(assessment: dict, evidence: dict, *, target_request_id: str | None) -> dict:
    """Check selected evidence, independent of tool order or call-ID spelling.

    Evidence is populated only by the executor. No model-written facts are
    accepted. An unknown document revision requires an explicit rule review.
    """
    if type(assessment) is not dict or set(assessment) != ASSESSMENT_FIELDS:
        return rejected("invalid_assessment")
    from .diagnostic_check import RULES, check_extended
    if type(assessment.get("cause")) is str and assessment["cause"] in RULES:
        return check_extended(assessment, evidence, target_request_id=target_request_id)
    for key in ("cause", "request_evidence_id", "account_evidence_id", "rule_citation"):
        if not isinstance(assessment[key], str) or not 1 <= len(assessment[key]) <= 160:
            return rejected("invalid_assessment")
    cause = assessment["cause"]
    if cause not in CAUSES:
        return rejected("unsupported_cause")
    follow_up = assessment["follow_up"]
    if (not isinstance(follow_up, list) or not 1 <= len(follow_up) <= 2
            or not all(isinstance(item, str) and item in FOLLOW_UPS for item in follow_up)
            or len(set(follow_up)) != len(follow_up)):
        return rejected("unsupported_follow_up")
    if "check_current_status" not in follow_up:
        return rejected("missing_current_status_caveat")
    if cause != "monthly_quota_exhausted" and "verify_quota_options" in follow_up:
        return rejected("irrelevant_follow_up")
    if target_request_id is None:
        return rejected("missing_target_request")

    selected = []
    for ref, tool, field, dataset in (
        ("request_evidence_id", "get_request_record", "record", "requests-v1"),
        ("account_evidence_id", "get_account_state", "account_state", "account-states-v1"),
    ):
        entry = evidence.get(assessment[ref], {})
        result = entry.get("result", {})
        if (entry.get("tool") != tool or result.get("status") != "ok"
                or result.get("simulation") is not True or result.get("dataset_version") != dataset
                or not isinstance(result.get(field), dict)):
            return rejected("missing_evidence")
        selected.append(result[field])
    request, account = selected
    if (request.get("request_id") != target_request_id
            or account.get("request_id") != target_request_id
            or request.get("occurred_at") != account.get("captured_at")
            or request.get("service_version") != "demo-api-v1"
            or account.get("service_version") != request.get("service_version")):
        return rejected("evidence_mismatch")

    integer_fields = ("rate_window_seconds", "rate_limit_per_window", "requests_in_window",
                      "monthly_quota_limit", "monthly_quota_used")
    if (any(type(account.get(key)) is not int or account[key] < 0 for key in integer_fields)
            or account["rate_window_seconds"] == 0):
        return rejected("invalid_account_facts")
    retry = request.get("retry_after_seconds")
    if retry is not None and (type(retry) is not int or retry < 0):
        return rejected("invalid_request_facts")
    try:
        if _time(account.get("quota_resets_at")) <= _time(request.get("occurred_at")):
            return rejected("invalid_time_evidence")
    except (ValueError, TypeError, OverflowError):
        return rejected("invalid_time_evidence")

    rule = CAUSES[cause]
    rate_exceeded = account["requests_in_window"] > account["rate_limit_per_window"]
    quota_exhausted = account["monthly_quota_used"] >= account["monthly_quota_limit"]
    numbers_support = ((quota_exhausted and not rate_exceeded) if cause == "monthly_quota_exhausted"
                       else (rate_exceeded and not quota_exhausted))
    if (request.get("http_status") != 429 or request.get("error_code") != rule["error_code"]
            or not numbers_support):
        return rejected("cause_not_supported")

    hits = [hit for entry in evidence.values() if entry.get("tool") == "search_docs"
            and entry.get("result", {}).get("status") == "ok"
            for hit in entry["result"].get("matches", [])]
    citation = assessment["rule_citation"]
    hit = next((item for item in hits if item.get("citation") == citation), None)
    if hit is None:
        return rejected("citation_not_observed")
    expected_citation = f"demo-errors-v1@{RULE_DOCUMENT_SHA256[:12]}:L{rule['line']}-L{rule['line']}"
    if (citation != expected_citation or hit.get("doc_id") != "demo-errors-v1"
            or hit.get("service_version") != request["service_version"]
            or hit.get("document_sha256") != RULE_DOCUMENT_SHA256
            or hit.get("line_start") != rule["line"] or hit.get("line_end") != rule["line"]
            or hit.get("excerpt", "").strip() != rule["row"]):
        return rejected("rule_not_supported")

    request_facts = {key: request[key] for key in
                     ("request_id", "occurred_at", "service_version", "http_status", "error_code")}
    request_facts["retry_after_seconds"] = retry
    account_facts = {key: account[key] for key in (*integer_fields, "captured_at", "quota_resets_at")}
    structured = {
        "scope": "historical_simulation",
        "facts": {"request": {"evidence_id": assessment["request_evidence_id"], "values": request_facts},
                  "account": {"evidence_id": assessment["account_evidence_id"], "values": account_facts}},
        "diagnosis": {"cause": cause, "label": rule["label"], "rule_citation": citation},
        "follow_up": [{"code": item, "status": "needs_verification", "text": FOLLOW_UPS[item]}
                      for item in follow_up],
    }
    answer = (
        f"已查证事实（模拟历史记录）：请求 {request['request_id']} 发生于 {request['occurred_at']}，"
        f"错误码为 {request['error_code']}。同一时刻的 {account['rate_window_seconds']} 秒窗口请求数/上限为 "
        f"{account['requests_in_window']}/{account['rate_limit_per_window']}，月度已用量/上限为 "
        f"{account['monthly_quota_used']}/{account['monthly_quota_limit']}。\n"
        f"诊断：该次历史请求的证据支持“{rule['label']}”。规则依据：{citation}。\n"
        f"时间范围：当时快照登记的配额重置时间为 {account['quota_resets_at']}；"
        "这一历史计划时间不能证明今天已经恢复。\n"
        + ("该次请求没有提供建议等待秒数；不能把空值解释为立即重试。\n" if retry is None
           else f"该次历史请求记录的建议等待秒数为 {retry}，不代表今天的等待时长。\n")
        + "待核实：\n" + "\n".join(f"- {FOLLOW_UPS[item]}" for item in follow_up)
    )
    return {"check": {"contract": ANSWER_CONTRACT, "passed": True, "issues": [],
                       "scope": "two_simulated_429_causes", "human_review": "pending"},
            "structured_answer": deepcopy(structured), "answer": answer, "citations": [citation]}
