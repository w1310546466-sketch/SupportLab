"""Checked, version-pinned rules for simulated auth/permission/upstream evidence."""

from copy import deepcopy
from hashlib import sha256
import re

from .scenarios import SNAPSHOT_FIELDS, SNAPSHOT_VERSION

DIAGNOSTIC_CONTRACT = "grounded-diagnostics-v1"
RULE_SHA256 = "b13a16f7fe346f4486995f1528d2a73a01ccf73643929353d9e7a6081e074a58"
ROW_HASHES = {
    11: "e987eb24dc903b6294378ffad22b87868970223ce9d8193c7aad0f6680ac9c29",
    12: "a8556bad4bce2ffe16de10dbdd058713700bdf139ed5b2c054ab58db4088ee67",
    13: "4dc337ec2120ef1b27b844027d04c46e306b3cdbca5f04570fea9390b96b2ca9",
    14: "8d4e4b1103a7bc21e6214dbf450cebeb6fc4c8a7a5e20649a0e5f607e4f4cb38",
    15: "3072431d035c1d682586945273d9875443912af80bdcfc6e6ef9acefc2f1dde4",
}
RULES = {
    "credential_revoked": {"error_code": "CREDENTIAL_REVOKED", "http_status": 401, "line": 11,
                           "label": "该次请求的凭据已被撤销"},
    "token_expired": {"error_code": "TOKEN_EXPIRED", "http_status": 401, "line": 12,
                      "label": "该次请求的凭据已过期"},
    "scope_missing": {"error_code": "SCOPE_MISSING", "http_status": 403, "line": 13,
                      "label": "该次请求缺少所需权限"},
    "upstream_incident_confirmed": {"error_code": "UPSTREAM_INCIDENT_CONFIRMED", "http_status": 503, "line": 14,
                                    "label": "该次请求关联上游故障事件"},
    "timeout_execution_unknown": {"error_code": "UPSTREAM_TIMEOUT_UNKNOWN", "http_status": 504, "line": 15,
                                   "label": "超时后的执行结果未知"},
}
NEXT_STEPS = {
    "credential_revoked": "请有权限的维护者核实当前凭据是否仍被撤销，以及是否需要重新签发；不要在对话里提供密钥。",
    "token_expired": "请有权限的维护者核实当前凭据有效期，再按服务规则处理；本程序不更换密钥。",
    "scope_missing": "请有权限的维护者核实所需权限与授权政策；本程序不修改权限，也不承诺能够获批。",
    "upstream_incident_confirmed": "请维护者核实当前事件状态及该请求的执行结果；关联事件不能证明没有执行，不自动重试。",
    "timeout_execution_unknown": "请维护者先核实该请求是否执行、是否有重复操作风险及幂等约定；不自动重试。",
}


def rejected(issue):
    return {"contract": DIAGNOSTIC_CONTRACT, "passed": False, "issues": [issue],
            "scope": "simulated_auth_permission_upstream", "human_review": "pending"}


def _identifier(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_:.-]{0,79}", value) is not None


def check_extended(assessment, evidence, *, target_request_id, allow_uncertain=False):
    from .answer_check import ASSESSMENT_FIELDS, FOLLOW_UPS, _time
    if (type(assessment) is not dict or set(assessment) != ASSESSMENT_FIELDS
            or any(type(assessment.get(key)) is not str or not 1 <= len(assessment[key]) <= 160
                   for key in ASSESSMENT_FIELDS - {"follow_up"})):
        return rejected("invalid_assessment")
    cause = assessment["cause"]
    if cause not in RULES:
        return rejected("unsupported_cause")
    if cause == "timeout_execution_unknown" and not allow_uncertain:
        return rejected("requires_execution_unknown_handoff")
    if assessment["follow_up"] != ["check_current_status"]:
        return rejected("unsupported_follow_up")
    if target_request_id is None:
        return rejected("missing_target_request")
    selected = []
    for ref, tool, field, dataset in (
        ("request_evidence_id", "get_request_record", "record", "requests-v1"),
        ("account_evidence_id", "get_account_state", "account_state", SNAPSHOT_VERSION),
    ):
        entry = evidence.get(assessment[ref], {})
        result = entry.get("result", {})
        if (entry.get("tool") != tool or result.get("status") != "ok" or result.get("simulation") is not True
                or result.get("dataset_version") != dataset or type(result.get(field)) is not dict
                or (entry.get("arguments") or {}).get("request_id") != target_request_id):
            return rejected("missing_evidence")
        selected.append(result[field])
    request, snapshot = selected
    if (request.get("request_id") != target_request_id or snapshot.get("request_id") != target_request_id
            or request.get("service_version") != "demo-api-v1" or snapshot.get("service_version") != request["service_version"]
            or snapshot.get("captured_at") != request.get("occurred_at")):
        return rejected("evidence_mismatch")
    rule, facts = RULES[cause], snapshot.get("diagnostic_facts")
    if (type(request.get("http_status")) is not int or request["http_status"] != rule["http_status"]
            or request.get("error_code") != rule["error_code"]):
        return rejected("cause_not_supported")
    if type(facts) is not dict or set(facts) != SNAPSHOT_FIELDS[rule["error_code"]]:
        return rejected("invalid_diagnostic_facts")
    retry = request.get("retry_after_seconds")
    if retry is not None and (type(retry) is not int or retry < 0):
        return rejected("invalid_request_facts")
    try:
        occurred = _time(request.get("occurred_at"))
        if cause == "credential_revoked":
            supported = facts["credential_status"] == "revoked"
        elif cause == "token_expired":
            supported = facts["credential_status"] == "expired" and _time(facts["credential_expires_at"]) <= occurred
        elif cause == "scope_missing":
            scopes = facts["granted_scopes"]
            supported = (facts["credential_status"] == "active" and _identifier(facts["required_scope"])
                         and type(scopes) is list and len(scopes) <= 32 and all(_identifier(item) for item in scopes)
                         and len(set(scopes)) == len(scopes) and facts["required_scope"] not in scopes)
        elif cause == "upstream_incident_confirmed":
            supported = (all(_identifier(facts[key]) for key in ("upstream_service", "incident_service", "incident_id"))
                         and facts["upstream_service"] == facts["incident_service"]
                         and _time(facts["incident_started_at"]) <= occurred
                         and (facts["incident_ended_at"] is None or occurred < _time(facts["incident_ended_at"]))
                         and facts["execution_state"] in ("unknown", "not_executed", "executed"))
        else:
            supported = facts["execution_state"] == "unknown" and (facts["idempotency_supported"] is None
                                                                   or type(facts["idempotency_supported"]) is bool)
    except (ValueError, TypeError, OverflowError):
        return rejected("invalid_time_evidence")
    if not supported:
        return rejected("cause_not_supported")
    citation = assessment["rule_citation"]
    hits = [hit for entry in evidence.values() if entry.get("tool") == "search_docs"
            and entry.get("result", {}).get("status") == "ok"
            and (entry.get("arguments") or {}).get("service_version") == "demo-api-v1"
            for hit in entry["result"].get("matches", [])]
    hit = next((item for item in hits if item.get("citation") == citation), None)
    if hit is None:
        return rejected("citation_not_observed")
    expected = f"demo-diagnostics-v1@{RULE_SHA256[:12]}:L{rule['line']}-L{rule['line']}"
    if (citation != expected or hit.get("doc_id") != "demo-diagnostics-v1" or hit.get("document_sha256") != RULE_SHA256
            or hit.get("service_version") != "demo-api-v1" or hit.get("line_start") != rule["line"]
            or hit.get("line_end") != rule["line"] or type(hit.get("excerpt")) is not str
            or sha256(hit["excerpt"].encode()).hexdigest() != ROW_HASHES[rule["line"]]):
        return rejected("rule_not_supported")
    # These facts are allowlisted and typed above; no arbitrary snapshot prose is echoed.
    details = {
        "credential_revoked": lambda: "凭据状态：revoked（已撤销）。",
        "token_expired": lambda: f"凭据状态：expired（已过期）；过期时间：{facts['credential_expires_at']}。",
        "scope_missing": lambda: f"凭据状态：active；所需权限：{facts['required_scope']}；当时授权：{', '.join(facts['granted_scopes']) or '无'}。",
        "upstream_incident_confirmed": lambda: f"上游：{facts['upstream_service']}；事件：{facts['incident_id']}；区间：{facts['incident_started_at']} 至 {facts['incident_ended_at'] or '结束时间未知'}；执行状态：{facts['execution_state']}。",
        "timeout_execution_unknown": lambda: "执行状态：unknown（未知）；幂等支持记录：" + {True: "支持", False: "不支持", None: "未知"}[facts["idempotency_supported"]] + "。",
    }[cause]()
    structured = {
        "scope": "historical_simulation",
        "facts": {"request": {"evidence_id": assessment["request_evidence_id"], "values": deepcopy(request)},
                  "account": {"evidence_id": assessment["account_evidence_id"], "values": deepcopy(snapshot)}},
        "diagnosis": {"cause": cause, "label": rule["label"], "rule_citation": citation},
        "evidence_summary": details,
        "follow_up": [{"code": "check_current_status", "status": "needs_verification", "text": FOLLOW_UPS["check_current_status"]},
                      {"code": "verify_with_maintainer", "status": "needs_verification", "text": NEXT_STEPS[cause]}],
    }
    answer = (f"已查证（模拟历史记录）：请求 {request['request_id']}，时间 {request['occurred_at']}，"
              f"返回 {request['http_status']} / {request['error_code']}。\n{details}\n"
              f"诊断：{rule['label']}。依据：{citation}。\n"
              f"待核实：{NEXT_STEPS[cause]}\n这些历史记录不能证明当前状态或服务已恢复。")
    return {"check": {"contract": DIAGNOSTIC_CONTRACT, "passed": True, "issues": [],
                      "scope": "simulated_auth_permission_upstream", "human_review": "pending"},
            "answer": answer, "structured_answer": structured, "citations": [citation]}
