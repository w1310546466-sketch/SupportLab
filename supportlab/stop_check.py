"""Evidence-backed stopping decisions and application-rendered handoff notes.

No model prose is published. A failed lookup proves only the lookup result,
not the existence of an account or the cause of an API failure.
"""

from .answer_check import CAUSES, check_assessment
from .diagnostic_check import RULES, check_extended


STOP_CONTRACT = "grounded-stop-v2"
REASONS = {
    "missing_request_id": ("clarify", "缺少要排查的请求编号", "请提供这次失败的请求编号；无需提供 API 密钥。"),
    "request_unavailable": ("clarify", "当前授权范围内没有查到请求记录", "请核对请求编号及所属服务/账号；查不到不能证明记录不存在，也不能证明它属于其他账号。"),
    "account_unavailable": ("handoff", "缺少该次请求对应的账号快照", "请由有权限的服务维护者补充同一请求、同一时刻和服务版本的账号用量或诊断记录。"),
    "evidence_conflict": ("handoff", "现有记录未通过一致性检查", "请由服务维护者核对请求错误码与同一时刻的账号快照，确认记录一致后再诊断；不要猜哪份记录正确。"),
    "documents_unavailable": ("handoff", "本次检索未取得可用于诊断的已核验规则", "请补充或核验该服务版本的规则文档，再检查它是否支持诊断。"),
    "unsupported_case": ("handoff", "超出现有已核验诊断规则的范围", "请由服务维护者提供此错误及版本的排查规则；现有工具不能据此确定根因。"),
    "execution_unknown": ("handoff", "超时后的执行结果未知，需人工核实", "请维护者核实该请求是否已执行、幂等约定及重复操作风险；不自动重试，不将未知当作未执行。"),
    "tool_failure": ("handoff", "工具未能正常完成查询", "请检查失败工具的参数、数据源或执行状态，修复后重新排查；查询失败不能证明业务故障原因。"),
}
TOOL_LABELS = {"get_request_record": "请求记录", "get_account_state": "账号快照", "search_docs": "文档检索"}
FAILURES = {"timeout", "storage_error", "tool_error", "invalid_arguments", "unknown_tool"}


def inspect_evidence(evidence, target_request_id):
    """Consider only queries for the trusted target, never a model-chosen other ID.

    The most recent attempt for each business tool determines its current state.
    Previous attempts remain in the loop trace. Document text is never echoed.
    """
    latest = {}
    for identifier, entry in evidence.items():
        arguments = entry.get("arguments") or {}
        if entry["tool"] in ("get_request_record", "get_account_state"):
            if target_request_id is not None and arguments.get("request_id") == target_request_id:
                latest[entry["tool"]] = (identifier, entry["result"])
    request_pair = latest.get("get_request_record")
    account_pair = latest.get("get_account_state")
    request = request_pair[1].get("record") if request_pair and request_pair[1]["status"] == "ok" else None
    account = account_pair[1].get("account_state") if account_pair and account_pair[1]["status"] == "ok" else None
    document_entries = [(identifier, entry["result"]) for identifier, entry in evidence.items()
                        if entry["tool"] == "search_docs" and request
                        and (entry.get("arguments") or {}).get("service_version") == request["service_version"]]
    if document_entries:
        latest["search_docs"] = document_entries[-1]

    allowed = set()
    if target_request_id is None:
        allowed.add("missing_request_id")
    if request_pair and request_pair[1]["status"] == "not_found":
        allowed.add("request_unavailable")
    if request and account_pair and account_pair[1]["status"] == "not_found":
        allowed.add("account_unavailable")
    latest_attempts = {entry["tool"]: entry for entry in evidence.values()}
    if any(entry["result"]["status"] in FAILURES for entry in latest_attempts.values()):
        allowed.add("tool_failure")
    rules = {**CAUSES, **RULES}
    cause = next((key for key, rule in rules.items() if request and rule["error_code"] == request["error_code"]), None)
    supported = (request and cause and request["http_status"] == rules[cause].get("http_status", 429)
                 and request["service_version"] == "demo-api-v1")
    if request and not supported:
        allowed.add("unsupported_case")
    consistency_issue = None
    verified_uncertainty = None
    if supported and account:
        citations = [hit["citation"] for _, result in document_entries if result["status"] == "ok"
                     for hit in result.get("matches", [])]
        checker = (lambda *args, **kwargs: check_extended(*args, **kwargs, allow_uncertain=True)) if cause in RULES else check_assessment
        checks = [checker({"cause": cause, "request_evidence_id": request_pair[0],
                                    "account_evidence_id": account_pair[0], "rule_citation": citation,
                                    "follow_up": ["check_current_status"]}, evidence,
                                   target_request_id=target_request_id)
                  for citation in (citations or ["not_observed"])]
        issues = [check.get("check", check)["issues"] for check in checks]
        if cause == "timeout_execution_unknown" and any(not row for row in issues):
            allowed.add("execution_unknown")
            verified_uncertainty = next(check["structured_answer"] for check in checks if check.get("check", {}).get("passed"))
        if not any(not row for row in issues):
            consistency_issue = issues[0][0]
            if consistency_issue not in ("citation_not_observed", "rule_not_supported"):
                allowed.add("evidence_conflict")
            elif document_entries:
                allowed.add("documents_unavailable")
    return {"latest": latest, "request": request, "account": account,
            "allowed": allowed, "consistency_issue": consistency_issue, "verified_uncertainty": verified_uncertainty}


def make_summary(evidence, target_request_id, *, reason, outcome="handoff"):
    state = inspect_evidence(evidence, target_request_id)
    known, gaps, steps, observations = [], [], [], []
    for tool, label in TOOL_LABELS.items():
        pair = state["latest"].get(tool)
        if pair is None:
            observations.append({"tool": tool, "status": "not_queried", "evidence_id": None})
            gaps.append(f"{label}：尚未针对当前请求取得查询结果。")
            continue
        identifier, result = pair
        status = result["status"]
        observations.append({"tool": tool, "status": status, "evidence_id": identifier})
        if status == "not_found":
            known.append(f"[{identifier}] {label}已查询，但本次没有找到可用记录。")
            gaps.append(f"{label}：已查但未找到；不能把缺失值填成 0。")
        elif status != "ok":
            known.append(f"[{identifier}] {label}本次未成功，工具状态为 {status}。")
            gaps.append(f"{label}：查询失败，内容未知。")
        elif tool == "get_request_record":
            row = state["request"]
            known.append(f"[{identifier}] 请求 {row['request_id']}，时间 {row['occurred_at']}，版本 {row['service_version']}，"
                         f"返回 {row['http_status']} / {row['error_code']}；这是记录中的线索，尚不是最终根因。")
        elif tool == "get_account_state":
            row = state["account"]
            if "diagnostic_facts" in row:
                known.append(f"[{identifier}] 已取得同请求诊断快照；时间 {row['captured_at']}，版本 {row['service_version']}；取得快照本身不证明结论成立。")
            else:
                known.append(f"[{identifier}] 快照时间 {row['captured_at']}，版本 {row['service_version']}；"
                             f"{row['rate_window_seconds']} 秒窗口请求数/上限 {row['requests_in_window']}/{row['rate_limit_per_window']}，"
                             f"月度已用量/上限 {row['monthly_quota_used']}/{row['monthly_quota_limit']}。")
        else:
            known.append(f"[{identifier}] 文档检索返回 {len(result['matches'])} 个片段；检索命中本身不证明诊断成立。")
    # Invalid parameters have no validated request ID. Report the execution
    # failure without pretending that a query for the current request occurred.
    for identifier, entry in evidence.items():
        if entry["result"]["status"] in FAILURES and identifier not in {pair[0] for pair in state["latest"].values()}:
            label = TOOL_LABELS.get(entry["tool"], "未登记工具")
            known.append(f"[{identifier}] {label}的一次调用失败，状态为 {entry['result']['status']}；不能据此确认当前请求的业务状态。")
    if not known:
        known.append("尚无针对当前请求的工具证据。")
    if state["verified_uncertainty"]:
        verified = state["verified_uncertainty"]
        known.append(verified["evidence_summary"] + "规则依据：" + verified["diagnosis"]["rule_citation"] + "。")
        gaps.append("还不知道这次超时请求最终是否执行，不能用超时记录推断未执行。")
    if "evidence_conflict" in state["allowed"]:
        gaps.append("请求与快照未通过一致性检查，现有证据不能支持确定诊断。")
    if reason in REASONS:
        _, title, step = REASONS[reason]
        steps.append(step)
    else:
        title = "本轮已停止，尚未形成通过检查的诊断"
        if reason in ("unsupported_assessment", "unsupported_resolution", "invalid_action", "invalid_citation",
                      "model_invalid_json", "model_repair_action_not_allowed"):
            steps.append("请检查模型输出与工具证据是否对应，再重新排查；本次未发布未经核实的结论。")
        elif reason in ("model_budget_exhausted", "tool_budget_exhausted", "turn_budget_exhausted", "model_deadline"):
            steps.append("本轮达到调用次数或时间上限；请根据已有记录补齐证据后重新排查，不自动追加调用。")
        else:
            steps.append("请根据轨迹中的停止原因检查模型接口或调用流程，恢复后重新排查。")
    for key in ("account_unavailable", "evidence_conflict", "request_unavailable", "tool_failure"):
        if key in state["allowed"] and REASONS[key][2] not in steps:
            steps.append(REASONS[key][2])
    if not gaps:
        gaps.append("仍缺少通过程序检查的最终诊断，不能据已有材料承诺恢复。")
    summary = {"scope": "historical_simulation", "reason": reason, "outcome": outcome,
               "observations": observations, "known": known, "gaps": gaps, "next_steps": steps,
               "diagnosis_confirmed": False, "ticket_created": False, "human_contacted": False,
               "uncertainty_evidence": state["verified_uncertainty"],
               "consistency_issue": state["consistency_issue"]}
    answer = (f"{'需要补充信息' if outcome == 'clarify' else '需人工核实'}：{title}。\n"
              + "已查证（模拟历史资料及本次查询状态）：\n" + "\n".join(f"- {item}" for item in known)
              + "\n仍缺少或需核实：\n" + "\n".join(f"- {item}" for item in gaps)
              + "\n如何补证据：\n" + "\n".join(f"- {item}" for item in steps)
              + "\n本次仅输出说明，未联系人工或创建工单。补充信息后需重新运行；历史资料不能证明当前账号状态。")
    return {"structured_stop": summary, "answer": answer}


def check_resolution(resolution, evidence, *, target_request_id, outcome):
    issues = []
    valid = type(resolution) is dict and set(resolution) == {"reason"} and isinstance(resolution["reason"], str)
    reason = resolution["reason"] if valid else None
    if not valid or reason not in REASONS:
        issues.append("invalid_resolution")
    elif REASONS[reason][0] != outcome:
        issues.append("wrong_stop_outcome")
    elif reason not in inspect_evidence(evidence, target_request_id)["allowed"]:
        issues.append("stop_reason_not_supported")
    check = {"contract": STOP_CONTRACT, "passed": not issues, "issues": issues,
             "proposed_reason": reason if reason in REASONS else "<invalid>", "human_review": "pending"}
    if issues:
        return {"stop_check": check}
    summary = make_summary(evidence, target_request_id, reason=reason, outcome=outcome)
    uncertainty = summary["structured_stop"]["uncertainty_evidence"]
    citations = [uncertainty["diagnosis"]["rule_citation"]] if uncertainty else []
    return {"stop_check": check, "citations": citations, **summary}
