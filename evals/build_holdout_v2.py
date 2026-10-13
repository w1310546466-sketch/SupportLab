"""Author 12 new compositions AFTER the repaired application was frozen; no runs."""

from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CAUSE = {"quota": "monthly_quota_exhausted", "revoked": "credential_revoked", "permission": "scope_missing"}
CODE = {"quota": "MONTHLY_QUOTA_EXHAUSTED", "revoked": "CREDENTIAL_REVOKED", "permission": "SCOPE_MISSING", "timeout": "UPSTREAM_TIMEOUT_UNKNOWN"}
families = {
    "retrieval_surface_availability_isolation": "只移走知识问答登记的支持指南：诊断工具的规则来源仍齐全，应能诊断；知识问答须失败；原字节还原后无需重建即可继续。区别于 v1 直接修改诊断规则并重建索引。",
    "explicit_docs_over_embedded_request": "同一请求编号置于 /docs 控制语句内应只查文档，置于普通消息则查请求；两种入口交替，检查目标状态与工具次数。区别于 v1 不含请求编号的文档问题。",
    "request_token_boundary_with_pending_resume": "已有诊断后输入粘连、大小写尾缀或超长编号，不能截出合法旧编号；需暂停，插入知识问题后用完整编号恢复。区别于 v1 多编号歧义与单工具参数校验。",
}
cases = []


def chat(text, equals, **extra):
    return {"op": "chat", "text": text, "expect": {"equals": equals, **extra}}


def answer(kind, rid):
    if kind == "timeout":
        return {"result.outcome": "handoff", "result.structured_stop.reason": "execution_unknown", "result.usage.tool_calls": 3}
    return {"result.outcome": "diagnosis", "result.structured_answer.diagnosis.cause": CAUSE[kind],
            "result.structured_answer.facts.request.values.request_id": rid, "result.usage.tool_calls": 3}


def doc(kind):
    return chat("/docs " + CODE[kind], {"result.outcome": "docs", "result.usage.tool_calls": 0}, evidence_contains=[CODE[kind]])


def add(family, n, title, steps):
    cases.append({"id": f"{family}-{n:02}", "family": family, "title": title,
                  "rationale": families[family], "steps": steps})


for n, kind in enumerate(("quota", "revoked", "permission", "timeout"), 1):
    rid = f"req_hold_v2a{n}"
    add("retrieval_surface_availability_isolation", n, f"支持指南暂缺时 {kind} 诊断与知识入口分别处理", [
        {"op": "seed", "template": kind, "request_id": rid}, doc(kind),
        {"op": "doc_remove", "file": "demo-support-guide-v1.md"}, chat(rid, answer(kind, rid)),
        chat("/docs " + CODE[kind], {"result.outcome": "clarify", "result.reason": "index_unavailable", "result.usage.tool_calls": 0}),
        {"op": "doc_restore", "file": "demo-support-guide-v1.md"}, doc(kind),
        chat("再查刚才那个请求", answer(kind, rid))])

for n, kind in enumerate(("quota", "revoked", "permission", "timeout"), 1):
    rid = f"req_hold_v2b{n}"
    add("explicit_docs_over_embedded_request", n, f"含 {kind} 请求编号的文档指令与普通消息切换", [
        {"op": "seed", "template": kind, "request_id": rid},
        chat(f"/docs {CODE[kind]} 的含义，示例编号 {rid}", {"result.outcome": "docs", "result.usage.tool_calls": 0}, evidence_contains=[CODE[kind]]),
        {"op": "state", "expect": {"equals": {"target": None}}},
        chat("请解释这个请求：" + rid, answer(kind, rid)),
        chat(f"/docs {CODE[kind]} 示例 {rid}", {"result.outcome": "docs", "result.usage.tool_calls": 0}, evidence_contains=[CODE[kind]]),
        {"op": "state", "expect": {"equals": {"target": rid, "usage.tool_calls": 3}}}])

for n in range(1, 5):
    rid = f"req_hold_v2c{n}"
    malformed = {1: "prefix" + rid, 2: rid + "UPPER", 3: rid + "a" * 70, 4: "x_" + rid}[n]
    add("request_token_boundary_with_pending_resume", n, f"编号边界变体 {n} 不得截断取证", [
        {"op": "seed", "template": "quota", "request_id": rid}, chat(rid, answer("quota", rid)),
        chat("帮我排查 " + malformed, {"result.outcome": "clarify", "result.pending_request_id": True, "result.usage.tool_calls": 0}),
        {"op": "state", "expect": {"equals": {"target": None, "status": "needs_input"}, "absent": ["summary.last_verified_cause"]}},
        doc("quota"), chat(rid, answer("quota", rid)),
        {"op": "state", "expect": {"equals": {"target": rid, "summary.last_verified_cause": "monthly_quota_exhausted"}}}])

assert len(cases) == 12
payload = {"dataset_version": "composition-holdout-v2-supplement", "simulation": True,
           "split": "prospective_composition_holdout", "created_at": datetime.now(timezone.utc).isoformat(),
           "freeze_record": "acceptance_candidate_v2.json", "human_review": "pending", "independent_authorship": False,
           "provenance": "修复 RAG 用量字段并冻结代码后另行编写；不复用 v1 八种完整轨迹模板。原语仍相同，AI 构造而非第三方盲测。",
           "families": families, "cases": cases}
if __name__ == "__main__":
    with (ROOT / "evals/holdout_v2.json").open("x", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print("已保存 12 个新组合场景，3 个新家族；未运行。")
