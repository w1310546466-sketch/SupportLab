"""Handwritten local decisions for learning the protocol; NOT an LLM or model eval."""


class DemoMockModel:
    def __init__(self, request_id: str | None):
        self.request_id = request_id

    @staticmethod
    def final(outcome, message, citations=None):
        return {"type": "final", "outcome": outcome, "message": message, "citations": citations or []}

    @staticmethod
    def call(number, name, **arguments):
        return {"type": "tool_call", "call_id": f"call_{number}", "name": name, "arguments": arguments}

    def next_action(self, messages: list[dict], tools: list[dict]) -> dict:
        # Rules below read only feedback delivered by the loop, never the DB/files.
        results = {item["name"]: item["content"] for item in messages if item["role"] == "tool"}
        if not self.request_id:
            return self.final("clarify", "请提供需要排查的请求编号，例如 req_demo_018。")
        if "get_request_record" not in results:
            return self.call(1, "get_request_record", request_id=self.request_id)
        request_result = results["get_request_record"]
        if request_result["status"] != "ok":
            if request_result["status"] in ("not_found", "invalid_arguments"):
                return self.final("clarify", "当前账号范围内无法取得该请求，请核对请求编号。")
            return self.final("handoff", "请求工具未提供可用证据，请人工检查工具错误。")
        request = request_result["record"]
        code = request["error_code"]
        if code not in ("RATE_LIMIT_EXCEEDED", "MONTHLY_QUOTA_EXHAUSTED"):
            return self.final("handoff", "本次模拟决策仅覆盖两种 429，其他错误需补充证据后处理。")
        if "get_account_state" not in results:
            return self.call(2, "get_account_state", request_id=self.request_id)
        account_result = results["get_account_state"]
        if account_result["status"] != "ok":
            return self.final("handoff", "未取得与请求对应的账号快照，证据不足，转人工。")
        if "search_docs" not in results:
            return self.call(3, "search_docs", query=code, service_version=request["service_version"], top_k=1)
        docs = results["search_docs"]
        if docs["status"] != "ok":
            return self.final("handoff", "未找到适用的文档规则，证据不足，转人工。")

        state = account_result["account_state"]
        hits = [hit for hit in docs["matches"]
                if hit["service_version"] == request["service_version"] and code in hit["excerpt"]]
        aligned = (state["request_id"] == request["request_id"]
                   and state["captured_at"] == request["occurred_at"]
                   and state["service_version"] == request["service_version"])
        rate_exceeded = state["requests_in_window"] > state["rate_limit_per_window"]
        quota_exhausted = state["monthly_quota_used"] >= state["monthly_quota_limit"]
        if not aligned or not hits:
            return self.final("handoff", "请求、快照或文档对应关系不完整，转人工。")
        if code == "RATE_LIMIT_EXCEEDED" and rate_exceeded and not quota_exhausted:
            reason = "频率限制"
        elif code == "MONTHLY_QUOTA_EXHAUSTED" and quota_exhausted and not rate_exceeded:
            reason = "月度配额耗尽"
        else:
            return self.final("handoff", "请求错误码与账号数字未形成一致证据，转人工核对。")
        answer = (
            f"模拟诊断：{reason}。请求 {request['request_id']} 的错误码为 {code}；"
            f"在 {state['captured_at']} 的对应快照中，{state['rate_window_seconds']} 秒窗口"
            f"请求数/上限为 {state['requests_in_window']}/{state['rate_limit_per_window']}，"
            f"月度已用量/上限为 {state['monthly_quota_used']}/{state['monthly_quota_limit']}。"
            "结论由本地规则模板生成，未调用真实模型。"
        )
        return self.final("diagnosis", answer, [hits[0]["citation"]])
