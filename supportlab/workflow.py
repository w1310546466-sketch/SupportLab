"""Fixed collection policy with the same real model and grounded final contract."""

from .agent import ModelFailure
from .deepseek import ADAPTER_PROMPT
from .fixed_flow import FixedFlowModel
from time import perf_counter

WORKFLOW_VERSION = "fixed-tools-llm-v1"
WORKFLOW_PROMPT = ADAPTER_PROMPT.replace(
    "根据用户提供的请求编号，自行选择可用工具收集证据。缺少编号应追问，不要猜编号。",
    "本次工具由应用固定流程执行，application_tool_results 是程序提供的实际执行反馈；根据用户编号和这些反馈作答，不猜编号。"
) + "\n固定取证流程已经结束；本轮只能提交最终结构化结论，不再申请工具调用。缺少、错误或冲突的证据仍按原契约处理。"


class ModelWorkflow:
    structured_stops = True
    control_mode = "fixed_workflow"
    workflow_version = WORKFLOW_VERSION

    def __init__(self, request_id, model):
        self.planner = FixedFlowModel(request_id)
        self.model = model
        self.execution_mode = model.execution_mode
        self.provider, self.model_name = model.provider, model.model_name
        self.decision_owner = "application_fixed_policy"
        self.synthesis_started = False
        self.started_at = None

    def next_action(self, messages, tools):
        if self.started_at is None:
            self.started_at = perf_counter()
        if perf_counter() - self.started_at >= self.model.limits.total_seconds:
            raise ModelFailure("model_deadline")
        if not self.synthesis_started:
            proposed = self.planner.next_action(messages, tools)
            if proposed["type"] == "tool_call":
                self.decision_owner = "application_fixed_policy"
                return proposed
            # Discard the planner's inferred answer. The LLM receives actual
            # feedback only; expected causes or stopping labels are not supplied.
            self.model.prime_from_application_history(messages, started_at=self.started_at)
            self.synthesis_started = True
        self.decision_owner = "model_final_synthesis"
        proposed = self.model.next_action(messages, tools)
        if proposed.get("type") != "final":
            raise ModelFailure("model_protocol_error")
        return proposed

    def usage_snapshot(self):
        return self.model.usage_snapshot()
