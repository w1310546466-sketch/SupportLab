"""Run from the project root: python -m supportlab request req_demo_017."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys
from time import perf_counter

from .agent import LoopLimits, run_loop
from .contracts import model_tool_schemas
from .execution import ToolExecutor
from .mock_model import DemoMockModel
from .retrieval import search_docs
from .storage import ACCOUNT_DATASET_VERSION, DATASET_VERSION, init_demo_db
from .tools import AppContext, get_account_state, get_request_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SupportLab 模拟业务诊断：mock 回放或 DeepSeek 真实模型")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-demo", help="将模拟 JSON 导入本地 SQLite；重复执行更新同编号记录")
    init.add_argument("--extended", action="store_true", help="额外导入认证、权限、上游故障与超时的 5 个模拟场景")
    request = commands.add_parser("request", help="在固定演示账号内查询请求记录")
    request.add_argument("request_id")
    account = commands.add_parser("account", help="查询该请求发生时的账号快照（非实时账号状态）")
    account.add_argument("request_id")
    docs = commands.add_parser("docs", help="按关键词或错误码检索模拟文档，返回可引用的原文")
    docs.add_argument("query")
    docs.add_argument("--service-version", default="demo-api-v1", help="服务版本，默认 demo-api-v1")
    docs.add_argument("--top-k", type=int, default=3, help="最多返回 1–5 个片段，默认 3")
    commands.add_parser("tool-schemas", help="查看模型可见的工具名称、说明及参数契约")
    agent = commands.add_parser("agent-mock", help="运行本地模拟决策循环（无真实模型调用）")
    agent.add_argument("request_id", nargs="?", help="缺少编号时演示追问")
    agent.add_argument("--max-tool-calls", type=int, default=4)
    agent.add_argument("--max-turns", type=int, default=8)
    agent.add_argument("--tool-timeout", type=float, default=2.0, help="每个工具进程的秒数上限，默认 2")
    commands.add_parser("check-model-config", help="仅检查本地 DeepSeek 配置，不显示密钥、不调用网络")
    live = commands.add_parser("agent", help="使用真实 DeepSeek 模型排查模拟请求，会产生 API 调用")
    live.add_argument("request_id", nargs="?", help="模拟请求编号；缺少时由模型追问")
    live.add_argument("--max-api-calls", type=int, default=4, help="最多 1–8 次模型请求；默认 4")
    live.add_argument("--max-tool-calls", type=int, default=4)
    live.add_argument("--max-turns", type=int, default=8)
    live.add_argument("--max-output-tokens", type=int, default=768)
    live.add_argument("--max-format-corrections", type=int, choices=(0, 1), default=1,
                      help="格式纠正最多 0 或 1 次，计入原模型调用上限；默认 1")
    live.add_argument("--output", type=Path, help="将不含密钥/思维过程的结果与轨迹保存为 JSON")
    evaluation = commands.add_parser("eval-mock", help="运行构造开发场景并检查行为；无真实模型调用")
    evaluation.add_argument("--case", dest="case_id", help="只运行指定场景，例如 quota-ok 或 missing-snapshot")
    evaluation.add_argument("--demo-wrong-answer", action="store_true", help="故意调换诊断结论，演示真实引用也可能配上错误答案")
    evaluation.add_argument("--output", type=Path, help="将完整模拟报告保存到指定 JSON 文件")
    evaluation.add_argument("--json", action="store_true", help="直接输出完整 JSON；默认显示中文摘要")
    answer_demo = commands.add_parser("answer-demo", help="离线演示答案证据检查：正确答案与三个故意错误；无模型调用")
    answer_demo.add_argument("--case", choices=("supported", "wrong-cause", "unsupported-advice", "missing-account"))
    answer_demo.add_argument("--output", type=Path, help="保存到项目 reports 目录下的 JSON 文件")
    stop_demo = commands.add_parser("stop-demo", help="隔离演示追问/转人工摘要；默认离线，不调用模型")
    stop_demo.add_argument("--case", choices=("missing-id", "missing-account", "conflicting-records", "tool-failure", "premature-stop"))
    stop_demo.add_argument("--live", action="store_true", help="仅限 missing-account，真实调用 DeepSeek，最多 4 次模型请求")
    stop_demo.add_argument("--output", type=Path, help="保存到项目 reports 目录下的 JSON 文件")
    recovery_demo = commands.add_parser("recovery-demo", help="离线演示格式纠正与停止边界；使用构造模型回复，不调用网络")
    recovery_demo.add_argument("--case", choices=("recovered", "still-invalid", "budget-used", "correction-disabled",
                                                "unsupported-reason", "tool-during-repair"))
    recovery_demo.add_argument("--output", type=Path, help="保存到项目 reports 目录下的 JSON 文件")
    chat = commands.add_parser("chat", help="多轮会话：工具、文档、历史、偏好；默认离线固定流程")
    chat.add_argument("--session", help="继续指定会话；省略则新建")
    chat.add_argument("--message", help="只发送一条消息；省略则进入交互")
    mode = chat.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="新建动态 Agent 真实会话；每次诊断最多 4 次模型请求，文档问答最多 2 次")
    mode.add_argument("--workflow", action="store_true", help="新建固定取证流程 + DeepSeek 结论会话，会调用真实模型")
    chat.add_argument("--json", action="store_true", help="打印含轨迹的完整结果")
    chat.add_argument("--output", type=Path, help="将本次结果保存到 reports/*.json；须同时使用 --message")
    commands.add_parser("sessions", help="查看本地会话及固定的运行方式")
    commands.add_parser("kb-build", help="建立或更新模拟文档的本地检索索引")
    app_demo = commands.add_parser("app-demo", help="隔离演示工具、文档、历史、偏好及续接；默认零 API")
    app_demo.add_argument("--live", action="store_true", help="真实验证续接诊断与文档问答，总计最多 6 次模型请求")
    app_demo.add_argument("--output", type=Path, help="保存到 reports/*.json")
    quality = commands.add_parser("eval-app", help="运行 20 个 RAG/多轮开发场景，零 API")
    quality.add_argument("--output", type=Path, help="保存到 reports/*.json")
    diagnostics = commands.add_parser("eval-diagnostics", help="认证/权限/上游/超时的 20 个开发场景；默认离线两种流程")
    diagnostics.add_argument("--case", dest="case_id", help="只运行一个案例，如 revoked、permission 或 timeout")
    diagnostics.add_argument("--live", action="store_true", help="须指定一个正常案例；动态 Agent 真实调用最多 4 次")
    diagnostics.add_argument("--output", type=Path, help="保存到 reports/*.json")
    compare = commands.add_parser("compare", help="固定流程与动态 Agent 对照；默认构造回复，只验证程序")
    compare.add_argument("--case", dest="case_id", choices=("rate-ok", "quota-ok", "missing-snapshot", "quota-conflict", "unsupported-401", "database-missing"))
    compare.add_argument("--live", action="store_true", help="真实对照须指定一个支持案例，总计最多 8 次模型请求")
    compare.add_argument("--output", type=Path, help="保存到 reports/*.json")
    args = parser.parse_args(argv)

    if args.command in ("chat", "sessions", "kb-build", "app-demo", "eval-app", "compare", "eval-diagnostics"):
        from .chat_cli import run_application
        return run_application(args)

    if args.command == "recovery-demo":
        from .recovery_demo import run_recovery_demo
        from .storage import PROJECT_ROOT
        target = args.output.resolve() if args.output else None
        if target and (target.parent != (PROJECT_ROOT / "reports").resolve() or target.suffix != ".json"):
            print("报告路径须为项目 reports 目录下的 .json 文件。")
            return 2
        report = run_recovery_demo(args.case)
        if target:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except OSError:
                print("报告保存失败。", file=sys.stderr)
                return 2
        print(f"离线格式恢复演示：{report['expected_behavior_count']}/{report['case_count']} 项符合预期；真实模型调用 0 次。")
        for case in report["cases"]:
            result = case["result"]
            recovery = result["usage"]["format_recovery"]
            print(f"[{case['id']}] {case['title']} → {result['reason']}")
            print(f"模拟模型请求 {result['usage']['simulated_model_requests']} 次，其中格式纠正 {recovery['correction_requests']} 次；工具 {result['usage']['tool_calls']} 次。")
            print(result["answer"])
        print("纠正后的格式与证据分别检查；构造回复不能证明真实模型恢复成功率。")
        return 0 if report["expected_behavior_count"] == report["case_count"] else 1

    if args.command == "stop-demo":
        from .stop_demo import run_stop_demo
        from .storage import PROJECT_ROOT
        from .configuration import ConfigurationError
        target = args.output.resolve() if args.output else None
        if target and (target.parent != (PROJECT_ROOT / "reports").resolve() or target.suffix != ".json"):
            print("报告路径须为项目 reports 目录下的 .json 文件。")
            return 2
        if args.live and args.case != "missing-account":
            print("真实试验仅支持 --case missing-account，每次最多 4 次模型请求。")
            return 2
        try:
            report = run_stop_demo(args.case, live=args.live)
        except (ConfigurationError, OSError, ValueError, sqlite3.Error, ImportError):
            print("演示未能准备，请检查本地配置、依赖及模拟资料。")
            return 2
        if target:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except OSError:
                print("报告保存失败；运行结果仍显示在终端。", file=sys.stderr)
        print(f"{'真实模型' if args.live else '离线'}停止说明演示：{report['expected_behavior_count']}/{report['case_count']} 项行为符合预期；模型调用 {report['model_api_calls']} 次。")
        for case in report["cases"]:
            print(f"[{case['id']}] {case['title']} → {case['result']['status']}")
            print(case["result"]["answer"])
        print("这是构造开发案例，不是测试集准确率。")
        return 0 if report["expected_behavior_count"] == report["case_count"] else 1

    if args.command == "answer-demo":
        from .answer_demo import run_answer_demo
        from .storage import PROJECT_ROOT
        target = args.output.resolve() if args.output else None
        if target and (target.parent != (PROJECT_ROOT / "reports").resolve() or target.suffix != ".json"):
            print("报告路径须为项目 reports 目录下的 .json 文件。")
            return 2
        report = run_answer_demo(args.case)
        if target:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except OSError:
                print("报告保存失败；运行结果仍显示在终端。", file=sys.stderr)
                return 2
        print(f"离线答案检查演示：{report['expected_behavior_count']}/{report['case_count']} 项行为符合预期；模型调用 0 次。")
        for case in report["cases"]:
            result = case["result"]
            print(f"[{case['id']}] {case['title']} → {result['status']}")
            print(result["answer"])
            if result.get("answer_check", {}).get("issues"):
                print("检查结果：" + ", ".join(result["answer_check"]["issues"]))
        print("这是构造的程序验证，不是真实模型准确率。")
        return 0 if report["expected_behavior_count"] == report["case_count"] else 1

    if args.command in ("check-model-config", "agent"):
        from .configuration import ConfigurationError, load_deepseek_config
        try:
            config = load_deepseek_config()
        except ConfigurationError as error:
            print(json.dumps({"status": "configuration_error", "message": str(error)}, ensure_ascii=False))
            return 2
        if args.command == "check-model-config":
            print(json.dumps(config.safe_summary(), ensure_ascii=False, indent=2))
            return 0
        from hashlib import sha256
        import importlib.metadata
        from .agent import SYSTEM_MESSAGE
        from .answer_check import ANSWER_CONTRACT
        from .stop_check import STOP_CONTRACT
        from .deepseek import ADAPTER_PROMPT, ADAPTER_VERSION, DeepSeekModel, ModelLimits
        from .format_recovery import FORMAT_RECOVERY_VERSION, REPAIR_MESSAGE
        from .evaluation import fingerprint
        from .retrieval import DEFAULT_DOCS_ROOT, DOCUMENTS
        from .storage import DEFAULT_ACCOUNT_FIXTURE, DEFAULT_FIXTURE, PROJECT_ROOT
        try:
            model_limits = ModelLimits(max_api_calls=args.max_api_calls, max_output_tokens=args.max_output_tokens,
                                       max_format_corrections=args.max_format_corrections)
            limits = LoopLimits(args.max_tool_calls, args.max_turns)
            sdk_version = importlib.metadata.version("openai")
        except (ValueError, importlib.metadata.PackageNotFoundError):
            print(json.dumps({"status": "configuration_error", "message": "请检查调用预算，并先安装 requirements.txt 中的 SDK。"}, ensure_ascii=False))
            return 2
        target = args.output.resolve() if args.output else None
        if target and (target.parent != (PROJECT_ROOT / "reports").resolve() or target.suffix != ".json"):
            print(json.dumps({"status": "report_error", "message": "报告路径须为项目 reports 目录下的 .json 文件。"}, ensure_ascii=False))
            return 2
        model = DeepSeekModel(config, limits=model_limits)
        result = run_loop(model, ToolExecutor(AppContext("acc_demo")),
                          user_message=f"请排查请求 {args.request_id or '（尚未提供）'}。",
                          limits=limits, target_request_id=args.request_id)
        result["reproducibility"] = {
            "adapter_version": ADAPTER_VERSION, "sdk_version": sdk_version,
            "answer_contract": ANSWER_CONTRACT, "stop_contract": STOP_CONTRACT,
            "code_sha256": fingerprint(list((PROJECT_ROOT / "supportlab").glob("*.py"))),
            "source_sha256": fingerprint([DEFAULT_FIXTURE, DEFAULT_ACCOUNT_FIXTURE,
                                           *(DEFAULT_DOCS_ROOT / document.filename for document in DOCUMENTS)]),
            "prompt_sha256": sha256((SYSTEM_MESSAGE + "\n" + ADAPTER_PROMPT).encode("utf-8")).hexdigest(),
            "format_recovery_version": FORMAT_RECOVERY_VERSION,
            "repair_prompt_sha256": sha256(REPAIR_MESSAGE.encode("utf-8")).hexdigest(),
            "case_id": args.request_id, "semantic_review": "pending", "cost": None,
        }
        if target:
            # Report output must not overwrite configuration, source code or fixtures.
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except OSError:
                print(json.dumps({"status": "report_error", "message": "无法保存报告，运行结果仍输出到终端。"}, ensure_ascii=False), file=sys.stderr)
        for entry in result.pop("trace"):
            print(json.dumps(entry, ensure_ascii=False), file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result["status"] == "handoff" else 0

    if args.command == "eval-mock":
        from .evaluation import format_report, run_evaluation
        try:
            result = run_evaluation(case_id=args.case_id, wrong_cause=args.demo_wrong_answer)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except (ValueError, OSError, KeyError, TypeError, sqlite3.Error):
            print(json.dumps({"status": "evaluation_error", "simulation": True,
                              "message": "请检查场景编号、开发集文件与报告路径。错误答案演示仅支持单个诊断场景。"}, ensure_ascii=False))
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2) if args.json else format_report(result))
        return 0 if result["automated_pass_count"] == result["case_count"] else 1

    if args.command == "tool-schemas":
        print(json.dumps(model_tool_schemas(), ensure_ascii=False, indent=2))
        return 0
    if args.command == "agent-mock":
        try:
            limits = LoopLimits(args.max_tool_calls, args.max_turns)
            executor = ToolExecutor(AppContext("acc_demo"), timeout_seconds=args.tool_timeout)
        except ValueError:
            print(json.dumps({"simulation": True, "execution_mode": "mock", "status": "invalid_arguments",
                              "message": "工具预算须为 0–20 整数，轮次须为 1–30 整数，工具时限须为 (0,30] 秒有限数。"}, ensure_ascii=False))
            return 2
        result = run_loop(DemoMockModel(args.request_id), executor,
                          user_message=f"请排查请求 {args.request_id or '（尚未提供）'}。", limits=limits)
        for entry in result.pop("trace"):
            print(json.dumps(entry, ensure_ascii=False), file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result["status"] == "handoff" else 0

    if args.command == "init-demo":
        try:
            counts = init_demo_db()
            if args.extended:
                from .scenarios import init_scenarios
                counts.update(init_scenarios())
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error):
            print(json.dumps({"simulation": True, "status": "setup_error",
                              "message": "初始化未全部完成，请检查模拟 JSON 的格式、对应时间及数据库文件；基础与扩展资料分别按事务导入。"},
                             ensure_ascii=False))
            return 1
        print(json.dumps({"simulation": True, "dataset_version": DATASET_VERSION,
                          "account_dataset_version": ACCOUNT_DATASET_VERSION,
                          "status": "ready", "imported_records": counts["request_records"],
                          "imported_account_states": counts["account_states"],
                          "imported_diagnostic_snapshots": counts.get("diagnostic_snapshots", 0)}, ensure_ascii=False))
        return 0

    # A fixed local session stands in for future authenticated application context.
    # Deliberately no --account-id flag on the tool command.
    context = AppContext(account_id="acc_demo")
    started = perf_counter()
    if args.command == "request":
        result = get_request_record(args.request_id, context=context)
        tool_name = "get_request_record"
        summary = result.get("record", {}).get("error_code", result["status"])
        arguments = {"request_id": args.request_id}
    elif args.command == "account":
        result = get_account_state(args.request_id, context=context)
        tool_name = "get_account_state"
        arguments = {"request_id": args.request_id}
        state = result.get("account_state")
        summary = result["status"] if state is None else {
            "captured_at": state["captured_at"],
            "requests_in_window": state["requests_in_window"],
            "rate_limit_per_window": state["rate_limit_per_window"],
            "monthly_quota_used": state["monthly_quota_used"],
            "monthly_quota_limit": state["monthly_quota_limit"],
        }
    else:
        result = search_docs(args.query, args.service_version, args.top_k)
        tool_name = "search_docs"
        # Log query length instead of arbitrary user text; never log excerpts.
        arguments = {"query_chars": len(args.query.strip()),
                     "service_version": args.service_version, "top_k": args.top_k}
        summary = {"match_count": len(result["matches"]),
                   "citations": [hit["citation"] for hit in result["matches"]]}
    status = result["status"]
    trace = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "simulation": True,
        "tool": tool_name,
        "dataset_version": result["dataset_version"],
        "validated_arguments": None if status == "invalid_arguments" else arguments,
        "status": status,
        "result_summary": summary,
        "error": None if status == "ok" else status,
        "elapsed_ms": round((perf_counter() - started) * 1000, 3),
        "model_called": False,
        "token_usage": None,
    }
    # stdout is the tool result; stderr is one JSON trace line, separately redirectable.
    print(json.dumps(trace, ensure_ascii=False), file=sys.stderr)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return {"ok": 0, "not_found": 1, "invalid_arguments": 2, "storage_error": 3}[status]


if __name__ == "__main__":
    # Predictable Chinese output in the Windows terminal and redirected log files.
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
