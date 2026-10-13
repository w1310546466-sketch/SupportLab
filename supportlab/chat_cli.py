"""Small human-facing interface for the persistent local application."""

import json
import sqlite3
import sys

from .conversation import ConversationService, HELP
from .knowledge import build_index
from .storage import DEFAULT_DB, PROJECT_ROOT, init_demo_db


def report_target(output):
    target = output.resolve() if output else None
    if target and (target.parent != (PROJECT_ROOT / "reports").resolve() or target.suffix != ".json"):
        raise ValueError("报告路径须为项目 reports 目录下的 .json 文件。")
    return target


def save_report(target, result):
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def show(result, *, full=False):
    if full:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result["answer"])
    usage = result["usage"]
    print(f"[会话 {result['session_id']} | 状态 {result['status']} | "
          f"模型请求 {usage['model_api_calls']} 次 | 工具 {usage.get('tool_calls', 0)} 次]")


def run_application(args):
    try:
        # Validate destination before opening a session or constructing a model.
        target = report_target(getattr(args, "output", None))
        if args.command == "chat" and target and args.message is None:
            raise ValueError("--output 须与 --message 同用。")
        if args.command == "kb-build":
            result = build_index()
            print(f"索引已更新：{result['document_count']} 篇模拟文档，{result['chunk_count']} 个可引用片段。")
            return 0
        if args.command == "eval-app":
            from .quality_eval import run_quality_evaluation
            report = run_quality_evaluation()
            save_report(target, report)
            print(f"RAG/多轮开发检查：{report['passed_count']}/{report['case_count']}；真实模型调用 0 次。")
            for row in report["cases"]:
                print(f"{'通过' if row['passed'] else '失败'} | {row['id']}")
            print("全部案例参与开发；不代表保留测试集准确率。")
            return 0 if report["passed_count"] == report["case_count"] else 1
        if args.command == "compare":
            from .comparison import run_comparison
            report = run_comparison(case_id=args.case_id, live=args.live)
            save_report(target, report)
            print("真实模型对照" if args.live else "离线对照程序验证（构造模型回复）")
            for name, row in report["summaries"].items():
                print(f"{name}：{row['passed']}/{row['cases']}，API {row['model_api_calls']} 次，工具 {row['tool_calls']} 次；"
                      f"运行耗时合计 {row['elapsed_ms_sum']} ms。")
            print("资料、权限与预算上限保持一致；这次结果不足以证明哪种架构更好。")
            return 0 if all(row["passed"] == row["cases"] for row in report["summaries"].values()) else 1
        if args.command == "eval-diagnostics":
            from .diagnostic_eval import run_diagnostic_evaluation
            report = run_diagnostic_evaluation(case_id=args.case_id, live=args.live)
            save_report(target, report)
            print(f"诊断开发验证：{report['case_count']} 个场景、{report['passed_runs']}/{report['run_count']} 次行为检查通过；API {report['model_api_calls']} 次。")
            for row in report["rows"]:
                print(f"{'通过' if row['grade']['passed'] else '失败'} | {row['id']} | {row['configuration']} | {row['result']['outcome']}")
            print("构造开发案例，尚非独立保留测试；离线结果不能代替真实模型评测。")
            return 0 if report["passed_runs"] == report["run_count"] else 1
        if args.command == "app-demo":
            from .application_demo import run_application_demo
            report = run_application_demo(live=args.live)
            save_report(target, report)
            print(f"{'真实模型' if args.live else '离线'}功能演示：{report['passed_checks']}/{report['check_count']} 项检查通过；"
                  f"模型请求 {report['model_api_calls']} 次。")
            for turn in report["turns"]:
                print(f"\n你：{turn['message']}\n助手：{turn['result']['answer']}")
            print("这是构造的功能演示，不是模型准确率评测。")
            return 0 if report["passed_checks"] == report["check_count"] else 1
        service = ConversationService()
        if args.command == "sessions":
            for session in service.store.sessions(service.actor):
                print(f"{session['id']} | {session['mode']} | {session['status']} | {session['target'] or '无当前请求'}")
            return 0
        if not DEFAULT_DB.exists():
            init_demo_db()
        session = service.new_session(args.session, mode="workflow" if args.workflow else ("agent" if args.live else None))
        if args.message is not None:
            result = service.chat(session["id"], args.message)
            save_report(target, result)
            show(result, full=args.json)
            return 1 if result["outcome"] == "handoff" else 0
        label = {"agent": "动态 Agent + 真实 DeepSeek（会产生 API 调用）",
                 "workflow": "固定流程 + 真实 DeepSeek（会产生 API 调用）", "fixed": "离线固定流程（零 API）"}[session["mode"]]
        print(f"SupportLab | 会话 {session['id']} | {label}\n{HELP}\n/exit 退出；/new 新建同模式会话。")
        while True:
            try:
                text = input("\n你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n会话已保存。")
                return 0
            if not text:
                continue
            if text == "/exit":
                print("会话已保存。")
                return 0
            if text == "/new":
                session = service.new_session(mode=session["mode"])
                print(f"已新建会话 {session['id']}。偏好保留，任务历史独立。")
                continue
            try:
                show(service.chat(session["id"], text), full=args.json)
            except ValueError as error:
                print(str(error))
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (OSError, sqlite3.Error, ImportError):
        print("本地资料、依赖或保存操作不可用，请检查安装与文件权限；没有自动重试模型调用。", file=sys.stderr)
        return 2
