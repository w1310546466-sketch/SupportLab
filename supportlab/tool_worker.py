"""Internal one-call worker. Not a model-visible tool or a general code runner."""

import json
from pathlib import Path
import sys

from .execution import error_result, invoke_tool
from .tools import AppContext


def main() -> None:
    try:
        raw = sys.stdin.read(8193)
        if len(raw) > 8192:
            raise ValueError("oversized internal request")
        payload = json.loads(raw)
        result = invoke_tool(payload["name"], payload["arguments"],
                             context=AppContext(payload["account_id"]),
                             db_path=Path(payload["db_path"]), docs_root=Path(payload["docs_root"]))
    except Exception:
        # Do not echo arguments, local paths, or raw exception details.
        result = error_result("tool_error", "内部工具执行失败。")
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
