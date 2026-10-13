"""Strict final-object parsing with diagnostics that contain no response text."""

import json


FORMAT_RECOVERY_VERSION = "final-format-v1"
REPAIR_MESSAGE = (
    "程序格式检查反馈：上一条最终回复未通过解析，已丢弃原文。"
    "请根据此前实际工具反馈重新组织最终答案，仅输出一个合法 JSON 对象。"
    "严格使用系统说明中的 diagnosis/assessment 或 clarify、handoff/resolution 格式。"
    "不得输出代码围栏、前后说明、重复字段或多个对象；不得调用工具，也不要输出内部思维过程。"
    "只能依据已有证据；缺少的事实不得编造。格式修正后仍须通过原来的证据检查。"
)


class FinalFormatError(ValueError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(diagnostic["category"])


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _constant(value):
    raise ValueError("nonstandard_constant")


def _check_depth(value):
    # Our final contracts need only a few levels. Use a fixed application bound
    # rather than depending on Python's platform-specific parser recursion limit.
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 16:
            raise ValueError("excessive_nesting")
        if isinstance(item, (dict, list)):
            children = item.values() if isinstance(item, dict) else item
            pending.extend((child, depth + 1) for child in children)


def parse_final_object(content):
    """Never strip wrappers, extract a guessed object, or log raw error text."""
    diagnostic = {"content_chars": len(content), "empty": not content.strip(),
                  "starts_with_code_fence": content.lstrip().startswith("```"),
                  "line": None, "column": None}
    try:
        if len(content) > 32768:
            raise ValueError("response_too_large")
        parsed = json.loads(content, object_pairs_hook=_object, parse_constant=_constant)
        _check_depth(parsed)
        if type(parsed) is not dict:
            raise ValueError("not_object")
        return parsed
    except json.JSONDecodeError as error:
        # JSONDecodeError.msg/doc may contain arbitrary text. Retain positions only.
        diagnostic.update(category="json_syntax", line=error.lineno, column=error.colno)
    except RecursionError:
        diagnostic.update(category="excessive_nesting")
    except ValueError as error:
        category = str(error)
        diagnostic.update(category=category if category in {
            "duplicate_key", "nonstandard_constant", "not_object", "response_too_large", "excessive_nesting"
        } else "json_syntax")
    raise FinalFormatError(diagnostic) from None
