"""Small inspectable routing policy; explicit request IDs and topic changes win."""

import re

ROUTING_VERSION = "intent-rules-v2"
REQUEST_PATTERN = r"(?<![a-zA-Z0-9_])req_[a-z0-9_]{1,60}(?![a-zA-Z0-9_])"
QUESTION_WORDS = ("是什么", "什么意思", "有何区别", "有什么区别", "解释", "说明", "含义", "文档", "为什么会", "原理", "怎么理解")
REFERENCES = ("刚才", "这个请求", "那个请求", "这次", "它", "继续", "再查")
DOC_FOLLOWUPS = ("详细", "简洁", "解释", "为什么", "继续说", "再说", "怎么办")


def request_ids(text):
    return list(dict.fromkeys(re.findall(REQUEST_PATTERN, text)))


def has_topic(text):
    return bool(re.search(r"\b[1-5][0-9]{2}\b|\b[A-Z]+_[A-Z_]+\b", text)) or any(
        term in text.lower() for term in ("配额", "频率", "权限", "认证", "余额", "超时", "quota", "authentication", "timeout"))


def choose_route(message, *, target=None, summary=None):
    summary = summary or {}
    ids = request_ids(message)

    def decision(route, chosen=target, reason=""):
        return {"route": route, "target": chosen, "route_reason": reason, "routing_version": ROUTING_VERSION}

    if message.startswith("/docs "):
        return decision("knowledge", reason="explicit_docs")
    if ids:
        return decision("diagnose" if len(ids) == 1 else "wait_request", ids[0] if len(ids) == 1 else None, "explicit_request")
    explicit_diagnosis = any(word in message for word in ("排查", "再查", "查一下", "检查一下", "诊断一下"))
    if not explicit_diagnosis and any(word in message for word in QUESTION_WORDS):
        return decision("knowledge", reason="information_question")
    if (not explicit_diagnosis and summary.get("last_outcome") == "docs"
            and any(word in message for word in DOC_FOLLOWUPS)):
        return decision("knowledge", reason="document_followup")
    if explicit_diagnosis or any(word in message for word in ("报错", "失败", "出错", "诊断", "继续")):
        chosen = target if any(word in message for word in REFERENCES) else None
        return decision("diagnose" if chosen else "wait_request", chosen, "request_investigation")
    if target and any(word in message for word in ("为什么", "刚才", "这个", "它")):
        return decision("diagnose", reason="request_followup")
    return decision("knowledge", reason="default_knowledge")


def document_query(message, task):
    question = message.removeprefix("/docs ").strip()
    if not 1 <= len(question) <= 256:
        raise ValueError("文档问题请控制在 1–256 字符，避免截断关键信息。")
    topic = question
    if not has_topic(question) and any(word in question for word in (*REFERENCES, *DOC_FOLLOWUPS)):
        topic = task.get("last_document_topic") if task.get("last_outcome") == "docs" else None
        topic = topic or task.get("last_verified_cause", "").upper() or question
        if topic != question:
            question = topic[:128] + " " + question[:127]
    return question, topic[:128]
