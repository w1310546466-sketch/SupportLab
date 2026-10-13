"""A deterministic keyword baseline: retrieve evidence, never execute document text."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from .storage import PROJECT_ROOT


DEFAULT_DOCS_ROOT = PROJECT_ROOT / "docs"
DOCS_DATASET_VERSION = "docs-v1"
RETRIEVAL_VERSION = "keyword-v1"
MAX_DOCUMENT_BYTES = 64 * 1024
MAX_EXCERPT_CHARS = 1200


@dataclass(frozen=True)
class DocumentSpec:
    doc_id: str
    service_version: str
    filename: str


# Application-owned source list. A query or a document cannot add files or tools.
DOCUMENTS = (
    DocumentSpec("demo-errors-v1", "demo-api-v1", "demo-api-v1.md"),
    DocumentSpec("demo-account-state-v1", "demo-api-v1", "demo-account-state-v1.md"),
    DocumentSpec("demo-diagnostics-v1", "demo-api-v1", "demo-diagnostics-v1.md"),
)


def _tokens(text: str) -> set[str]:
    """Keep whole identifiers/codes; use adjacent character pairs for Chinese."""
    tokens = set()
    for word in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.casefold()):
        if re.fullmatch(r"[a-z0-9_]+", word):
            tokens.add(word)
        else:
            tokens.update(word[index:index + 2] for index in range(len(word) - 1))
    return tokens


def _term_weight(term: str) -> int:
    if "_" in term:  # Whole error codes and field names are more specific.
        return 10
    if re.fullmatch(r"[1-5][0-9]{2}", term):
        return 3
    return 1


def _table_separator(line: str) -> bool:
    cells = line.strip().strip("|").split("|")
    return len(cells) > 1 and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _chunks(lines: list[str]):
    """Preserve paragraphs and individual table rows with exact source lines."""
    section = ""
    table_header = None
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            table_header = None
            index += 1
            continue
        if stripped.startswith("#"):
            section = stripped.lstrip("# ")
            index += 1
            continue
        if stripped.startswith("- ") and "：" in stripped:
            index += 1  # Document metadata is not diagnosis evidence.
            continue
        if index + 1 < len(lines) and _table_separator(lines[index + 1]):
            table_header = {"line": index + 1, "text": lines[index]}
            index += 2
            continue
        start = index
        if stripped.startswith("|"):
            index += 1
        else:
            table_header = None
            while index < len(lines) and lines[index].strip():
                if lines[index].lstrip().startswith(("#", "|")):
                    break
                index += 1
        excerpt = "\n".join(lines[start:index])
        if len(excerpt) > MAX_EXCERPT_CHARS:
            raise ValueError("Document paragraph exceeds the excerpt budget.")
        yield {"line_start": start + 1, "line_end": index, "excerpt": excerpt,
               "section": section, "table_header": table_header}


def search_docs(
    query: str, service_version: str, top_k: int = 3, *, docs_root: Path = DEFAULT_DOCS_ROOT
) -> dict:
    """Only query, service_version and top_k are external tool arguments.

    docs_root and DOCUMENTS are controlled by the application. All returned text
    is untrusted reference material; lexical scores are not confidence scores.
    """
    result = {"simulation": True, "dataset_version": DOCS_DATASET_VERSION,
              "retrieval_version": RETRIEVAL_VERSION, "content_trust": "untrusted_document"}
    if (not isinstance(query, str) or not 1 <= len(query.strip()) <= 256
            or not isinstance(service_version, str)
            or re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,63}", service_version) is None
            or type(top_k) is not int or not 1 <= top_k <= 5):
        return {**result, "status": "invalid_arguments", "matches": [],
                "message": "query 须为 1–256 字符，service_version 须为版本标识，top_k 须为 1–5 的整数。"}

    documents = [doc for doc in DOCUMENTS if doc.service_version == service_version]
    if not documents:
        return {**result, "status": "not_found", "matches": [],
                "reason": "unsupported_version", "message": "没有该服务版本的资料，不回退到其他版本。"}

    query_terms = _tokens(query)
    matches = []
    try:
        root = docs_root.resolve()
        for document_order, doc in enumerate(documents):
            path = (root / doc.filename).resolve()
            if not path.is_relative_to(root):
                raise ValueError("Document must remain in the application docs directory.")
            with path.open("rb") as source:
                raw = source.read(MAX_DOCUMENT_BYTES + 1)
            if len(raw) > MAX_DOCUMENT_BYTES:
                raise ValueError("Document exceeds the source budget.")
            text = raw.decode("utf-8")
            # Metadata confirms the code-owned registry; it cannot change it.
            if (re.findall(r"^- 文档 ID：`([^`]+)`\s*$", text, re.MULTILINE) != [doc.doc_id]
                    or re.findall(r"^- 服务版本：`([^`]+)`\s*$", text, re.MULTILINE) != [doc.service_version]):
                raise ValueError("Source metadata does not match the document registry.")
            digest = sha256(raw).hexdigest()
            for chunk in _chunks(text.splitlines()):
                matched_terms = query_terms & _tokens(chunk["excerpt"])
                score = sum(_term_weight(term) for term in matched_terms)
                if score == 0:
                    continue
                matches.append({
                    **chunk, "doc_id": doc.doc_id, "service_version": doc.service_version,
                    "source": f"docs/{doc.filename}", "document_sha256": digest,
                    "citation": f"{doc.doc_id}@{digest[:12]}:L{chunk['line_start']}-L{chunk['line_end']}",
                    "score": score, "matched_terms": sorted(matched_terms),
                    "_document_order": document_order,
                })
    except (OSError, UnicodeError, ValueError):
        return {**result, "status": "storage_error", "matches": [],
                "message": "文档不可用或格式不符；请检查应用登记的模拟文档。"}

    # No frequency bonus: repeating a term cannot raise its score.
    matches.sort(key=lambda hit: (-hit["score"], hit["_document_order"], hit["line_start"]))
    selected = matches[:top_k]
    for hit in selected:
        del hit["_document_order"]
    return {**result, "status": "ok" if selected else "not_found", "matches": selected,
            "message": "返回相关原文，是否支持结论仍需核对。" if selected else "没有匹配证据，应补充信息或转人工。"}
