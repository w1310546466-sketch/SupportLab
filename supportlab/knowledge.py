"""Versioned local lexical RAG index with source-line citations and freshness checks."""

from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3

from .retrieval import DEFAULT_DOCS_ROOT, DOCUMENTS, DocumentSpec, MAX_DOCUMENT_BYTES, _chunks, _tokens, _term_weight
from .storage import PROJECT_ROOT, open_readonly_db


INDEX_VERSION = "fts-keyword-v3"
DEFAULT_INDEX = PROJECT_ROOT / "data" / "knowledge.sqlite3"
KNOWLEDGE_DOCUMENTS = (*DOCUMENTS, DocumentSpec("demo-support-guide-v1", "demo-api-v1", "demo-support-guide-v1.md"))
ALIASES = {"太快": "RATE_LIMIT_EXCEEDED", "频繁": "RATE_LIMIT_EXCEEDED", "次数用完": "MONTHLY_QUOTA_EXHAUSTED",
           "配额耗尽": "MONTHLY_QUOTA_EXHAUSTED", "月度额度": "MONTHLY_QUOTA_EXHAUSTED",
           "authentication": "401 身份认证", "permission denied": "403 PERMISSION_DENIED",
           "quota exhausted": "MONTHLY_QUOTA_EXHAUSTED", "rate limit": "RATE_LIMIT_EXCEEDED", "timeout": "超时"}
# Generic question wording must not count as subject evidence. This is a lexical
# filter, not a claim of general semantic relevance or calibrated confidence.
COMMON_QUERY_TERMS = {"api", "支持", "文档", "说明", "请问", "请说", "明文", "档里", "里的", "参数",
                      "这个", "服务", "如何", "什么", "为什", "么会", "怎么", "怎样", "可以", "是否", "的是",
                      "一下", "详细", "解释", "细解", "释一", "是什", "么意", "意思", "the", "a", "is", "what", "please"}
# Split generic phrases before Chinese bigrams so “这个服务” cannot accidentally
# match “整个服务” through the meaningless fragment “个服”. Preserve topical words.
GENERIC_PHRASES = ("请说明", "文档里的", "这个服务", "这个", "如何", "怎么样", "今天", "请问")


def _sources(docs_root):
    root = Path(docs_root).resolve()
    sources = []
    for document in KNOWLEDGE_DOCUMENTS:
        path = (root / document.filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Invalid registered document path")
        with path.open("rb") as source:
            raw = source.read(MAX_DOCUMENT_BYTES + 1)
        if len(raw) > MAX_DOCUMENT_BYTES:
            raise ValueError("Document exceeds source limit")
        text = raw.decode("utf-8")
        if (re.findall(r"^- 文档 ID：`([^`]+)`\s*$", text, re.M) != [document.doc_id]
                or re.findall(r"^- 服务版本：`([^`]+)`\s*$", text, re.M) != [document.service_version]):
            raise ValueError("Document metadata mismatch")
        sources.append((document, sha256(raw).hexdigest(), text))
    return sources


def _manifest(sources):
    return {document.filename: digest for document, digest, _ in sources}


def build_index(*, index_path=DEFAULT_INDEX, docs_root=DEFAULT_DOCS_ROOT):
    sources = _sources(docs_root)
    rows = []
    for document, digest, content in sources:
        for chunk in _chunks(content.splitlines()):
            hit = {**chunk, "doc_id": document.doc_id, "service_version": document.service_version,
                   "source": f"docs/{document.filename}", "document_sha256": digest,
                   "citation": f"{document.doc_id}@{digest[:12]}:L{chunk['line_start']}-L{chunk['line_end']}"}
            rows.append((" ".join(sorted(_tokens(chunk["excerpt"]))), json.dumps(hit, ensure_ascii=False)))
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(index_path, timeout=2)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS knowledge_meta (id INTEGER PRIMARY KEY CHECK(id=1), manifest TEXT, version TEXT)")
        db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks USING fts5(tokens, hit UNINDEXED)")
        db.execute("DELETE FROM knowledge_chunks")
        db.executemany("INSERT INTO knowledge_chunks(tokens,hit) VALUES(?,?)", rows)
        db.execute("INSERT OR REPLACE INTO knowledge_meta VALUES(1,?,?)", (json.dumps(_manifest(sources)), INDEX_VERSION))
    return {"status": "ready", "index_version": INDEX_VERSION, "document_count": len(sources), "chunk_count": len(rows),
            "manifest": _manifest(sources), "simulation": True}


def retrieve(query, *, service_version="demo-api-v1", top_k=3, index_path=DEFAULT_INDEX, docs_root=DEFAULT_DOCS_ROOT):
    result = {"simulation": True, "index_version": INDEX_VERSION, "matches": [], "content_trust": "untrusted_document"}
    if (not isinstance(query, str) or not 1 <= len(query.strip()) <= 256
            or type(top_k) is not int or not 1 <= top_k <= 5
            or not isinstance(service_version, str) or re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,63}", service_version) is None):
        return {**result, "status": "invalid_arguments"}
    try:
        sources = _sources(docs_root)
        with closing(open_readonly_db(Path(index_path))) as db:
            metadata = db.execute("SELECT manifest,version FROM knowledge_meta WHERE id=1").fetchone()
            if metadata is None or metadata[1] != INDEX_VERSION or json.loads(metadata[0]) != _manifest(sources):
                return {**result, "status": "stale_index"}
            expansions = [code for phrase, code in ALIASES.items() if phrase in query.lower()]
            normalized = query
            for phrase in GENERIC_PHRASES:
                normalized = normalized.replace(phrase, " ")
            terms = _tokens(normalized + " " + " ".join(expansions)) - COMMON_QUERY_TERMS
            if not terms:
                return {**result, "status": "not_found"}
            expression = " OR ".join('"' + term + '"' for term in sorted(terms))
            candidates = db.execute("SELECT hit,bm25(knowledge_chunks) FROM knowledge_chunks WHERE knowledge_chunks MATCH ? ORDER BY bm25(knowledge_chunks) LIMIT 200", (expression,)).fetchall()
        hits = []
        for raw, bm25 in candidates:
            hit = json.loads(raw)
            if hit["service_version"] != service_version:
                continue
            matched = terms & _tokens(hit["excerpt"])
            score = sum(_term_weight(term) for term in matched)
            if score:
                hits.append({**hit, "score": score, "bm25": bm25, "matched_terms": sorted(matched)})
        hits.sort(key=lambda hit: (-hit["score"], hit["bm25"], hit["source"], hit["line_start"]))
        return {**result, "status": "ok" if hits else "not_found", "matches": hits[:top_k],
                "query_expansions": expansions, "query_terms": sorted(terms), "source_manifest": _manifest(sources)}
    except (OSError, ValueError, sqlite3.Error):
        return {**result, "status": "index_unavailable"}
