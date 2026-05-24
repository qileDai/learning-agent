import math
import re
from collections import Counter
from typing import Any

from langchain_core.documents import Document

from app.rag.graph_store import load_graph_index

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#.-]{2,}")


def text_tokens(text: str) -> list[str]:
    raw = (text or "").strip().lower()
    tokens: list[str] = []
    tokens.extend(_TOKEN_RE.findall(raw))
    chinese = re.sub(r"[^\u4e00-\u9fff]", "", raw)
    for n in (2, 3, 4):
        for i in range(max(0, len(chinese) - n + 1)):
            gram = chinese[i : i + n]
            if gram:
                tokens.append(gram)
    return tokens


def estimate_tokens(text: str) -> int:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0
    ascii_chars = sum(1 for ch in cleaned if ord(ch) < 128)
    non_ascii_chars = len(cleaned) - ascii_chars
    return max(1, math.ceil(ascii_chars / 4) + non_ascii_chars)


def truncate_by_budget(text: str, max_tokens: int) -> str:
    content = (text or "").strip()
    if estimate_tokens(content) <= max_tokens:
        return content
    chars = max(80, max_tokens * 3)
    truncated = content[:chars].rstrip()
    while estimate_tokens(truncated) > max_tokens and len(truncated) > 80:
        chars = max(80, int(len(truncated) * 0.85))
        truncated = truncated[:chars].rstrip()
    return truncated + "…"


def compress_lines(text: str, max_tokens: int) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    kept: list[str] = []
    spent = 0
    for line in lines:
        cost = estimate_tokens(line)
        if kept and spent + cost > max_tokens:
            break
        if not kept and cost > max_tokens:
            return truncate_by_budget(line, max_tokens)
        kept.append(line)
        spent += cost
    return "\n".join(kept)


def expand_query(question: str) -> dict[str, Any]:
    graph = load_graph_index()
    concepts = graph.get("concepts", {})
    lower_question = (question or "").casefold()
    expansions: list[str] = []
    matched_concepts: list[str] = []
    route_subjects: Counter[str] = Counter()

    for name, meta in concepts.items():
        aliases = [name, *(meta.get("aliases") or [])]
        matched = False
        for alias in aliases:
            alias_text = str(alias).strip()
            if alias_text and alias_text.casefold() in lower_question:
                matched = True
                break
        if not matched:
            continue
        matched_concepts.append(name)
        subject = str(meta.get("subject") or "").strip()
        if subject:
            route_subjects[subject] += 1
        for alias in aliases:
            alias_text = str(alias).strip()
            if alias_text and alias_text.casefold() not in lower_question:
                expansions.append(alias_text)

    unique_expansions: list[str] = []
    seen: set[str] = set()
    for item in expansions:
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_expansions.append(item)

    expanded_question = question
    if unique_expansions:
        expanded_question = f"{question} {' '.join(unique_expansions[:6])}".strip()

    return {
        "question": question,
        "expanded_question": expanded_question,
        "matched_concepts": matched_concepts,
        "query_expansions": unique_expansions[:6],
        "route_subjects": [subject for subject, _ in route_subjects.most_common(3)],
    }


def lexical_score(question: str, content: str, metadata: dict[str, Any] | None = None) -> float:
    metadata = metadata or {}
    query_tokens = text_tokens(question)
    if not query_tokens:
        return 0.0
    content_text = " ".join(
        [
            content or "",
            str(metadata.get("chapter") or ""),
            str(metadata.get("summary") or ""),
            " ".join(str(item) for item in metadata.get("concepts") or []),
            " ".join(str(item) for item in metadata.get("aliases") or []),
        ]
    )
    doc_tokens = text_tokens(content_text)
    if not doc_tokens:
        return 0.0
    query_counter = Counter(query_tokens)
    doc_counter = Counter(doc_tokens)
    overlap = 0.0
    strong_hits = 0
    for token, count in query_counter.items():
        if token not in doc_counter:
            continue
        overlap += min(count, doc_counter[token])
        if len(token) >= 3:
            strong_hits += 1
    if overlap == 0:
        return 0.0
    coverage = overlap / max(len(query_tokens), 1)
    density = overlap / max(len(doc_tokens), 1)
    concept_bonus = 0.0
    concepts = {str(item).strip().lower() for item in metadata.get("concepts") or [] if str(item).strip()}
    for token in query_counter:
        if token in concepts:
            concept_bonus += 0.08
    return round(coverage * 0.75 + density * 1.5 + strong_hits * 0.06 + concept_bonus, 4)


def diversify_documents(documents: list[Document], final_k: int, max_per_source: int) -> list[Document]:
    picked: list[Document] = []
    per_source: Counter[str] = Counter()
    seen_signatures: set[tuple[str, str]] = set()
    for doc in documents:
        meta = dict(doc.metadata or {})
        source = str(meta.get("source") or "unknown")
        signature = (source, re.sub(r"\s+", " ", doc.page_content[:180]).strip())
        if signature in seen_signatures:
            continue
        if per_source[source] >= max_per_source:
            continue
        seen_signatures.add(signature)
        per_source[source] += 1
        picked.append(doc)
        if len(picked) >= final_k:
            break
    return picked
