import json
import math
import re
from collections import Counter
from pathlib import Path
from time import time
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.rag.graph_store import load_graph_index

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#.-]{2,}")
_ANALYSIS_HINTS = ("企业级", "方案", "架构", "优化", "设计", "治理", "评估", "总结", "策略", "路线", "落地", "系统")
_COMPLEX_HINTS = ("区别", "对比", "为什么", "原因", "如何", "怎么", "步骤", "实现", "原理", "流程", "注意", "举例")
_SENTENCE_SPLIT_RE = re.compile(r"[。！？!?；;\n]+")


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


def normalize_question(question: str) -> str:
    return " ".join(text_tokens(question))


def token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(text_tokens(left))
    right_tokens = set(text_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / max(min(len(left_tokens), len(right_tokens)), 1), 4)


def classify_query(question: str) -> dict[str, Any]:
    text = (question or "").strip()
    route_type = "simple"
    matched_hints: list[str] = []
    if settings.retrieval_strategy_router_enabled:
        for hint in _ANALYSIS_HINTS:
            if hint in text:
                matched_hints.append(hint)
        if matched_hints or len(text) >= 24:
            route_type = "analysis"
        elif any(hint in text for hint in _COMPLEX_HINTS) or len(text) >= 12:
            route_type = "complex"
    vector_k = settings.retrieval_vector_k
    lexical_k = settings.retrieval_lexical_k
    final_k = settings.retrieval_final_k
    max_per_source = settings.retrieval_max_per_source
    if route_type == "complex":
        vector_k += 4
        lexical_k += 4
        final_k += 1
    elif route_type == "analysis":
        vector_k += 8
        lexical_k += 8
        final_k += 2
        max_per_source += 1
    return {
        "route_type": route_type,
        "matched_hints": matched_hints[:6],
        "vector_k": vector_k,
        "lexical_k": lexical_k,
        "final_k": final_k,
        "max_per_source": max_per_source,
    }


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

    route_plan = classify_query(question)
    return {
        "question": question,
        "expanded_question": expanded_question,
        "matched_concepts": matched_concepts,
        "query_expansions": unique_expansions[:6],
        "route_subjects": [subject for subject, _ in route_subjects.most_common(3)],
        **route_plan,
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


def score_document_coverage(question: str, content: str, metadata: dict[str, Any] | None = None, matched_concepts: list[str] | None = None) -> float:
    metadata = metadata or {}
    base = lexical_score(question, content, metadata)
    overlap = token_overlap_ratio(question, content)
    concept_bonus = 0.0
    concepts = {str(item).strip() for item in metadata.get("concepts") or [] if str(item).strip()}
    if matched_concepts and concepts:
        concept_bonus += len(set(matched_concepts) & concepts) * 0.08
    return round(base * 0.55 + overlap * 0.45 + concept_bonus, 4)


def reciprocal_rank_fusion(rank_positions: dict[str, int], constant: int = 60) -> float:
    score = 0.0
    for position in rank_positions.values():
        if position < 0:
            continue
        score += 1.0 / (constant + position + 1)
    return round(score, 6)


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


def get_score_weights(route_type: str, retry_count: int = 0, retry_strategy: str | None = None) -> dict[str, float]:
    weights: dict[str, float] = {
        "vector": 1.0,
        "lexical": 0.92,
        "graph": 0.86,
        "route": 0.72,
        "coverage": 1.05,
        "intent": 0.66,
        "consensus": 1.0,
        "rrf": 10.0,
    }
    if route_type == "complex":
        weights.update({"lexical": 1.0, "graph": 0.95, "coverage": 1.12, "intent": 0.72})
    elif route_type == "analysis":
        weights.update({"lexical": 1.08, "graph": 1.02, "coverage": 1.18, "intent": 0.82, "route": 0.8})
    if retry_count >= 1:
        weights["graph"] += 0.06
        weights["coverage"] += 0.04
        weights["consensus"] += 0.08
    if retry_strategy == "widen_retrieval":
        weights["lexical"] += 0.12
        weights["graph"] += 0.08
        weights["route"] += 0.05
    elif retry_strategy == "focus_coverage":
        weights["coverage"] += 0.16
        weights["intent"] += 0.1
    elif retry_strategy == "query_rewrite":
        weights["vector"] += 0.06
        weights["lexical"] += 0.06
    return {key: round(value, 4) for key, value in weights.items()}


def _cache_path() -> Path:
    return Path(settings.retrieval_cache_file)


def _load_retrieval_cache() -> list[dict[str, Any]]:
    path = _cache_path()
    if not settings.retrieval_cache_enabled or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _save_retrieval_cache(entries: list[dict[str, Any]]) -> None:
    if not settings.retrieval_cache_enabled:
        return
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries[: settings.retrieval_cache_max_entries], ensure_ascii=False), encoding="utf-8")


def _serialize_documents(documents: list[Document]) -> list[dict[str, Any]]:
    return [{"page_content": doc.page_content, "metadata": dict(doc.metadata or {})} for doc in documents]


def _deserialize_documents(items: list[dict[str, Any]]) -> list[Document]:
    return [Document(page_content=str(item.get("page_content") or ""), metadata=dict(item.get("metadata") or {})) for item in items]


def get_cached_retrieval(question: str, route_type: str) -> tuple[list[Document], dict[str, Any]] | None:
    if not settings.retrieval_cache_enabled:
        return None
    normalized = normalize_question(question)
    if not normalized:
        return None
    best_entry: dict[str, Any] | None = None
    best_score = 0.0
    for entry in _load_retrieval_cache():
        if str(entry.get("route_type") or "") != route_type:
            continue
        entry_question = str(entry.get("question") or "")
        entry_normalized = str(entry.get("normalized_question") or "")
        if not entry_normalized:
            continue
        similarity = 1.0 if entry_normalized == normalized else token_overlap_ratio(normalized, entry_normalized)
        if question.strip() == entry_question.strip():
            similarity = 1.0
        if similarity < settings.retrieval_cache_similarity_threshold or similarity <= best_score:
            continue
        best_entry = entry
        best_score = similarity
    if best_entry is None:
        return None
    graph_result = dict(best_entry.get("graph_result") or {})
    retrieval_summary = dict(graph_result.get("retrieval_summary") or {})
    retrieval_summary["cache_hit"] = True
    retrieval_summary["cache_similarity"] = round(best_score, 4)
    graph_result["retrieval_summary"] = retrieval_summary
    return _deserialize_documents(list(best_entry.get("documents") or [])), graph_result


def save_cached_retrieval(question: str, route_type: str, documents: list[Document], graph_result: dict[str, Any]) -> None:
    if not settings.retrieval_cache_enabled or not documents:
        return
    normalized = normalize_question(question)
    if not normalized:
        return
    entries = _load_retrieval_cache()
    compact_result = dict(graph_result or {})
    retrieval_summary = dict(compact_result.get("retrieval_summary") or {})
    retrieval_summary["cache_hit"] = False
    compact_result["retrieval_summary"] = retrieval_summary
    compact_result.pop("documents", None)
    new_entry = {
        "question": question,
        "normalized_question": normalized,
        "route_type": route_type,
        "saved_at": int(time()),
        "documents": _serialize_documents(documents),
        "graph_result": compact_result,
    }
    filtered = [entry for entry in entries if str(entry.get("normalized_question") or "") != normalized or str(entry.get("route_type") or "") != route_type]
    filtered.insert(0, new_entry)
    filtered.sort(key=lambda item: int(item.get("saved_at") or 0), reverse=True)
    _save_retrieval_cache(filtered)


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text or "")]
    return [part for part in parts if len(part) >= 4]


def validate_answer_grounding(question: str, answer: str, references: list[str], concepts: list[str] | None = None) -> dict[str, Any]:
    answer_text = (answer or "").strip()
    reference_text = "\n".join(item.strip() for item in references if item and item.strip())
    concept_text = " ".join(concepts or [])
    answer_overlap = token_overlap_ratio(answer_text, f"{reference_text} {concept_text}")
    question_overlap = token_overlap_ratio(question, answer_text)
    concept_bonus = 0.0
    answer_lower = answer_text.casefold()
    for concept in concepts or []:
        concept_text_item = str(concept).strip()
        if concept_text_item and concept_text_item.casefold() in answer_lower:
            concept_bonus += 0.04
    sentences = _split_sentences(answer_text)
    supported_claims = 0
    unsupported_claims = 0
    weak_sentences: list[str] = []
    for sentence in sentences:
        sentence_score = token_overlap_ratio(sentence, f"{reference_text} {concept_text}")
        if sentence_score >= settings.retrieval_answer_grounding_sentence_threshold:
            supported_claims += 1
        else:
            unsupported_claims += 1
            weak_sentences.append(sentence[:48])
    citation_coverage = round(supported_claims / max(len(sentences), 1), 4) if sentences else 0.0
    unsupported_penalty = min(unsupported_claims * 0.03, 0.15)
    score = round(answer_overlap * 0.56 + question_overlap * 0.2 + citation_coverage * 0.24 + concept_bonus - unsupported_penalty, 4)
    score = max(0.0, score)
    grounded = score >= settings.retrieval_answer_grounding_threshold and citation_coverage >= 0.34
    return {
        "grounded": grounded,
        "grounding_score": score,
        "reference_overlap": round(answer_overlap, 4),
        "question_overlap": round(question_overlap, 4),
        "citation_coverage": citation_coverage,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "weak_sentences": weak_sentences[:3],
    }
