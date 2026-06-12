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
_DEFINITION_HINTS = ("是什么", "什么是", "定义", "含义", "概念", "介绍")
_PROCESS_HINTS = ("如何", "怎么", "步骤", "流程", "实现", "做法", "方法")
_COMPARISON_HINTS = ("区别", "对比", "比较", "不同", "优缺点", "联系")
_ANALYSIS_QUESTION_HINTS = ("为什么", "原因", "影响", "分析", "评估", "总结")
_ADVICE_HINTS = ("建议", "怎么学", "怎么做", "入门", "注意什么", "推荐")
_ASPECT_KEYWORDS = {
    "定义": ("是", "指", "定义", "含义", "概念"),
    "核心要点": ("核心", "要点", "特点", "本质", "关键"),
    "例子": ("例如", "比如", "示例", "举例"),
    "步骤": ("步骤", "首先", "然后", "最后", "第"),
    "条件": ("条件", "前提", "适用", "要求"),
    "注意事项": ("注意", "避免", "不要", "建议"),
    "对比维度": ("维度", "方面", "从", "对比"),
    "关键差异": ("区别", "不同", "差异", "而"),
    "结论": ("因此", "总之", "结论", "所以"),
    "原因": ("原因", "因为", "导致", "影响"),
    "关键依据": ("依据", "根据", "表明", "说明"),
    "建议": ("建议", "可以", "推荐", "优先"),
    "直接答案": ("是", "为", "通常", "一般"),
    "关键事实": ("包括", "主要", "通常", "常见"),
}


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


def infer_answer_type(question: str) -> str:
    text = (question or "").strip()
    if not text:
        return "fact"
    if any(hint in text for hint in _COMPARISON_HINTS):
        return "comparison"
    if any(hint in text for hint in _PROCESS_HINTS):
        return "process"
    if any(hint in text for hint in _ANALYSIS_QUESTION_HINTS):
        return "analysis"
    if any(hint in text for hint in _DEFINITION_HINTS):
        return "definition"
    if any(hint in text for hint in _ADVICE_HINTS):
        return "advice"
    return "fact"


def expected_answer_aspects(question: str, answer_type: str | None = None) -> list[str]:
    answer_type = answer_type or infer_answer_type(question)
    if answer_type == "definition":
        return ["定义", "核心要点", "例子"]
    if answer_type == "process":
        return ["步骤", "条件", "注意事项"]
    if answer_type == "comparison":
        return ["对比维度", "关键差异", "结论"]
    if answer_type == "analysis":
        return ["结论", "原因", "关键依据"]
    if answer_type == "advice":
        return ["建议", "步骤", "注意事项"]
    return ["直接答案", "关键事实"]


def classify_query(question: str) -> dict[str, Any]:
    text = (question or "").strip()
    answer_type = infer_answer_type(text)
    route_type = "simple"
    matched_hints: list[str] = []
    router_features: list[str] = [answer_type]
    if settings.retrieval_strategy_router_enabled:
        for hint in _ANALYSIS_HINTS:
            if hint in text:
                matched_hints.append(hint)
        if answer_type in {"analysis", "comparison"} or matched_hints or len(text) >= 24:
            route_type = "analysis"
        elif answer_type == "process" or any(hint in text for hint in _COMPLEX_HINTS) or len(text) >= 12:
            route_type = "complex"
    if len(text) >= 36:
        router_features.append("long_query")
    if re.search(r"\d", text):
        router_features.append("contains_number")
    if "图" in text or "表" in text:
        router_features.append("needs_structure")
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
        "answer_type": answer_type,
        "expected_aspects": expected_answer_aspects(text, answer_type),
        "router_features": router_features,
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


def _knowledge_version() -> str:
    paths = [
        Path(settings.knowledge_metadata_file),
        Path(settings.vector_index_dir) / "documents.json",
        Path(settings.vector_index_dir) / "embeddings.npy",
    ]
    stamps = [str(int(path.stat().st_mtime)) for path in paths if path.exists()]
    return "|".join(stamps) if stamps else "unknown"


def _cache_policy(route_type: str, answer_type: str) -> dict[str, Any]:
    base_threshold = float(settings.retrieval_cache_similarity_threshold)
    if route_type == "analysis" or answer_type in {"analysis", "comparison"}:
        return {
            "policy": "strict",
            "risk": "high",
            "min_similarity": min(1.0, base_threshold + settings.retrieval_cache_strict_similarity_delta),
            "exact_only": settings.retrieval_cache_high_risk_exact_only,
        }
    if route_type == "complex" or answer_type == "process":
        return {
            "policy": "balanced",
            "risk": "medium",
            "min_similarity": min(1.0, base_threshold + settings.retrieval_cache_strict_similarity_delta / 2),
            "exact_only": False,
        }
    return {
        "policy": "fast",
        "risk": "low",
        "min_similarity": base_threshold,
        "exact_only": False,
    }


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


def get_cached_retrieval(question: str, route_type: str, answer_type: str = "fact") -> tuple[list[Document], dict[str, Any]] | None:
    if not settings.retrieval_cache_enabled:
        return None
    normalized = normalize_question(question)
    if not normalized:
        return None
    policy = _cache_policy(route_type, answer_type)
    best_entry: dict[str, Any] | None = None
    best_score = 0.0
    knowledge_version = _knowledge_version()
    for entry in _load_retrieval_cache():
        if str(entry.get("route_type") or "") != route_type:
            continue
        if str(entry.get("answer_type") or "fact") != answer_type:
            continue
        if str(entry.get("knowledge_version") or "") != knowledge_version:
            continue
        entry_question = str(entry.get("question") or "")
        entry_normalized = str(entry.get("normalized_question") or "")
        if not entry_normalized:
            continue
        similarity = 1.0 if entry_normalized == normalized else token_overlap_ratio(normalized, entry_normalized)
        if question.strip() == entry_question.strip():
            similarity = 1.0
        if policy["exact_only"] and similarity < 1.0:
            continue
        if similarity < policy["min_similarity"] or similarity <= best_score:
            continue
        best_entry = entry
        best_score = similarity
    if best_entry is None:
        return None
    graph_result = dict(best_entry.get("graph_result") or {})
    retrieval_summary = dict(graph_result.get("retrieval_summary") or {})
    retrieval_summary["cache_hit"] = True
    retrieval_summary["cache_similarity"] = round(best_score, 4)
    retrieval_summary["cache_policy"] = policy["policy"]
    retrieval_summary["cache_risk"] = policy["risk"]
    graph_result["retrieval_summary"] = retrieval_summary
    return _deserialize_documents(list(best_entry.get("documents") or [])), graph_result


def save_cached_retrieval(question: str, route_type: str, documents: list[Document], graph_result: dict[str, Any], *, answer_type: str = "fact") -> None:
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
        "answer_type": answer_type,
        "knowledge_version": _knowledge_version(),
        "saved_at": int(time()),
        "documents": _serialize_documents(documents),
        "graph_result": compact_result,
    }
    filtered = [
        entry
        for entry in entries
        if str(entry.get("normalized_question") or "") != normalized
        or str(entry.get("route_type") or "") != route_type
        or str(entry.get("answer_type") or "fact") != answer_type
    ]
    filtered.insert(0, new_entry)
    filtered.sort(key=lambda item: int(item.get("saved_at") or 0), reverse=True)
    _save_retrieval_cache(filtered)


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text or "")]
    return [part for part in parts if len(part) >= 4]


def _aspect_keywords(aspect: str) -> tuple[str, ...]:
    return _ASPECT_KEYWORDS.get(aspect, (aspect,))


def _aspect_hit(answer_text: str, aspect: str) -> bool:
    if aspect in answer_text:
        return True
    keywords = _aspect_keywords(aspect)
    if any(keyword and keyword in answer_text for keyword in keywords):
        return True
    return token_overlap_ratio(aspect, answer_text) >= 0.5


def extract_relevant_facts(question: str, references: list[str], aspects: list[str] | None = None, limit: int = 6) -> list[str]:
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    joined_aspects = " ".join(aspects or [])
    for reference in references:
        for sentence in _split_sentences(reference):
            compact = re.sub(r"\s+", " ", sentence).strip()
            if len(compact) < 8 or compact in seen:
                continue
            seen.add(compact)
            score = token_overlap_ratio(question, compact) * 0.62 + token_overlap_ratio(joined_aspects, compact) * 0.38
            if aspects and any(_aspect_hit(compact, aspect) for aspect in aspects):
                score += 0.12
            if len(compact) >= 18:
                score += 0.02
            scored.append((round(score, 4), compact))
    scored.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
    return [sentence for score, sentence in scored if score > 0][: max(1, limit)]


def get_grounding_threshold(answer_type: str) -> tuple[float, float]:
    base = float(settings.retrieval_answer_grounding_threshold)
    if answer_type in {"definition", "fact"}:
        return round(base + 0.04, 4), 0.34
    if answer_type in {"process", "analysis", "comparison"}:
        return round(base + 0.02, 4), 0.38
    if answer_type == "advice":
        return round(max(0.18, base - 0.02), 4), 0.28
    return base, 0.34


def validate_answer_grounding(
    question: str,
    answer: str,
    references: list[str],
    concepts: list[str] | None = None,
    *,
    answer_type: str | None = None,
    expected_aspects: list[str] | None = None,
    facts: list[str] | None = None,
) -> dict[str, Any]:
    answer_text = (answer or "").strip()
    reference_text = "\n".join(item.strip() for item in references if item and item.strip())
    concept_text = " ".join(concepts or [])
    answer_type = answer_type or infer_answer_type(question)
    expected_aspects = expected_aspects or expected_answer_aspects(question, answer_type)
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
    covered_aspects = [aspect for aspect in expected_aspects if _aspect_hit(answer_text, aspect)]
    missing_aspects = [aspect for aspect in expected_aspects if aspect not in covered_aspects]
    aspect_coverage = round(len(covered_aspects) / max(len(expected_aspects), 1), 4) if expected_aspects else 1.0
    facts = facts or []
    used_facts = 0
    for fact in facts:
        if token_overlap_ratio(fact, answer_text) >= 0.3 or fact in answer_text:
            used_facts += 1
    fact_coverage = round(used_facts / max(len(facts), 1), 4) if facts else 0.0
    unsupported_penalty = min(unsupported_claims * 0.03, 0.15)
    aspect_bonus = aspect_coverage * 0.18
    fact_bonus = fact_coverage * 0.12
    score = round(answer_overlap * 0.46 + question_overlap * 0.18 + citation_coverage * 0.18 + concept_bonus + aspect_bonus + fact_bonus - unsupported_penalty, 4)
    score = max(0.0, score)
    score_threshold, citation_floor = get_grounding_threshold(answer_type)
    grounded = score >= score_threshold and citation_coverage >= citation_floor and aspect_coverage >= 0.34
    return {
        "grounded": grounded,
        "grounding_score": score,
        "reference_overlap": round(answer_overlap, 4),
        "question_overlap": round(question_overlap, 4),
        "citation_coverage": citation_coverage,
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims,
        "weak_sentences": weak_sentences[:3],
        "answer_type": answer_type,
        "aspect_coverage": aspect_coverage,
        "missing_aspects": missing_aspects[:4],
        "fact_coverage": fact_coverage,
        "used_facts": used_facts,
    }
