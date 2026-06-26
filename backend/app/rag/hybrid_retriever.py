from collections import defaultdict
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.rag.elastic_store import elasticsearch_enabled, lexical_search as elasticsearch_lexical_search
from app.rag.graph_store import search_graph
from app.rag.retrieval_optimizer import (
    diversify_documents,
    expand_query,
    get_cached_retrieval,
    get_score_weights,
    lexical_score,
    reciprocal_rank_fusion,
    save_cached_retrieval,
    score_document_coverage,
)
from app.rag.vector_store import load_index_documents, similarity_search_with_scores


def _doc_key(doc: Document) -> tuple[str, str]:
    meta = dict(doc.metadata or {})
    chunk_id = str(meta.get("chunk_id") or "").strip()
    if chunk_id:
        return str(meta.get("source", "unknown")).strip(), chunk_id
    return str(meta.get("source", "unknown")).strip(), doc.page_content[:160]


def _route_boost(meta: dict[str, Any], route_subjects: list[str], source_scores: dict[str, float], matched_concepts: list[str]) -> float:
    source = str(meta.get("source", "")).strip()
    concepts = {str(item).strip() for item in meta.get("concepts") or [] if str(item).strip()}
    subject = str(meta.get("subject") or "").strip()
    boost = float(source_scores.get(source, 0.0))
    if route_subjects and subject in route_subjects:
        boost += 0.12
    if matched_concepts and concepts:
        overlap = len(concepts & set(matched_concepts))
        boost += overlap * 0.08
    return round(boost, 4)


def _intent_boost(meta: dict[str, Any], route_type: str) -> float:
    concepts = [str(item).strip() for item in meta.get("concepts") or [] if str(item).strip()]
    summary = str(meta.get("summary") or "")
    chapter = str(meta.get("chapter") or "")
    if route_type == "analysis":
        return round(min(len(concepts), 4) * 0.04 + (0.08 if summary or chapter else 0.0), 4)
    if route_type == "complex":
        return round(min(len(concepts), 3) * 0.03 + (0.05 if chapter else 0.0), 4)
    return round(0.04 if concepts else 0.0, 4)


def _build_lexical_candidates(expanded_question: str, route_subjects: list[str], lexical_k: int) -> list[tuple[Document, float, int]]:
    if elasticsearch_enabled():
        results = elasticsearch_lexical_search(expanded_question, limit=lexical_k, route_subjects=route_subjects)
        if results:
            return results
    candidates: list[tuple[Document, float]] = []
    for doc in load_index_documents():
        meta = dict(doc.metadata or {})
        score = lexical_score(expanded_question, doc.page_content, meta)
        if score <= 0:
            continue
        if route_subjects and str(meta.get("subject") or "").strip() in route_subjects:
            score += 0.08
        candidates.append((Document(page_content=doc.page_content, metadata=meta), round(score, 4)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [(doc, score, rank) for rank, (doc, score) in enumerate(candidates[:lexical_k])]


def _backfill_context(question: str, final_docs: list[Document], matched_concepts: list[str]) -> list[Document]:
    normal_docs = [doc for doc in final_docs if str(doc.metadata.get("file_type") or "") != "graph"]
    if not normal_docs:
        return final_docs
    seeds = normal_docs[:2]
    target_sources = {str(doc.metadata.get("source") or "").strip() for doc in seeds if str(doc.metadata.get("source") or "").strip()}
    target_chapters = {str(doc.metadata.get("chapter") or "").strip() for doc in seeds if str(doc.metadata.get("chapter") or "").strip()}
    existing_keys = {_doc_key(doc) for doc in final_docs}
    candidates: list[tuple[float, Document]] = []
    for doc in load_index_documents():
        key = _doc_key(doc)
        if key in existing_keys:
            continue
        meta = dict(doc.metadata or {})
        source = str(meta.get("source") or "").strip()
        chapter = str(meta.get("chapter") or "").strip()
        if source not in target_sources and chapter not in target_chapters:
            continue
        coverage = score_document_coverage(question, doc.page_content, meta, matched_concepts)
        if coverage <= 0:
            continue
        boost = 0.0
        if source in target_sources:
            boost += 0.18
        if chapter and chapter in target_chapters:
            boost += 0.1
        if matched_concepts and set(matched_concepts) & set(meta.get("concepts") or []):
            boost += 0.08
        meta["coverage_score"] = round(max(float(meta.get("coverage_score") or 0.0), coverage + boost), 4)
        candidates.append((coverage + boost, Document(page_content=doc.page_content, metadata=meta)))
    candidates.sort(key=lambda item: (-item[0], -len(item[1].page_content)))
    supplemented = list(final_docs)
    for _, doc in candidates[:2]:
        supplemented.append(doc)
    return supplemented


def hybrid_retrieve(question: str, vector_k: int | None = None, *, retry_count: int = 0, retry_strategy: str | None = None) -> tuple[list[Document], dict]:
    query_plan = expand_query(question)
    route_type = str(query_plan.get("route_type") or "simple")
    answer_type = str(query_plan.get("answer_type") or "fact")
    vector_k = max(int(vector_k or 0), int(query_plan.get("vector_k") or settings.retrieval_vector_k))
    lexical_k = int(query_plan.get("lexical_k") or settings.retrieval_lexical_k)
    final_k = int(query_plan.get("final_k") or settings.retrieval_final_k)
    max_per_source = int(query_plan.get("max_per_source") or settings.retrieval_max_per_source)

    if retry_count >= 1 or retry_strategy == "widen_retrieval":
        vector_k += 4
        lexical_k += 4
        final_k += 1
    if retry_count >= 2:
        max_per_source += 1
        final_k += 1

    score_weights = get_score_weights(route_type, retry_count=retry_count, retry_strategy=retry_strategy)

    cached = get_cached_retrieval(question, route_type, answer_type)
    if cached is not None and retry_count == 0:
        documents, graph_result = cached
        retrieval_summary = dict(graph_result.get("retrieval_summary") or {})
        retrieval_summary.setdefault("route_type", route_type)
        retrieval_summary.setdefault("answer_type", answer_type)
        retrieval_summary.setdefault("router_features", query_plan.get("router_features") or [])
        retrieval_summary.setdefault("intent_profile", query_plan.get("intent_profile") or {})
        retrieval_summary.setdefault("vector_k", vector_k)
        retrieval_summary.setdefault("lexical_k", lexical_k)
        retrieval_summary.setdefault("rerank_window", settings.retrieval_rerank_window)
        retrieval_summary.setdefault("final_k", final_k)
        retrieval_summary.setdefault("cache_hit", True)
        retrieval_summary.setdefault("max_per_source", max_per_source)
        retrieval_summary["retry_count"] = retry_count
        retrieval_summary["retry_strategy"] = retry_strategy or "none"
        retrieval_summary["score_profile"] = score_weights
        graph_result["retrieval_summary"] = retrieval_summary
        graph_result.setdefault("query_plan", query_plan)
        return documents, graph_result

    graph_result = search_graph(query_plan["expanded_question"])
    source_scores = graph_result.get("source_scores", {})
    matched_concepts = [item.get("name", "") for item in graph_result.get("matched_concepts", []) if item.get("name")]
    route_subjects = query_plan.get("route_subjects") or [item.get("subject", "") for item in graph_result.get("matched_concepts", []) if item.get("subject")]

    candidate_map: dict[tuple[str, str], Document] = {}
    candidate_scores: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "vector": 0.0,
            "lexical": 0.0,
            "graph": 0.0,
            "route": 0.0,
            "coverage": 0.0,
            "intent": 0.0,
            "rrf": 0.0,
            "consensus": 0.0,
        }
    )
    rank_positions: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)

    documents: list[Document] = list(graph_result.get("documents", []))
    graph_docs = len(documents)
    for doc in documents:
        candidate_map[_doc_key(doc)] = doc

    vector_queries = [question]
    if query_plan["expanded_question"] != question:
        vector_queries.append(query_plan["expanded_question"])

    vector_rank = 0
    for query in vector_queries:
        for doc, distance in similarity_search_with_scores(query, k=vector_k):
            meta = dict(doc.metadata or {})
            route = _route_boost(meta, route_subjects, source_scores, matched_concepts)
            vector_score = max(0.0, 1.3 - float(distance))
            coverage = score_document_coverage(question, doc.page_content, meta, matched_concepts)
            intent = _intent_boost(meta, route_type)
            key = _doc_key(doc)
            if key not in candidate_map:
                meta["score"] = round(float(distance) - route, 4)
                meta["retrieval_mode"] = "vector"
                if matched_concepts and "graph_matched_concepts" not in meta:
                    meta["graph_matched_concepts"] = matched_concepts
                candidate_map[key] = Document(page_content=doc.page_content, metadata=meta)
            candidate_scores[key]["vector"] = max(candidate_scores[key]["vector"], round(vector_score, 4))
            candidate_scores[key]["graph"] = max(candidate_scores[key]["graph"], round(float(source_scores.get(str(meta.get("source", "")).strip(), 0.0)), 4))
            candidate_scores[key]["route"] = max(candidate_scores[key]["route"], route)
            candidate_scores[key]["coverage"] = max(candidate_scores[key]["coverage"], coverage)
            candidate_scores[key]["intent"] = max(candidate_scores[key]["intent"], intent)
            current_rank = rank_positions[key].get("vector", vector_rank)
            rank_positions[key]["vector"] = min(current_rank, vector_rank)
            vector_rank += 1

    lexical_ranked = _build_lexical_candidates(query_plan["expanded_question"], route_subjects, lexical_k)
    for doc, lexical, rank in lexical_ranked:
        meta = dict(doc.metadata or {})
        route = _route_boost(meta, route_subjects, source_scores, matched_concepts)
        normalized_lexical = round(min(float(lexical), 4.0), 4)
        coverage = score_document_coverage(question, doc.page_content, meta, matched_concepts)
        intent = _intent_boost(meta, route_type)
        key = _doc_key(doc)
        if key not in candidate_map:
            meta["score"] = round(max(0.0, 1.0 - normalized_lexical * 0.18), 4)
            meta["retrieval_mode"] = "lexical"
            if matched_concepts and "graph_matched_concepts" not in meta:
                meta["graph_matched_concepts"] = matched_concepts
            candidate_map[key] = Document(page_content=doc.page_content, metadata=meta)
        candidate_scores[key]["lexical"] = max(candidate_scores[key]["lexical"], normalized_lexical)
        candidate_scores[key]["graph"] = max(candidate_scores[key]["graph"], round(float(source_scores.get(str(meta.get("source", "")).strip(), 0.0)), 4))
        candidate_scores[key]["route"] = max(candidate_scores[key]["route"], route)
        candidate_scores[key]["coverage"] = max(candidate_scores[key]["coverage"], coverage)
        candidate_scores[key]["intent"] = max(candidate_scores[key]["intent"], intent)
        current_rank = rank_positions[key].get("lexical", rank)
        rank_positions[key]["lexical"] = min(current_rank, rank)

    ranked: list[Document] = []
    for key, doc in candidate_map.items():
        if str(doc.metadata.get("file_type")) == "graph":
            ranked.append(doc)
            continue
        meta = dict(doc.metadata or {})
        score_parts = candidate_scores[key]
        channel_count = sum(1 for name in ("vector", "lexical", "graph") if score_parts[name] > 0)
        score_parts["consensus"] = round(channel_count * 0.06, 4)
        score_parts["rrf"] = reciprocal_rank_fusion(rank_positions.get(key, {}))
        final_rank_score = round(
            score_parts["vector"] * score_weights["vector"]
            + score_parts["lexical"] * score_weights["lexical"]
            + score_parts["graph"] * score_weights["graph"]
            + score_parts["route"] * score_weights["route"]
            + score_parts["coverage"] * score_weights["coverage"]
            + score_parts["intent"] * score_weights["intent"]
            + score_parts["consensus"] * score_weights["consensus"]
            + score_parts["rrf"] * score_weights["rrf"],
            4,
        )
        modes: list[str] = []
        if score_parts["vector"] > 0:
            modes.append("vector")
        if score_parts["lexical"] > 0:
            modes.append("lexical")
        if score_parts["graph"] > 0 or score_parts["route"] > 0:
            modes.append("graph")
        meta["retrieval_mode"] = "+".join(modes) if modes else str(meta.get("retrieval_mode") or "vector")
        meta["rank_score"] = final_rank_score
        meta["coverage_score"] = score_parts["coverage"]
        meta["retrieval_debug"] = {
            "vector": score_parts["vector"],
            "lexical": score_parts["lexical"],
            "graph": score_parts["graph"],
            "route": score_parts["route"],
            "coverage": score_parts["coverage"],
            "intent": score_parts["intent"],
            "consensus": score_parts["consensus"],
            "rrf": score_parts["rrf"],
            "weights": score_weights,
        }
        meta["score"] = round(float(meta.get("score", 1.0)) - score_parts["graph"] * 0.2 - score_parts["lexical"] * 0.05 - score_parts["coverage"] * 0.08, 4)
        if matched_concepts and "graph_matched_concepts" not in meta:
            meta["graph_matched_concepts"] = matched_concepts
        if route_subjects:
            meta["route_subjects"] = route_subjects
        meta["route_type"] = route_type
        ranked.append(Document(page_content=doc.page_content, metadata=meta))

    graph_docs_list = [doc for doc in ranked if str(doc.metadata.get("file_type")) == "graph"]
    normal_docs = [doc for doc in ranked if str(doc.metadata.get("file_type")) != "graph"]
    normal_docs.sort(
        key=lambda item: (
            -float(item.metadata.get("rank_score", 0.0)),
            -float(item.metadata.get("coverage_score", 0.0)),
            float(item.metadata.get("score", 1.0)),
            str(item.metadata.get("source", "")),
        )
    )

    rerank_window = min(len(normal_docs), settings.retrieval_rerank_window + max(retry_count * 4, 0))
    diversified = diversify_documents(normal_docs[:rerank_window], final_k, max_per_source)
    final_docs = [*graph_docs_list[:1], *diversified]
    final_docs = _backfill_context(question, final_docs, matched_concepts)

    graph_result["query_plan"] = query_plan
    graph_result["retrieval_summary"] = {
        "query_expansions": query_plan.get("query_expansions", []),
        "route_subjects": route_subjects,
        "route_type": route_type,
        "answer_type": answer_type,
        "router_features": query_plan.get("router_features") or [],
        "intent_profile": query_plan.get("intent_profile") or {},
        "graph_documents": graph_docs,
        "vector_candidates": len([doc for doc in ranked if "vector" in str(doc.metadata.get("retrieval_mode", ""))]),
        "lexical_candidates": len([doc for doc in ranked if "lexical" in str(doc.metadata.get("retrieval_mode", ""))]),
        "final_candidates": len(final_docs),
        "max_per_source": max_per_source,
        "vector_k": vector_k,
        "lexical_k": lexical_k,
        "final_k": final_k,
        "rerank_window": rerank_window,
        "cache_hit": False,
        "cache_similarity": 0.0,
        "cache_policy": "strict" if route_type == "analysis" else "balanced" if route_type == "complex" else "fast",
        "cache_risk": "high" if route_type == "analysis" else "medium" if route_type == "complex" else "low",
        "retry_count": retry_count,
        "retry_strategy": retry_strategy or "none",
        "score_profile": score_weights,
        "planner_queries": [],
        "selected_by": "pending",
        "selection_confidence": 0.0,
        "evidence_sources": [str(doc.metadata.get("source") or "") for doc in final_docs if str(doc.metadata.get("source") or "")],
    }
    save_cached_retrieval(question, route_type, final_docs, graph_result, answer_type=answer_type)
    return final_docs, graph_result
