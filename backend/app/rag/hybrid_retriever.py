"""混合检索主流程。

调用关系：
1. graph/nodes.py -> hybrid_retrieve：回答链路的检索入口
2. hybrid_retrieve -> expand_query：先做查询扩展与路由
3. hybrid_retrieve -> search_graph：先拿图谱概念、关系、来源分数
4. hybrid_retrieve -> similarity_search_with_scores：查向量召回
5. hybrid_retrieve -> _build_lexical_candidates：查 ES / 本地词法召回
6. hybrid_retrieve -> reciprocal_rank_fusion / diversify_documents：做融合排序与多样性控制
7. hybrid_retrieve -> save_cached_retrieval：缓存最终结果

这个模块是整个企业级 RAG 的“编排层”，负责把图谱、向量、关键词三路结果合成最终候选。
"""

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
    lexical_score,
    reciprocal_rank_fusion,
    save_cached_retrieval,
    score_document_coverage,
)
from app.rag.vector_store import load_index_documents, similarity_search_with_scores


def _doc_key(doc: Document) -> tuple[str, str]:
    """生成候选文档主键，用于多路召回结果合并去重。"""
    meta = dict(doc.metadata or {})
    chunk_id = str(meta.get("chunk_id") or "").strip()
    if chunk_id:
        return str(meta.get("source", "unknown")).strip(), chunk_id
    return str(meta.get("source", "unknown")).strip(), doc.page_content[:160]


def _route_boost(meta: dict[str, Any], route_subjects: list[str], source_scores: dict[str, float], matched_concepts: list[str]) -> float:
    """计算图谱路由带来的额外加分。"""
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
    """根据查询类型给结构化资料增加意图适配分。"""
    concepts = [str(item).strip() for item in meta.get("concepts") or [] if str(item).strip()]
    summary = str(meta.get("summary") or "")
    chapter = str(meta.get("chapter") or "")
    if route_type == "analysis":
        return round(min(len(concepts), 4) * 0.04 + (0.08 if summary or chapter else 0.0), 4)
    if route_type == "complex":
        return round(min(len(concepts), 3) * 0.03 + (0.05 if chapter else 0.0), 4)
    return round(0.04 if concepts else 0.0, 4)


def _build_lexical_candidates(expanded_question: str, route_subjects: list[str], lexical_k: int) -> list[tuple[Document, float, int]]:
    """构建关键词召回候选。

    优先走 Elasticsearch；如果 ES 不可用，则回退到本地文档做轻量词法打分。
    """
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


def hybrid_retrieve(question: str, vector_k: int | None = None) -> tuple[list[Document], dict]:
    """执行混合检索。

    主流程：
    1. 查询扩展和路由
    2. 语义缓存命中判断
    3. 图谱检索
    4. 向量召回
    5. 关键词召回
    6. 多源融合排序
    7. 多样性裁剪
    8. 回写检索摘要和缓存
    """
    query_plan = expand_query(question)
    route_type = str(query_plan.get("route_type") or "simple")
    vector_k = max(int(vector_k or 0), int(query_plan.get("vector_k") or settings.retrieval_vector_k))
    lexical_k = int(query_plan.get("lexical_k") or settings.retrieval_lexical_k)
    final_k = int(query_plan.get("final_k") or settings.retrieval_final_k)
    max_per_source = int(query_plan.get("max_per_source") or settings.retrieval_max_per_source)

    # 先看语义缓存，命中则直接返回，降低重复检索成本。
    cached = get_cached_retrieval(question, route_type)
    if cached is not None:
        documents, graph_result = cached
        retrieval_summary = dict(graph_result.get("retrieval_summary") or {})
        retrieval_summary.setdefault("route_type", route_type)
        retrieval_summary.setdefault("vector_k", vector_k)
        retrieval_summary.setdefault("lexical_k", lexical_k)
        retrieval_summary.setdefault("rerank_window", settings.retrieval_rerank_window)
        retrieval_summary.setdefault("final_k", final_k)
        retrieval_summary.setdefault("cache_hit", True)
        retrieval_summary.setdefault("max_per_source", max_per_source)
        graph_result["retrieval_summary"] = retrieval_summary
        graph_result.setdefault("query_plan", query_plan)
        return documents, graph_result

    # 图谱先行：它提供概念命中、关系扩展，以及来源 source 的先验分数。
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

    # 图谱文档本身也会参与最终生成，用于给 LLM 提供关系上下文。
    documents: list[Document] = list(graph_result.get("documents", []))
    graph_docs = len(documents)
    for doc in documents:
        candidate_map[_doc_key(doc)] = doc

    # 向量召回会同时尝试原问题和扩展问题，提升语义召回率。
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
            existing = candidate_map.get(key)
            if existing is None:
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

    # 关键词召回补齐“明确术语 / 概念名 / 章节名”这类语义检索不稳定的场景。
    lexical_ranked = _build_lexical_candidates(query_plan["expanded_question"], route_subjects, lexical_k)
    for doc, lexical, rank in lexical_ranked:
        meta = dict(doc.metadata or {})
        route = _route_boost(meta, route_subjects, source_scores, matched_concepts)
        normalized_lexical = round(min(float(lexical), 4.0), 4)
        coverage = score_document_coverage(question, doc.page_content, meta, matched_concepts)
        intent = _intent_boost(meta, route_type)
        key = _doc_key(doc)
        existing = candidate_map.get(key)
        if existing is None:
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

        # consensus 表示同一 chunk 被多路召回同时命中，通常更可靠。
        channel_count = sum(1 for name in ("vector", "lexical", "graph") if score_parts[name] > 0)
        score_parts["consensus"] = round(channel_count * 0.06, 4)
        score_parts["rrf"] = reciprocal_rank_fusion(rank_positions.get(key, {}))

        # 最终排序分数 = 多路召回强度 + 图谱路由 + 覆盖度 + 意图匹配 + RRF。
        final_rank_score = round(
            score_parts["vector"]
            + score_parts["lexical"]
            + score_parts["graph"]
            + score_parts["route"]
            + score_parts["coverage"]
            + score_parts["intent"]
            + score_parts["consensus"]
            + score_parts["rrf"] * 10,
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
        }

        # score 字段保留“越小越接近”的兼容语义，供旧逻辑继续使用。
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

    # 先取 rerank_window，再做 source 多样性控制，避免全被同一资料占满。
    rerank_window = min(len(normal_docs), settings.retrieval_rerank_window)
    diversified = diversify_documents(normal_docs[:rerank_window], final_k, max_per_source)
    final_docs = [*graph_docs_list[:1], *diversified]

    graph_result["query_plan"] = query_plan
    graph_result["retrieval_summary"] = {
        "query_expansions": query_plan.get("query_expansions", []),
        "route_subjects": route_subjects,
        "route_type": route_type,
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
    }
    save_cached_retrieval(question, route_type, final_docs, graph_result)
    return final_docs, graph_result
