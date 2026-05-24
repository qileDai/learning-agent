from collections import defaultdict
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.rag.graph_store import search_graph
from app.rag.retrieval_optimizer import diversify_documents, expand_query, lexical_score
from app.rag.vector_store import load_index_documents, similarity_search_with_scores


def _doc_key(doc: Document) -> tuple[str, str]:
    meta = dict(doc.metadata or {})
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
    return boost


def _build_lexical_candidates(question: str, expanded_question: str, route_subjects: list[str]) -> list[tuple[Document, float]]:
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
    return candidates[: settings.retrieval_lexical_k]


def hybrid_retrieve(question: str, vector_k: int | None = None) -> tuple[list[Document], dict]:
    vector_k = vector_k or settings.retrieval_vector_k
    query_plan = expand_query(question)
    graph_result = search_graph(query_plan["expanded_question"])
    source_scores = graph_result.get("source_scores", {})
    matched_concepts = [item.get("name", "") for item in graph_result.get("matched_concepts", []) if item.get("name")]
    route_subjects = query_plan.get("route_subjects") or [item.get("subject", "") for item in graph_result.get("matched_concepts", []) if item.get("subject")]

    candidate_map: dict[tuple[str, str], Document] = {}
    candidate_scores: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"vector": 0.0, "lexical": 0.0, "graph": 0.0, "route": 0.0})

    documents: list[Document] = list(graph_result.get("documents", []))
    graph_docs = len(documents)
    for doc in documents:
        candidate_map[_doc_key(doc)] = doc

    vector_queries = [question]
    if query_plan["expanded_question"] != question:
        vector_queries.append(query_plan["expanded_question"])

    for query in vector_queries:
        for doc, distance in similarity_search_with_scores(query, k=vector_k):
            meta = dict(doc.metadata or {})
            route = _route_boost(meta, route_subjects, source_scores, matched_concepts)
            vector_score = max(0.0, 1.3 - float(distance))
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
            candidate_scores[key]["route"] = max(candidate_scores[key]["route"], round(route, 4))

    for doc, lexical in _build_lexical_candidates(question, query_plan["expanded_question"], route_subjects):
        meta = dict(doc.metadata or {})
        route = _route_boost(meta, route_subjects, source_scores, matched_concepts)
        key = _doc_key(doc)
        existing = candidate_map.get(key)
        if existing is None:
            meta["score"] = round(max(0.0, 1.0 - lexical * 0.18), 4)
            meta["retrieval_mode"] = "lexical"
            if matched_concepts and "graph_matched_concepts" not in meta:
                meta["graph_matched_concepts"] = matched_concepts
            candidate_map[key] = Document(page_content=doc.page_content, metadata=meta)
        candidate_scores[key]["lexical"] = max(candidate_scores[key]["lexical"], lexical)
        candidate_scores[key]["graph"] = max(candidate_scores[key]["graph"], round(float(source_scores.get(str(meta.get("source", "")).strip(), 0.0)), 4))
        candidate_scores[key]["route"] = max(candidate_scores[key]["route"], round(route, 4))

    ranked: list[Document] = []
    for key, doc in candidate_map.items():
        if str(doc.metadata.get("file_type")) == "graph":
            ranked.append(doc)
            continue
        meta = dict(doc.metadata or {})
        score_parts = candidate_scores[key]
        final_rank_score = round(score_parts["vector"] + score_parts["lexical"] + score_parts["graph"] + score_parts["route"], 4)
        source = str(meta.get("source", "")).strip()
        modes: list[str] = []
        if score_parts["vector"] > 0:
            modes.append("vector")
        if score_parts["lexical"] > 0:
            modes.append("lexical")
        if score_parts["graph"] > 0 or score_parts["route"] > 0:
            modes.append("graph")
        meta["retrieval_mode"] = "+".join(modes) if modes else str(meta.get("retrieval_mode") or "vector")
        meta["rank_score"] = final_rank_score
        meta["retrieval_debug"] = {
            "vector": score_parts["vector"],
            "lexical": score_parts["lexical"],
            "graph": score_parts["graph"],
            "route": score_parts["route"],
        }
        meta["score"] = round(float(meta.get("score", 1.0)) - score_parts["graph"] * 0.2 - score_parts["lexical"] * 0.05, 4)
        if matched_concepts and "graph_matched_concepts" not in meta:
            meta["graph_matched_concepts"] = matched_concepts
        if route_subjects:
            meta["route_subjects"] = route_subjects
        ranked.append(Document(page_content=doc.page_content, metadata=meta))

    graph_docs_list = [doc for doc in ranked if str(doc.metadata.get("file_type")) == "graph"]
    normal_docs = [doc for doc in ranked if str(doc.metadata.get("file_type")) != "graph"]
    normal_docs.sort(key=lambda item: (-float(item.metadata.get("rank_score", 0.0)), float(item.metadata.get("score", 1.0)), str(item.metadata.get("source", ""))))
    diversified = diversify_documents(normal_docs, settings.retrieval_final_k, settings.retrieval_max_per_source)
    final_docs = [*graph_docs_list[:1], *diversified]

    graph_result["query_plan"] = query_plan
    graph_result["retrieval_summary"] = {
        "query_expansions": query_plan.get("query_expansions", []),
        "route_subjects": route_subjects,
        "graph_documents": graph_docs,
        "vector_candidates": len([doc for doc in ranked if "vector" in str(doc.metadata.get("retrieval_mode", ""))]),
        "lexical_candidates": len([doc for doc in ranked if "lexical" in str(doc.metadata.get("retrieval_mode", ""))]),
        "final_candidates": len(final_docs),
        "max_per_source": settings.retrieval_max_per_source,
    }
    return final_docs, graph_result
