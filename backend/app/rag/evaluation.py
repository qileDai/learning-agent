from __future__ import annotations

from typing import Any

from app.observability import record_eval_run
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.retrieval_optimizer import token_overlap_ratio, validate_answer_grounding


def evaluate_retrieval(cases: list[dict[str, Any]], top_k: int = 3) -> dict[str, Any]:
    record_eval_run("retrieval")
    items: list[dict[str, Any]] = []
    hit_at_k = 0
    reciprocal_rank_sum = 0.0
    grounding_sum = 0.0
    for index, case in enumerate(cases):
        question = str(case.get("question") or "").strip()
        if not question:
            continue
        expected_sources = [str(item).strip() for item in case.get("expected_sources") or [] if str(item).strip()]
        expected_terms = [str(item).strip() for item in case.get("expected_terms") or [] if str(item).strip()]
        documents, graph_result = hybrid_retrieve(question)
        docs = documents[:top_k]
        matched_rank = None
        sources = []
        best_term_overlap = 0.0
        for rank, doc in enumerate(docs, start=1):
            source = str(doc.metadata.get("source") or "")
            sources.append(source)
            text = doc.page_content
            overlap = max([token_overlap_ratio(term, text) for term in expected_terms], default=0.0)
            best_term_overlap = max(best_term_overlap, overlap)
            if matched_rank is None and ((expected_sources and source in expected_sources) or (expected_terms and overlap >= 0.5)):
                matched_rank = rank
        hit = matched_rank is not None
        if hit:
            hit_at_k += 1
            reciprocal_rank_sum += 1.0 / matched_rank
        gold_answer = str(case.get("gold_answer") or "").strip()
        if gold_answer:
            validation = validate_answer_grounding(question, gold_answer, [doc.page_content for doc in docs], expected_terms)
            grounding_sum += float(validation.get("grounding_score") or 0.0)
        else:
            validation = {"grounded": False, "grounding_score": 0.0, "reference_overlap": 0.0, "question_overlap": 0.0}
        items.append(
            {
                "index": index,
                "question": question,
                "expected_sources": expected_sources,
                "expected_terms": expected_terms,
                "retrieved_sources": sources,
                "hit": hit,
                "matched_rank": matched_rank,
                "best_term_overlap": round(best_term_overlap, 4),
                "retrieval_summary": graph_result.get("retrieval_summary") or {},
                "answer_validation": validation,
            }
        )
    total = len(items)
    return {
        "total": total,
        "hit_at_k": round(hit_at_k / total, 4) if total else 0.0,
        "mrr": round(reciprocal_rank_sum / total, 4) if total else 0.0,
        "avg_grounding_score": round(grounding_sum / total, 4) if total else 0.0,
        "items": items,
    }


def evaluate_answer(payload: dict[str, Any]) -> dict[str, Any]:
    record_eval_run("answer")
    question = str(payload.get("question") or "").strip()
    answer = str(payload.get("answer") or "").strip()
    references = [str(item).strip() for item in payload.get("references") or [] if str(item).strip()]
    concepts = [str(item).strip() for item in payload.get("concepts") or [] if str(item).strip()]
    validation = validate_answer_grounding(question, answer, references, concepts)
    validation["reference_count"] = len(references)
    validation["concept_count"] = len(concepts)
    return validation
