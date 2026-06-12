from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings
from app.observability import record_eval_run
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.retrieval_optimizer import expected_answer_aspects, infer_answer_type, token_overlap_ratio, validate_answer_grounding
from app.task_store import list_tasks


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
        answer_type = infer_answer_type(question)
        expected_aspects = expected_answer_aspects(question, answer_type)
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
            validation = validate_answer_grounding(
                question,
                gold_answer,
                [doc.page_content for doc in docs],
                expected_terms,
                answer_type=answer_type,
                expected_aspects=expected_aspects,
            )
            grounding_sum += float(validation.get("grounding_score") or 0.0)
        else:
            validation = {
                "grounded": False,
                "grounding_score": 0.0,
                "reference_overlap": 0.0,
                "question_overlap": 0.0,
                "answer_type": answer_type,
                "aspect_coverage": 0.0,
                "missing_aspects": expected_aspects,
                "fact_coverage": 0.0,
                "used_facts": 0,
            }
        items.append(
            {
                "index": index,
                "question": question,
                "answer_type": answer_type,
                "expected_aspects": expected_aspects,
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
    answer_type = infer_answer_type(question)
    validation = validate_answer_grounding(
        question,
        answer,
        references,
        concepts,
        answer_type=answer_type,
        expected_aspects=expected_answer_aspects(question, answer_type),
    )
    validation["reference_count"] = len(references)
    validation["concept_count"] = len(concepts)
    return validation


def export_failed_cases(limit: int = 50, write_file: bool = True) -> dict[str, Any]:
    record_eval_run("failure_samples")
    tasks = list_tasks(kind="chat", limit=max(limit * 4, limit))
    items: list[dict[str, Any]] = []
    by_reason: dict[str, int] = {}
    for task in tasks:
        result = dict(task.get("result") or {})
        question = str((task.get("payload") or {}).get("question") or "").strip()
        if not question:
            continue
        status = str(task.get("status") or "")
        answer = str(result.get("final_answer") or "").strip()
        validation = dict(result.get("answer_validation") or {})
        grounded = bool(validation.get("grounded"))
        missing_aspects = [str(item).strip() for item in validation.get("missing_aspects") or [] if str(item).strip()]
        reason_code = str(result.get("critic_reason_code") or task.get("error_code") or "UNKNOWN").strip() or "UNKNOWN"
        is_failure = status in {"failed", "timeout"} or not grounded or bool(missing_aspects) or not answer
        if not is_failure:
            continue
        item = {
            "task_id": task.get("task_id"),
            "status": status,
            "question": question,
            "answer": answer,
            "reason_code": reason_code,
            "reason": str(result.get("critic_reason") or task.get("error_message") or "").strip(),
            "answer_type": str(validation.get("answer_type") or infer_answer_type(question)),
            "missing_aspects": missing_aspects,
            "grounding_score": float(validation.get("grounding_score") or 0.0),
            "citation_coverage": float(validation.get("citation_coverage") or 0.0),
            "fact_coverage": float(validation.get("fact_coverage") or 0.0),
            "retrieval_summary": result.get("retrieval_summary") or {},
            "execution_trace": result.get("execution_trace") or [],
            "updated_at": task.get("updated_at"),
        }
        items.append(item)
        by_reason[reason_code] = by_reason.get(reason_code, 0) + 1
        if len(items) >= limit:
            break
    payload = {
        "total": len(items),
        "by_reason": dict(sorted(by_reason.items(), key=lambda item: (-item[1], item[0]))),
        "items": items,
    }
    if write_file:
        path = Path(settings.evaluation_failure_export_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["file"] = str(path)
    return payload
