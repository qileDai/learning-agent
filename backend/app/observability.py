from __future__ import annotations

from time import perf_counter
from typing import Any

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

HTTP_REQUESTS_TOTAL = Counter("education_agent_http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_REQUEST_LATENCY_SECONDS = Histogram("education_agent_http_request_latency_seconds", "HTTP request latency", ["method", "path"])
RETRIEVAL_TOTAL = Counter("education_agent_retrieval_total", "Retrieval operations", ["route_type", "cache_hit"])
RETRIEVAL_FINAL_CANDIDATES = Histogram("education_agent_retrieval_final_candidates", "Final retrieval candidate count", ["route_type"])
ANSWER_GROUNDING_SCORE = Histogram("education_agent_answer_grounding_score", "Answer grounding score")
EVAL_RUN_TOTAL = Counter("education_agent_eval_run_total", "Evaluation API runs", ["mode"])


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = perf_counter()
        response = await call_next(request)
        path = request.url.path
        method = request.method
        status = str(response.status_code)
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status).inc()
        HTTP_REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(perf_counter() - start)
        return response


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def record_retrieval_metrics(summary: dict[str, Any]) -> None:
    route_type = str(summary.get("route_type") or "simple")
    cache_hit = str(bool(summary.get("cache_hit"))).lower()
    RETRIEVAL_TOTAL.labels(route_type=route_type, cache_hit=cache_hit).inc()
    final_candidates = int(summary.get("final_candidates") or 0)
    RETRIEVAL_FINAL_CANDIDATES.labels(route_type=route_type).observe(final_candidates)


def record_answer_validation(validation: dict[str, Any]) -> None:
    if not validation:
        return
    ANSWER_GROUNDING_SCORE.observe(float(validation.get("grounding_score") or 0.0))


def record_eval_run(mode: str) -> None:
    EVAL_RUN_TOTAL.labels(mode=mode).inc()
