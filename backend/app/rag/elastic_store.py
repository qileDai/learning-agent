from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from app.config import settings


def elasticsearch_enabled() -> bool:
    return bool(settings.elasticsearch_enabled and settings.elasticsearch_url and settings.elasticsearch_index)


def _get_client():
    from elasticsearch import Elasticsearch

    kwargs: dict[str, Any] = {"verify_certs": settings.elasticsearch_verify_certs}
    if settings.elasticsearch_api_key:
        kwargs["api_key"] = settings.elasticsearch_api_key
    elif settings.elasticsearch_username and settings.elasticsearch_password:
        kwargs["basic_auth"] = (settings.elasticsearch_username, settings.elasticsearch_password)
    return Elasticsearch(settings.elasticsearch_url, **kwargs)


def _index_mapping() -> dict[str, Any]:
    return {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "page_content": {"type": "text"},
                "source": {"type": "keyword"},
                "subject": {"type": "keyword"},
                "chapter": {"type": "text"},
                "summary": {"type": "text"},
                "file_type": {"type": "keyword"},
                "concepts": {"type": "keyword"},
                "aliases": {"type": "keyword"},
            }
        }
    }


def reset_index() -> None:
    if not elasticsearch_enabled():
        return
    try:
        client = _get_client()
        if client.indices.exists(index=settings.elasticsearch_index):
            client.indices.delete(index=settings.elasticsearch_index)
    except Exception:
        return


def ingest_documents(documents: list[Document], reset: bool = False) -> None:
    if not elasticsearch_enabled() or not documents:
        return
    try:
        from elasticsearch import helpers

        client = _get_client()
        if reset and client.indices.exists(index=settings.elasticsearch_index):
            client.indices.delete(index=settings.elasticsearch_index)
        if not client.indices.exists(index=settings.elasticsearch_index):
            client.indices.create(index=settings.elasticsearch_index, **_index_mapping())
        actions = []
        for doc in documents:
            meta = dict(doc.metadata or {})
            chunk_id = str(meta.get("chunk_id") or "")
            if not chunk_id:
                continue
            actions.append(
                {
                    "_index": settings.elasticsearch_index,
                    "_id": chunk_id,
                    "_source": {
                        "chunk_id": chunk_id,
                        "page_content": doc.page_content,
                        "source": str(meta.get("source") or ""),
                        "subject": str(meta.get("subject") or ""),
                        "chapter": str(meta.get("chapter") or ""),
                        "summary": str(meta.get("summary") or ""),
                        "file_type": str(meta.get("file_type") or ""),
                        "concepts": [str(item).strip() for item in meta.get("concepts") or [] if str(item).strip()],
                        "aliases": [str(item).strip() for item in meta.get("aliases") or [] if str(item).strip()],
                        "metadata": meta,
                    },
                }
            )
        if actions:
            helpers.bulk(client, actions, refresh=True)
    except Exception:
        return


def lexical_search(query: str, limit: int = 10, route_subjects: list[str] | None = None) -> list[tuple[Document, float, int]]:
    if not elasticsearch_enabled() or not query.strip():
        return []
    route_subjects = route_subjects or []
    should_query: list[dict[str, Any]] = [
        {
            "multi_match": {
                "query": query,
                "fields": [
                    "page_content^3",
                    "chapter^2",
                    "summary^2",
                    "concepts^4",
                    "aliases^3",
                ],
                "type": "best_fields",
            }
        }
    ]
    request_query: dict[str, Any] = {"bool": {"should": should_query, "minimum_should_match": 1}}
    if route_subjects:
        request_query["bool"]["filter"] = [{"terms": {"subject": route_subjects}}]
    try:
        client = _get_client()
        response = client.search(index=settings.elasticsearch_index, query=request_query, size=limit)
        hits = response.get("hits", {}).get("hits", [])
    except Exception:
        return []
    results: list[tuple[Document, float, int]] = []
    for rank, hit in enumerate(hits):
        source = hit.get("_source") or {}
        metadata = dict(source.get("metadata") or {})
        metadata.setdefault("chunk_id", source.get("chunk_id"))
        metadata.setdefault("source", source.get("source"))
        metadata.setdefault("subject", source.get("subject"))
        metadata.setdefault("chapter", source.get("chapter"))
        metadata.setdefault("summary", source.get("summary"))
        metadata.setdefault("file_type", source.get("file_type"))
        metadata.setdefault("concepts", source.get("concepts") or [])
        metadata.setdefault("aliases", source.get("aliases") or [])
        results.append((Document(page_content=str(source.get("page_content") or ""), metadata=metadata), float(hit.get("_score") or 0.0), rank))
    return results
