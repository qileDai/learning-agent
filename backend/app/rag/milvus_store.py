from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.documents import Document

from app.config import settings


_VECTOR_FIELD = "vector"
_PRIMARY_FIELD = "chunk_id"
_OUTPUT_FIELDS = ["page_content", "source", "subject", "chapter", "summary", "file_type", "concepts", "aliases"]


def milvus_enabled() -> bool:
    return bool(settings.milvus_enabled and settings.milvus_uri and settings.milvus_collection)


def _get_client():
    from pymilvus import MilvusClient

    kwargs: dict[str, Any] = {"uri": settings.milvus_uri}
    if settings.milvus_token:
        kwargs["token"] = settings.milvus_token
    return MilvusClient(**kwargs)


def _collection_exists(client) -> bool:
    try:
        return bool(client.has_collection(collection_name=settings.milvus_collection))
    except TypeError:
        return bool(client.has_collection(settings.milvus_collection))


def _ensure_collection(client, dim: int, reset: bool = False) -> None:
    from pymilvus import DataType

    exists = _collection_exists(client)
    if reset and exists:
        client.drop_collection(collection_name=settings.milvus_collection)
        exists = False
    if exists:
        return
    schema = client.create_schema(auto_id=False, enable_dynamic_fields=True)
    schema.add_field(field_name=_PRIMARY_FIELD, datatype=DataType.VARCHAR, is_primary=True, max_length=128)
    schema.add_field(field_name=_VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=dim)
    client.create_collection(collection_name=settings.milvus_collection, schema=schema, consistency_level="Strong")


def reset_collection() -> None:
    if not milvus_enabled():
        return
    try:
        client = _get_client()
        if _collection_exists(client):
            client.drop_collection(collection_name=settings.milvus_collection)
    except Exception:
        return


def ingest_vectors(documents: list[Document], vectors: np.ndarray, reset: bool = False) -> None:
    if not milvus_enabled() or not documents or vectors.size == 0:
        return
    try:
        client = _get_client()
        _ensure_collection(client, int(vectors.shape[1]), reset=reset)
        rows: list[dict[str, Any]] = []
        for index, doc in enumerate(documents):
            meta = dict(doc.metadata or {})
            chunk_id = str(meta.get("chunk_id") or "")
            if not chunk_id:
                continue
            rows.append(
                {
                    _PRIMARY_FIELD: chunk_id,
                    _VECTOR_FIELD: vectors[index].tolist(),
                    "page_content": doc.page_content,
                    "source": str(meta.get("source") or ""),
                    "subject": str(meta.get("subject") or ""),
                    "chapter": str(meta.get("chapter") or ""),
                    "summary": str(meta.get("summary") or ""),
                    "file_type": str(meta.get("file_type") or ""),
                    "concepts": [str(item).strip() for item in meta.get("concepts") or [] if str(item).strip()],
                    "aliases": [str(item).strip() for item in meta.get("aliases") or [] if str(item).strip()],
                }
            )
        if rows:
            client.insert(collection_name=settings.milvus_collection, data=rows)
    except Exception:
        return


def search_vectors(query_vector: np.ndarray, limit: int = 10) -> list[tuple[Document, float]]:
    if not milvus_enabled() or query_vector.size == 0:
        return []
    try:
        client = _get_client()
        if not _collection_exists(client):
            return []
        response = client.search(
            collection_name=settings.milvus_collection,
            anns_field=_VECTOR_FIELD,
            data=[query_vector.tolist()],
            limit=limit,
            search_params={"metric_type": "COSINE", "params": {}},
            output_fields=_OUTPUT_FIELDS,
        )
    except Exception:
        return []
    hits = response[0] if response else []
    results: list[tuple[Document, float]] = []
    for hit in hits:
        entity = hit.get("entity") or {}
        metadata = {
            "chunk_id": entity.get(_PRIMARY_FIELD) or hit.get("id"),
            "source": entity.get("source", ""),
            "subject": entity.get("subject"),
            "chapter": entity.get("chapter"),
            "summary": entity.get("summary"),
            "file_type": entity.get("file_type"),
            "concepts": list(entity.get("concepts") or []),
            "aliases": list(entity.get("aliases") or []),
        }
        distance = float(hit.get("distance") or 0.0)
        results.append((Document(page_content=str(entity.get("page_content") or ""), metadata=metadata), 1.0 - distance))
    return results
