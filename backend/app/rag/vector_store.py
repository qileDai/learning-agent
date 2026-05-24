from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.rag.elastic_store import ingest_documents as ingest_elasticsearch_documents
from app.rag.elastic_store import reset_index as reset_elasticsearch_index
from app.rag.milvus_store import ingest_vectors as ingest_milvus_vectors
from app.rag.milvus_store import milvus_enabled, reset_collection as reset_milvus_collection, search_vectors as milvus_search_vectors

_INDEX_DIR = Path(settings.vector_index_dir)
_EMBEDDINGS_FILE = _INDEX_DIR / "embeddings.npy"
_DOCS_FILE = _INDEX_DIR / "documents.json"


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key or "dummy",
        openai_api_base=settings.openai_api_base,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _chunk_id(doc: Document, index: int) -> str:
    meta = dict(doc.metadata or {})
    base = f"{meta.get('source', 'unknown')}::{index}::{doc.page_content[:160]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def _prepare_chunks(documents: list[Document]) -> list[Document]:
    chunks = split_documents(documents)
    prepared: list[Document] = []
    for index, chunk in enumerate(chunks):
        meta = dict(chunk.metadata or {})
        meta.setdefault("chunk_id", _chunk_id(chunk, index))
        prepared.append(Document(page_content=chunk.page_content, metadata=meta))
    return prepared


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _load_index() -> tuple[list[dict[str, Any]], np.ndarray | None]:
    if not _DOCS_FILE.exists() or not _EMBEDDINGS_FILE.exists():
        return [], None
    docs = json.loads(_DOCS_FILE.read_text(encoding="utf-8"))
    matrix = np.load(_EMBEDDINGS_FILE)
    if not docs or matrix.size == 0:
        return [], None
    return docs, matrix


def load_index_documents() -> list[Document]:
    docs, _ = _load_index()
    return [Document(page_content=item["page_content"], metadata=item.get("metadata", {})) for item in docs]


def _save_index(docs: list[dict[str, Any]], matrix: np.ndarray | None) -> None:
    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _DOCS_FILE.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    if matrix is not None and matrix.size:
        np.save(_EMBEDDINGS_FILE, matrix)
    elif _EMBEDDINGS_FILE.exists():
        _EMBEDDINGS_FILE.unlink()


def reset_index() -> None:
    for path in (_DOCS_FILE, _EMBEDDINGS_FILE):
        if path.exists():
            path.unlink()
    reset_elasticsearch_index()
    reset_milvus_collection()


def ingest_documents(documents: list[Document], reset: bool = False) -> int:
    if not documents:
        return 0
    chunks = _prepare_chunks(documents)
    if reset:
        reset_index()
    texts = [c.page_content for c in chunks]
    vectors = np.array(get_embeddings().embed_documents(texts), dtype=np.float32)
    vectors = _normalize(vectors)
    new_docs = [{"page_content": c.page_content, "metadata": c.metadata or {}} for c in chunks]

    existing_docs, existing_matrix = _load_index()
    if existing_matrix is None or not existing_docs or reset:
        merged_docs, merged_matrix = new_docs, vectors
    else:
        merged_docs = existing_docs + new_docs
        merged_matrix = np.vstack([existing_matrix, vectors])
    _save_index(merged_docs, merged_matrix)
    ingest_elasticsearch_documents(chunks, reset=reset)
    ingest_milvus_vectors(chunks, vectors, reset=reset)
    return len(chunks)


def _embed_query(query: str) -> np.ndarray:
    query_vec = np.array(get_embeddings().embed_query(query), dtype=np.float32)
    return query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)


def _search_local(query_vec: np.ndarray, k: int) -> list[tuple[Document, float]]:
    docs, matrix = _load_index()
    if not docs or matrix is None:
        return []
    distances = 1.0 - (matrix @ query_vec)
    top_k = min(k, len(docs))
    indices = np.argsort(distances)[:top_k]
    return [
        (
            Document(page_content=docs[i]["page_content"], metadata=docs[i].get("metadata", {})),
            float(distances[i]),
        )
        for i in indices
    ]


def _search(query: str, k: int) -> list[tuple[Document, float]]:
    query_vec = _embed_query(query)
    if milvus_enabled():
        results = milvus_search_vectors(query_vec, limit=k)
        if results:
            return results
    return _search_local(query_vec, k=k)


def similarity_search(query: str, k: int = 8) -> list[Document]:
    return [doc for doc, _ in _search(query, k=k)]


def similarity_search_with_scores(query: str, k: int = 12) -> list[tuple[Document, float]]:
    try:
        return _search(query, k=k)
    except Exception:
        docs = similarity_search(query, k=k)
        return [(d, 1.0) for d in docs]
