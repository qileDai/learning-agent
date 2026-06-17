"""向量索引与多后端写入模块。

调用关系：
1. ingest.py -> ingest_documents：入库时调用，负责 chunk 切分与向量写入
2. ingest_documents -> ingest_elasticsearch_documents：同步写入关键词索引
3. ingest_documents -> ingest_milvus_vectors：同步写入 Milvus 向量库
4. hybrid_retriever.py -> similarity_search / similarity_search_with_scores：在线检索时调用

这个模块既保留本地向量索引能力，也负责接入企业级向量库 Milvus。
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_openai import OpenAIEmbeddings
except Exception:
    OpenAIEmbeddings = None

from app.config import settings
from app.rag.elastic_store import ingest_documents as ingest_elasticsearch_documents
from app.rag.elastic_store import reset_index as reset_elasticsearch_index
from app.rag.milvus_store import ingest_vectors as ingest_milvus_vectors
from app.rag.milvus_store import milvus_enabled, reset_collection as reset_milvus_collection, search_vectors as milvus_search_vectors

_INDEX_DIR = Path(settings.vector_index_dir)
_EMBEDDINGS_FILE = _INDEX_DIR / "embeddings.npy"
_DOCS_FILE = _INDEX_DIR / "documents.json"
_VECTOR_DIM = 1536
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#.-]{2,}|[\u4e00-\u9fff]")
_INDEX_LOCK = threading.RLock()
_INDEX_CACHE_VERSION: tuple[int, int] | None = None
_INDEX_CACHE_DOCS: list[dict[str, Any]] = []
_INDEX_CACHE_MATRIX: np.ndarray | None = None


class _FallbackEmbeddings:
    def __init__(self, model: str, api_key: str, api_base: str) -> None:
        self.model = model

    def _embed_text(self, text: str) -> list[float]:
        vec = np.zeros(_VECTOR_DIM, dtype=np.float32)
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            vec[0] = 1.0
            return vec.tolist()
        for index, token in enumerate(tokens):
            slot = hash(token) % _VECTOR_DIM
            sign = 1.0 if hash(f"{token}:{index % 7}") % 2 == 0 else -1.0
            weight = 1.0 + min(index, 6) * 0.03
            vec[slot] += sign * weight
        norm = max(float(np.linalg.norm(vec)), 1e-12)
        return (vec / norm).tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_text(query)


def get_embeddings() -> Any:
    """返回统一使用的 Embedding 模型实例。"""
    if OpenAIEmbeddings is not None:
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            openai_api_key=settings.openai_api_key or "dummy",
            openai_api_base=settings.openai_api_base,
        )
    return _FallbackEmbeddings(
        model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
        api_base=settings.openai_api_base,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """按固定 chunk 大小切分文档。

    这里的 chunk 是后续向量检索、关键词检索和图谱构建的共同基础粒度。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def _chunk_id(doc: Document, index: int) -> str:
    """为每个 chunk 生成稳定 ID，方便多后端统一引用。"""
    meta = dict(doc.metadata or {})
    base = f"{meta.get('source', 'unknown')}::{index}::{doc.page_content[:160]}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))


def _prepare_chunks(documents: list[Document]) -> list[Document]:
    """切分文档并补齐 chunk_id 元数据。"""
    chunks = split_documents(documents)
    prepared: list[Document] = []
    for index, chunk in enumerate(chunks):
        meta = dict(chunk.metadata or {})
        meta.setdefault("chunk_id", _chunk_id(chunk, index))
        prepared.append(Document(page_content=chunk.page_content, metadata=meta))
    return prepared


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """做 L2 归一化，便于使用余弦相似度检索。"""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _index_version() -> tuple[int, int] | None:
    if not _DOCS_FILE.exists() or not _EMBEDDINGS_FILE.exists():
        return None
    return int(_DOCS_FILE.stat().st_mtime_ns), int(_EMBEDDINGS_FILE.stat().st_mtime_ns)


def _invalidate_index_cache() -> None:
    global _INDEX_CACHE_VERSION, _INDEX_CACHE_DOCS, _INDEX_CACHE_MATRIX
    _INDEX_CACHE_VERSION = None
    _INDEX_CACHE_DOCS = []
    _INDEX_CACHE_MATRIX = None


def _load_index() -> tuple[list[dict[str, Any]], np.ndarray | None]:
    """读取本地向量索引，作为默认存储和回退路径。"""
    global _INDEX_CACHE_VERSION, _INDEX_CACHE_DOCS, _INDEX_CACHE_MATRIX
    version = _index_version()
    if version is None:
        with _INDEX_LOCK:
            _invalidate_index_cache()
        return [], None
    with _INDEX_LOCK:
        if _INDEX_CACHE_VERSION == version and _INDEX_CACHE_DOCS and _INDEX_CACHE_MATRIX is not None:
            return list(_INDEX_CACHE_DOCS), _INDEX_CACHE_MATRIX
        docs = json.loads(_DOCS_FILE.read_text(encoding="utf-8"))
        matrix = np.load(_EMBEDDINGS_FILE)
        if not docs or matrix.size == 0:
            _invalidate_index_cache()
            return [], None
        _INDEX_CACHE_VERSION = version
        _INDEX_CACHE_DOCS = list(docs)
        _INDEX_CACHE_MATRIX = matrix
        return list(_INDEX_CACHE_DOCS), _INDEX_CACHE_MATRIX


def load_index_documents() -> list[Document]:
    """返回本地索引中的全部 Document，供本地词法召回回退使用。"""
    docs, _ = _load_index()
    return [Document(page_content=item["page_content"], metadata=item.get("metadata", {})) for item in docs]


def _save_index(docs: list[dict[str, Any]], matrix: np.ndarray | None) -> None:
    """持久化本地文档和向量矩阵。"""
    global _INDEX_CACHE_VERSION, _INDEX_CACHE_DOCS, _INDEX_CACHE_MATRIX
    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with _INDEX_LOCK:
        _DOCS_FILE.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
        if matrix is not None and matrix.size:
            np.save(_EMBEDDINGS_FILE, matrix)
            _INDEX_CACHE_MATRIX = np.array(matrix, copy=True)
        elif _EMBEDDINGS_FILE.exists():
            _EMBEDDINGS_FILE.unlink()
            _INDEX_CACHE_MATRIX = None
        _INDEX_CACHE_DOCS = list(docs)
        _INDEX_CACHE_VERSION = _index_version()


def reset_index() -> None:
    """重置本地索引，并联动清空 Elasticsearch / Milvus。"""
    with _INDEX_LOCK:
        for path in (_DOCS_FILE, _EMBEDDINGS_FILE):
            if path.exists():
                path.unlink()
        _invalidate_index_cache()
    reset_elasticsearch_index()
    reset_milvus_collection()


def ingest_documents(documents: list[Document], reset: bool = False) -> int:
    """把知识文档写入本地向量索引，并同步写入 ES / Milvus。

    这是离线入库阶段最核心的函数，承担“切分 -> 向量化 -> 多后端写入”的职责。
    """
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

    # 本地索引用于开发环境和 Milvus 不可用时的降级检索。
    _save_index(merged_docs, merged_matrix)

    # 企业级后端同步写入：一个 chunk 会同时进入 ES 和 Milvus。
    ingest_elasticsearch_documents(chunks, reset=reset)
    ingest_milvus_vectors(chunks, vectors, reset=reset)
    return len(chunks)


def _embed_query(query: str) -> np.ndarray:
    """把用户查询转为归一化向量。"""
    query_vec = np.array(get_embeddings().embed_query(query), dtype=np.float32)
    return query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)


def _search_local(query_vec: np.ndarray, k: int) -> list[tuple[Document, float]]:
    """本地向量检索回退路径。"""
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
    """统一向量检索入口。

    调用顺序：
    - 先查 Milvus
    - Milvus 未启用或无结果时回退本地索引
    """
    query_vec = _embed_query(query)
    if milvus_enabled():
        results = milvus_search_vectors(query_vec, limit=k)
        if results:
            return results
    return _search_local(query_vec, k=k)


def similarity_search(query: str, k: int = 8) -> list[Document]:
    """只返回文档结果，用于简单语义召回场景。"""
    try:
        return [doc for doc, _ in _search(query, k=k)]
    except Exception:
        return []


def similarity_search_with_scores(query: str, k: int = 12) -> list[tuple[Document, float]]:
    """返回文档和分数，用于混合排序阶段。"""
    try:
        return _search(query, k=k)
    except Exception:
        return []
