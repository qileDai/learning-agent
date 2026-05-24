"""RAG 入库入口。

调用关系：
1. ingest_knowledge_base -> load_directory：读取知识目录中的原始文档
2. ingest_knowledge_base -> ingest_documents：切分文档并写入本地向量索引 / Milvus / Elasticsearch
3. ingest_knowledge_base -> build_graph_index：基于文档元数据构建 JSON 图谱 / Neo4j 图谱

这个模块的职责是把“原始知识文件”统一变成后续检索链路可消费的多路索引。
"""

from pathlib import Path

from app.config import settings
from app.rag.graph_store import active_graph_backend, build_graph_index
from app.rag.loaders import load_directory
from app.rag.vector_store import ingest_documents


def ingest_knowledge_base(reset: bool = False) -> dict:
    """执行整套知识库入库流程。

    参数：
    - reset=True 时会清空已有索引并重建

    返回：
    - 入库后的统计信息，便于前端和运维确认当前知识库状态
    """
    knowledge_dir = Path(settings.knowledge_dir)
    if not knowledge_dir.exists():
        return {"status": "error", "message": f"Knowledge dir not found: {knowledge_dir}"}

    # 第一步：加载知识目录，得到带元数据的原始文档对象。
    documents = load_directory(knowledge_dir)
    sources = sorted({d.metadata.get("source", "") for d in documents if d.metadata.get("source")})

    # 第二步：切分并写入向量索引，同时同步到 Elasticsearch / Milvus。
    count = ingest_documents(documents, reset=reset)

    # 第三步：构建知识图谱，并按配置同步到 Neo4j。
    graph = build_graph_index(documents)

    by_type: dict[str, int] = {}
    subjects: dict[str, int] = {}
    for doc in documents:
        ft = doc.metadata.get("file_type", "unknown")
        by_type[ft] = by_type.get(ft, 0) + 1
        subject = doc.metadata.get("subject") or "未标注"
        subjects[subject] = subjects.get(subject, 0) + 1

    return {
        "status": "ok",
        "files": len(sources),
        "sources": sources,
        "documents": len(documents),
        "chunks": count,
        "by_type": by_type,
        "subjects": subjects,
        "graph": graph.get("stats", {}),
        "graph_backend": active_graph_backend(),
        "xinli_included": "xinli.md" in sources or "xinli.docx" in sources,
    }
