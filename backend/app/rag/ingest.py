from pathlib import Path

from app.config import settings
from app.rag.graph_store import active_graph_backend, build_graph_index
from app.rag.loaders import load_directory
from app.rag.vector_store import ingest_documents


def ingest_knowledge_base(reset: bool = False) -> dict:
    knowledge_dir = Path(settings.knowledge_dir)
    if not knowledge_dir.exists():
        return {"status": "error", "message": f"Knowledge dir not found: {knowledge_dir}"}
    documents = load_directory(knowledge_dir)
    sources = sorted({d.metadata.get("source", "") for d in documents if d.metadata.get("source")})
    count = ingest_documents(documents, reset=reset)
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
