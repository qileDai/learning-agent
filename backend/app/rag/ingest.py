from pathlib import Path

from app.config import settings
from app.rag.vector_store import ingest_documents
from app.rag.loaders import load_directory


def ingest_knowledge_base(reset: bool = False) -> dict:
    knowledge_dir = Path(settings.knowledge_dir)
    if not knowledge_dir.exists():
        return {"status": "error", "message": f"Knowledge dir not found: {knowledge_dir}"}
    documents = load_directory(knowledge_dir)
    sources = sorted({d.metadata.get("source", "") for d in documents if d.metadata.get("source")})
    count = ingest_documents(documents, reset=reset)
    by_type: dict[str, int] = {}
    for doc in documents:
        ft = doc.metadata.get("file_type", "unknown")
        by_type[ft] = by_type.get(ft, 0) + 1
    return {
        "status": "ok",
        "files": len(sources),
        "sources": sources,
        "documents": len(documents),
        "chunks": count,
        "by_type": by_type,
        "xinli_included": "xinli.md" in sources or "xinli.docx" in sources,
    }
