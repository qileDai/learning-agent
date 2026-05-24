from langchain_core.documents import Document

from app.rag.graph_store import search_graph
from app.rag.vector_store import similarity_search_with_scores


def hybrid_retrieve(question: str, vector_k: int = 15) -> tuple[list[Document], dict]:
    graph_result = search_graph(question)
    source_scores = graph_result.get("source_scores", {})
    matched_concepts = [item.get("name", "") for item in graph_result.get("matched_concepts", []) if item.get("name")]

    documents: list[Document] = list(graph_result.get("documents", []))
    seen: set[tuple[str, str]] = {
        (str(doc.metadata.get("source", "")), doc.page_content[:120]) for doc in documents
    }

    for doc, distance in similarity_search_with_scores(question, k=vector_k):
        meta = dict(doc.metadata or {})
        source = str(meta.get("source", "")).strip()
        concepts = list(meta.get("concepts") or [])
        boost = float(source_scores.get(source, 0.0))
        if matched_concepts and concepts:
            overlap = len(set(concepts) & set(matched_concepts))
            boost += overlap * 0.08
        adjusted_score = round(float(distance) - boost, 4)
        key = (source, doc.page_content[:120])
        if key in seen:
            continue
        seen.add(key)
        meta["score"] = adjusted_score
        meta["retrieval_mode"] = "hybrid" if boost > 0 else "vector"
        if matched_concepts and "graph_matched_concepts" not in meta:
            meta["graph_matched_concepts"] = matched_concepts
        documents.append(Document(page_content=doc.page_content, metadata=meta))

    return documents, graph_result
