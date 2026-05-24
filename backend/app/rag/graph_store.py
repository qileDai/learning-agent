import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.rag.metadata_registry import load_metadata_registry, normalize_source

_GRAPH_DIR = Path(settings.graph_index_dir)
_GRAPH_FILE = _GRAPH_DIR / "graph.json"


def _dedup_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _dedup_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for edge in edges:
        key = (str(edge.get("source", "")).strip(), str(edge.get("relation", "")).strip(), str(edge.get("target", "")).strip())
        if not all(key) or key in seen:
            continue
        seen.add(key)
        result.append(
            {
                "source": key[0],
                "relation": key[1],
                "target": key[2],
                "subject": str(edge.get("subject", "")).strip(),
                "evidence_sources": _dedup_strings(list(edge.get("evidence_sources") or [])),
            }
        )
    return result


def _registry_concepts_from_source(meta: dict[str, Any]) -> list[str]:
    return _dedup_strings(list(meta.get("concepts") or []))


def _concept_aliases(name: str, meta: dict[str, Any]) -> list[str]:
    return _dedup_strings([name, *(meta.get("aliases") or [])])


def build_graph_index(documents: list[Document]) -> dict[str, Any]:
    registry = load_metadata_registry()
    registry_sources = registry.get("sources", {})
    registry_concepts = registry.get("concepts", {})
    registry_relations = registry.get("relations", [])

    source_nodes: dict[str, dict[str, Any]] = {}
    concept_nodes: dict[str, dict[str, Any]] = {}
    concept_to_sources: dict[str, set[str]] = {}

    for doc in documents:
        meta = dict(doc.metadata or {})
        source = normalize_source(str(meta.get("source", "unknown")))
        source_meta = dict(registry_sources.get(source, {}))
        concepts = _dedup_strings(list(meta.get("concepts") or source_meta.get("concepts") or []))
        source_info = source_nodes.setdefault(
            source,
            {
                "source": source,
                "subject": str(meta.get("subject") or source_meta.get("subject") or "").strip(),
                "grade": str(meta.get("grade") or source_meta.get("grade") or "").strip(),
                "chapter": str(meta.get("chapter") or source_meta.get("chapter") or "").strip(),
                "difficulty": str(meta.get("difficulty") or source_meta.get("difficulty") or "").strip(),
                "summary": str(meta.get("summary") or source_meta.get("summary") or "").strip(),
                "concepts": concepts,
                "document_count": 0,
                "chunk_count": 0,
            },
        )
        source_info["document_count"] += 1
        source_info["chunk_count"] += 1
        if concepts:
            source_info["concepts"] = _dedup_strings([*source_info.get("concepts", []), *concepts])

        for concept in source_info.get("concepts", []):
            concept_meta = dict(registry_concepts.get(concept, {}))
            concept_info = concept_nodes.setdefault(
                concept,
                {
                    "name": concept,
                    "subject": str(concept_meta.get("subject") or source_info.get("subject") or "").strip(),
                    "description": str(concept_meta.get("description") or "").strip(),
                    "aliases": _concept_aliases(concept, concept_meta),
                    "chapters": [],
                    "sources": [],
                },
            )
            concept_info["aliases"] = _dedup_strings([*concept_info.get("aliases", []), *_concept_aliases(concept, concept_meta)])
            if source_info.get("chapter"):
                concept_info["chapters"] = _dedup_strings([*concept_info.get("chapters", []), source_info["chapter"]])
            concept_info["sources"] = _dedup_strings([*concept_info.get("sources", []), source])
            concept_to_sources.setdefault(concept, set()).add(source)

    for concept, meta in registry_concepts.items():
        concept_info = concept_nodes.setdefault(
            concept,
            {
                "name": concept,
                "subject": str(meta.get("subject", "")).strip(),
                "description": str(meta.get("description", "")).strip(),
                "aliases": _concept_aliases(concept, meta),
                "chapters": [],
                "sources": [],
            },
        )
        concept_info["aliases"] = _dedup_strings([*concept_info.get("aliases", []), *_concept_aliases(concept, meta)])
        concept_info["sources"] = _dedup_strings([*concept_info.get("sources", []), *sorted(concept_to_sources.get(concept, set()))])

    edges = _dedup_edges(registry_relations)
    graph = {
        "concepts": dict(sorted(concept_nodes.items())),
        "sources": dict(sorted(source_nodes.items())),
        "relations": edges,
        "stats": {
            "concepts": len(concept_nodes),
            "sources": len(source_nodes),
            "relations": len(edges),
        },
    }
    _GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    _GRAPH_FILE.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return graph


def load_graph_index() -> dict[str, Any]:
    if not _GRAPH_FILE.exists():
        return {"concepts": {}, "sources": {}, "relations": [], "stats": {"concepts": 0, "sources": 0, "relations": 0}}
    return json.loads(_GRAPH_FILE.read_text(encoding="utf-8"))


def graph_overview(limit: int = 12) -> dict[str, Any]:
    graph = load_graph_index()
    concepts = list(graph.get("concepts", {}).values())[:limit]
    sources = list(graph.get("sources", {}).values())[:limit]
    return {
        "stats": graph.get("stats", {}),
        "concepts": concepts,
        "sources": sources,
        "relations": graph.get("relations", [])[:limit],
    }


def _concept_match_score(question: str, concept: str, aliases: list[str]) -> float:
    text = (question or "").casefold()
    score = 0.0
    for alias in aliases:
        alias_text = str(alias).strip().casefold()
        if not alias_text:
            continue
        if alias_text == text:
            score = max(score, 1.0)
        elif alias_text in text:
            score = max(score, min(0.98, 0.45 + len(alias_text) * 0.08))
    if concept.casefold() in text:
        score = max(score, min(1.0, 0.5 + len(concept) * 0.08))
    return round(score, 4)


def search_graph(question: str, limit_sources: int = 4, limit_relations: int = 8) -> dict[str, Any]:
    graph = load_graph_index()
    concepts = graph.get("concepts", {})
    sources = graph.get("sources", {})
    relations = graph.get("relations", [])

    matched: list[dict[str, Any]] = []
    for name, meta in concepts.items():
        score = _concept_match_score(question, name, list(meta.get("aliases") or []))
        if score > 0:
            matched.append({"name": name, "score": score, "subject": meta.get("subject", "")})
    matched.sort(key=lambda item: (-item["score"], -len(item["name"]), item["name"]))
    matched = matched[:5]
    matched_names = [item["name"] for item in matched]

    relation_hits: list[dict[str, Any]] = []
    related_scores: dict[str, float] = {}
    for edge in relations:
        source = str(edge.get("source", "")).strip()
        target = str(edge.get("target", "")).strip()
        if source in matched_names or target in matched_names:
            relation_hits.append(edge)
            if source in matched_names and target not in matched_names:
                related_scores[target] = max(related_scores.get(target, 0.0), 0.62)
            if target in matched_names and source not in matched_names:
                related_scores[source] = max(related_scores.get(source, 0.0), 0.58)
    relation_hits = relation_hits[:limit_relations]

    source_scores: dict[str, float] = {}
    for source, meta in sources.items():
        concepts_in_source = set(meta.get("concepts") or [])
        score = 0.0
        for item in matched:
            if item["name"] in concepts_in_source:
                score += 0.32 + item["score"] * 0.18
        for concept, boost in related_scores.items():
            if concept in concepts_in_source:
                score += boost * 0.2
        chapter = str(meta.get("chapter") or "")
        summary = str(meta.get("summary") or "")
        text = f"{chapter} {summary}"
        for item in matched:
            if item["name"] in text:
                score += 0.12
        if score > 0:
            source_scores[source] = round(score, 4)

    top_sources = sorted(source_scores.items(), key=lambda item: (-item[1], item[0]))[:limit_sources]
    related_concepts = [name for name, _ in sorted(related_scores.items(), key=lambda item: (-item[1], item[0]))[:6]]

    graph_documents: list[Document] = []
    if matched_names:
        lines = [f"问题涉及概念：{'、'.join(matched_names)}。"]
        if related_concepts:
            lines.append(f"图谱关联概念：{'、'.join(related_concepts)}。")
        if relation_hits:
            relation_text = "；".join(
                f"{edge['source']}—{edge['relation']}→{edge['target']}" for edge in relation_hits[:5]
            )
            lines.append(f"关键关系：{relation_text}。")
        if top_sources:
            source_text = "；".join(
                f"{source}（图谱得分 {score}）" for source, score in top_sources
            )
            lines.append(f"优先参考资料：{source_text}。")
        dominant_subject = next((item.get("subject", "") for item in matched if item.get("subject")), "")
        graph_documents.append(
            Document(
                page_content="\n".join(lines),
                metadata={
                    "source": "graph://knowledge-map",
                    "file_type": "graph",
                    "subject": dominant_subject,
                    "chapter": "知识图谱检索",
                    "concepts": _dedup_strings([*matched_names, *related_concepts]),
                    "retrieval_mode": "graph",
                    "score": 0.0,
                },
            )
        )

    return {
        "matched_concepts": matched,
        "related_concepts": related_concepts,
        "relation_hits": relation_hits,
        "source_scores": source_scores,
        "top_sources": [{"source": source, "score": score} for source, score in top_sources],
        "documents": graph_documents,
        "stats": graph.get("stats", {}),
    }
