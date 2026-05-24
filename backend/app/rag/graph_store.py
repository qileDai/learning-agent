"""知识图谱存储与图谱检索模块。

调用关系：
1. ingest.py -> build_graph_index：离线构建图谱
2. retrieval_optimizer.py -> load_graph_index：查询扩展时读取图谱概念
3. hybrid_retriever.py -> search_graph：在线检索时先走图谱召回
4. 图谱后端支持 JSON fallback 和 Neo4j，两者对外暴露同一个 search_graph 接口

这个模块的目标是把图谱从“静态数据”升级成真正参与召回和排序的检索通道。
"""

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from app.config import settings
from app.rag.metadata_registry import load_metadata_registry, normalize_source

_GRAPH_DIR = Path(settings.graph_index_dir)
_GRAPH_FILE = _GRAPH_DIR / "graph.json"


def _dedup_strings(items: list[str]) -> list[str]:
    """对字符串列表去重并保留顺序。"""
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
    """对图谱关系边去重。"""
    seen: set[tuple[str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for edge in edges:
        key = (
            str(edge.get("source", "")).strip(),
            str(edge.get("relation", "")).strip(),
            str(edge.get("target", "")).strip(),
        )
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


def _concept_aliases(name: str, meta: dict[str, Any]) -> list[str]:
    """拼接概念本名和别名集合。"""
    return _dedup_strings([name, *(meta.get("aliases") or [])])


def _build_graph_payload(documents: list[Document]) -> dict[str, Any]:
    """基于知识文档和元数据注册表构建图谱载荷。"""
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
    return {
        "concepts": dict(sorted(concept_nodes.items())),
        "sources": dict(sorted(source_nodes.items())),
        "relations": edges,
        "stats": {
            "concepts": len(concept_nodes),
            "sources": len(source_nodes),
            "relations": len(edges),
        },
    }


def _use_neo4j() -> bool:
    """判断是否配置为 Neo4j 后端。"""
    return settings.graph_store_backend.strip().casefold() == "neo4j"


def _neo4j_ready() -> bool:
    """判断 Neo4j 连接参数是否齐全。"""
    return bool(settings.neo4j_uri and settings.neo4j_user and settings.neo4j_password)


def active_graph_backend() -> str:
    """返回当前生效的图谱后端。"""
    return "neo4j" if _use_neo4j() and _neo4j_ready() else "json"


def _get_neo4j_driver():
    """创建 Neo4j Driver。"""
    from neo4j import GraphDatabase

    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))


def _graph_fallback() -> dict[str, Any]:
    """读取本地图谱 JSON，作为统一回退路径。"""
    if not _GRAPH_FILE.exists():
        return {"concepts": {}, "sources": {}, "relations": [], "stats": {"concepts": 0, "sources": 0, "relations": 0}}
    return json.loads(_GRAPH_FILE.read_text(encoding="utf-8"))


def _persist_graph_json(graph: dict[str, Any]) -> None:
    """把图谱持久化为本地 JSON。"""
    _GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    _GRAPH_FILE.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")


def _sync_graph_to_neo4j(graph: dict[str, Any]) -> None:
    """把图谱同步到 Neo4j。

    离线构建时会把 Source、Concept、Relation 三类结构写入图数据库。
    """
    if active_graph_backend() != "neo4j":
        return
    try:
        driver = _get_neo4j_driver()
        with driver:
            with driver.session(database=settings.neo4j_database) as session:
                session.run("CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE")
                session.run("CREATE CONSTRAINT source_name_unique IF NOT EXISTS FOR (n:Source) REQUIRE n.source IS UNIQUE")
                session.run("MATCH (n:KnowledgeGraph) DETACH DELETE n")
                for meta in graph.get("sources", {}).values():
                    session.run(
                        """
                        MERGE (s:KnowledgeGraph:Source {source: $source})
                        SET s.subject = $subject,
                            s.grade = $grade,
                            s.chapter = $chapter,
                            s.difficulty = $difficulty,
                            s.summary = $summary,
                            s.concepts = $concepts,
                            s.document_count = $document_count,
                            s.chunk_count = $chunk_count
                        """,
                        **meta,
                    )
                for meta in graph.get("concepts", {}).values():
                    session.run(
                        """
                        MERGE (c:KnowledgeGraph:Concept {name: $name})
                        SET c.subject = $subject,
                            c.description = $description,
                            c.aliases = $aliases,
                            c.chapters = $chapters,
                            c.sources = $sources
                        """,
                        **meta,
                    )
                    for source in meta.get("sources", []):
                        session.run(
                            """
                            MATCH (c:KnowledgeGraph:Concept {name: $concept})
                            MATCH (s:KnowledgeGraph:Source {source: $source})
                            MERGE (c)-[:MENTIONED_IN]->(s)
                            """,
                            concept=meta.get("name"),
                            source=source,
                        )
                for edge in graph.get("relations", []):
                    session.run(
                        """
                        MATCH (a:KnowledgeGraph:Concept {name: $source})
                        MATCH (b:KnowledgeGraph:Concept {name: $target})
                        MERGE (a)-[r:RELATED {relation: $relation, target_name: $target}]->(b)
                        SET r.subject = $subject,
                            r.evidence_sources = $evidence_sources
                        """,
                        **edge,
                    )
    except Exception:
        return


def _load_graph_from_neo4j() -> dict[str, Any]:
    """从 Neo4j 反查当前图谱。"""
    if active_graph_backend() != "neo4j":
        return _graph_fallback()
    try:
        driver = _get_neo4j_driver()
        with driver:
            with driver.session(database=settings.neo4j_database) as session:
                concept_records = list(
                    session.run(
                        """
                        MATCH (c:KnowledgeGraph:Concept)
                        RETURN c.name AS name,
                               c.subject AS subject,
                               c.description AS description,
                               coalesce(c.aliases, []) AS aliases,
                               coalesce(c.chapters, []) AS chapters,
                               coalesce(c.sources, []) AS sources
                        ORDER BY name
                        """
                    )
                )
                source_records = list(
                    session.run(
                        """
                        MATCH (s:KnowledgeGraph:Source)
                        RETURN s.source AS source,
                               s.subject AS subject,
                               s.grade AS grade,
                               s.chapter AS chapter,
                               s.difficulty AS difficulty,
                               s.summary AS summary,
                               coalesce(s.concepts, []) AS concepts,
                               coalesce(s.document_count, 0) AS document_count,
                               coalesce(s.chunk_count, 0) AS chunk_count
                        ORDER BY source
                        """
                    )
                )
                relation_records = list(
                    session.run(
                        """
                        MATCH (a:KnowledgeGraph:Concept)-[r:RELATED]->(b:KnowledgeGraph:Concept)
                        RETURN a.name AS source,
                               r.relation AS relation,
                               b.name AS target,
                               coalesce(r.subject, '') AS subject,
                               coalesce(r.evidence_sources, []) AS evidence_sources
                        ORDER BY source, relation, target
                        """
                    )
                )
        concepts = {
            str(record["name"]): {
                "name": str(record["name"]),
                "subject": str(record["subject"] or "").strip(),
                "description": str(record["description"] or "").strip(),
                "aliases": _dedup_strings(list(record["aliases"] or [])),
                "chapters": _dedup_strings(list(record["chapters"] or [])),
                "sources": _dedup_strings(list(record["sources"] or [])),
            }
            for record in concept_records
        }
        sources = {
            str(record["source"]): {
                "source": str(record["source"]),
                "subject": str(record["subject"] or "").strip(),
                "grade": str(record["grade"] or "").strip(),
                "chapter": str(record["chapter"] or "").strip(),
                "difficulty": str(record["difficulty"] or "").strip(),
                "summary": str(record["summary"] or "").strip(),
                "concepts": _dedup_strings(list(record["concepts"] or [])),
                "document_count": int(record["document_count"] or 0),
                "chunk_count": int(record["chunk_count"] or 0),
            }
            for record in source_records
        }
        relations = [
            {
                "source": str(record["source"]),
                "relation": str(record["relation"]),
                "target": str(record["target"]),
                "subject": str(record["subject"] or "").strip(),
                "evidence_sources": _dedup_strings(list(record["evidence_sources"] or [])),
            }
            for record in relation_records
        ]
        return {
            "concepts": concepts,
            "sources": sources,
            "relations": relations,
            "stats": {
                "concepts": len(concepts),
                "sources": len(sources),
                "relations": len(relations),
            },
        }
    except Exception:
        return _graph_fallback()


def build_graph_index(documents: list[Document]) -> dict[str, Any]:
    """统一构建图谱，并同步 JSON / Neo4j。"""
    graph = _build_graph_payload(documents)
    _persist_graph_json(graph)
    _sync_graph_to_neo4j(graph)
    return graph


def load_graph_index() -> dict[str, Any]:
    """读取图谱，优先读取当前生效后端。"""
    graph = _load_graph_from_neo4j() if active_graph_backend() == "neo4j" else _graph_fallback()
    if graph.get("concepts") or graph.get("sources") or graph.get("relations"):
        return graph
    return _graph_fallback()


def graph_overview(limit: int = 12) -> dict[str, Any]:
    """返回图谱概览，用于 API 展示和调试。"""
    graph = load_graph_index()
    concepts = list(graph.get("concepts", {}).values())[:limit]
    sources = list(graph.get("sources", {}).values())[:limit]
    return {
        "stats": graph.get("stats", {}),
        "backend": active_graph_backend(),
        "concepts": concepts,
        "sources": sources,
        "relations": graph.get("relations", [])[:limit],
    }


def _concept_match_score(question: str, concept: str, aliases: list[str]) -> float:
    """根据问题和概念/别名的匹配程度打分。"""
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


def _build_graph_document(
    matched_names: list[str],
    related_concepts: list[str],
    relation_hits: list[dict[str, Any]],
    top_sources: list[tuple[str, float]],
    dominant_subject: str,
    backend: str,
) -> list[Document]:
    """把图谱结果整理成一个特殊的 graph document，供后续生成阶段直接使用。"""
    if not matched_names:
        return []
    lines = [f"问题涉及概念：{'、'.join(matched_names)}。"]
    if related_concepts:
        lines.append(f"图谱关联概念：{'、'.join(related_concepts)}。")
    if relation_hits:
        relation_text = "；".join(f"{edge['source']}—{edge['relation']}→{edge['target']}" for edge in relation_hits[:5])
        lines.append(f"关键关系：{relation_text}。")
    if top_sources:
        source_text = "；".join(f"{source}（图谱得分 {score}）" for source, score in top_sources)
        lines.append(f"优先参考资料：{source_text}。")
    return [
        Document(
            page_content="\n".join(lines),
            metadata={
                "source": "graph://neo4j-cypher" if backend == "neo4j" else "graph://knowledge-map",
                "file_type": "graph",
                "subject": dominant_subject,
                "chapter": "知识图谱检索",
                "concepts": _dedup_strings([*matched_names, *related_concepts]),
                "retrieval_mode": "graph",
                "score": 0.0,
            },
        )
    ]


def _question_terms(question: str) -> list[str]:
    """从问题中抽取可用于图谱检索的关键词项。"""
    tokens = re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]{2,}", question or "")
    tokens = [token.casefold() for token in tokens if len(token.strip()) >= 2]
    return _dedup_strings(tokens)[:10]


def _search_graph_json(question: str, limit_sources: int = 4, limit_relations: int = 8) -> dict[str, Any]:
    """JSON 图谱回退检索。

    逻辑：
    - 先匹配概念
    - 再找相关关系
    - 最后给资料来源打分
    """
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
    dominant_subject = next((item.get("subject", "") for item in matched if item.get("subject")), "")

    return {
        "backend": "json",
        "matched_concepts": matched,
        "related_concepts": related_concepts,
        "relation_hits": relation_hits,
        "source_scores": source_scores,
        "top_sources": [{"source": source, "score": score} for source, score in top_sources],
        "documents": _build_graph_document(matched_names, related_concepts, relation_hits, top_sources, dominant_subject, "json"),
        "stats": graph.get("stats", {}),
    }


def _search_graph_neo4j(question: str, limit_sources: int = 4, limit_relations: int = 8) -> dict[str, Any]:
    """Neo4j 图谱检索。

    调用顺序：
    1. 先用 Cypher 匹配命中概念
    2. 再扩展 RELATED 关系
    3. 再回查 MENTIONED_IN 来源
    4. 把图谱结果转成 graph document 参与后续混合检索
    """
    terms = _question_terms(question)
    lower_question = (question or "").casefold()
    driver = _get_neo4j_driver()
    with driver:
        with driver.session(database=settings.neo4j_database) as session:
            matched_records = list(
                session.run(
                    """
                    MATCH (c:KnowledgeGraph:Concept)
                    WHERE toLower(c.name) CONTAINS $question
                       OR $question CONTAINS toLower(c.name)
                       OR ANY(alias IN coalesce(c.aliases, []) WHERE toLower(alias) CONTAINS $question OR $question CONTAINS toLower(alias))
                       OR ANY(term IN $terms WHERE toLower(c.name) CONTAINS term OR ANY(alias IN coalesce(c.aliases, []) WHERE toLower(alias) CONTAINS term))
                    RETURN c.name AS name,
                           coalesce(c.subject, '') AS subject,
                           coalesce(c.aliases, []) AS aliases,
                           CASE
                               WHEN toLower(c.name) = $question THEN 1.0
                               WHEN ANY(alias IN coalesce(c.aliases, []) WHERE toLower(alias) = $question) THEN 0.98
                               ELSE 0.42 + 0.08 * size([term IN $terms WHERE toLower(c.name) CONTAINS term OR ANY(alias IN coalesce(c.aliases, []) WHERE toLower(alias) CONTAINS term)])
                           END AS score
                    ORDER BY score DESC, size(c.name) DESC, c.name ASC
                    LIMIT 5
                    """,
                    question=lower_question,
                    terms=terms,
                )
            )
            matched = [
                {"name": str(record["name"]), "score": round(float(record["score"] or 0.0), 4), "subject": str(record["subject"] or "")}
                for record in matched_records
                if record.get("name")
            ]
            matched_names = [item["name"] for item in matched]
            if not matched_names:
                return _search_graph_json(question, limit_sources=limit_sources, limit_relations=limit_relations)

            relation_records = list(
                session.run(
                    """
                    MATCH (a:KnowledgeGraph:Concept)-[r:RELATED]->(b:KnowledgeGraph:Concept)
                    WHERE a.name IN $matched_names OR b.name IN $matched_names
                    RETURN a.name AS source,
                           b.name AS target,
                           coalesce(r.relation, '关联') AS relation,
                           coalesce(r.subject, '') AS subject,
                           coalesce(r.evidence_sources, []) AS evidence_sources,
                           CASE
                               WHEN a.name IN $matched_names AND b.name IN $matched_names THEN 0.76
                               WHEN a.name IN $matched_names THEN 0.66
                               ELSE 0.6
                           END AS related_score
                    ORDER BY related_score DESC, source ASC, target ASC
                    LIMIT $limit_relations
                    """,
                    matched_names=matched_names,
                    limit_relations=limit_relations,
                )
            )
            relation_hits = [
                {
                    "source": str(record["source"]),
                    "relation": str(record["relation"]),
                    "target": str(record["target"]),
                    "subject": str(record["subject"] or ""),
                    "evidence_sources": _dedup_strings(list(record["evidence_sources"] or [])),
                }
                for record in relation_records
            ]
            related_scores: dict[str, float] = {}
            for record in relation_records:
                source = str(record["source"])
                target = str(record["target"])
                boost = round(float(record["related_score"] or 0.0), 4)
                if source in matched_names and target not in matched_names:
                    related_scores[target] = max(related_scores.get(target, 0.0), boost)
                if target in matched_names and source not in matched_names:
                    related_scores[source] = max(related_scores.get(source, 0.0), max(0.0, boost - 0.04))

            source_records = list(
                session.run(
                    """
                    MATCH (c:KnowledgeGraph:Concept)-[:MENTIONED_IN]->(s:KnowledgeGraph:Source)
                    WHERE c.name IN $matched_names OR c.name IN $related_names
                    WITH s,
                         collect(DISTINCT c.name) AS hit_concepts,
                         count(DISTINCT CASE WHEN c.name IN $matched_names THEN c END) AS matched_count,
                         count(DISTINCT c) AS total_hits
                    RETURN s.source AS source,
                           coalesce(s.subject, '') AS subject,
                           coalesce(s.chapter, '') AS chapter,
                           coalesce(s.summary, '') AS summary,
                           coalesce(s.concepts, []) AS concepts,
                           matched_count,
                           total_hits,
                           0.28 * matched_count + 0.14 * total_hits + CASE WHEN toLower(s.chapter) CONTAINS $question THEN 0.12 ELSE 0 END + CASE WHEN toLower(s.summary) CONTAINS $question THEN 0.08 ELSE 0 END AS score
                    ORDER BY score DESC, source ASC
                    LIMIT $limit_sources
                    """,
                    matched_names=matched_names,
                    related_names=list(related_scores.keys()),
                    question=lower_question,
                    limit_sources=limit_sources,
                )
            )
            source_scores = {str(record["source"]): round(float(record["score"] or 0.0), 4) for record in source_records if record.get("source")}
            top_sources = sorted(source_scores.items(), key=lambda item: (-item[1], item[0]))[:limit_sources]
            related_concepts = [name for name, _ in sorted(related_scores.items(), key=lambda item: (-item[1], item[0]))[:6]]
            dominant_subject = next((item.get("subject", "") for item in matched if item.get("subject")), "")
            return {
                "backend": "neo4j",
                "matched_concepts": matched,
                "related_concepts": related_concepts,
                "relation_hits": relation_hits,
                "source_scores": source_scores,
                "top_sources": [{"source": source, "score": score} for source, score in top_sources],
                "documents": _build_graph_document(matched_names, related_concepts, relation_hits, top_sources, dominant_subject, "neo4j"),
                "stats": load_graph_index().get("stats", {}),
            }


def search_graph(question: str, limit_sources: int = 4, limit_relations: int = 8) -> dict[str, Any]:
    """统一图谱检索入口。

    对上层来说不需要区分 JSON 还是 Neo4j，只需要拿到统一结构的图谱检索结果。
    """
    if active_graph_backend() == "neo4j":
        try:
            return _search_graph_neo4j(question, limit_sources=limit_sources, limit_relations=limit_relations)
        except Exception:
            return _search_graph_json(question, limit_sources=limit_sources, limit_relations=limit_relations)
    return _search_graph_json(question, limit_sources=limit_sources, limit_relations=limit_relations)
