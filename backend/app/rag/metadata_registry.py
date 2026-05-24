import json
from pathlib import Path
from typing import Any

from app.config import settings


def normalize_source(source: str) -> str:
    return source.replace("\\", "/").strip()


def load_metadata_registry() -> dict[str, Any]:
    path = Path(settings.knowledge_metadata_file)
    if not path.exists():
        return {"sources": {}, "concepts": {}, "relations": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = {
        normalize_source(key): value
        for key, value in (data.get("sources") or {}).items()
        if isinstance(value, dict)
    }
    concepts = {
        str(key).strip(): value
        for key, value in (data.get("concepts") or {}).items()
        if isinstance(value, dict)
    }
    relations = [
        item
        for item in (data.get("relations") or [])
        if isinstance(item, dict) and item.get("source") and item.get("target") and item.get("relation")
    ]
    return {"sources": sources, "concepts": concepts, "relations": relations}


def source_metadata_for(source: str) -> dict[str, Any]:
    return dict(load_metadata_registry().get("sources", {}).get(normalize_source(source), {}))
