import re
from typing import Any

_EN_RE = re.compile(r"[a-zA-Z]{2,}")


def _tokens(text: str) -> set[str]:
    text = (text or "").strip().lower()
    tokens: set[str] = set()
    for w in _EN_RE.findall(text):
        tokens.add(w)
    chinese = re.sub(r"[^\u4e00-\u9fff]", "", text)
    if len(chinese) < 2:
        return tokens
    for n in (2, 3, 4):
        for i in range(len(chinese) - n + 1):
            gram = chinese[i : i + n]
            tokens.add(gram)
    return tokens


def keyword_overlap(question: str, content: str) -> float:
    q = _tokens(question)
    c = _tokens(content)
    if not q or not c:
        return 0.0
    hit = q & c
    if not hit:
        return 0.0
    strong = [t for t in hit if len(t) >= 3]
    if strong:
        return min(1.0, 0.25 + len(strong) * 0.12)
    return min(1.0, len(hit) / max(min(len(q), 24), 1))


def is_chunk_relevant_to_question(
    question: str,
    content: str,
    distance: float | None = None,
    metadata: dict[str, Any] | None = None,
    matched_concepts: list[str] | None = None,
) -> bool:
    metadata = metadata or {}
    concepts = [str(item).strip() for item in metadata.get("concepts") or [] if str(item).strip()]
    aliases = [str(item).strip() for item in metadata.get("aliases") or [] if str(item).strip()]
    concept_text = " ".join([*concepts, *aliases])

    q = _tokens(question)
    c = _tokens(content)
    hit = q & c
    if any(len(t) >= 3 for t in hit):
        return True
    if len(hit) >= 4:
        return True
    if keyword_overlap(question, content) >= 0.15:
        return True
    if concept_text and keyword_overlap(question, concept_text) >= 0.12:
        return True
    if matched_concepts and set(matched_concepts) & set(concepts):
        return True
    if metadata.get("file_type") == "graph" and matched_concepts:
        return True
    if distance is not None and distance < 0.55 and len(hit) >= 2:
        return True
    return False
