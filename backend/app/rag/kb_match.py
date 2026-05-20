"""判断检索结果是否与用户问题真正匹配（用于分流：走知识库 or 走大模型）。"""
import re

_EN_RE = re.compile(r"[a-zA-Z]{2,}")


def _tokens(text: str) -> set[str]:
    """中文 2~4 字 n-gram + 英文单词，避免整句被当成一个词。"""
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


def is_chunk_relevant_to_question(question: str, content: str, distance: float | None = None) -> bool:
    q = _tokens(question)
    c = _tokens(content)
    hit = q & c
    if any(len(t) >= 3 for t in hit):
        return True
    if len(hit) >= 4:
        return True
    if keyword_overlap(question, content) >= 0.15:
        return True
    # 向量非常接近且有一定字面重叠
    if distance is not None and distance < 0.55 and len(hit) >= 2:
        return True
    return False
