import zipfile
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def _load_pdf(path: Path) -> list[Document]:
    reader = PdfReader(str(path))
    docs: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"page": i}))
    return docs


def _load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8").strip()
    return [Document(page_content=text)] if text else []


def _load_docx(path: Path) -> list[Document]:
    """加载 Word；若扩展名为 .docx 但实际为纯文本，则按 UTF-8 读取。"""
    if not zipfile.is_zipfile(path):
        return _load_text(path)
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    text = "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    return [Document(page_content=text)] if text else []


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in (".docx", ".doc"):
        return _load_docx(path)
    if suffix in (".md", ".markdown", ".txt"):
        return _load_text(path)
    return []


def load_directory(directory: Path) -> list[Document]:
    docs: list[Document] = []
    patterns = ("**/*.pdf", "**/*.docx", "**/*.md", "**/*.markdown", "**/*.txt")
    for pattern in patterns:
        for path in directory.glob(pattern):
            if not path.is_file():
                continue
            try:
                loaded = load_file(path)
                for doc in loaded:
                    doc.metadata["source"] = str(path.relative_to(directory))
                    doc.metadata["file_type"] = path.suffix.lower().lstrip(".")
                docs.extend(loaded)
            except Exception as exc:
                print(f"[ingest] skip {path}: {exc}")
    return docs
