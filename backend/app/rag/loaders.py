import zipfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def _load_docx(path: Path) -> list[Document]:
    """加载 Word；若扩展名为 .docx 但实际为纯文本，则按 UTF-8 读取。"""
    if not zipfile.is_zipfile(path):
        return TextLoader(str(path), encoding="utf-8").load()
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    text = "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
    return [Document(page_content=text)] if text else []


def load_file(path: Path) -> list[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in (".docx", ".doc"):
        return _load_docx(path)
    if suffix in (".md", ".markdown"):
        return TextLoader(str(path), encoding="utf-8").load()
    if suffix in (".txt",):
        return TextLoader(str(path), encoding="utf-8").load()
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
