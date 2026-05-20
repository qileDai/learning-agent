from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        openai_api_key=settings.openai_api_key or "dummy",
        openai_api_base=settings.openai_api_base,
    )


def get_vector_store() -> Chroma:
    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name="education_kb",
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_persist_dir,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_documents(documents: list[Document], reset: bool = False) -> int:
    if not documents:
        return 0
    chunks = split_documents(documents)
    store = get_vector_store()
    if reset:
        try:
            store.delete_collection()
        except Exception:
            pass
        store = get_vector_store()
    store.add_documents(chunks)
    return len(chunks)


def similarity_search(query: str, k: int = 8) -> list[Document]:
    store = get_vector_store()
    return store.similarity_search(query, k=k)


def similarity_search_with_scores(query: str, k: int = 12) -> list[tuple[Document, float]]:
    store = get_vector_store()
    try:
        return store.similarity_search_with_score(query, k=k)
    except Exception:
        docs = store.similarity_search(query, k=k)
        return [(d, 1.0) for d in docs]
