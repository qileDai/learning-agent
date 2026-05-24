from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    vector_index_dir: str = str(BACKEND_ROOT / "data" / "vector_index")
    graph_index_dir: str = str(BACKEND_ROOT / "data" / "graph_index")
    static_dir: str = str(BACKEND_ROOT / "static")
    knowledge_dir: str = str(BACKEND_ROOT / "data" / "knowledge")
    knowledge_metadata_file: str = str(BACKEND_ROOT / "data" / "knowledge_metadata.json")

    retrieval_vector_k: int = 18
    retrieval_lexical_k: int = 16
    retrieval_final_k: int = 8
    retrieval_max_per_source: int = 2
    retrieval_chunk_budget_tokens: int = 260
    retrieval_graph_budget_tokens: int = 120

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000"

    daily_push_cron_hour: int = 8
    daily_push_cron_minute: int = 0

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
