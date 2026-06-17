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
    openai_model_definition: str = ""
    openai_model_process: str = ""
    openai_model_comparison: str = ""
    openai_model_analysis: str = ""
    openai_model_advice: str = ""
    openai_model_fact: str = ""
    openai_model_greeting: str = ""

    auth_enabled: bool = False
    auth_tokens: str = ""
    default_tenant_id: str = "public"
    require_tenant_header: bool = False

    vector_index_dir: str = str(BACKEND_ROOT / "data" / "vector_index")
    graph_index_dir: str = str(BACKEND_ROOT / "data" / "graph_index")
    graph_store_backend: str = "neo4j"
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"

    elasticsearch_enabled: bool = False
    elasticsearch_url: str = "http://127.0.0.1:9200"
    elasticsearch_api_key: str = ""
    elasticsearch_username: str = ""
    elasticsearch_password: str = ""
    elasticsearch_index: str = "education-agent-knowledge"
    elasticsearch_verify_certs: bool = False

    milvus_enabled: bool = False
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str = ""
    milvus_collection: str = "education_agent_chunks"

    static_dir: str = str(BACKEND_ROOT / "static")
    media_output_dir: str = str(BACKEND_ROOT / "static" / "generated_media")
    knowledge_dir: str = str(BACKEND_ROOT / "data" / "knowledge")
    knowledge_metadata_file: str = str(BACKEND_ROOT / "data" / "knowledge_metadata.json")

    retrieval_vector_k: int = 18
    retrieval_lexical_k: int = 16
    retrieval_final_k: int = 8
    retrieval_max_per_source: int = 2
    retrieval_chunk_budget_tokens: int = 260
    retrieval_graph_budget_tokens: int = 120
    retrieval_strategy_router_enabled: bool = True
    retrieval_rerank_window: int = 24

    retrieval_cache_enabled: bool = True
    retrieval_cache_file: str = str(BACKEND_ROOT / "data" / "graph_index" / "retrieval_cache.json")
    retrieval_cache_similarity_threshold: float = 0.84
    retrieval_cache_max_entries: int = 200
    retrieval_cache_strict_similarity_delta: float = 0.08
    retrieval_cache_high_risk_exact_only: bool = True

    retrieval_answer_grounding_threshold: float = 0.24
    retrieval_answer_grounding_sentence_threshold: float = 0.1

    task_store_backend: str = "sqlite"
    task_store_file: str = str(BACKEND_ROOT / "data" / "runtime" / "tasks.json")
    task_store_db_file: str = str(BACKEND_ROOT / "data" / "runtime" / "tasks.sqlite3")
    task_store_migrate_legacy_json: bool = True
    task_event_limit: int = 200

    graph_max_steps: int = 3
    graph_task_timeout_seconds: int = 180
    graph_auto_select_min_score_gap: float = 0.18
    graph_auto_select_complex_gap: float = 0.28
    answer_fact_limit: int = 7

    media_job_timeout_seconds: int = 1800
    media_job_max_refresh_attempts: int = 30

    image_generation_provider: str = "demo"
    image_generation_api_url: str = ""
    image_generation_api_key: str = ""
    image_generation_model: str = "stable-diffusion-xl"

    video_generation_provider: str = "demo"
    video_generation_api_url: str = ""
    video_generation_status_url: str = ""
    video_generation_api_key: str = ""
    video_generation_model: str = "gen4.5"
    media_poll_interval_seconds: int = 3

    stability_api_base_url: str = "https://api.stability.ai"
    stability_api_key: str = ""
    stability_image_model: str = "core"
    stability_output_format: str = "png"

    runway_api_base_url: str = "https://api.dev.runwayml.com"
    runway_api_key: str = ""
    runway_api_version: str = "2024-11-06"
    runway_image_model: str = "gen4_image"
    runway_video_model: str = "gen4.5"
    runway_image_ratio: str = "1280:720"
    runway_video_ratio: str = "1280:720"

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000"

    daily_push_cron_hour: int = 8
    daily_push_cron_minute: int = 0
    evaluation_failure_export_file: str = str(BACKEND_ROOT / "data" / "runtime" / "failed_chat_cases.json")
    release_gate_hit_at_k_min: float = 0.6
    release_gate_mrr_min: float = 0.45
    release_gate_grounding_score_min: float = 0.2
    release_gate_max_failure_cases: int = 12

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
