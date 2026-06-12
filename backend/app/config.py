from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 后端根目录，其他默认路径都基于这里展开
BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI API Key，用于问答生成与 embedding
    openai_api_key: str = ""
    # OpenAI 兼容接口地址，支持代理或兼容服务
    openai_api_base: str = "https://api.openai.com/v1"
    # 默认对话模型
    openai_model: str = "gpt-4o-mini"
    # 默认向量模型
    openai_embedding_model: str = "text-embedding-3-small"

    # 向量索引目录
    vector_index_dir: str = str(BACKEND_ROOT / "data" / "vector_index")
    # 图索引目录
    graph_index_dir: str = str(BACKEND_ROOT / "data" / "graph_index")
    # 图存储后端，默认走 neo4j
    graph_store_backend: str = "neo4j"
    # Neo4j 连接地址
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    # Neo4j 用户名
    neo4j_user: str = "neo4j"
    # Neo4j 密码
    neo4j_password: str = ""
    # Neo4j 数据库名
    neo4j_database: str = "neo4j"

    # 是否启用 Elasticsearch 词法检索
    elasticsearch_enabled: bool = False
    # Elasticsearch 服务地址
    elasticsearch_url: str = "http://127.0.0.1:9200"
    # Elasticsearch API Key
    elasticsearch_api_key: str = ""
    # Elasticsearch 用户名
    elasticsearch_username: str = ""
    # Elasticsearch 密码
    elasticsearch_password: str = ""
    # Elasticsearch 索引名
    elasticsearch_index: str = "education-agent-knowledge"
    # 是否校验证书，开发环境默认关闭
    elasticsearch_verify_certs: bool = False

    # 是否启用 Milvus 向量库
    milvus_enabled: bool = False
    # Milvus 服务地址
    milvus_uri: str = "http://127.0.0.1:19530"
    # Milvus 鉴权 token
    milvus_token: str = ""
    # Milvus 集合名
    milvus_collection: str = "education_agent_chunks"

    # 静态资源目录
    static_dir: str = str(BACKEND_ROOT / "static")
    # 生成图片/视频等素材输出目录
    media_output_dir: str = str(BACKEND_ROOT / "static" / "generated_media")
    # 知识库原始文件目录
    knowledge_dir: str = str(BACKEND_ROOT / "data" / "knowledge")
    # 知识库元数据文件
    knowledge_metadata_file: str = str(BACKEND_ROOT / "data" / "knowledge_metadata.json")

    # 向量召回条数
    retrieval_vector_k: int = 18
    # 关键词召回条数
    retrieval_lexical_k: int = 16
    # 最终保留候选条数
    retrieval_final_k: int = 8
    # 单个来源最多保留多少条，避免单一文档霸榜
    retrieval_max_per_source: int = 2
    # 普通 chunk 的 token 预算
    retrieval_chunk_budget_tokens: int = 260
    # 图谱上下文的 token 预算
    retrieval_graph_budget_tokens: int = 120
    # 是否启用检索路由器，根据问题复杂度动态调参
    retrieval_strategy_router_enabled: bool = True
    # 重排窗口大小，控制进入 rerank 的候选范围
    retrieval_rerank_window: int = 24

    # 是否启用检索缓存
    retrieval_cache_enabled: bool = True
    # 检索缓存文件位置
    retrieval_cache_file: str = str(BACKEND_ROOT / "data" / "graph_index" / "retrieval_cache.json")
    # 缓存复用的基础相似度阈值
    retrieval_cache_similarity_threshold: float = 0.84
    # 缓存最多保留多少条记录
    retrieval_cache_max_entries: int = 200
    # 严格缓存策略相比基础阈值额外提高的相似度
    retrieval_cache_strict_similarity_delta: float = 0.08
    # 高风险问题是否仅允许 exact match 复用缓存
    retrieval_cache_high_risk_exact_only: bool = True

    # 回答 grounded 总分阈值
    retrieval_answer_grounding_threshold: float = 0.24
    # 单句被视为有证据支撑的阈值
    retrieval_answer_grounding_sentence_threshold: float = 0.1

    # 任务存储后端，默认 sqlite
    task_store_backend: str = "sqlite"
    # 旧版 json 任务存储路径
    task_store_file: str = str(BACKEND_ROOT / "data" / "runtime" / "tasks.json")
    # sqlite 任务库文件
    task_store_db_file: str = str(BACKEND_ROOT / "data" / "runtime" / "tasks.sqlite3")
    # 启动时是否迁移旧 json 数据到 sqlite
    task_store_migrate_legacy_json: bool = True
    # 单任务最大事件保留数
    task_event_limit: int = 200

    # LangGraph 最大循环步数
    graph_max_steps: int = 3
    # 单个图任务最大执行秒数
    graph_task_timeout_seconds: int = 180
    # 简单问题自动选证据所需的最小分差
    graph_auto_select_min_score_gap: float = 0.18
    # 复杂问题自动选证据所需的更高分差
    graph_auto_select_complex_gap: float = 0.28
    # 从证据中最多抽取多少条事实用于生成
    answer_fact_limit: int = 7

    # 媒体任务超时时间
    media_job_timeout_seconds: int = 1800
    # 媒体任务最大轮询次数
    media_job_max_refresh_attempts: int = 30

    # 图片生成提供方
    image_generation_provider: str = "demo"
    # 图片生成接口地址
    image_generation_api_url: str = ""
    # 图片生成接口密钥
    image_generation_api_key: str = ""
    # 图片生成模型名
    image_generation_model: str = "stable-diffusion-xl"

    # 视频生成提供方
    video_generation_provider: str = "demo"
    # 视频生成接口地址
    video_generation_api_url: str = ""
    # 视频状态查询接口地址
    video_generation_status_url: str = ""
    # 视频生成接口密钥
    video_generation_api_key: str = ""
    # 视频生成模型名
    video_generation_model: str = "gen4.5"
    # 媒体轮询间隔秒数
    media_poll_interval_seconds: int = 3

    # Stability API 基础地址
    stability_api_base_url: str = "https://api.stability.ai"
    # Stability API Key
    stability_api_key: str = ""
    # Stability 图片模型
    stability_image_model: str = "core"
    # Stability 输出格式
    stability_output_format: str = "png"

    # Runway API 基础地址
    runway_api_base_url: str = "https://api.dev.runwayml.com"
    # Runway API Key
    runway_api_key: str = ""
    # Runway API 版本
    runway_api_version: str = "2024-11-06"
    # Runway 图片模型
    runway_image_model: str = "gen4_image"
    # Runway 视频模型
    runway_video_model: str = "gen4.5"
    # Runway 图片默认比例
    runway_image_ratio: str = "1280:720"
    # Runway 视频默认比例
    runway_video_ratio: str = "1280:720"

    # 后端监听地址
    host: str = "0.0.0.0"
    # 后端监听端口
    port: int = 8000
    # 允许跨域的前端来源列表，逗号分隔
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8000,http://127.0.0.1:8000"

    # 每日推送定时任务小时
    daily_push_cron_hour: int = 8
    # 每日推送定时任务分钟
    daily_push_cron_minute: int = 0
    # 失败问答样本导出文件，用于回流评测
    evaluation_failure_export_file: str = str(BACKEND_ROOT / "data" / "runtime" / "failed_chat_cases.json")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
