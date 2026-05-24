# 企业级 RAG 技术方案文档

## 1. 文档目标

本文档用于系统梳理当前项目中已经落地的企业级 RAG 方案，说明整体架构、核心组件、检索与生成链路、可观测性设计，以及每个模块在系统中的实际作用。

这套方案不是单一的“向量检索 + 大模型回答”，而是围绕企业知识库常见问题做的完整增强：

- 提升召回率
- 提升回答准确率
- 控制上下文长度与 token 成本
- 支持图谱增强检索
- 支持可观测性与评测
- 支持多后端存储与可回退运行

---

## 2. 方案目标与设计原则

### 2.1 业务目标

面向企业级知识库问答，系统需要解决以下典型问题：

1. 用户问题表达不稳定，同一知识点有多种问法、简称、别名
2. 单一向量召回容易漏掉关键词、公式、术语、专有名词
3. 单一词法召回又难以覆盖语义相近但表达不同的问题
4. 单篇文档切分后容易出现相似 chunk 过多，影响结果多样性
5. 图谱关系无法充分利用时，复杂问题容易缺失上下文关联
6. 提示词上下文过长时，推理成本高、噪音大、稳定性差
7. 系统上线后若没有指标和评测，难以持续优化

### 2.2 设计原则

本方案遵循以下原则：

- 混合检索优先，而不是依赖单一路径
- 图谱增强优先，而不是把知识图谱做成独立摆设
- 可降级运行优先，而不是强依赖单一基础设施
- 可解释与可观测优先，而不是只返回最终答案
- 成本受控优先，而不是无限堆上下文

---

## 3. 总体架构

当前企业级 RAG 架构可以概括为：

**文档接入 -> 元数据增强 -> Chunk 切分 -> 向量化 -> 向量库 / 关键词索引 / 图谱索引三路构建 -> 在线混合检索 -> 多路融合排序 -> 上下文压缩 -> LLM 生成 -> 指标埋点与评测**

### 3.1 架构分层

#### 接入层
- 负责读取知识库目录中的 PDF、DOCX、Markdown 等文件
- 解析文件内容并抽取元数据

#### 索引层
- 向量索引：本地向量索引 + Milvus
- 关键词索引：Elasticsearch
- 图谱索引：JSON 图谱 + Neo4j

#### 检索层
- 查询扩展
- 路由分类
- 图谱概念匹配
- 向量召回
- 词法召回
- 多路融合排序
- 多样性控制
- 语义缓存

#### 生成层
- 选择最合适证据片段
- 压缩图谱上下文
- 控制 token 预算
- 生成最终回答
- 做答案 grounding 校验

#### 观测层
- Prometheus 指标埋点
- 检索摘要输出
- 评测接口输出 Hit@K / MRR / grounding 结果

---

## 4. 核心模块与作用说明

## 4.1 配置中心

文件：`backend/app/config.py`

作用：
- 统一管理模型配置、检索参数、图数据库、Elasticsearch、Milvus、Prometheus 相关配置
- 让系统支持“本地开发 / 单机运行 / 企业环境部署”三种模式切换

关键配置分组：

### 模型配置
- `openai_api_key`
- `openai_api_base`
- `openai_model`
- `openai_embedding_model`

作用：
- 控制生成模型与 Embedding 模型的来源

### 检索参数配置
- `retrieval_vector_k`
- `retrieval_lexical_k`
- `retrieval_final_k`
- `retrieval_max_per_source`
- `retrieval_chunk_budget_tokens`
- `retrieval_graph_budget_tokens`
- `retrieval_rerank_window`
- `retrieval_strategy_router_enabled`
- `retrieval_cache_enabled`

作用：
- 控制召回深度、融合窗口、最终候选数量、单来源上限、上下文预算与缓存策略

### Elasticsearch 配置
- `elasticsearch_enabled`
- `elasticsearch_url`
- `elasticsearch_index`
- `elasticsearch_api_key`
- `elasticsearch_username`
- `elasticsearch_password`

作用：
- 打开或关闭关键词召回能力
- 用于接入企业已有 ES 集群

### Milvus 配置
- `milvus_enabled`
- `milvus_uri`
- `milvus_token`
- `milvus_collection`

作用：
- 打开或关闭 Milvus 向量库能力
- 支持把向量检索从本地文件索引升级到企业级向量数据库

### Neo4j 配置
- `graph_store_backend`
- `neo4j_uri`
- `neo4j_user`
- `neo4j_password`
- `neo4j_database`

作用：
- 控制图谱存储与检索使用 JSON 还是 Neo4j
- 支持从轻量图谱逐步升级到真实图数据库

---

## 4.2 知识入库模块

文件：`backend/app/rag/ingest.py`

作用：
- 作为知识库构建入口，统一调度文档读取、向量入库、图谱构建
- 一次入库，同时构建三类索引：向量索引、关键词索引、图谱索引

入库输出内容包括：
- 文件数
- 文档数
- chunk 数
- 文件类型分布
- 学科分布
- 图谱统计
- 当前图谱后端

价值：
- 把原来分散的构建过程统一成一个标准入库流程
- 便于知识库刷新、重建与批量接入

---

## 4.3 向量存储模块

文件：`backend/app/rag/vector_store.py`

作用：
- 负责文档切分、Embedding 生成、向量索引持久化、向量召回
- 在架构上承担语义召回主通道

### 当前实现特点

#### 文档切分
- 使用 `RecursiveCharacterTextSplitter`
- 控制 chunk size 与 overlap

作用：
- 保证知识片段既具备语义完整性，又不会过长

#### 本地向量索引
- 使用 `documents.json + embeddings.npy` 持久化

作用：
- 作为默认向量存储与本地回退方案
- 在 Milvus 不可用时仍可运行

#### 向量归一化
- 对文档向量和查询向量做归一化

作用：
- 提高相似度计算稳定性

#### Milvus 集成
- 入库时可同步写入 Milvus
- 查询时优先走 Milvus，失败后自动回退本地索引

作用：
- 让系统具备企业级向量数据库能力
- 同时保留低门槛本地运行体验

---

## 4.4 Milvus 向量库模块

文件：`backend/app/rag/milvus_store.py`

作用：
- 封装 Milvus 的 collection 创建、向量写入、向量搜索能力

### 主要功能

#### Collection 管理
- 自动检测 collection 是否存在
- 支持 reset 时重建 collection
- 自动创建主键字段和向量字段

作用：
- 降低部署初始化成本
- 避免手工建表出错

#### 向量写入
- 把 chunk_id、向量和关键元数据一起写入 Milvus

作用：
- 检索结果可以直接回传原始知识片段，不需要额外关联

#### 向量检索
- 使用 `COSINE` 度量做 ANN 检索
- 返回内容字段与元数据字段

作用：
- 提供高性能语义召回能力
- 为混合排序提供向量分值基础

---

## 4.5 Elasticsearch 关键词召回模块

文件：`backend/app/rag/elastic_store.py`

作用：
- 提供关键词召回能力，补足纯向量检索对术语、专名、精确关键词的短板

### 主要功能

#### 索引结构
索引中存储以下关键信息：
- `page_content`
- `source`
- `subject`
- `chapter`
- `summary`
- `concepts`
- `aliases`

作用：
- 让 ES 不只搜索正文，还能搜索章节、摘要、概念、别名等结构化信息

#### 文档写入
- 入库时把 chunk 同步写入 ES

作用：
- 实现知识库内容的全文检索能力

#### 词法检索
- 使用 `multi_match`
- 对正文、章节、摘要、概念、别名设置不同权重
- 支持按学科做 filter

作用：
- 提升关键词场景下的召回率与排序精度
- 与语义召回形成互补

---

## 4.6 图谱存储与检索模块

文件：`backend/app/rag/graph_store.py`

作用：
- 负责知识图谱的构建、JSON 持久化、Neo4j 同步、图谱检索
- 是当前 Graph RAG 能力的核心承载模块

### 图谱模型

#### Concept 节点
包含：
- 概念名
- 学科
- 描述
- 别名
- 章节
- 来源列表

作用：
- 表达知识中的核心实体与术语

#### Source 节点
包含：
- 来源文件
- 学科
- 年级
- 章节
- 难度
- 摘要
- 概念集合

作用：
- 把知识片段和知识来源组织起来

#### 关系边
包含：
- source
- relation
- target
- subject
- evidence_sources

作用：
- 表示概念之间的知识关联
- 支撑复杂问题中的上下游推理

### 双后端设计

#### JSON 图谱
作用：
- 轻量运行模式
- 无 Neo4j 环境时仍可提供基础图谱能力

#### Neo4j 图谱
作用：
- 提供真实图数据库能力
- 支持 Cypher 检索与关系扩展

### Neo4j Cypher 检索升级点

当前不是简单把概念导入 Neo4j，而是把在线检索也升级为真实 Cypher 过程：

1. 先对问题做词项抽取
2. 在 Neo4j 中匹配概念名和别名
3. 通过 `RELATED` 关系扩展相关概念
4. 通过 `MENTIONED_IN` 找到命中的来源文档
5. 对来源做图谱打分
6. 生成图谱上下文文档参与最终混合检索

作用：
- 让图谱从“展示用数据”变成真正参与召回和排序的检索通道
- 对复杂问题、跨概念问题、上下游问题更有效

---

## 4.7 检索优化器模块

文件：`backend/app/rag/retrieval_optimizer.py`

作用：
- 承载查询理解、轻量词法计算、上下文预算控制、缓存与答案校验等横向能力
- 是企业级 RAG 中非常关键的“策略层”模块

### 主要能力

#### 文本分词与 token 估算
- `text_tokens`
- `estimate_tokens`

作用：
- 为词法召回、答案校验、预算裁剪提供统一基础

#### 查询分类
- `classify_query`

作用：
- 按问题复杂度自动路由为 `simple / complex / analysis`
- 动态调整 `vector_k / lexical_k / final_k / max_per_source`

业务价值：
- 简单问题不浪费资源
- 复杂问题自动扩大召回范围

#### 查询扩展
- `expand_query`

作用：
- 根据图谱中的概念与别名扩展用户问题
- 输出 `query_expansions` 与 `route_subjects`

业务价值：
- 提升同义词、简称、别名场景下的召回率

#### 词法评分
- `lexical_score`
- `score_document_coverage`

作用：
- 为本地 fallback 词法召回与融合排序提供稳定评分

#### RRF 融合
- `reciprocal_rank_fusion`

作用：
- 融合多个召回通道的排名位置
- 防止某一路分值过大完全压制其他路径

#### 多样性控制
- `diversify_documents`

作用：
- 控制单个 source 不要占据过多候选位
- 避免相似 chunk 过多

#### 语义缓存
- `get_cached_retrieval`
- `save_cached_retrieval`

作用：
- 对相似问题复用已有检索结果
- 降低重复查询时的计算成本

#### 答案 grounding 校验
- `validate_answer_grounding`

作用：
- 校验回答和参考证据之间的重合度
- 为评测与监控提供基础分数

---

## 4.8 混合检索模块

文件：`backend/app/rag/hybrid_retriever.py`

作用：
- 企业级 RAG 在线检索主入口
- 负责把多种召回通道组合成统一的候选结果集

### 在线检索流程

#### 第一步：查询扩展与路由规划
- 调用 `expand_query`
- 获取扩展问句、命中概念、学科路由、检索参数

作用：
- 决定这次检索要走多深、重点搜什么

#### 第二步：语义缓存命中判断
- 若命中缓存，直接复用历史检索结果

作用：
- 降低重复请求的延迟和成本

#### 第三步：图谱检索
- 通过图谱先找到命中概念、相关概念、优先来源

作用：
- 为后续向量与词法召回提供结构化先验

#### 第四步：向量召回
- 对原问题和扩展问题执行向量搜索

作用：
- 覆盖语义近似问题

#### 第五步：关键词召回
- 优先调用 Elasticsearch
- 如果 ES 不可用，则回退本地 lexical scoring

作用：
- 覆盖关键词、术语、精确表达问题

#### 第六步：多路打分融合
融合信号包括：
- vector 分数
- lexical 分数
- graph 分数
- route boost
- coverage 分数
- intent 分数
- consensus 分数
- RRF 分数

作用：
- 把不同召回路径的优点聚合起来
- 让真正相关的文档排到前面

#### 第七步：去重与多样性控制
- 相同文档签名去重
- 限制单来源候选数
- rerank window 内再做 diversify

作用：
- 提升结果覆盖面与稳定性

#### 第八步：输出检索摘要
输出字段包括：
- `query_expansions`
- `route_subjects`
- `route_type`
- `graph_documents`
- `vector_candidates`
- `lexical_candidates`
- `final_candidates`
- `max_per_source`
- `vector_k`
- `lexical_k`
- `final_k`
- `rerank_window`
- `cache_hit`
- `cache_similarity`

作用：
- 让检索链路可解释、可调优、可评估

---

## 4.9 问答工作流与生成模块

文件：
- `backend/app/graph/nodes.py`
- `backend/app/graph/workflow.py`

作用：
- 负责把检索结果转换成最终可用于大模型回答的上下文
- 承担“检索 -> 证据选择 -> 生成 -> 校验”的工作流编排

### 关键节点说明

#### retrieve_node
作用：
- 调用混合检索
- 过滤真正相关的 chunk
- 对 chunk 做预算裁剪
- 对图谱上下文做压缩
- 记录检索指标

#### generate_answer_node
作用：
- 从候选知识片段中选择最终证据
- 构造严格受控的提示词
- 调用大模型生成回答
- 记录答案 grounding 分数

#### generate_answer_llm_node
作用：
- 在知识库未命中时走大模型直答兜底

### 生成阶段的成本控制

当前生成阶段不是把所有召回结果直接塞给模型，而是做了以下控制：

- 普通知识片段：按 chunk token 预算裁剪
- 图谱关系上下文：按 graph token 预算压缩
- 默认只保留单条高质量证据片段作为主参考

作用：
- 显著降低 token 成本
- 减少无关噪音
- 提升回答稳定性与可控性

---

## 4.10 API 层

文件：`backend/app/api/routes.py`

作用：
- 提供系统对外能力入口
- 把企业级 RAG 能力、图谱能力、评测能力暴露为标准接口

### 核心接口

#### 知识入库
- `POST /api/ingest`

作用：
- 重新构建知识库索引与图谱

#### 对话问答
- `POST /api/chat/start`
- `POST /api/chat/resume`
- `GET /api/chat/state/{thread_id}`

作用：
- 提供完整问答流程和候选证据选择流程

#### 图谱能力
- `GET /api/graph/overview`
- `GET /api/graph/search`

作用：
- 提供图谱浏览和图谱检索能力

#### 评测能力
- `POST /api/eval/retrieval`
- `POST /api/eval/answer`

作用：
- 对召回效果和答案 grounding 效果进行离线评估

---

## 4.11 可观测性模块

文件：`backend/app/observability.py`

作用：
- 提供 Prometheus 指标定义、中间件埋点、检索与回答相关指标记录

### 当前指标

#### HTTP 指标
- `education_agent_http_requests_total`
- `education_agent_http_request_latency_seconds`

作用：
- 监控接口调用量与请求耗时

#### 检索指标
- `education_agent_retrieval_total`
- `education_agent_retrieval_final_candidates`

作用：
- 监控检索调用次数、路由类型、缓存命中情况、最终候选规模

#### 回答质量指标
- `education_agent_answer_grounding_score`

作用：
- 监控回答与证据的一致性水平

#### 评测运行指标
- `education_agent_eval_run_total`

作用：
- 监控评测接口调用情况

### 降级设计
- 当 `prometheus_client` 未安装时，系统自动降级为 no-op 指标对象
- 应用仍然可以启动，不会因观测依赖缺失而整体不可用

作用：
- 保证企业能力增强不会破坏基础可用性

---

## 5. 评测方案设计

文件：`backend/app/rag/evaluation.py`

作用：
- 提供离线评测能力，用于验证检索效果和答案 grounding 效果

### 5.1 检索评测

接口：`POST /api/eval/retrieval`

输入：
- question
- expected_sources
- expected_terms
- gold_answer

输出：
- Hit@K
- MRR
- 平均 grounding 分数
- 每条样本的召回结果明细

作用：
- 用于评估知识库召回质量
- 支持后续做参数调优、回归验证和版本对比

### 5.2 答案评测

接口：`POST /api/eval/answer`

输入：
- question
- answer
- references
- concepts

输出：
- grounded
- grounding_score
- reference_overlap
- question_overlap

作用：
- 用于判断当前答案是否真正建立在证据之上

---

## 6. 为什么要同时接入 Elasticsearch、Milvus、Neo4j

## 6.1 Elasticsearch 的作用

适合解决的问题：
- 关键词强依赖问题
- 术语查询
- 公式、专有名词、章节名、摘要字段搜索

如果没有 ES：
- 纯向量召回对精确关键词问题容易不稳定

所以它的定位是：
**关键词召回通道与结构化文本匹配通道**

## 6.2 Milvus 的作用

适合解决的问题：
- 大规模 chunk 语义召回
- 企业级向量存储与检索性能要求

如果没有 Milvus：
- 本地索引适合小规模与开发环境，但不适合更大规模知识库的生产检索

所以它的定位是：
**企业级向量数据库与语义召回主通道**

## 6.3 Neo4j 的作用

适合解决的问题：
- 复杂问题中的概念关联
- 上下游知识推导
- 从命中概念找到关联概念和优先资料来源

如果没有 Neo4j：
- 图谱只能停留在轻量 JSON 匹配层，无法充分发挥关系检索能力

所以它的定位是：
**图谱增强检索与关系推理通道**

---

## 7. 在线链路全流程说明

一次用户问答在系统中的实际执行过程如下：

1. 用户发起问题
2. 系统判断是否是问候语，若不是则进入知识检索
3. 对问题做查询扩展与复杂度路由
4. 尝试命中语义缓存
5. 执行图谱检索，识别相关概念、关系和高优先级来源
6. 执行向量召回，优先 Milvus，失败则本地回退
7. 执行关键词召回，优先 Elasticsearch，失败则本地回退
8. 把多路候选做融合排序
9. 控制单来源候选数并做多样性筛选
10. 输出候选片段给工作流
11. 裁剪证据上下文与图谱上下文
12. 生成最终回答
13. 记录 grounding 分数与检索指标
14. 返回回答、检索摘要、图谱摘要、验证结果

这条链路的核心价值是：
- 先尽量找全
- 再尽量排准
- 最后尽量低成本地回答

---

## 8. 当前方案相对基础版 RAG 的升级点

相比基础版“本地切片 + 向量检索 + 直接回答”，当前方案的升级点如下：

1. 从单一向量召回升级到混合召回
2. 从静态图谱展示升级到真实图谱检索
3. 从本地向量索引升级到可接入 Milvus
4. 从本地词法打分升级到可接入 Elasticsearch
5. 从黑盒问答升级到可解释的检索摘要输出
6. 从只关注答案升级到加入 grounding 校验
7. 从无监控升级到 Prometheus 指标埋点
8. 从无评测升级到可做离线回归评测
9. 从简单长上下文拼接升级到 token 预算控制
10. 从单一运行方式升级到多后端可回退架构

---

## 9. 当前方案的工程价值

### 对业务侧的价值
- 提高问答命中率
- 提高回答可信度
- 更适合知识问答、培训问答、制度问答、内部手册问答等企业场景

### 对算法侧的价值
- 可以持续调优召回参数
- 可以针对不同问题类型做检索策略优化
- 可以通过评测数据闭环优化系统

### 对工程侧的价值
- 组件清晰，可扩展
- 基础设施可替换，可回退
- 支持观测、调优、评估、上线迭代

---

## 10. 后续可继续增强的方向

1. 引入重排模型，对 topN 候选做二次 rerank
2. 把查询理解从规则扩展到 LLM Router 或分类模型
3. 对 Elasticsearch 增加中文分词器和自定义同义词词典
4. 对 Milvus 增加索引参数调优和分区设计
5. 对 Neo4j 增加更细粒度的关系类型与路径查询
6. 引入评测集管理、A/B 评估和版本基线对比
7. 接入 Grafana 做检索与回答质量看板
8. 增加知识时效性管理和增量更新能力

---

## 11. 结论

当前项目已经从基础版 RAG 演进为一套具备企业级雏形的混合检索增强问答系统。

它的核心不是单一模型效果，而是通过以下组合能力形成稳定的知识问答系统：

- Elasticsearch 做关键词召回
- Milvus 做向量召回
- Neo4j 做图谱关系检索
- 混合排序做统一决策
- token 预算控制做成本治理
- Prometheus 与评测接口做可观测和持续优化

从工程视角看，这已经是一套可继续扩展、可上线迭代、可逐步演进的企业级 RAG 技术方案。
