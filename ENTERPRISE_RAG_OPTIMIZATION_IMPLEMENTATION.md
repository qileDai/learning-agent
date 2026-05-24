# 企业级 RAG 优化实现文档

## 1. 目标

本次优化的目标不是简单继续堆功能，而是围绕企业级知识库常见的三个核心问题进行增强：

- 提升召回率
- 提升回答准确率
- 降低大模型 token 成本

当前项目原本已经具备轻量 Graph RAG 能力，本次在此基础上继续增强为更接近企业级检索链路的方案：

**查询扩展 + 图谱增强 + 词法召回 + 向量召回 + 多路融合排序 + 去重与多样性控制 + token 预算裁剪 + 检索可观测性**

---

## 2. 本次新增与改造概览

### 2.1 查询扩展

新增文件：
- `backend/app/rag/retrieval_optimizer.py`

实现内容：
- 基于图谱中的概念和别名，对用户问题做轻量 query expansion
- 自动识别问题中命中的概念
- 自动扩展常见别名和同义表达
- 输出 `route_subjects`，作为后续检索的学科路由参考

价值：
- 提升同义问法、简称问法、概念别名问法下的召回率
- 让系统具备企业级知识库里常见的“术语归一化”能力

---

### 2.2 词法召回

新增文件：
- `backend/app/rag/retrieval_optimizer.py`

改造文件：
- `backend/app/rag/hybrid_retriever.py`
- `backend/app/rag/vector_store.py`

实现内容：
- 在原有向量召回基础上，增加轻量 lexical retrieval
- 对文档内容、章节、摘要、概念、别名进行 token 化
- 使用 token overlap 和覆盖率构造轻量词法评分
- 从本地索引文档中构造 lexical top candidates

价值：
- 解决纯向量召回对关键词、术语、公式、专有名词不稳定的问题
- 企业级知识库中，大量问题都具有强关键词属性，词法召回是必要补充

---

### 2.3 多路融合排序

改造文件：
- `backend/app/rag/hybrid_retriever.py`

实现内容：
- 将以下多路信号统一融合：
  - vector score
  - lexical score
  - graph source score
  - route subject boost
- 为每个候选文档计算 `rank_score`
- 记录 `retrieval_debug`，保留不同召回通道的分值构成

价值：
- 不再依赖单一路径召回
- 企业级知识库检索的关键不是只“找到”，而是把最对的内容排到前面

---

### 2.4 去重与多样性控制

新增文件：
- `backend/app/rag/retrieval_optimizer.py`

改造文件：
- `backend/app/rag/hybrid_retriever.py`

实现内容：
- 增加文档签名去重，避免重复 chunk 重复进入最终候选
- 增加 `max_per_source`，限制单个 source 占据过多候选位
- 最终候选经过 diversify 处理后再输出

价值：
- 避免某一份文档因为切出很多相似 chunk 而淹没其他高质量来源
- 提升最终候选集的信息覆盖面和结果稳定性

---

### 2.5 Token 成本控制

新增文件：
- `backend/app/rag/retrieval_optimizer.py`

改造文件：
- `backend/app/config.py`
- `backend/app/graph/nodes.py`

实现内容：
- 增加 `estimate_tokens` 近似估算 token
- 增加 `truncate_by_budget`，对最终传给模型的 chunk 做预算裁剪
- 增加 `compress_lines`，对图谱上下文做压缩
- 新增配置项：
  - `retrieval_chunk_budget_tokens`
  - `retrieval_graph_budget_tokens`

当前策略：
- 普通知识片段进入生成阶段前先截断到预算范围
- 图谱上下文只保留预算内最重要的关系行
- 检索阶段返回给前端的候选内容也已做压缩，避免无意义长文本

价值：
- 直接降低每次问答进入 LLM 的 token 消耗
- 在保证证据可用的前提下控制推理成本
- 更适合企业级多轮问答和高频调用场景

---

### 2.6 检索可观测性

改造文件：
- `backend/app/graph/state.py`
- `backend/app/graph/nodes.py`
- `backend/app/api/routes.py`

实现内容：
- 新增 `RetrievalSummary`
- 把检索链路中的关键统计信息放进状态与 API 返回结果
- 当前输出包括：
  - `query_expansions`
  - `route_subjects`
  - `graph_documents`
  - `vector_candidates`
  - `lexical_candidates`
  - `final_candidates`
  - `max_per_source`
  - `chunk_budget_tokens`
  - `graph_budget_tokens`

价值：
- 便于观察召回链路是否生效
- 便于后续做离线评估、线上调优和排查误召回问题
- 企业级系统不能只有结果，还要能解释“为什么召回成这样”

---

## 3. 代码实现说明

### 3.1 配置增强

文件：
- `backend/app/config.py`

新增配置：
- `retrieval_vector_k`
- `retrieval_lexical_k`
- `retrieval_final_k`
- `retrieval_max_per_source`
- `retrieval_chunk_budget_tokens`
- `retrieval_graph_budget_tokens`

作用：
- 控制召回深度、最终候选规模、单文档上限和 token 预算

---

### 3.2 索引读取增强

文件：
- `backend/app/rag/vector_store.py`

新增能力：
- `load_index_documents()`

作用：
- 从已经构建好的本地索引中直接取出 `Document` 列表
- 为 lexical retrieval 提供数据基础
- 避免重复解析源文件，提高检索阶段效率

---

### 3.3 检索优化器

文件：
- `backend/app/rag/retrieval_optimizer.py`

新增能力：
- `text_tokens()`
- `estimate_tokens()`
- `truncate_by_budget()`
- `compress_lines()`
- `expand_query()`
- `lexical_score()`
- `diversify_documents()`

职责说明：
- 统一承载查询扩展、词法评分、token 预算和候选去重逻辑
- 使检索优化从业务流程代码中拆分出来，便于扩展和维护

---

### 3.4 混合检索增强

文件：
- `backend/app/rag/hybrid_retriever.py`

升级前：
- 只有 graph boost + vector search

升级后：
- 查询扩展
- 图谱检索
- 多次向量召回（原问题 + 扩展问题）
- 词法召回
- route subject boost
- graph source boost
- 多路融合排序
- 去重与多样性控制
- 输出 `retrieval_summary`

这一步是本次最核心的企业级增强点。

---

### 3.5 生成阶段成本控制

文件：
- `backend/app/graph/nodes.py`

升级内容：
- 检索后对候选片段进行预算裁剪
- 图谱上下文进行压缩
- 生成 prompt 时只保留 token 预算内的证据
- 返回信息里增加 query expansion 和混合召回提示

效果：
- 模型看到的上下文更干净
- 不再把无上限长文本直接塞进 prompt
- 兼顾效果与成本

---

### 3.6 API 输出增强

文件：
- `backend/app/api/routes.py`

新增返回字段：
- `retrieval_summary`

出现在以下接口中：
- `POST /api/chat/start`
- `POST /api/chat/resume`
- `GET /api/chat/state/{thread_id}`

这样前端或外部系统可以直接观察：
- 检索是否命中 query expansion
- 是否走了 lexical 候选
- 最终保留了多少候选
- 当前 token 预算是多少

---

## 4. 本次优化如何提升企业级能力

### 4.1 提升召回率

本次主要通过以下手段提升召回率：
- query expansion
- graph concept alias 扩展
- lexical retrieval
- expanded query 再次向量召回
- route subject boost

适用场景：
- 用户问法不标准
- 使用简称、别名、缩写
- 强术语问题
- 概念关系问题

---

### 4.2 提升准确率

本次主要通过以下手段提升准确率：
- 图谱来源分数参与排序
- 词法得分参与排序
- 多路融合 rank score
- 单 source 候选上限
- 候选去重和多样性控制
- 相关性过滤继续保留

适用场景：
- 避免“相似但不相关”的结果排前面
- 避免单一文档大量近似 chunk 污染结果
- 提高前排候选的稳定性

---

### 4.3 控制 token 成本

本次主要通过以下手段降低成本：
- chunk token 预算
- graph token 预算
- 图谱按行压缩
- 仅传入预算内证据给模型

适用场景：
- 高频问答
- 大模型计费敏感场景
- 企业级知识库长文档问答

---

## 5. 当前仍可继续升级的方向

虽然这次已经把系统往企业级推进了一步，但如果继续往生产级演进，建议后续再补：

### 5.1 BM25 / 倒排索引
- 当前 lexical retrieval 是轻量实现
- 如果知识规模继续扩大，建议升级为真正的 BM25 或搜索引擎方案

### 5.2 Reranker
- 目前已经有多路融合排序
- 后续可以再引入 reranker 做精排，进一步提升前排准确率

### 5.3 分层索引
- 当前还是 chunk 级主导
- 后续可升级为文档级、章节级、chunk 级三层索引

### 5.4 权限过滤
- 企业内部知识库必须支持 ACL、部门、租户和密级过滤

### 5.5 离线评估集
- 建议建立标准 query / gold doc / gold chunk 数据集
- 评估 Recall@K、Precision@K、MRR、NDCG 等指标

---

## 6. 验证结果

本次已完成以下校验：

- VS Code 诊断无报错
- 后端代码通过：
  - `python -m compileall repo/backend/app`

说明：
- 当前检索优化与 token 预算逻辑已具备静态可用性
- 如需进一步验证效果，建议在真实知识数据上做对比测试：
  - 优化前 vs 优化后命中率
  - 优化前 vs 优化后前 3 条相关性
  - 平均 prompt token 下降幅度

---

## 7. 一句话总结

本次优化把原有轻量 Graph RAG 从“可用版本”推进到了更接近企业级检索架构的形态：

**通过查询扩展、词法召回、图谱增强、多路融合排序、去重多样性控制和 token 预算压缩，同时提升了召回率、回答准确率和成本控制能力。**
