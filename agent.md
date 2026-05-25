# RAG 面试回答稿

---

## 目录

- [面试前：简历写法](#零面试前的准备--简历写法)
- [第一层：一句话说清楚（30秒）](#一第一层一句话说清楚30-秒-pitch)
- [第二层：分层讲架构（2分钟完整版）](#二第二层分层讲架构2-分钟完整版)
- [第三层：深挖技术细节（面试官追问集锦）](#三第三层深挖技术细节面试官追问集锦)
- [收尾：最后一分钟总结](#四收尾最后一分钟总结)
- [附录A：架构设计图](#附录-a架构设计图)
- [附录B：元数据领域迁移（交易Agent为例）](#附录-b元数据领域迁移以交易-agent-为例)

---

## 零、面试前的准备 — 简历写法

### 项目经验栏（推荐版，3-4 行）

> **企业级混合检索 Graph RAG 知识库问答系统** | 后端负责人
>
> - 设计并实现了一套完整的企业级 RAG 体系，涵盖离线入库（PDF/Word/Markdown 解析、元数据增强、多索引构建）和在线问答（查询扩展、混合检索、融合排序、生成校验）全链路
> - 构建了**向量（Milvus）+ 关键词（Elasticsearch）+ 图谱（Neo4j）三路混合检索架构**，通过 8 维融合排序（语义、词法、图谱路由、覆盖度、意图、共识、RRF）和多样性控制，解决企业知识库中术语别名漏召回、概念关系难表达等问题
> - 引入语义缓存、Token 预算控制（chunk 260 / graph 120 tokens）、Grounding 校验等策略，在提升回答可信度的同时控制 LLM 调用成本
> - 对接 Prometheus 指标监控与离线评测体系（Hit@K / MRR），全链路关键组件（ES / Milvus / Neo4j）均设计降级回退，保证系统高可用

### 技能栏关键词

`RAG` `Graph RAG` `混合检索` `Milvus` `Elasticsearch` `Neo4j` `LangChain` `LangGraph` `向量检索` `RRF` `查询扩展` `语义缓存` `Token 预算` `Grounding` `Prometheus`

### 项目描述栏（详细版，作品集用）

> **企业级混合检索 Graph RAG 知识库问答系统**
>
> 基于 LangChain + LangGraph 构建的面向企业知识库场景的智能问答系统。区别于基础版"向量检索 + LLM"，本系统实现了**查询扩展 → 图谱先行 → 向量+关键词双路混合召回 → 8 维融合排序 → 多样性控制 → Token 预算压缩 → Grounding 校验 → Prometheus 监控**的完整工程链路。
>
> **核心技术栈**：Python / FastAPI / LangChain / LangGraph / OpenAI Embeddings / Milvus / Elasticsearch / Neo4j / Prometheus
>
> **主要工作**：
> 1. **离线入库管道**：统一解析 PDF/DOCX/Markdown/TXT 四类知识文件，注入 subject/chapter/summary/concepts/aliases/relations 结构化元数据，同步构建向量索引（本地 numpy + Milvus）、关键词索引（Elasticsearch）、知识图谱索引（JSON + Neo4j Cypher）
> 2. **在线检索链路**：设计查询扩展（基于图谱别名自动补充同义词）和问题路由（按复杂度分 simple/complex/analysis 三级动态调整召回深度），语义缓存命中时可跳过全量检索
> 3. **混合检索与融合排序**：图谱先行提供概念关系与来源优先级，向量（Milvus 余弦相似度）+ 关键词（ES multi_match 字段加权）双路并行召回，最终以 vector / lexical / graph / route / coverage / intent / consensus / RRF 8 个维度做融合排序
> 4. **生成质量保障**：Token 预算控制（普通 chunk 260 tokens + 图谱 120 tokens 分别压缩），Grounding 校验（计算答案与证据 token 重叠率判断可信度），结合检索摘要与回答质量指标输出
> 5. **可观测与降级**：Prometheus 记录请求量/延迟/缓存命中/grounding 分数，离线评测输出 Hit@K / MRR；ES / Milvus / Neo4j / prometheus_client 均设计无感降级回退

---

## 一、第一层：一句话说清楚（30 秒 Pitch）

> 面试官："介绍一下你这个 RAG 项目。"

我做的是一套面向企业知识库场景的混合检索 Graph RAG 系统。跟基础版"向量检索 + 大模型回答"不一样，我把它升级成了**查询扩展 → 图谱先行 → 向量+关键词双路召回 → 8 维融合排序 → 多样性控制 → Token 压缩 → Grounding 校验 → 指标监控**的完整链路。核心解决四个问题：**召得全、排得准、答得稳、成本可控**。

---

## 二、第二层：分层讲架构（2 分钟完整版）

> 面试官："具体怎么做的？展开讲讲。"

### 2.1 一句话概括整体架构

整套系统分四层：**接入层（API）→ 编排层（LangGraph）→ 检索层（三路混合检索 + 8 维排序）→ 数据层（多格式解析 + 三路建索引）**，外加横切的监控和评测。

### 2.2 离线入库（知识怎么进去的）

第一步，**统一接入**——PDF、Word、Markdown、TXT 四种格式全部解析成统一文档对象。

第二步，**元数据增强**——给每份资料注入 subject、chapter、summary、concepts、aliases、relations 六个结构化字段。这一步是整个系统的基础，后续的路由、排序、图谱构建全靠这些标签。

第三步，**切块向量化**——RecursiveCharacterTextSplitter 按 800 字切块、120 字重叠，text-embedding-3-small 生成向量。

第四步，**三路同步建索引**——同一个 chunk 同时写入三个地方：
- 向量索引：本地 numpy + Milvus
- 关键词索引：Elasticsearch
- 图谱索引：JSON + Neo4j

本质上就是把"一堆文件"变成"三种可检索的知识结构"。

### 2.3 在线问答（用户提问后发生了什么）

不是一次 topK 向量检索就完事，而是走一个完整的 10 步决策链路：

| 步骤 | 做什么 | 解决什么问题 |
|------|--------|-------------|
| ① 查询扩展 + 路由 | 补别名、判复杂度，定召回深度 | 问法不标准导致漏召回 |
| ② 语义缓存 | 查是否已有相似问题结果 | 高频问题免检索 |
| ③ 图谱先行 | 拿概念关系 + 来源优先级 | 给后续召回提供元信息 |
| ④ 向量召回 | Milvus 语义相似度匹配 | 字面不同但意思接近的问题 |
| ⑤ 关键词召回 | ES multi_match 字段加权 | 专有名词精确匹配 |
| ⑥ 8 维融合排序 | 多角度综合打分 | 不偏信任何一路 |
| ⑦ 多样性控制 | 每份资料限 2 条 + 去重 | 避免单一来源偏差 |
| ⑧ Token 预算压缩 | chunk 260 / graph 120 tokens 截断 | 控制调用成本 |
| ⑨ LLM 生成 | 基于证据拼接 prompt | 严格基于检索结果回答 |
| ⑩ Grounding 校验 | 答案 vs 证据 token 重叠率 | 防止幻觉 |

### 2.4 跟基础版 RAG 的核心差异

| | 基础版 RAG | 我的版本 |
|---|-----------|---------|
| 检索 | 一次向量 topK | 三路混合检索 + 图谱先行 |
| 排序 | 按余弦距离 | 8 维融合排序 + 多样性控制 |
| 成本 | 无控制 | 语义缓存 + Token 预算 |
| 质量 | 无校验 | Grounding 校验 |
| 运维 | 无 | Prometheus + 评测体系 + 全链路降级 |

基础版是 **"切块→向量→topK→prompt→回答"**，我做的是 **"多索引构建→多路混合检索→多因子融合排序→成本控制→质量校验→指标监控"**，是一个完整的工程体系。

---

## 三、第三层：深挖技术细节（面试官追问集锦）

> 这一层以一问一答形式组织，回答按"先说结论 → 再讲原理 → 最后举个例子"的结构。

---

### Q1：为什么要三路混合检索？一路不够吗？

**先说结论**：一路只解决一个问题，三路互补。

向量检索解决**语义相似**——"怎么算加速度"能召回"F=ma 的推导"。但它对专有名词不稳定，"元素周期表"可能被映射到不相关的语义空间。

关键词检索解决**精确匹配**——术语、标题这些向量不擅长的事。

图谱检索解决**关系推理**——"导数依赖极限"这种结构化关联，向量算不出来。

三路合在一起不但各自取长补短，在融合排序阶段还能**互相印证**——一个 chunk 同时被三路命中，consensus 分就高，排序更靠前。

**举个例子**：用户问"牛二是什么？"，向量能找到讲牛顿定律的段落，关键词能精确命中"牛顿第二定律"这个标题，图谱能告诉系统"牛二"是"牛顿第二定律"的别名——三路协同才能做到不遗漏。

---

### Q2：8 维融合排序是什么意思？具体怎么算的？

**先说结论**：不从单一维度打分，而是从 8 个不同角度分别打分，最后加起来——让多路证据互相印证。

**8 个维度是**：

| 维度 | 衡量什么 | 来源 |
|------|---------|------|
| vector | 语义有多接近 | Milvus 余弦距离 |
| lexical | 关键词有多少重合 | ES BM25 / 本地词法 |
| graph | 来源在图谱里优先级多高 | 图谱检索 source_scores |
| route | 学科是否匹配路由目标 | subject 命中 + 概念重叠 |
| coverage | 内容对问题的覆盖程度 | query tokens vs doc tokens 覆盖率 |
| intent | 问题复杂度与文档结构是否匹配 | 问题分类 simple/complex/analysis |
| consensus | 同时被几路召回命中 | 多路命中计数 × 0.06 |
| rrf | 在各路内的排名位置 | 倒数排名融合 × 10 |

**计算公式**：
```
最终分 = (1.3 - 余弦距离)           ← vector
       + ES BM25 归一化分            ← lexical
       + source_scores 图谱分        ← graph
       + _route_boost               ← route
       + score_document_coverage    ← coverage
       + _intent_boost              ← intent
       + 命中路数 × 0.06            ← consensus
       + RRF × 10                   ← rrf
```

**为什么需要 8 维？** 因为只靠语义相似度有盲区——一个写了"牛顿第二定律"但没有真正解释内涵的 chunk，可能比措辞不同但讲得更清楚的 chunk 语义分更高。多维度交叉验证才能把真正好的排在前面。

---

### Q3：为什么图谱要先跑，不能三个通道并行？

**先说结论**：有数据依赖。图谱的输出是向量和关键词需要用到的元信息。

图谱检索输出的三个结果：
- `matched_concepts`：命中了哪些概念
- `source_scores`：每个来源的优先级分
- `route_subjects`：建议搜索哪些学科

这些信息后续会被 `_route_boost` 用来给每个 chunk 算路由加分，被 `_build_lexical_candidates` 用来做 subject 过滤。如果三个通道完全并行，这些信息就丢了。

**实际编排是**：图谱先行 → 拿到元信息 → 向量和关键词并行执行 → 融合排序汇总。这是有依赖的编排，不是无脑并行。

---

### Q4：多样性控制是什么？为什么需要？

**先说结论**：同一份资料最多保留 2 个 chunk，防止一本教材霸占全部候选。

做法：先取排序后的前 24 个候选，然后按 source 计数——同一份资料（如同一本教材）最多保留 2 个 chunk，同时做内容签名去重。

**为什么需要**：如果不做多样性控制，排序完直接取 topK，可能出现"一本教材的前 8 个段落全在候选里"。用户问跨学科问题，结果回答全部基于同一本书，视角单一、容易片面。

---

### Q5：语义缓存是怎么做的？为什么缓存检索结果而不是最终答案？

**先说结论**：缓存检索结果而非答案，因为答案受后续步骤影响，缓存风险大。

缓存策略：每次检索完成后把 question、normalized_question、route_type、documents、graph_result 存下来。下次提问时先归一化问题，转词袋，找 route_type 一致且 token 重叠率 > 0.84 的条目。命中直接返回，跳过图谱检索、向量召回、关键词召回三步。

**为什么缓存检索结果而不是答案**：最终答案还受用户选的证据、prompt 调整、知识库更新等影响，直接缓存答案可能返回过时内容。检索结果相对稳定，缓存更安全。

缓存上限 200 条，超过按时间淘汰最旧的。

---

### Q6：Token 预算控制 260 和 120 怎么定的？

**先说结论**：工程上的合理估计，不是精确理论值。

普通 chunk 设 260 tokens（约 180-200 个中文字），保证一段完整概念不被截断——"牛顿第二定律是...（后面没了）"这种截断不能出现。

图谱上下文设 120 tokens（约 80-100 个中文字），因为关系信息密度极高，一行就是一条独立信息，3-5 条关键关系 + 优先来源就够，多了反而是噪声。

**要调优的话**，应该在评测集上跑不同预算组合，找 grounding 分不再提升但 token 成本开始陡增的拐点。

---

### Q7：Grounding 校验是什么意思？怎么判断回答不可信？

**先说结论**：算答案中有多少词跟参考证据重叠——重叠高 = grounded，重叠低 = 可能幻觉。

**三步计算**：

```
grounding_score = 答案 vs 证据 token 重叠率 × 0.72    ← 主力
                + 答案 vs 问题 token 重叠率 × 0.28    ← 防止跑题
                + 概念命中加分                         ← 每个概念 +0.04
```

分数 ≥ 0.18 → `grounded: true`；低于 0.18 → `grounded: false`。

**定位**：轻量辅助判断，不是硬拦截。可能误判——答案用自己的话重述了证据，但 token 不重叠。所以阈值设得低，先拦住明显跑偏的回答。更严谨的可以做 NLI 模型判断，但成本高。

---

### Q8：所有关键组件都有降级回退，如果一个都没有能跑吗？

**先说结论**：能。裸跑也完整工作。

| 组件不可用 | 降级方案 |
|-----------|---------|
| Milvus | 本地 numpy 矩阵乘余弦距离 |
| Elasticsearch | 本地 `lexical_score` 词法打分 |
| Neo4j | 本地 JSON 图谱文件遍历匹配 |
| prometheus_client | 所有指标降级为 no-op |

这就是说：拿这套代码在任何装好 Python 的机器上直接跑，不需要先搭基础设施。等数据量和并发上来，再把 Milvus、ES、Neo4j 逐个接上，不用改核心代码。

---

### Q9：8 维排序的权重系数怎么调的？

**先说结论**：不是拍脑袋，各维度原始值本身就处于不同量级，直接累加已经形成了隐含权重。

- vector 分在 0-1 范围
- lexical 分在 0-4 范围
- graph 分在 0-1 范围
- rrf 分通常很小，所以乘 10 拉到和其他维度可比

当前策略是各维度保持原始量级直接累加。如果后续要精细调参，在评测集上对每个维度跑网格搜索加系数，找 Hit@K 最优的组合。

---

### Q10：如果知识库有 10 万份文档，扛得住吗？

**先说结论**：架构已经考虑了规模化路径，核心链路不会成为瓶颈。

- Milvus 替代本地 numpy → 分布式向量检索
- ES 替代本地词法 → 分布式关键词检索
- 向量和关键词独立并行，不会互相阻塞
- 语义缓存降低高频查询开销
- 图谱离线构建，大规模下可改增量更新

---

### Q11：离线入库时三路建索引，一个 chunk 同时进三个库，写入性能怎么保证？

**先说结论**：离线入库本身是一次性的批量操作，不以写入延迟为约束。

一次性批量建索引，三路写入可以并发——向量索引和 ES 索引在 `vector_store.ingest_documents()` 里同步完成，图谱索引在 `graph_store.build_graph_index()` 里独立执行。入库耗时取决于文件总量，但这对在线链路无影响。

---

### Q12：你提到元数据有 subject、chapter、concepts 等字段，这些数据怎么来的？

**先说结论**：当前是人工预定义在 `knowledge_metadata.json` 里，是系统运行的基础配置。

每个知识文件对应一条 source 条目，包含文件名、学科、章节、摘要、概念列表、别名列表。concepts 和 relations 也在同一个文件里集中管理。查询扩展读的是 concepts 里的 aliases，图谱构建读的是 sources + concepts + relations。

**后续可以自动化**：用 LLM 对新增文档自动提取摘要、概念、关系，人工审核后入库，减少手工维护成本。

---

## 四、收尾：最后一分钟总结

> 面试官："还有什么要补充的吗？"

总结一下，我做的这套系统不是"接个 API 调个 prompt"的 Demo，而是从**知识接入 → 结构化建模 → 多索引构建 → 混合检索 → 融合排序 → 成本控制 → 质量校验 → 指标监控**的完整工程闭环。

核心价值不是"能回答"，而是能在企业场景里做到：**语义不丢、术语不漏、关系不丢、成本可控、质量可验证、系统可运营**。

如果后续继续迭代，优先级做三件事：一是评测集建设，用真实业务问题驱动排序参数调优；二是增量索引更新，避免每次全量重建；三是检索路由的 LLM 化，用模型做路由决策而不是关键词正则。

---

## 附录 A：架构设计图

### A.1 系统分层架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         接入层 (API)                              │
│  /api/chat  /api/ingest  /api/eval  /api/metrics                 │
├──────────────────────────────────────────────────────────────────┤
│                         编排层 (LangGraph)                        │
│  state.py       → AgentState（统一状态定义）                       │
│  workflow.py    → 状态图编排                                     │
│  nodes.py       → 检索节点 / 生成节点 / 寒暄节点                   │
├──────────────────────────────────────────────────────────────────┤
│                         检索层 (RAG Core)                         │
│  hybrid_retriever.py  → 三路混合召回编排，8维融合排序              │
│  retrieval_optimizer.py → 查询扩展/路由/缓存/预算/校验             │
├──────────┬──────────────────┬────────────────────────────────────┤
│  向量通道  │   关键词通道       │   图谱通道                         │
│  Milvus   │   Elasticsearch   │   Neo4j / JSON                     │
│  numpy ↓  │   lexical_score ↓ │   graph_store ↓                   │
├──────────┴──────────────────┴────────────────────────────────────┤
│                         数据层                                    │
│  loaders.py          → PDF/DOCX/MD 解析                          │
│  metadata_registry.py → 结构化元数据管理                          │
│  vector_store.py     → 切块/向量化/多后端入库                     │
│  ingest.py           → 离线入库总入口                             │
├──────────────────────────────────────────────────────────────────┤
│                         横切层                                    │
│  evaluation.py     → 检索评测 + 回答评测                          │
│  observability.py  → Prometheus 指标 + 全链路降级                  │
│  config.py         → 统一配置管理                                 │
└──────────────────────────────────────────────────────────────────┘
```

### A.2 离线入库流程

```
knowledge_metadata.json          知识文件目录
        │                            │
        ▼                            ▼
  metadata_registry              loaders.py
        │                            │
        └──────────┬─────────────────┘
                   ▼
            ingest_knowledge_base()         ← ingest.py
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
  vector_store   elastic    graph_store
  .ingest_       _store     .build_graph
  documents()    .ingest_   _index()
        │        documents()
        │          │          │
        ▼          ▼          ▼
   ┌────────┐  ┌────────┐  ┌────────────────┐
   │ 切块    │  │ 写 ES  │  │ 构建 Source 节点 │
   │ 向量化  │  │ 索引   │  │ 构建 Concept 节点│
   ├────────┤  └────────┘  │ 构建 Relation 边 │
   │写 numpy │              ├────────────────┤
   │写 Milvus│              │ 写 JSON 图谱    │
   └────────┘              │ 同步 Neo4j      │
                           └────────────────┘
最终产物：
  data/vector_index/   ← 本地向量 + 文档
  data/graph_index/    ← JSON 图谱
  Milvus collection    ← 向量库
  ES index             ← 关键词索引
  Neo4j KnowledgeGraph ← 图数据库
```

### A.3 在线问答流程

```
用户提问 "牛二是什么？"
        │
        ▼
  hybrid_retrieve(question)
        │
        ├─① expand_query → 别名扩展 + 路由分类
        │     "牛二" → "牛顿第二定律" "F=ma"  |  route_type = "simple"
        │
        ├─② get_cached_retrieval → 语义缓存命中？→ 直接返回 [跳过③④⑤]
        │
        ├─③ search_graph → 图谱先行
        │     Neo4j: MATCH concept WHERE name CONTAINS ...
        │     返回 matched_concepts, source_scores
        │
        ├─④ similarity_search → 向量召回（原问 + 扩展问各查一次）
        │     Milvus / numpy 回退
        │
        ├─⑤ _build_lexical_candidates → 关键词召回
        │     ES multi_match(concepts^4, page_content^3, ...) / 本地回退
        │
        ├─⑥ 8维融合排序 → vector + lexical + graph + route
        │                    + coverage + intent + consensus + rrf
        │
        ├─⑦ diversify_documents → 每 source 最多 2 条 + 去重
        │
        └─⑧ save_cached_retrieval → 写回缓存
        │
        ▼
  generate_answer_node
        │
        ├─⑨ truncate_by_budget(260) + compress_lines(120)
        ├─⑩ ChatOpenAI 生成
        └─⑩ validate_answer_grounding
              score = 答案vs证据重叠率×0.72 + 答案vs问题重叠率×0.28 + 概念加分
              ≥ 0.18 → grounded
        │
        ▼
  返回 final_answer + retrieval_summary + answer_validation
        │
        ▼
  observability.py → Prometheus 指标记录
```

### A.4 模块调用关系图

```
                        ┌─────────────┐
                        │  main.py    │
                        │  FastAPI    │
                        └──────┬──────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐   ┌────────────┐   ┌──────────────┐
        │ /ingest  │   │  /chat      │   │ /eval        │
        │          │   │             │   │ /metrics     │
        └────┬─────┘   └──────┬──────┘   └──────┬───────┘
             │                │                  │
             ▼                ▼                  │
    ┌────────────┐   ┌──────────────┐   ┌───────┴───────┐
    │ ingest.py  │   │ workflow.py  │   │ evaluation.py │
    │            │   │              │   │ observability │
    └──┬──┬──┬───┘   └──────┬───────┘   │   .py         │
       │  │  │               │           └───────────────┘
       │  │  │               ▼
       │  │  │      ┌────────────────┐
       │  │  │      │  nodes.py      │
       │  │  │      │  retrieve_node─┼──────────────┐
       │  │  │      │  generate_node │              │
       │  │  │      └────────────────┘              │
       │  │  │                                      ▼
       │  │  │                         ┌─────────────────────┐
       │  │  │                         │ hybrid_retriever.py │
       │  │  │                         │  hybrid_retrieve()  │
       │  │  │                         └────────┬────────────┘
       │  │  │                                  │
       │  │  │          ┌───────────────────────┼───────────────────────┐
       │  │  │          │                       │                       │
       │  │  │          ▼                       ▼                       ▼
       │  │  │  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
       │  │  │  │ retrieval_   │    │ graph_store.py   │    │ vector_store.py  │
       │  │  │  │ optimizer.py │    │ search_graph()   │    │ similarity_      │
       │  │  │  │              │    │                  │    │ search_with_     │
       │  │  │  │ expand_query │    │ ┌──────────────┐ │    │ scores()         │
       │  │  │  │ classify_    │    │ │ Neo4j Cypher │ │    │                  │
       │  │  │  │ query        │    │ │ 检索         │ │    │ ┌──────────────┐ │
       │  │  │  │ get/set      │    │ ├──────────────┤ │    │ │ Milvus       │ │
       │  │  │  │ cache        │    │ │ JSON 遍历    │ │    │ │ 检索         │ │
       │  │  │  │ lexical_     │    │ │ 回退         │ │    │ ├──────────────┤ │
       │  │  │  │ score        │    │ └──────────────┘ │    │ │ numpy 回退   │ │
       │  │  │  │ rrf/diversify│    └──────────────────┘    │ └──────────────┘ │
       │  │  │  │ grounding    │                            └──────────────────┘
       │  │  │  └──────────────┘
       │  │  │
       │  │  │  ┌──────────────────┐
       │  │  └──│ elastic_store.py │
       │  │     │ lexical_search() │
       │  │     └──────────────────┘
       │  │
       │  │     ┌──────────────────┐
       │  └─────│ vector_store.py  │
       │        │ ingest_documents │
       │        │   ├─ numpy       │
       │        │   ├─ Milvus      │
       │        │   └─ ES同步      │
       │        └──────────────────┘
       │
       │        ┌──────────────────┐
       └────────│ graph_store.py   │
                │ build_graph_index│
                │   ├─ JSON图谱    │
                │   └─ Neo4j同步   │
                └──────────────────┘
                      │
                      ▼
                ┌──────────────────┐
                │ metadata_        │
                │ registry.py      │
                │ (knowledge_      │
                │  metadata.json)  │
                └──────────────────┘
                      │
                      ▼
                ┌──────────────────┐
                │ loaders.py       │
                │ PDF/DOCX/MD/TXT  │
                └──────────────────┘
```

### A.5 核心设计决策与取舍

| 设计决策 | 选择 | 为什么 | 代价 |
|---------|------|--------|------|
| 检索顺序 | 图谱先行 | 提供 source_scores，后续通道可复用 | 多一次图谱查询延迟 |
| 缓存粒度 | 检索结果而非答案 | 答案受后续步骤影响，缓存风险高 | 仍需走生成阶段 |
| 排序维度 | 8维而非单维 | 单维无法同时表达语义、术语、关系、覆盖度 | 调参复杂度高 |
| 图谱后端 | Neo4j + JSON 双轨 | Neo4j 做图遍历，JSON 做降级 | 维护两套检索逻辑 |
| 向量后端 | Milvus + numpy 双轨 | Milvus 做企业级检索，numpy 做裸机可用 | 初始化需同步两份 |
| 路由方式 | 关键词正则 | 零延迟，不消耗 token | 覆盖度有限 |
| 查询扩展 | 基于图谱别名 | 无需 LLM 调用，稳定可控 | 依赖 metadata 质量 |
| 预算控制 | 硬截断 260/120 | 简单可靠 | 可能丢失段尾关键信息 |

---

## 附录 B：元数据领域迁移（以交易 Agent 为例）

> 面试官："你这套 RAG 如果换到金融交易场景，元数据怎么改？"

**核心思路**：元数据跟着领域语义走，不跟着技术栈走。技术架构全部复用，改的只是 schema 和路由决策逻辑。

### 领域改造对照

| 维度 | 教育知识库 | 交易 Agent |
|------|----------|-----------|
| 顶层分类 | `subject` | `asset_class` + `market` |
| 二级定位 | `chapter` | `sector`（行业/板块） |
| 内容摘要 | `summary` | `signal_type` + `data_origin` |
| 核心标签 | `concepts` | `tickers` + `indicators` + `strategies` |
| 关系类型 | 包含、前置依赖 | 产业链、宏观传导、策略依赖、季节效应 |
| 特有字段 | `grade`, `difficulty` | `time_sensitivity`, `confidence`, `frequency`, `sentiment` |

### 交易 Agent metadata 示例

```json
{
  "sources": {
    "report_maotai_2025Q1.pdf": {
      "asset_class": "stock",
      "market": "A股",
      "sector": "白酒",
      "tickers": ["600519"],
      "report_type": "财报",
      "publish_date": "2025-04-28",
      "time_sensitivity": "quarterly",
      "data_origin": "公司公告",
      "confidence": "high",
      "summary": "贵州茅台2025年一季度营收同比增长12%，净利润增长15%，直销占比提升至55%。",
      "indicators": ["营收增速", "净利润增速", "直销占比", "毛利率"],
      "strategies": ["价值投资", "消费龙头"],
      "aliases": ["贵州茅台", "茅台", "600519.SH", "600519"]
    },
    "news_fed_rate_may25.txt": {
      "asset_class": "macro",
      "market": "美国",
      "event_type": "央行政策",
      "publish_date": "2025-05-15T02:00:00Z",
      "time_sensitivity": "intraday",
      "data_origin": "美联储官网",
      "sentiment": "hawkish",
      "summary": "美联储5月维持利率不变，点阵图暗示年内仍有加息空间。"
    }
  },
  "relations": [
    { "source": "贵州茅台", "relation": "属于", "target": "白酒行业", "relation_type": "产业链" },
    { "source": "CPI", "relation": "影响", "target": "消费板块", "relation_type": "宏观传导", "direction": "正向", "lag": "1-3月" },
    { "source": "美联储加息", "relation": "利空", "target": "成长股", "relation_type": "宏观传导", "direction": "负向" },
    { "source": "动量策略", "relation": "依赖指标", "target": "20日收益率", "relation_type": "策略依赖" }
  ]
}
```

### 检索链路的关键差异

| 步骤 | 教育 | 交易 |
|------|------|------|
| 查询扩展 | 补概念别名 | 补 ticker ↔ 名称映射、指标别名 |
| 路由 | 按 subject | 按 asset_class + market |
| 图谱检索 | 前置依赖、包含关系 | 产业链上下游、宏观传导路径 |
| 排序加分 | chapter/summary 命中 | **时效性加分**（今天的 > 三个月前的）、**置信度加分**（官方 > 社交媒体） |
| Grounding | 与参考证据对齐 | 与原始数据源对齐（财报数字不能错） |

### 扩展到其他领域的通用公式

```
metadata = 路由字段（大类分类）
         + 定位字段（细粒度定位）
         + 时效/置信字段（质量控制）
         + 业务标签（概念/指标/标的）
         + 别名映射（多形态输入）
         + 关系网络（领域特有的结构关系）
```

| 领域 | 核心分类 | 特有字段 | 关系类型 |
|------|---------|---------|---------|
| 教育 | subject | grade, difficulty | 包含、前置依赖 |
| 交易 | asset_class, market | tickers, time_sensitivity, confidence | 产业链、宏观传导、策略依赖 |
| 医疗 | department, disease_type | drug_names, symptoms, contraindications | 适应症、禁忌、药物相互作用 |
| 法律 | law_type, jurisdiction | case_numbers, statute_ids, effective_date | 引用、推翻、修订、管辖 |
| 电商 | product_category, platform | sku, brand, price_range | 竞品、互补品、替代关系 |