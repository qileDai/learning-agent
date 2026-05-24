# 教育类智能体 (Education Agent)

LangGraph 多智能体协同 · 轻量本地向量检索（numpy）· OpenAI 中转站兼容 · 每日 AI 推送 · React 前端

## 架构

```
用户提问 → [检索节点] 本地向量索引相似度搜索
         → [人工选择] LangGraph interrupt，前端勾选片段
         → [回答节点] 基于选中片段生成教育辅导回答
```

## 知识库（多样本）

| 类型 | 示例文件 |
|------|----------|
| Markdown | Python、历史、化学、地理、语文、微积分 |
| Word | 细胞结构、英语时态 |
| PDF | 二次方程、牛顿定律 |

## 快速开始

### 1. 后端

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt

copy .env.example .env   # 填入 OPENAI_API_KEY 与 OPENAI_API_BASE（中转站）
python scripts/create_static_samples.py   # 无需 pip，生成 PDF/Word 样本
python scripts/self_test_minimal.py       # 结构自测
python scripts/self_test.py               # 完整自测（需先 pip install）
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

首次使用在前端点击 **「重建向量索引」**，或：

```bash
curl -X POST http://127.0.0.1:8000/api/ingest -H "Content-Type: application/json" -d "{\"reset\": true}"
```

### 2. 前端

```bash
cd frontend
npm install
npm run dev
```

浏览器打开 http://localhost:5173

### Docker（前后端一体）

```bash
# 在项目根目录（含 Dockerfile 的目录）
docker build -t learning-agent .
docker run --rm -p 8000:8000 -p 22:22 \
  -e OPENAI_API_KEY=sk-xxx \
  -e OPENAI_API_BASE=https://你的中转地址/v1 \
  learning-agent
```

浏览器打开 http://localhost:8000（API 仍为 `/api/*`）。首次使用在前端点击 **「重建向量索引」**。

### 3. OpenAI 中转站

`.env` 示例：

```env
OPENAI_API_KEY=sk-xxx
OPENAI_API_BASE=https://你的中转地址/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

兼容所有 OpenAI API 格式的服务。

## API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/ingest` | 入库知识库 |
| POST | `/api/chat/start` | 开始对话（检索后等待选择） |
| POST | `/api/chat/resume` | 提交选中片段并生成回答 |
| GET | `/api/daily-push/latest` | 最新每日推送 |
| POST | `/api/daily-push/generate` | 手动生成今日安排 |

## 自测

```bash
cd backend
python scripts/self_test.py
```

验证：文档加载、图节点结构、PDF/Word 样本生成。

## 目录结构

```
agent/
├── backend/
│   ├── app/
│   │   ├── graph/      # LangGraph 工作流
│   │   ├── rag/        # 轻量向量索引 + 多格式加载
│   │   ├── scheduler/  # 每日推送
│   │   └── api/
│   ├── data/knowledge/
│   └── scripts/
└── frontend/
    └── src/
```
