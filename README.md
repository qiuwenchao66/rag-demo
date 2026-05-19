# 企业知识库问答系统

一个基于 RAG（Retrieval-Augmented Generation）的企业知识库问答项目，支持多知识库隔离、文档上传解析、多轮对话、按需检索，以及 `BM25 + FAISS + Cross-Encoder Rerank` 的混合检索链路。

项目当前采用：
- `Streamlit` 构建前端交互
- `Sentence Transformers` 生成本地向量
- `FAISS` 执行语义检索
- `BM25` 执行关键词检索
- `Cross-Encoder Rerank` 对候选结果重排
- `DeepSeek API` 负责问题改写、检索决策与答案生成

## 项目特性

- 支持 `PDF / DOCX / TXT` 文档上传与解析
- 支持多知识库隔离管理
- 支持本地向量索引构建与持久化
- 支持 `BM25 + 向量检索` 混合召回
- 支持 `Rerank` 重排序优化上下文质量
- 支持多轮对话与追问改写
- 支持按需检索，避免连续追问时重复检索
- 支持答案来源展示与上下文落盘
- 提供完整测试用例

## 系统架构

项目核心链路如下：

1. 上传文档
2. 文档解析与文本切分
3. 同时构建：
   - `FAISS` 向量索引
   - `BM25` 关键词索引
4. 用户提问后，系统判断是否需要重新检索
5. 如需检索，先执行问题改写，再走：
   - 向量检索
   - BM25 检索
   - 合并去重
   - Rerank 重排
6. 基于最终上下文调用大模型生成答案
7. 保存对话历史、来源信息和检索模式

详细架构文档见：

- [中文架构说明](docs/系统架构说明.md)
- [English Architecture Doc](docs/system-architecture.md)

## 项目结构

```text
rag-demo/
├─ app.py
├─ README.md
├─ requirements.txt
├─ docs/
│  ├─ system-architecture.md
│  └─ 系统架构说明.md
├─ src/
│  ├─ bm25db.py
│  ├─ config.py
│  ├─ knowledge_base.py
│  ├─ llm_service.py
│  ├─ loaders.py
│  ├─ qa_chain.py
│  ├─ reranker.py
│  ├─ retrieval.py
│  ├─ splitter.py
│  └─ vectordb.py
└─ tests/
```

## 核心模块说明

- `app.py`
  - Streamlit 页面入口
  - 负责文档上传、知识库构建、提问与聊天记录展示

- `src/loaders.py`
  - 加载 `PDF / DOCX / TXT` 文档内容

- `src/splitter.py`
  - 文本切分
  - 为每个 chunk 添加 `chunk_index` 和 `chunk_id`

- `src/vectordb.py`
  - 构建和加载 FAISS 向量索引

- `src/bm25db.py`
  - 构建和加载 BM25 索引

- `src/retrieval.py`
  - 执行混合检索与结果去重

- `src/reranker.py`
  - 使用 cross-encoder 对候选结果重排

- `src/qa_chain.py`
  - 串联检索决策、问题改写、混合检索、rerank、上下文构造与答案生成

- `src/llm_service.py`
  - 封装 DeepSeek API 调用逻辑

- `src/knowledge_base.py`
  - 管理知识库目录与聊天记录持久化

## 知识库目录结构

每个知识库会独立保存为一个目录：

```text
knowledge_bases/
  <knowledge_base_id>/
    metadata.json
    raw_docs/
    vector_store/
    bm25_store.json
    chat_history.json
```

这样可以做到：
- 知识库之间索引隔离
- 原始文档隔离
- 对话历史隔离

## 环境要求

- Python `3.10+`
- Windows / macOS / Linux

推荐使用虚拟环境运行。

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd rag-demo
```

### 2. 创建虚拟环境并安装依赖

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. 配置环境变量

复制示例配置：

Windows:

```bash
copy .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

然后在 `.env` 中填写你的 `DeepSeek API Key`。

### 4. 启动项目

```bash
streamlit run app.py
```

默认访问地址：

```text
http://localhost:8501
```

## 环境变量说明

| 变量名 | 说明 | 默认值 |
|---|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key | 空 |
| `DEEPSEEK_BASE_URL` | DeepSeek 接口地址 | `https://api.deepseek.com` |
| `CHAT_MODEL` | 对话模型名称 | `deepseek-v4-flash` |
| `EMBEDDING_MODEL` | 本地向量模型 | `BAAI/bge-small-zh-v1.5` |
| `RERANKER_MODEL` | 本地重排模型 | `BAAI/bge-reranker-base` |
| `EMBEDDING_BATCH_SIZE` | 向量化批大小 | `16` |
| `VECTOR_STORE_DIR` | 向量索引目录 | `data/vector_store` |
| `RAW_DOCS_DIR` | 原始文档目录 | `data/raw_docs` |
| `CHUNK_SIZE` | 文本切分大小 | `500` |
| `CHUNK_OVERLAP` | 文本切分重叠 | `100` |
| `TOP_K` | 最终返回上下文数 | `4` |
| `VECTOR_TOP_K` | 向量检索候选数 | `8` |
| `BM25_TOP_K` | BM25 检索候选数 | `8` |

## 使用流程

1. 创建一个新的知识库
2. 上传企业文档
3. 点击“构建知识库”
4. 等待系统完成：
   - 文档解析
   - 文本切分
   - 向量索引构建
   - BM25 索引构建
5. 在聊天框中提问
6. 查看回答与来源片段

## 检索策略说明

项目当前使用混合检索：

- `BM25`
  - 更适合匹配关键词、制度名称、编号、字段名

- `FAISS 向量检索`
  - 更适合匹配语义相关、改写表达和自然语言提问

- `Rerank`
  - 对 BM25 与向量召回的候选结果再做相关性排序
  - 提高最终喂给大模型的上下文质量

- `按需检索`
  - 对“继续解释”“总结一下”“展开说说”这类问题优先复用上一轮上下文
  - 减少重复检索和额外延迟

## 测试

运行全部测试：

```bash
pytest
```

当前项目已覆盖：
- 文档加载
- 文本切分
- 知识库目录管理
- BM25 索引
- 向量索引
- 混合检索
- Rerank 接口
- 问答链路
- 聊天记录持久化


