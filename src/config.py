from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class AppConfig:
    deepseek_api_key: str
    deepseek_base_url: str
    chat_model: str
    embedding_model: str
    reranker_model: str
    vector_store_dir: Path
    raw_docs_dir: Path
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 4
    vector_top_k: int = 8
    bm25_top_k: int = 8
    embedding_batch_size: int = 16


def load_config() -> AppConfig:
    load_dotenv()
    return AppConfig(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        chat_model=os.getenv("CHAT_MODEL", "deepseek-v4-flash"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base"),
        vector_store_dir=Path(os.getenv("VECTOR_STORE_DIR", "data/vector_store")),
        raw_docs_dir=Path(os.getenv("RAW_DOCS_DIR", "data/raw_docs")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        top_k=int(os.getenv("TOP_K", "4")),
        vector_top_k=int(os.getenv("VECTOR_TOP_K", "8")),
        bm25_top_k=int(os.getenv("BM25_TOP_K", "8")),
        embedding_batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")),
    )
