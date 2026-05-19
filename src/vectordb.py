from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class EmbeddingsAdapter(Embeddings):
    def __init__(self, client: Any) -> None:
        self._client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


def _normalize_embeddings(embeddings: Any) -> Embeddings:
    if isinstance(embeddings, Embeddings):
        return embeddings
    return EmbeddingsAdapter(embeddings)


def build_vector_store(
    documents: list[Document],
    embeddings: Any,
    persist_dir: Path | None = None,
) -> FAISS:
    if not documents:
        raise ValueError("No documents available for vector store creation.")

    store = FAISS.from_documents(documents, _normalize_embeddings(embeddings))
    if persist_dir is not None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        store.save_local(str(persist_dir))
    return store


def load_vector_store(persist_dir: Path, embeddings: Any) -> FAISS:
    return FAISS.load_local(
        str(persist_dir),
        _normalize_embeddings(embeddings),
        allow_dangerous_deserialization=True,
    )


def search_vector_store(store: FAISS, query: str, top_k: int = 4) -> list[Document]:
    return store.similarity_search(query, k=top_k)
