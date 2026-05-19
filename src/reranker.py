from __future__ import annotations

from typing import Protocol

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class SupportsRerank(Protocol):
    def rerank(self, query: str, documents: list[Document], top_k: int = 4) -> list[Document]: ...


class LocalReranker:
    def __init__(self, model_name: str) -> None:
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: list[Document], top_k: int = 4) -> list[Document]:
        if not documents:
            return []

        pairs = [(query, document.page_content) for document in documents]
        scores = self._model.predict(pairs)
        ranked = sorted(
            zip(documents, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [document for document, _score in ranked[:top_k]]


def rerank_documents(
    query: str,
    documents: list[Document],
    reranker: SupportsRerank,
    top_k: int = 4,
) -> list[Document]:
    return reranker.rerank(query=query, documents=documents, top_k=top_k)
