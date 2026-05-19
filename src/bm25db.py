from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class BM25Store:
    documents: list[Document]
    model: BM25Okapi

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        if not self.documents:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.model.get_scores(query_tokens)
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        results: list[Document] = []
        for index, score in ranked:
            results.append(self.documents[index])
            if len(results) >= top_k:
                break
        return results


def _serialize_documents(documents: list[Document]) -> list[dict]:
    return [
        {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }
        for document in documents
    ]


def _deserialize_documents(payload: list[dict]) -> list[Document]:
    return [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in payload
    ]


def build_bm25_store(documents: list[Document], persist_path: Path | None = None) -> BM25Store:
    if not documents:
        raise ValueError("No documents available for BM25 store creation.")

    tokenized_corpus = [tokenize(document.page_content) for document in documents]
    store = BM25Store(documents=documents, model=BM25Okapi(tokenized_corpus))
    if persist_path is not None:
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        persist_path.write_text(
            json.dumps(_serialize_documents(documents), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return store


def load_bm25_store(persist_path: Path) -> BM25Store:
    payload = json.loads(persist_path.read_text(encoding="utf-8"))
    documents = _deserialize_documents(payload)
    tokenized_corpus = [tokenize(document.page_content) for document in documents]
    return BM25Store(documents=documents, model=BM25Okapi(tokenized_corpus))
