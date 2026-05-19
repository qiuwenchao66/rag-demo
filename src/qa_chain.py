from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.retrieval import hybrid_retrieve
from src.reranker import rerank_documents


def build_context(documents: list[Document]) -> str:
    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        source = Path(document.metadata.get("source", "unknown")).name
        chunk_index = document.metadata.get("chunk_index", "n/a")
        blocks.append(
            f"[Source {index}] {source} (chunk {chunk_index})\n{document.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def latest_cached_context(chat_history: list[dict[str, Any]] | None) -> tuple[str, list[dict[str, Any]]]:
    if not chat_history:
        return "", []

    for message in reversed(chat_history):
        if message.get("role") != "assistant":
            continue
        context = str(message.get("context", "")).strip()
        sources = message.get("sources", [])
        if context:
            return context, sources
    return "", []


def answer_question(
    question: str,
    vector_store: Any,
    bm25_store: Any,
    reranker: Any,
    chat_service: Any,
    top_k: int = 4,
    vector_top_k: int = 8,
    bm25_top_k: int = 8,
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    retrieval_question = question
    used_retrieval = True
    sources: list[dict[str, Any]]

    if chat_history and not chat_service.should_retrieve(question=question, chat_history=chat_history):
        context, sources = latest_cached_context(chat_history)
        used_retrieval = False
        if not context:
            used_retrieval = True
    else:
        context = ""
        sources = []

    if used_retrieval:
        if chat_history:
            retrieval_question = chat_service.rewrite_question(
                question=question,
                chat_history=chat_history,
            )

        candidates = hybrid_retrieve(
            query=retrieval_question,
            vector_store=vector_store,
            bm25_store=bm25_store,
            vector_top_k=vector_top_k,
            bm25_top_k=bm25_top_k,
        )
        documents = rerank_documents(
            query=retrieval_question,
            documents=candidates,
            reranker=reranker,
            top_k=top_k,
        )
        context = build_context(documents)
        sources = [
            {
                "source": Path(doc.metadata.get("source", "unknown")).name,
                "chunk_index": doc.metadata.get("chunk_index"),
                "content": doc.page_content.strip(),
            }
            for doc in documents
        ]

    answer = chat_service.generate_answer(
        question=question,
        context=context,
        chat_history=chat_history,
    )
    return {
        "answer": answer,
        "sources": sources,
        "context": context,
        "retrieval_question": retrieval_question,
        "used_retrieval": used_retrieval,
    }
