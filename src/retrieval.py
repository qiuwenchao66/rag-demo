from __future__ import annotations

from langchain_core.documents import Document


def _document_key(document: Document) -> str:
    source = document.metadata.get("source")
    chunk_index = document.metadata.get("chunk_index")
    if source is not None and chunk_index is not None:
        return f"{source}#{chunk_index}"
    chunk_id = document.metadata.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return document.page_content


def dedup_documents(documents: list[Document]) -> list[Document]:
    seen: set[str] = set()
    deduped: list[Document] = []
    for document in documents:
        key = _document_key(document)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(document)
    return deduped


def hybrid_retrieve(
    query: str,
    vector_store: object,
    bm25_store: object,
    vector_top_k: int = 8,
    bm25_top_k: int = 8,
) -> list[Document]:
    vector_documents = vector_store.similarity_search(query, k=vector_top_k)
    bm25_documents = bm25_store.search(query, top_k=bm25_top_k)
    return dedup_documents(vector_documents + bm25_documents)
