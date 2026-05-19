from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    non_blank = [doc for doc in documents if doc.page_content.strip()]
    if not non_blank:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_documents(non_blank)
    for index, chunk in enumerate(chunks):
        source = str(chunk.metadata.get("source", "unknown"))
        chunk.metadata["chunk_index"] = index
        chunk.metadata["chunk_id"] = f"{source}#{index}"
    return [chunk for chunk in chunks if chunk.page_content.strip()]
