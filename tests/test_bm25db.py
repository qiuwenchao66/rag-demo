from pathlib import Path

from langchain_core.documents import Document

from src.bm25db import build_bm25_store, load_bm25_store


def test_build_and_search_bm25_store(tmp_path: Path) -> None:
    documents = [
        Document(
            page_content="Leave requests must be submitted three business days in advance.",
            metadata={"source": "hr.txt", "chunk_index": 0, "chunk_id": "hr#0"},
        ),
        Document(
            page_content="Passwords must be at least twelve characters long.",
            metadata={"source": "security.txt", "chunk_index": 0, "chunk_id": "security#0"},
        ),
    ]

    persist_path = tmp_path / "bm25_store.json"
    store = build_bm25_store(documents, persist_path=persist_path)

    results = store.search("leave request advance", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] == "hr.txt"
    assert persist_path.exists()


def test_load_bm25_store_restores_documents(tmp_path: Path) -> None:
    documents = [
        Document(
            page_content="Managers approve annual leave requests.",
            metadata={"source": "hr.txt", "chunk_index": 1, "chunk_id": "hr#1"},
        )
    ]
    persist_path = tmp_path / "bm25_store.json"
    build_bm25_store(documents, persist_path=persist_path)

    store = load_bm25_store(persist_path)
    results = store.search("approve annual leave", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "hr#1"
