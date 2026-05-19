from pathlib import Path

from langchain_core.documents import Document

from src.vectordb import build_vector_store, search_vector_store


class FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [
            1.0 if "年假" in text else 0.0,
            1.0 if "密码" in text else 0.0,
            0.0,
        ]


def test_build_vector_store_returns_searchable_index(tmp_path: Path) -> None:
    docs = [
        Document(page_content="年假最少提前 3 天申请", metadata={"source": "hr.txt"}),
        Document(page_content="密码至少 12 位", metadata={"source": "security.txt"}),
    ]

    store = build_vector_store(docs, FakeEmbeddings(), persist_dir=tmp_path)
    results = search_vector_store(store, "年假怎么申请", top_k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] == "hr.txt"
