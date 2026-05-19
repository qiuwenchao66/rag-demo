from langchain_core.documents import Document

from src.retrieval import dedup_documents, hybrid_retrieve
from src.reranker import rerank_documents


class FakeVectorStore:
    def similarity_search(self, query: str, k: int = 4):
        assert query == "leave approver"
        assert k == 2
        return [
            Document(
                page_content="Managers approve leave requests.",
                metadata={"source": "hr.txt", "chunk_index": 0, "chunk_id": "hr#0"},
            ),
            Document(
                page_content="Leave requests must be submitted in advance.",
                metadata={"source": "hr.txt", "chunk_index": 1, "chunk_id": "hr#1"},
            ),
        ]


class FakeBM25Store:
    def search(self, query: str, top_k: int = 5):
        assert query == "leave approver"
        assert top_k == 3
        return [
            Document(
                page_content="Managers approve leave requests.",
                metadata={"source": "hr.txt", "chunk_index": 0, "chunk_id": "hr#0"},
            ),
            Document(
                page_content="Directors approve budget changes.",
                metadata={"source": "finance.txt", "chunk_index": 0, "chunk_id": "finance#0"},
            ),
        ]


class FakeReranker:
    def rerank(self, query: str, documents: list[Document], top_k: int = 2):
        assert query == "leave approver"
        assert len(documents) == 3
        assert top_k == 2
        return [documents[2], documents[0]]


def test_dedup_documents_uses_chunk_id() -> None:
    documents = [
        Document(page_content="A", metadata={"chunk_id": "one"}),
        Document(page_content="B", metadata={"chunk_id": "one"}),
        Document(page_content="C", metadata={"chunk_id": "two"}),
    ]

    deduped = dedup_documents(documents)

    assert [doc.page_content for doc in deduped] == ["A", "C"]


def test_hybrid_retrieve_merges_vector_and_bm25_results() -> None:
    results = hybrid_retrieve(
        query="leave approver",
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        vector_top_k=2,
        bm25_top_k=3,
    )

    assert len(results) == 3
    assert [doc.metadata["chunk_id"] for doc in results] == ["hr#0", "hr#1", "finance#0"]


def test_rerank_documents_reorders_candidates() -> None:
    documents = hybrid_retrieve(
        query="leave approver",
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        vector_top_k=2,
        bm25_top_k=3,
    )

    reranked = rerank_documents(
        query="leave approver",
        documents=documents,
        reranker=FakeReranker(),
        top_k=2,
    )

    assert [doc.metadata["chunk_id"] for doc in reranked] == ["finance#0", "hr#0"]
