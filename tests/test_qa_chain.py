from langchain_core.documents import Document

from src.qa_chain import answer_question, build_context


class FakeVectorStore:
    def similarity_search(self, query: str, k: int = 4):
        assert query == "What is the approver for a leave request?"
        assert k == 3
        return [
            Document(
                page_content="Submit leave requests at least three business days in advance.",
                metadata={"source": "employee_handbook.txt", "chunk_index": 0},
            ),
            Document(
                page_content="Your direct manager must approve the request.",
                metadata={"source": "employee_handbook.txt", "chunk_index": 1},
            ),
        ]


class FakeBM25Store:
    def search(self, query: str, top_k: int = 5):
        assert query == "What is the approver for a leave request?"
        assert top_k == 3
        return [
            Document(
                page_content="Your direct manager must approve the request.",
                metadata={"source": "employee_handbook.txt", "chunk_index": 1, "chunk_id": "employee_handbook#1"},
            ),
            Document(
                page_content="Approvers may delegate routine requests.",
                metadata={"source": "employee_handbook.txt", "chunk_index": 2, "chunk_id": "employee_handbook#2"},
            ),
        ]


class FakeReranker:
    def rerank(self, query: str, documents: list[Document], top_k: int = 2):
        assert query == "What is the approver for a leave request?"
        assert top_k == 2
        assert len(documents) == 3
        return [documents[1], documents[0]]


class FakeChatService:
    def should_retrieve(
        self,
        question: str,
        chat_history: list[dict] | None = None,
    ) -> bool:
        assert question == "Who approves it?"
        assert chat_history == [{"role": "user", "content": "What is the leave policy?"}]
        return True

    def rewrite_question(
        self,
        question: str,
        chat_history: list[dict] | None = None,
    ) -> str:
        assert question == "Who approves it?"
        assert chat_history == [{"role": "user", "content": "What is the leave policy?"}]
        return "What is the approver for a leave request?"

    def generate_answer(
        self,
        question: str,
        context: str,
        chat_history: list[dict] | None = None,
    ) -> str:
        assert question == "Who approves it?"
        assert "Submit leave requests at least three business days in advance." in context
        assert chat_history == [{"role": "user", "content": "What is the leave policy?"}]
        return "Submit the leave request three business days in advance and wait for manager approval."


def test_build_context_formats_sources() -> None:
    docs = [
        Document(
            page_content="Passwords must be changed every 90 days.",
            metadata={"source": "policy.txt", "chunk_index": 0},
        )
    ]

    context = build_context(docs)

    assert "[Source 1] policy.txt" in context
    assert "Passwords must be changed every 90 days." in context


def test_answer_question_returns_answer_sources_and_uses_history() -> None:
    response = answer_question(
        question="Who approves it?",
        vector_store=FakeVectorStore(),
        bm25_store=FakeBM25Store(),
        reranker=FakeReranker(),
        chat_service=FakeChatService(),
        top_k=2,
        vector_top_k=3,
        bm25_top_k=3,
        chat_history=[{"role": "user", "content": "What is the leave policy?"}],
    )

    assert "three business days" in response["answer"]
    assert len(response["sources"]) == 2
    assert response["sources"][0]["source"] == "employee_handbook.txt"


def test_answer_question_uses_original_question_without_history() -> None:
    class PlainVectorStore:
        def similarity_search(self, query: str, k: int = 4):
            assert query == "How do I request leave?"
            return [
                Document(
                    page_content="Submit leave requests at least three business days in advance.",
                    metadata={"source": "employee_handbook.txt", "chunk_index": 0},
                )
            ]

    class PlainBM25Store:
        def search(self, query: str, top_k: int = 5):
            assert query == "How do I request leave?"
            return []

    class PlainReranker:
        def rerank(self, query: str, documents: list[Document], top_k: int = 2):
            return documents[:top_k]

    class PlainChatService:
        def rewrite_question(self, question: str, chat_history: list[dict] | None = None) -> str:
            raise AssertionError("rewrite_question should not be called without history")

        def should_retrieve(self, question: str, chat_history: list[dict] | None = None) -> bool:
            return True

        def generate_answer(
            self,
            question: str,
            context: str,
            chat_history: list[dict] | None = None,
        ) -> str:
            assert question == "How do I request leave?"
            return "Answer"

    response = answer_question(
        question="How do I request leave?",
        vector_store=PlainVectorStore(),
        bm25_store=PlainBM25Store(),
        reranker=PlainReranker(),
        chat_service=PlainChatService(),
        top_k=2,
        vector_top_k=2,
        bm25_top_k=2,
    )

    assert response["answer"] == "Answer"


def test_answer_question_reuses_cached_context_without_retrieval() -> None:
    class NoSearchRetriever:
        def similarity_search(self, query: str, k: int = 4):
            raise AssertionError("similarity_search should not be called")

    class NoSearchBM25Store:
        def search(self, query: str, top_k: int = 5):
            raise AssertionError("search should not be called")

    class NoSearchReranker:
        def rerank(self, query: str, documents: list[Document], top_k: int = 2):
            raise AssertionError("rerank should not be called")

    class CachedContextChatService:
        def should_retrieve(self, question: str, chat_history: list[dict] | None = None) -> bool:
            assert question == "Can you summarize that?"
            return False

        def rewrite_question(self, question: str, chat_history: list[dict] | None = None) -> str:
            raise AssertionError("rewrite_question should not be called")

        def generate_answer(
            self,
            question: str,
            context: str,
            chat_history: list[dict] | None = None,
        ) -> str:
            assert context == "Cached context from previous retrieval."
            return "Short summary"

    response = answer_question(
        question="Can you summarize that?",
        vector_store=NoSearchRetriever(),
        bm25_store=NoSearchBM25Store(),
        reranker=NoSearchReranker(),
        chat_service=CachedContextChatService(),
        top_k=2,
        vector_top_k=2,
        bm25_top_k=2,
        chat_history=[
            {
                "role": "assistant",
                "content": "Detailed answer",
                "sources": [{"source": "employee_handbook.txt", "chunk_index": 0, "content": "cached source"}],
                "context": "Cached context from previous retrieval.",
            }
        ],
    )

    assert response["answer"] == "Short summary"
    assert response["context"] == "Cached context from previous retrieval."
    assert response["sources"][0]["source"] == "employee_handbook.txt"
    assert response["used_retrieval"] is False
