import requests

from src.config import AppConfig
from src.llm_service import ChatService, DeepSeekAPIError, LocalEmbeddings


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def post(self, endpoint, payload):
        self.calls.append((endpoint, payload))
        return self.responses[len(self.calls) - 1]


class FakeVector:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.calls = []

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        self.calls.append(
            {
                "texts": texts,
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
                "show_progress_bar": show_progress_bar,
            }
        )
        return FakeVector([[0.1, 0.2] for _ in texts])


def build_config() -> AppConfig:
    return AppConfig(
        deepseek_api_key="token",
        deepseek_base_url="https://api.deepseek.com",
        chat_model="deepseek-v4-flash",
        embedding_model="BAAI/bge-small-zh-v1.5",
        reranker_model="BAAI/bge-reranker-base",
        vector_store_dir=None,  # type: ignore[arg-type]
        raw_docs_dir=None,  # type: ignore[arg-type]
        embedding_batch_size=2,
    )


def test_chat_service_uses_deepseek_chat_endpoint() -> None:
    service = ChatService(build_config())
    service._client = FakeClient(  # type: ignore[assignment]
        [{"choices": [{"message": {"content": "answer"}}]}]
    )

    answer = service.generate_answer(
        "question",
        "context",
        chat_history=[{"role": "user", "content": "previous"}],
    )

    assert answer == "answer"
    endpoint, payload = service._client.calls[0]
    assert endpoint == "/chat/completions"
    assert payload["model"] == "deepseek-v4-flash"
    assert "Conversation history" in payload["messages"][1]["content"]
    assert "User: previous" in payload["messages"][1]["content"]


def test_chat_service_rewrites_follow_up_question_for_retrieval() -> None:
    service = ChatService(build_config())
    service._client = FakeClient(  # type: ignore[assignment]
        [{"choices": [{"message": {"content": "What is the approver for a leave request?"}}]}]
    )

    rewritten = service.rewrite_question(
        "Who approves it?",
        chat_history=[
            {"role": "user", "content": "How do I request leave?"},
            {"role": "assistant", "content": "Submit it three days in advance."},
        ],
    )

    assert rewritten == "What is the approver for a leave request?"
    endpoint, payload = service._client.calls[0]
    assert endpoint == "/chat/completions"
    assert "Rewrite the latest user question" in payload["messages"][0]["content"]
    assert "Who approves it?" in payload["messages"][1]["content"]


def test_chat_service_can_skip_retrieval_for_follow_up() -> None:
    service = ChatService(build_config())
    service._client = FakeClient(  # type: ignore[assignment]
        [{"choices": [{"message": {"content": "NO"}}]}]
    )

    should_retrieve = service.should_retrieve(
        "Can you explain that in simpler terms?",
        chat_history=[
            {"role": "user", "content": "How do I request leave?"},
            {"role": "assistant", "content": "Submit it three days in advance."},
        ],
    )

    assert should_retrieve is False
    endpoint, payload = service._client.calls[0]
    assert endpoint == "/chat/completions"
    assert "Decide whether the latest user question requires new retrieval" in payload["messages"][0]["content"]


def test_local_embeddings_use_sentence_transformer() -> None:
    embeddings = LocalEmbeddings.__new__(LocalEmbeddings)
    embeddings._model_name = "BAAI/bge-small-zh-v1.5"
    embeddings._batch_size = 2
    embeddings._model = FakeSentenceTransformer(embeddings._model_name)

    vector = embeddings.embed_query("hello")

    assert vector == [0.1, 0.2]
    call = embeddings._model.calls[0]
    assert call["texts"] == ["hello"]
    assert call["batch_size"] == 2
    assert call["normalize_embeddings"] is True


def test_client_raises_friendly_error_for_unauthorized() -> None:
    class FakeResponse:
        status_code = 401
        text = "Unauthorized"

        def raise_for_status(self) -> None:
            raise requests.HTTPError("401 Client Error", response=self)

        def json(self):
            return {"message": "Invalid API key"}

    class FakeRequests:
        @staticmethod
        def post(*args, **kwargs):
            return FakeResponse()

    from src import llm_service

    client = llm_service.DeepSeekClient(build_config())
    original_post = llm_service.requests.post
    llm_service.requests.post = FakeRequests.post  # type: ignore[assignment]
    try:
        try:
            client.post("/chat/completions", {"model": "x"})
        except DeepSeekAPIError as exc:
            assert exc.status_code == 401
            assert "authentication failed" in str(exc)
        else:
            raise AssertionError("Expected DeepSeekAPIError")
    finally:
        llm_service.requests.post = original_post  # type: ignore[assignment]
