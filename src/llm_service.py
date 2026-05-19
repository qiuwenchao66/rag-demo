from __future__ import annotations

from typing import Any

import requests
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import AppConfig


SYSTEM_PROMPT = """You are an enterprise knowledge base assistant.
Answer only from the provided context.
If the context is insufficient, say that the knowledge base does not contain enough information.
Keep the answer concise and concrete."""

QUESTION_REWRITE_PROMPT = """Rewrite the latest user question into a standalone retrieval query.
Preserve the original meaning.
Use the conversation history only to resolve omitted references.
Return only the rewritten question."""

RETRIEVAL_DECISION_PROMPT = """Decide whether the latest user question requires new retrieval from the knowledge base.
Reply with YES if new retrieval is needed.
Reply with NO if the assistant can answer using the existing conversation and previously retrieved context only.
Return only YES or NO."""


def format_chat_history(chat_history: list[dict[str, Any]] | None) -> str:
    if not chat_history:
        return ""

    lines: list[str] = []
    for message in chat_history:
        role = message.get("role", "user")
        speaker = "Assistant" if role == "assistant" else "User"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


class DeepSeekAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class DeepSeekClient:
    def __init__(self, config: AppConfig) -> None:
        self._base_url = config.deepseek_base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {config.deepseek_api_key}",
            "Content-Type": "application/json",
        }

    def post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.post(
                f"{self._base_url}{endpoint}",
                headers=self._headers,
                json=payload,
                timeout=60,
            )
        except requests.exceptions.SSLError as exc:
            raise DeepSeekAPIError(
                "A TLS/SSL error occurred while connecting to DeepSeek API.",
            ) from exc
        except requests.RequestException as exc:
            raise DeepSeekAPIError(f"Request to DeepSeek API failed: {exc}") from exc

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            body = exc.response.text.strip() if exc.response is not None else ""
            if status_code == 401:
                raise DeepSeekAPIError(
                    "DeepSeek API authentication failed. Check DEEPSEEK_API_KEY.",
                    status_code=status_code,
                ) from exc
            raise DeepSeekAPIError(
                f"DeepSeek API request failed (HTTP {status_code}). Response: {body or 'empty response'}",
                status_code=status_code,
            ) from exc
        return response.json()


class ChatService:
    def __init__(self, config: AppConfig) -> None:
        self._client = DeepSeekClient(config)
        self._model = config.chat_model

    def generate_answer(
        self,
        question: str,
        context: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> str:
        history_block = format_chat_history(chat_history)
        user_content = f"Conversation history:\n{history_block}\n\n" if history_block else ""
        user_content += f"Question: {question}\n\nContext:\n{context}"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
        }
        data = self._client.post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    def should_retrieve(
        self,
        question: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> bool:
        if not chat_history:
            return True

        history_block = format_chat_history(chat_history)
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": RETRIEVAL_DECISION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_block}\n\n"
                        f"Latest user question:\n{question}"
                    ),
                },
            ],
            "temperature": 0,
        }
        data = self._client.post("/chat/completions", payload)
        decision = data["choices"][0]["message"]["content"].strip().upper()
        return decision != "NO"

    def rewrite_question(
        self,
        question: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> str:
        if not chat_history:
            return question

        history_block = format_chat_history(chat_history)
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": QUESTION_REWRITE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_block}\n\n"
                        f"Latest user question:\n{question}"
                    ),
                },
            ],
            "temperature": 0,
        }
        data = self._client.post("/chat/completions", payload)
        rewritten_question = data["choices"][0]["message"]["content"].strip()
        return rewritten_question or question


class LocalEmbeddings(Embeddings):
    def __init__(self, config: AppConfig) -> None:
        self._model_name = config.embedding_model
        self._batch_size = config.embedding_batch_size
        self._model = SentenceTransformer(self._model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(config: AppConfig) -> LocalEmbeddings:
    return LocalEmbeddings(config)
