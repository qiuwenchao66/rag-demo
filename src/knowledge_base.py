from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


METADATA_FILE = "metadata.json"
CHAT_HISTORY_FILE = "chat_history.json"
VECTOR_STORE_DIR = "vector_store"
RAW_DOCS_DIR = "raw_docs"
BM25_STORE_FILE = "bm25_store.json"


@dataclass(slots=True)
class KnowledgeBasePaths:
    id: str
    name: str
    base_dir: Path
    vector_store_dir: Path
    bm25_store_path: Path
    raw_docs_dir: Path
    chat_history_path: Path


def _slugify(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return normalized or "knowledge-base"


def _metadata_path(base_dir: Path) -> Path:
    return base_dir / METADATA_FILE


def _read_metadata(base_dir: Path) -> dict[str, Any]:
    return json.loads(_metadata_path(base_dir).read_text(encoding="utf-8"))


def _write_metadata(base_dir: Path, payload: dict[str, Any]) -> None:
    _metadata_path(base_dir).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _build_paths(base_dir: Path, knowledge_base_id: str, knowledge_base_name: str) -> KnowledgeBasePaths:
    return KnowledgeBasePaths(
        id=knowledge_base_id,
        name=knowledge_base_name,
        base_dir=base_dir,
        vector_store_dir=base_dir / VECTOR_STORE_DIR,
        bm25_store_path=base_dir / BM25_STORE_FILE,
        raw_docs_dir=base_dir / RAW_DOCS_DIR,
        chat_history_path=base_dir / CHAT_HISTORY_FILE,
    )


def create_knowledge_base(root_dir: Path, name: str) -> KnowledgeBasePaths:
    root_dir.mkdir(parents=True, exist_ok=True)
    normalized_name = name.strip()
    base_id = _slugify(normalized_name)
    candidate = base_id
    suffix = 2
    while (root_dir / candidate).exists():
        candidate = f"{base_id}-{suffix}"
        suffix += 1

    base_dir = root_dir / candidate
    base_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "id": candidate,
        "name": normalized_name,
        "created_at": datetime.now(UTC).isoformat(),
    }
    _write_metadata(base_dir, metadata)
    paths = _build_paths(base_dir, candidate, normalized_name)
    paths.vector_store_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_docs_dir.mkdir(parents=True, exist_ok=True)
    paths.chat_history_path.write_text("[]", encoding="utf-8")
    return paths


def list_knowledge_bases(root_dir: Path) -> list[KnowledgeBasePaths]:
    if not root_dir.exists():
        return []

    items: list[KnowledgeBasePaths] = []
    for child in sorted(root_dir.iterdir(), key=lambda entry: entry.name):
        if not child.is_dir():
            continue
        metadata_path = _metadata_path(child)
        if not metadata_path.exists():
            continue
        metadata = _read_metadata(child)
        items.append(_build_paths(child, metadata["id"], metadata["name"]))
    return items


def build_knowledge_base_paths(root_dir: Path, knowledge_base_id: str) -> KnowledgeBasePaths:
    base_dir = root_dir / knowledge_base_id
    if not base_dir.exists():
        raise FileNotFoundError(f"Knowledge base '{knowledge_base_id}' does not exist.")
    metadata = _read_metadata(base_dir)
    return _build_paths(base_dir, metadata["id"], metadata["name"])


def load_chat_history(chat_history_path: Path) -> list[dict[str, Any]]:
    if not chat_history_path.exists():
        return []
    return json.loads(chat_history_path.read_text(encoding="utf-8"))


def append_chat_message(chat_history_path: Path, message: dict[str, Any]) -> None:
    history = load_chat_history(chat_history_path)
    history.append(message)
    chat_history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
