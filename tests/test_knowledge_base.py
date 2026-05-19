from pathlib import Path

from src.knowledge_base import (
    append_chat_message,
    build_knowledge_base_paths,
    create_knowledge_base,
    list_knowledge_bases,
    load_chat_history,
)


def test_create_knowledge_base_persists_metadata_and_paths(tmp_path: Path) -> None:
    knowledge_base = create_knowledge_base(tmp_path, "HR Policies")

    assert knowledge_base.name == "HR Policies"
    assert knowledge_base.base_dir.exists()
    assert knowledge_base.vector_store_dir == knowledge_base.base_dir / "vector_store"
    assert knowledge_base.bm25_store_path == knowledge_base.base_dir / "bm25_store.json"
    assert knowledge_base.raw_docs_dir == knowledge_base.base_dir / "raw_docs"
    assert knowledge_base.chat_history_path == knowledge_base.base_dir / "chat_history.json"


def test_create_knowledge_base_avoids_duplicate_ids(tmp_path: Path) -> None:
    first = create_knowledge_base(tmp_path, "HR Policies")
    second = create_knowledge_base(tmp_path, "HR Policies")

    assert first.id == "hr-policies"
    assert second.id == "hr-policies-2"


def test_list_knowledge_bases_reads_saved_metadata(tmp_path: Path) -> None:
    create_knowledge_base(tmp_path, "HR Policies")
    create_knowledge_base(tmp_path, "Security")

    knowledge_bases = list_knowledge_bases(tmp_path)

    assert [item.name for item in knowledge_bases] == ["HR Policies", "Security"]


def test_chat_history_is_stored_per_knowledge_base(tmp_path: Path) -> None:
    knowledge_base = create_knowledge_base(tmp_path, "HR Policies")

    append_chat_message(
        knowledge_base.chat_history_path,
        {
            "role": "user",
            "content": "How do I request leave?",
            "sources": [],
        },
    )
    append_chat_message(
        knowledge_base.chat_history_path,
        {
            "role": "assistant",
            "content": "Submit the request three days in advance.",
            "sources": [{"source": "handbook.txt", "chunk_index": 0, "content": "..."}],
        },
    )

    history = load_chat_history(knowledge_base.chat_history_path)

    assert [item["role"] for item in history] == ["user", "assistant"]
    assert history[1]["sources"][0]["source"] == "handbook.txt"


def test_build_knowledge_base_paths_returns_selected_paths(tmp_path: Path) -> None:
    knowledge_base = create_knowledge_base(tmp_path, "HR Policies")

    resolved = build_knowledge_base_paths(tmp_path, knowledge_base.id)

    assert resolved.id == knowledge_base.id
    assert resolved.name == "HR Policies"
