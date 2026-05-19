from pathlib import Path

import pytest

from src.loaders import load_document, load_documents


def test_load_document_reads_txt_file(tmp_path: Path) -> None:
    file_path = tmp_path / "policy.txt"
    file_path.write_text("员工需佩戴工牌。", encoding="utf-8")

    document = load_document(file_path)

    assert document.page_content == "员工需佩戴工牌。"
    assert document.metadata["source"] == str(file_path)
    assert document.metadata["file_type"] == "txt"


def test_load_document_rejects_unsupported_file_type(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("# title", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(file_path)


def test_load_documents_skips_empty_documents(tmp_path: Path) -> None:
    first = tmp_path / "filled.txt"
    second = tmp_path / "empty.txt"
    first.write_text("有效内容", encoding="utf-8")
    second.write_text("   ", encoding="utf-8")

    documents = load_documents([first, second])

    assert len(documents) == 1
    assert documents[0].metadata["source"] == str(first)
