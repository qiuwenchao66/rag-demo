from langchain_core.documents import Document

from src.splitter import split_documents


def test_split_documents_preserves_source_metadata() -> None:
    documents = [
        Document(
            page_content="A" * 700,
            metadata={"source": "handbook.txt", "file_type": "txt"},
        )
    ]

    chunks = split_documents(documents, chunk_size=300, chunk_overlap=50)

    assert len(chunks) >= 2
    assert chunks[0].metadata["source"] == "handbook.txt"
    assert chunks[0].metadata["file_type"] == "txt"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_id"] == "handbook.txt#0"


def test_split_documents_omits_blank_chunks() -> None:
    chunks = split_documents(
        [Document(page_content="   ", metadata={"source": "blank.txt"})]
    )

    assert chunks == []
