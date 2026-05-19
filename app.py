from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src.bm25db import build_bm25_store, load_bm25_store
from src.config import load_config
from src.knowledge_base import (
    append_chat_message,
    build_knowledge_base_paths,
    create_knowledge_base,
    list_knowledge_bases,
    load_chat_history,
)
from src.llm_service import ChatService, DeepSeekAPIError, build_embeddings
from src.loaders import SUPPORTED_SUFFIXES, load_documents
from src.qa_chain import answer_question
from src.reranker import LocalReranker
from src.splitter import split_documents
from src.vectordb import build_vector_store, load_vector_store


def save_uploaded_files(
    uploaded_files: list[st.runtime.uploaded_file_manager.UploadedFile],
    target_dir: Path,
) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for uploaded_file in uploaded_files:
        path = target_dir / uploaded_file.name
        path.write_bytes(uploaded_file.getbuffer())
        saved_paths.append(path)
    return saved_paths


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def knowledge_bases_root(config_path: Path) -> Path:
    return config_path.parent / "knowledge_bases"


@st.cache_resource(show_spinner=False)
def get_embeddings(config_signature: tuple[str, int]):
    config = load_config()
    return build_embeddings(config)


@st.cache_resource(show_spinner=False)
def get_reranker(model_name: str) -> LocalReranker:
    return LocalReranker(model_name)


def knowledge_base_ready(vector_store_dir: Path, bm25_store_path: Path) -> bool:
    return (
        vector_store_dir.exists()
        and any(vector_store_dir.iterdir())
        and bm25_store_path.exists()
        and bm25_store_path.stat().st_size > 0
    )


def ensure_selected_knowledge_base(root_dir: Path) -> str | None:
    knowledge_bases = list_knowledge_bases(root_dir)
    if not knowledge_bases:
        return None

    current_id = st.session_state.get("current_knowledge_base_id")
    available_ids = {item.id for item in knowledge_bases}
    if current_id in available_ids:
        return current_id

    selected_id = knowledge_bases[0].id
    st.session_state["current_knowledge_base_id"] = selected_id
    return selected_id


def format_knowledge_base_label(knowledge_bases: list[Any], knowledge_base_id: str) -> str:
    for item in knowledge_bases:
        if item.id == knowledge_base_id:
            return item.name
    return knowledge_base_id


def render_chat_history(chat_history: list[dict[str, Any]]) -> None:
    for message in chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                mode = message.get("retrieval_mode")
                if mode:
                    st.caption(f"Retrieval mode: {mode}")
                for source in message.get("sources", []):
                    with st.expander(f'{source["source"]} / chunk {source["chunk_index"]}'):
                        st.write(source["content"])


def main() -> None:
    config = load_config()
    kb_root = knowledge_bases_root(config.vector_store_dir)

    st.set_page_config(page_title="Enterprise Knowledge Base QA", page_icon=":books:", layout="wide")
    st.title("Enterprise Knowledge Base QA")
    st.caption("Build isolated indexes per knowledge base with multi-turn chat, hybrid retrieval, and reranking.")

    with st.sidebar:
        st.subheader("Knowledge Base")
        knowledge_bases = list_knowledge_bases(kb_root)
        selected_id = ensure_selected_knowledge_base(kb_root)

        if knowledge_bases:
            knowledge_base_ids = [item.id for item in knowledge_bases]
            if selected_id not in knowledge_base_ids:
                selected_id = knowledge_base_ids[0]
                st.session_state["current_knowledge_base_id"] = selected_id

            selected_id = st.selectbox(
                "Current knowledge base",
                options=knowledge_base_ids,
                index=knowledge_base_ids.index(selected_id),
                format_func=lambda kb_id: format_knowledge_base_label(knowledge_bases, kb_id),
            )
            st.session_state["current_knowledge_base_id"] = selected_id
        else:
            st.info("Create a knowledge base first.")

        new_knowledge_base_name = st.text_input(
            "New knowledge base name",
            placeholder="Example: HR policies",
        )
        if st.button("Create knowledge base", use_container_width=True):
            if not new_knowledge_base_name.strip():
                st.error("Enter a knowledge base name first.")
            else:
                created = create_knowledge_base(kb_root, new_knowledge_base_name)
                st.session_state["current_knowledge_base_id"] = created.id
                st.rerun()

        st.divider()
        st.subheader("Retrieval Settings")
        top_k = st.slider("Final Top-K", min_value=1, max_value=8, value=config.top_k)
        vector_top_k = st.slider("Vector Top-K", min_value=1, max_value=16, value=config.vector_top_k)
        bm25_top_k = st.slider("BM25 Top-K", min_value=1, max_value=16, value=config.bm25_top_k)
        chunk_size = st.number_input(
            "Chunk size",
            min_value=200,
            max_value=1200,
            value=config.chunk_size,
            step=50,
        )
        chunk_overlap = st.number_input(
            "Chunk overlap",
            min_value=0,
            max_value=400,
            value=config.chunk_overlap,
            step=20,
        )

    selected_id = st.session_state.get("current_knowledge_base_id")
    if not selected_id:
        st.stop()

    knowledge_base = build_knowledge_base_paths(kb_root, selected_id)
    st.subheader(knowledge_base.name)

    controls_col, action_col = st.columns([3, 1])
    with controls_col:
        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, or TXT documents for the current knowledge base",
            type=[suffix.removeprefix(".") for suffix in sorted(SUPPORTED_SUFFIXES)],
            accept_multiple_files=True,
        )
    with action_col:
        st.write("")
        st.write("")
        if st.button("Clear chat", use_container_width=True):
            knowledge_base.chat_history_path.write_text("[]", encoding="utf-8")
            st.rerun()

    if st.button("Build knowledge base", type="primary", disabled=not uploaded_files):
        if not config.deepseek_api_key:
            st.error("Missing DeepSeek API key. Set DEEPSEEK_API_KEY in .env first.")
            return

        try:
            with st.spinner("Parsing documents and building indexes..."):
                reset_directory(knowledge_base.raw_docs_dir)
                reset_directory(knowledge_base.vector_store_dir)
                if knowledge_base.bm25_store_path.exists():
                    knowledge_base.bm25_store_path.unlink()

                saved_paths = save_uploaded_files(uploaded_files, knowledge_base.raw_docs_dir)
                documents = load_documents(saved_paths)
                chunks = split_documents(
                    documents,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                embeddings = get_embeddings((config.embedding_model, config.embedding_batch_size))
                build_vector_store(chunks, embeddings, persist_dir=knowledge_base.vector_store_dir)
                build_bm25_store(chunks, persist_path=knowledge_base.bm25_store_path)
            st.success(f"Knowledge base built successfully with {len(chunks)} chunks.")
        except DeepSeekAPIError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Knowledge base build failed: {exc}")
            return

    chat_history = load_chat_history(knowledge_base.chat_history_path)
    render_chat_history(chat_history)

    prompt = st.chat_input(f"Ask a question about {knowledge_base.name}")
    if not prompt:
        return

    if not knowledge_base_ready(knowledge_base.vector_store_dir, knowledge_base.bm25_store_path):
        st.error("The current knowledge base is not indexed yet. Upload documents and build it first.")
        return

    if not config.deepseek_api_key:
        st.error("Missing DeepSeek API key. Set DEEPSEEK_API_KEY in .env first.")
        return

    with st.chat_message("user"):
        st.write(prompt)

    try:
        with st.spinner("Retrieving context and generating answer..."):
            embeddings = get_embeddings((config.embedding_model, config.embedding_batch_size))
            vector_store = load_vector_store(knowledge_base.vector_store_dir, embeddings)
            bm25_store = load_bm25_store(knowledge_base.bm25_store_path)
            reranker = get_reranker(config.reranker_model)
            response = answer_question(
                question=prompt.strip(),
                vector_store=vector_store,
                bm25_store=bm25_store,
                reranker=reranker,
                chat_service=ChatService(config),
                top_k=top_k,
                vector_top_k=vector_top_k,
                bm25_top_k=bm25_top_k,
                chat_history=chat_history,
            )
    except DeepSeekAPIError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"Question answering failed: {exc}")
        return

    retrieval_mode = "hybrid" if response["used_retrieval"] else "cached"

    append_chat_message(
        knowledge_base.chat_history_path,
        {
            "role": "user",
            "content": prompt.strip(),
            "sources": [],
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )
    append_chat_message(
        knowledge_base.chat_history_path,
        {
            "role": "assistant",
            "content": response["answer"],
            "sources": response["sources"],
            "context": response["context"],
            "retrieval_question": response["retrieval_question"],
            "used_retrieval": response["used_retrieval"],
            "retrieval_mode": retrieval_mode,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )

    with st.chat_message("assistant"):
        st.write(response["answer"])
        st.caption(f"Retrieval mode: {retrieval_mode}")
        for source in response["sources"]:
            with st.expander(f'{source["source"]} / chunk {source["chunk_index"]}'):
                st.write(source["content"])


if __name__ == "__main__":
    main()
