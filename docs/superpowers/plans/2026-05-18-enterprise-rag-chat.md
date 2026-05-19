# Enterprise RAG Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable MVP that ingests PDF, DOCX, and TXT files, creates a local FAISS index, answers questions with an OpenAI-compatible LLM, and shows sources in a Streamlit UI.

**Architecture:** Keep the backend modular in `src/` so document loading, chunking, vector indexing, and QA orchestration stay independently testable. Use Streamlit as the single UI surface and persist the FAISS store locally under `data/vector_store/`.

**Tech Stack:** Python, Streamlit, LangChain text splitter + FAISS, OpenAI-compatible chat/embedding APIs, PyPDF2, python-docx, pytest

---

### Task 1: Scaffold the project

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `README.md`
- Create: `src/__init__.py`
- Create: `data/raw_docs/.gitkeep`
- Create: `data/vector_store/.gitkeep`
- Create: `demo_docs/employee_handbook.txt`

- [ ] **Step 1: Create dependency and environment files**
- [ ] **Step 2: Add README usage instructions**
- [ ] **Step 3: Create runtime directories and one demo document**

### Task 2: Implement and test document loading

**Files:**
- Create: `src/loaders.py`
- Create: `tests/test_loaders.py`

- [ ] **Step 1: Write failing tests for TXT and unsupported file handling**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement document loading helpers**
- [ ] **Step 4: Run targeted tests and verify pass**

### Task 3: Implement and test chunking and retrieval config

**Files:**
- Create: `src/splitter.py`
- Create: `src/config.py`
- Create: `tests/test_splitter.py`

- [ ] **Step 1: Write failing tests for chunk generation and metadata preservation**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement splitter/config**
- [ ] **Step 4: Run targeted tests and verify pass**

### Task 4: Implement and test vector store helpers

**Files:**
- Create: `src/vectordb.py`
- Create: `tests/test_vectordb.py`

- [ ] **Step 1: Write failing tests with a fake embeddings client**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement FAISS build/save/load/search helpers**
- [ ] **Step 4: Run targeted tests and verify pass**

### Task 5: Implement and test QA orchestration

**Files:**
- Create: `src/llm_service.py`
- Create: `src/qa_chain.py`
- Create: `tests/test_qa_chain.py`

- [ ] **Step 1: Write failing tests for prompt assembly and source formatting**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Implement chat service and QA orchestration**
- [ ] **Step 4: Run targeted tests and verify pass**

### Task 6: Build the Streamlit app

**Files:**
- Create: `app.py`

- [ ] **Step 1: Wire upload, indexing, and question submission flow**
- [ ] **Step 2: Surface answer, sources, and status/error states**

### Task 7: Final verification

**Files:**
- Verify: `tests/`
- Verify: `app.py`

- [ ] **Step 1: Run full test suite**
- [ ] **Step 2: Smoke-check Streamlit import path**
- [ ] **Step 3: Review README against actual environment variables and commands**
