# Phase 01 — Foundation, Scope, and Data Provenance

## Objective

Make the repository safe to develop, reproducible, and explicit about its corpus before changing RAG behavior.

## Scope decision

Implement the Core Track only:

- Text-extractable PDF guidelines.
- Local embeddings and a persisted vector store.
- Local Ollama generation.
- FastAPI backend.
- Streamlit frontend.

Do not add YOLO, image uploads, OCR pipelines, or multimodal prompting unless every Core Track exit gate has passed and there is time remaining.

## Tasks

### 1. Establish repository hygiene

Create a root .gitignore before initializing the real Git repository. It must exclude at least:

    .venv/
    __pycache__/
    *.py[cod]
    .env
    *.log
    .pytest_cache/
    .ipynb_checkpoints/
    reference/
    backend/data/vector_store/
    models/

Decide whether raw PDFs can be committed:

- NICE and NHLBI are small.
- GINA is about 17 MB, below GitHub's hard single-file limit but large enough to require a deliberate decision.
- Confirm each source's redistribution terms. If any document should not be redistributed, ignore data/raw and document a download process instead.

Do not commit secrets, the virtual environment, downloaded model weights, the reference clone, or generated caches.

### 2. Normalize required names and directories

Rename:

    notebooks/Rag_pipline.ipynb

to:

    notebooks/rag_pipeline.ipynb

Create the planned backend, frontend, eval, and docs/images directories. Add package __init__.py files where imports require them.

The rename must preserve notebook content and history. Do not keep both spellings.

### 3. Record corpus provenance

Create data/manifest.json with one entry per source:

- stable document_id
- file_name
- display_name
- publisher or organization
- document version or publication year
- source_url
- retrieval/download date
- SHA-256
- page_count
- file_size_bytes
- redistribution note
- expected format

Use the exact local document versions. In particular:

- Identify NICE as NG245, not the reference repository's older NG80 label.
- Identify the exact NHLBI edition/date.
- Identify GINA 2026 and its official source URL.

Open representative pages from the beginning, middle, tables, and end of each PDF. Record extraction artifacts, page-number offsets, multi-column ordering problems, and whether citations should use PDF page number or printed page number. The simplest defensible rule is to cite the one-based PDF page used in stored metadata and state that rule in README.

### 4. Fix the GINA parser blocker

Choose and test one loader path:

- Preferred minimal change: keep pypdf and add a pinned cryptography dependency that can decrypt the AES-protected file.
- Acceptable alternative: use PyMuPDF directly, adapted from the reference ProcessController, if it extracts the document reliably.

After the fix, compute for every PDF:

- pages discovered
- pages extracted
- empty pages
- pages under the OCR threshold
- parsing exceptions
- total extracted characters

Fail the notebook clearly if a required file cannot be parsed. Do not silently index only two of three guidelines.

### 5. Define configuration contracts

Create environment examples without secrets.

Backend settings:

| Variable | Purpose | Suggested default |
|---|---|---|
| APP_NAME | API display name | Asthma Guideline RAG |
| OLLAMA_BASE_URL | Local Ollama endpoint | http://localhost:11434 |
| OLLAMA_MODEL | Local generation model | llama3.2:1b |
| VECTOR_STORE_PATH | Persisted Chroma path | data/vector_store/chroma |
| RAG_CONFIG_PATH | Exported config | data/rag_config.json |
| FRONTEND_ORIGINS | Comma-separated allowed origins | http://localhost:8501 |
| LOG_LEVEL | Application log level | INFO |
| RETRIEVAL_TOP_K | Candidate count | loaded from exported config |
| REQUEST_TIMEOUT_SECONDS | Ollama timeout | 600 |

Frontend settings:

| Variable | Purpose | Suggested default |
|---|---|---|
| API_BASE_URL | FastAPI base URL | http://localhost:8000 |
| API_TIMEOUT_SECONDS | Friendly request timeout | 660 |

Use pydantic-settings in the backend and python-dotenv or environment reads in the frontend.

### 6. Pin dependencies by component

Create separate requirements files:

- backend/requirements.txt
- frontend/requirements.txt
- optionally requirements-dev.txt for notebook and tests

Required backend packages include FastAPI, Uvicorn, Pydantic, pydantic-settings, Ollama client, Chroma, sentence-transformers, pytest, and httpx. Add bm25s only if Phase 03's local hybrid index is implemented. Add cryptography if pypdf remains the loader.

Pin exact versions only after the full flow passes. Until then, use compatible bounds to avoid freezing a broken environment.

## Reference reuse

Adapt:

- Source metadata ideas from reference/RAG_AI_Hackathon/scripts/ingest_guidelines.py.
- Page normalization and loader handling from src/controllers/ProccesController.py.
- Settings organization from src/helpers/config.py.

Do not copy:

- Reference database settings.
- External API keys and provider configuration.
- Project IDs or upload endpoints.

## Verification

Run and record:

    python --version
    ollama --version
    git --version
    ollama list

Then run a corpus inspection that exits non-zero if:

- any manifest file is missing
- any hash differs
- any required PDF cannot be parsed
- every page of a document is empty

## Exit gate

- The notebook has the exact required name.
- The manifest covers all three PDFs with source URLs and hashes.
- GINA extraction succeeds in the chosen runtime.
- .gitignore prevents accidental submission of the virtual environment, secrets, reference clone, model weights, and generated caches.
- Environment examples contain every required runtime variable and no real secret.
- The Core Track decision and citation page-number convention are documented.

## Deliverables

- .gitignore
- data/manifest.json
- notebooks/rag_pipeline.ipynb
- backend/.env.example
- frontend/.env.example
- component requirements files
- THIRD_PARTY_NOTICES.md
