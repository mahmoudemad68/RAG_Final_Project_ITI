# Phase 08 — Documentation, Release, and Presentation

## Objective

Package the verified project so a stranger can run it, understand its design and limitations, inspect the API, and reproduce the demonstrated results.

## Administrative clarification

The guide contains two internal conflicts:

- Its property table says the deadline is 4 days, while the Goal section says 6 days.
- Its property table says individual assignment, while the Goal section says individual or a team of 2–3.

Confirm these points with the instructor. Until clarified, plan to the stricter assumptions: a four-day schedule and independently authored submission.

## Root README requirements

Create README.md with all required sections.

### 1. Overview

- Product purpose.
- Asthma guideline domain.
- Core Track declaration.
- Educational-use and medical disclaimer.
- Evidence-only behavior.

### 2. Architecture diagram

Use Mermaid or an image that shows:

    PDFs → cleaning/chunking → embeddings → Chroma
      → retrieval → safety/confidence → Ollama → citations
      → FastAPI → Streamlit

Distinguish offline notebook/index building from online query serving.

### 3. Tech stack

List exact selected versions for:

- Python
- Jupyter
- pypdf or PyMuPDF
- Sentence Transformers and BGE-base-en-v1.5
- Chroma
- optional BM25 library
- Ollama and selected model
- FastAPI/Uvicorn
- Streamlit
- pytest/httpx

### 4. Project structure

Show the final tree without .venv, reference clone, model cache, large generated internals, or secret files.

### 5. Domain and data

For every guideline, include:

- title/version
- organization
- official source URL
- page count
- retrieval date
- redistribution note

Explain whether raw files are committed and how to obtain them when absent. State that citations use one-based PDF page numbers.

### 6. Setup

Provide copy-paste steps for:

1. clone
2. create/activate virtual environment
3. install notebook/dev dependencies
4. obtain source PDFs
5. run the notebook or obtain the included small artifact
6. install/start Ollama
7. pull the configured model
8. configure backend environment
9. start FastAPI
10. configure frontend environment
11. start Streamlit

Include Windows and Linux/macOS activation differences where useful.

### 7. Environment variables

Include tables for every backend and frontend variable, whether required, default, and example. Never include a real secret.

### 8. API reference

Document:

- GET /health
- POST /query
- request/response examples
- validation and service error behavior
- curl command
- Swagger URL

### 9. Evaluation

Publish locally generated results:

- question count and categories
- retrieval method comparison
- selected Top-K
- retrieval metrics
- answer correctness
- citation metrics
- safety/refusal metrics
- known failure cases

Link to the executed notebook and result files. State the machine/model used for latency.

### 10. Screenshots

Include:

- Streamlit question and cited answer.
- Evidence/source display.
- Swagger POST /query success.
- Optional safe refusal example.

Use compressed images beneath docs/images.

### 11. Limitations

Cover:

- corpus limited to three asthma guideline files
- guideline date/version conflicts
- extraction limitations in tables/multi-column text
- local model quality and hardware dependence
- deterministic verifier limitations
- no diagnosis or personalized prescribing
- not a clinical decision system

### 12. Attribution

Create THIRD_PARTY_NOTICES.md:

- Identify the reference repository URL and audited commit.
- State which design patterns were adapted.
- Preserve any required copyright/license notice.
- State that the downloaded clone is not included.

Because the reference clone has a README-level MIT declaration but no standalone LICENSE file, confirm reuse permission or ownership before copying implementation text. Independent-assignment rules remain stricter than software licensing.

## Vector-store schema documentation

The submission asks for a database schema. Chroma is the selected vector database, so document:

- collection name
- embedding dimension
- cosine distance
- chunk ID format
- document text field
- every metadata field and type
- corpus fingerprint
- artifact schema version
- rebuild procedure

Place this in README or docs/vector-store-schema.md. Include a sample record with shortened invented text.

## Docker and deployment

The required backend Dockerfile must be documented.

For a local Docker run:

- mount or copy the persisted vector store
- point OLLAMA_BASE_URL to the correct host/service
- expose port 8000
- configure the Streamlit origin

An optional docker-compose.yml may start backend and frontend, but do not add a database or external provider. Ollama can remain a documented host prerequisite if GPU/runtime constraints make containerization impractical.

## GitHub publication

Before the first real commit:

- confirm .gitignore
- remove any secret
- confirm reference/ is ignored
- confirm .venv is ignored
- confirm large model and vector artifacts policy
- confirm raw-PDF redistribution policy

Then:

    git init
    git add .
    git status
    git commit -m "Build asthma guideline RAG assistant"
    git branch -M main
    git remote add origin https://github.com/USERNAME/rag-assistant-app.git
    git push -u origin main

Inspect git status and staged files before every commit. The current environment placeholder named .git is not a functional repository; create a real repository only when ready.

Use small, meaningful commits if time permits:

1. foundation and notebook repair
2. retrieval and evaluation
3. backend and tests
4. frontend
5. documentation and screenshots

## Fresh-clone release gate

Clone into a new directory and follow only README. Verify:

- dependency install
- artifact availability/build
- Ollama model setup
- backend health
- Swagger request
- frontend request
- cited answer
- insufficient-evidence response
- safe refusal
- tests

Record the tested operating system, Python version, Ollama version, and model.

## Presentation plan

Prepare a concise live demo:

1. Show architecture and corpus.
2. Show the notebook evaluation summary.
3. Ask a supported question and open its cited source.
4. Ask a multi-document comparison.
5. Ask an out-of-scope or emergency question to show the safety path.
6. Show Swagger or tests.
7. State limitations honestly.

Prepare fallbacks:

- pre-warm Ollama before the demo
- keep verified screenshots
- keep a short recorded successful flow
- avoid rebuilding the index live
- keep one known-fast question

## Recorded walkthrough

Record:

- project goal
- architecture
- data and notebook
- backend endpoints
- frontend flow
- grounding/citations
- evaluation
- safety behavior
- setup instructions

Do not show private environment files, local usernames, tokens, or unrelated desktop content.

## Final deliverables checklist

- [x] notebooks/rag_pipeline.ipynb runs top to bottom.
- [x] Notebook contains cleaning, chunking, embeddings, retrieval, at least ten evaluated questions, and failure analysis.
- [x] backend/ has required structure, artifacts, .env.example, requirements, Dockerfile, and passing tests.
- [x] GET /health and POST /query work.
- [x] frontend/ has working Streamlit chat and .env.example.
- [x] Persisted vector store loads without request-time rebuild.
- [x] README contains every guide-required section.
- [x] Database/vector metadata schema is documented.
- [x] Evaluation results are local and reproducible.
- [x] Screenshots exist.
- [x] .venv, .env, reference clone, caches, raw PDFs, and model files are excluded.
- [ ] Public GitHub repository is usable from a fresh clone (owner account and push required).
- [x] End-to-end local demo works.
- [ ] Live presentation is rehearsed by the student.
- [ ] Recorded walkthrough is completed by the student.

## Exit gate

- A stranger completes setup using README only.
- GitHub contains no reference clone or secret.
- All guide deliverables are present.
- Final screenshots/results match the final code.
- The presentation and recording demonstrate the same verified build.
