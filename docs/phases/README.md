# Asthma RAG Graduation Project — Implementation Plan

## Purpose

This directory is the implementation source of truth for completing the Core Track defined in graduation_project_guide.md.

The plan is based on:

- A full audit of the current workspace on 2026-09-21.
- The graduation guide requirements.
- The downloaded reference repository at reference/RAG_AI_Hackathon.
- Reference commit 121b5ad80e26e08f0f40c9bd34e23fad27f09d20.

The target product is a local, evidence-grounded asthma guideline assistant using:

- Python 3.10 or newer.
- A reproducible Jupyter notebook.
- BGE-base English embeddings with resumable batch checkpoints.
- A persisted Chroma vector store.
- Local Ollama generation.
- FastAPI.
- Streamlit.
- Page-level citations and medical safety guardrails.

This is the Core Track. YOLO and image ingestion are not required and must not delay the required text RAG deliverables.

## Current conclusion

The implementation and local acceptance work are complete as of 2026-09-22.
The replacement `notebooks/rag_pipeline.ipynb` executes top-to-bottom, exports
515 chunks and 515 Chroma records, and contains saved retrieval and answer
evaluation outputs. FastAPI, Streamlit, persisted artifacts, tests, screenshots,
and release documentation are present.

Measured results are Hit Rate@5 0.909, MRR 0.556, expected-document context
relevance 0.800, grounded rate 1.000, citation faithfulness 1.000, safety
accuracy 1.000, refusal accuracy 1.000, and zero unsupported numeric claims.
Thirty automated tests pass. See [implementation status](implementation-status.md)
for exact evidence and the owner-controlled submission gates.

## Required execution order

1. [Phase 00 — Current-state audit](00-current-state-audit.md)
2. [Phase 01 — Foundation, scope, and data provenance](01-foundation-scope-and-data.md)
3. [Phase 02 — Notebook ingestion and chunking](02-notebook-ingestion-and-chunking.md)
4. [Phase 03 — Embeddings, persistence, and retrieval](03-embeddings-persistence-and-retrieval.md)
5. [Phase 04 — Grounded generation, safety, and evaluation](04-grounded-generation-safety-and-evaluation.md)
6. [Phase 05 — FastAPI backend](05-fastapi-backend.md)
7. [Phase 06 — Streamlit frontend](06-streamlit-frontend.md)
8. [Phase 07 — Tests and quality gates](07-tests-and-quality-gates.md)
9. [Phase 08 — Documentation, release, and presentation](08-documentation-release-and-presentation.md)

Detailed reference decisions are recorded in [Reference repository reuse map](reference-reuse-map.md).

Each phase must pass its exit gate before the next phase is treated as complete. Retrieval and safety metrics must be measured rather than claimed.

## Target repository layout

    .
    ├── .gitignore
    ├── README.md
    ├── THIRD_PARTY_NOTICES.md
    ├── graduation_project_guide.md
    ├── data/
    │   ├── manifest.json
    │   └── raw/
    ├── notebooks/
    │   └── rag_pipeline.ipynb
    ├── eval/
    │   ├── retrieval_questions.json
    │   ├── answer_cases.json
    │   └── results/
    ├── backend/
    │   ├── app/
    │   │   ├── main.py
    │   │   ├── api/routes/query.py
    │   │   ├── core/config.py
    │   │   ├── schemas/query.py
    │   │   ├── services/retrieval.py
    │   │   ├── services/generation.py
    │   │   └── utils/logging_config.py
    │   ├── data/
    │   │   ├── rag_config.json
    │   │   ├── notebook_evaluation.csv
    │   │   └── vector_store/
    │   ├── tests/test_query.py
    │   ├── requirements.txt
    │   ├── .env.example
    │   └── Dockerfile
    ├── frontend/
    │   ├── app.py
    │   ├── api_client.py
    │   ├── requirements.txt
    │   └── .env.example
    ├── docs/
    │   ├── phases/
    │   └── images/
    └── reference/
        └── RAG_AI_Hackathon/

The reference directory is for local study only. It must be excluded from the final submission repository.

## Architecture contract

    Raw official PDFs
        → parse and inspect
        → clean while preserving page provenance
        → section-aware chunks
        → dense embeddings
        → persisted Chroma
        → dense retrieval plus optional local lexical fusion
        → confidence gate
        → grounded Ollama prompt
        → citation and numeric-claim checks
        → FastAPI response
        → Streamlit answer and sources

Non-negotiable rules:

- The backend loads persisted artifacts during FastAPI lifespan; it never rebuilds the corpus per request.
- The generator receives only selected retrieved evidence and cannot silently answer from general model knowledge.
- Every returned source must map to a real indexed chunk and page.
- Weak or missing evidence produces an explicit insufficient-evidence answer.
- Emergency and personalized prescribing questions are refused or redirected safely.
- The notebook remains the reproducible artifact that creates the vector store and evaluation output.
- Environment-specific values come from environment variables.
- The required interface remains GET /health and POST /query.

## Reuse policy

The reference repository is useful, but it must not be copied wholesale. The graduation guide explicitly requires independent implementation and prohibits copied submissions.

Reuse means:

- Study a pattern.
- Rewrite it for the required Chroma, Ollama, FastAPI, and Streamlit architecture.
- Add tests that prove the rewritten behavior.
- Record attribution in THIRD_PARTY_NOTICES.md when code or substantial logic is adapted.
- Do not commit the cloned repository.

The reference README declares MIT, but the downloaded repository does not contain a standalone LICENSE file. Confirm permission or ownership before copying source text. Academic originality rules still apply even if reuse is legally permitted.

## Definition of done

The project is complete only when all of the following are true:

- notebooks/rag_pipeline.ipynb runs after Kernel → Restart & Run All.
- The notebook reports document/page inspection, chunking rationale, retrieval tests, a ten-or-more-question evaluation table, failure analysis, and export validation.
- The persisted vector store and configuration are produced once and loaded by the backend.
- GET /health and POST /query work and are documented in Swagger.
- The backend tests include a successful query and invalid input returning 422.
- Streamlit shows a loading state, answer, citations, evidence, and friendly failure state.
- The full local question → API → retrieval → Ollama → cited answer flow works.
- README setup instructions succeed in a clean clone.
- Secrets, the virtual environment, the reference clone, logs, large raw files, and large generated artifacts are excluded as appropriate.
- Evaluation results and screenshots are committed.
- The live demo and recorded walkthrough are prepared.
