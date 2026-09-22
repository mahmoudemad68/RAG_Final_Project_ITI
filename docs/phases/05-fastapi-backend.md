# Phase 05 — FastAPI Backend

## Objective

Implement the exact graduation-guide API shape around the persisted RAG artifacts, with one-time startup loading, typed validation, controlled errors, and testable service boundaries.

## Required structure

    backend/
    ├── app/
    │   ├── __init__.py
    │   ├── main.py
    │   ├── api/
    │   │   ├── __init__.py
    │   │   └── routes/query.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   ├── config.py
    │   │   └── safety_rules.json
    │   ├── schemas/
    │   │   ├── __init__.py
    │   │   └── query.py
    │   ├── services/
    │   │   ├── __init__.py
    │   │   ├── retrieval.py
    │   │   └── generation.py
    │   └── utils/
    │       ├── __init__.py
    │       └── logging_config.py
    ├── data/
    │   ├── rag_config.json
    │   ├── chunks.jsonl
    │   └── vector_store/
    ├── tests/
    │   ├── __init__.py
    │   └── test_query.py
    ├── requirements.txt
    ├── .env.example
    └── Dockerfile

## Runtime ownership

### main.py

- Create the FastAPI application.
- Define lifespan.
- Load settings once.
- Configure logging.
- Load and validate the retrieval service once.
- Create the Ollama generation service once.
- Store services in app.state.
- Add environment-driven CORS.
- Include the query router.
- Release resources on shutdown where required.

Do not instantiate SentenceTransformer, open Chroma, or validate the whole corpus in every request.

### core/config.py

Use pydantic-settings and resolve paths relative to the backend directory, not the caller's current working directory.

Validate:

- model names are non-empty
- configured artifacts exist
- retrieval limits are positive and bounded
- frontend origins are parsed as a list
- timeouts are positive

Never hard-code a production origin or secret.

### services/retrieval.py

Own:

- loading rag_config.json
- verifying schema version and corpus fingerprint
- loading the embedding model
- opening persisted Chroma
- optionally loading persisted BM25
- query expansion
- dense or hybrid search
- score conversion and fusion
- deduplication
- retrieval result typing
- evidence confidence calculation

Provide a small interface:

    retrieve(question, top_k=None) -> list[RetrievedChunk]

Loading failures should make readiness false and fail startup with an actionable log.

### services/generation.py

Own:

- risk classification
- evidence selection and context budgeting
- evidence gate
- prompt construction
- Ollama call
- citation verification
- numeric claim verification
- one correction attempt
- safe fallback
- public response construction

Provide:

    answer(question, retrieved_chunks) -> QueryResponse

Keep provider-specific Ollama calls behind a small client method so unit tests can mock it.

## API schemas

### QueryRequest

Required field:

    question: str

Validation:

- strip surrounding whitespace
- reject empty/whitespace-only input
- minimum meaningful length
- maximum length, such as 2,000 characters
- forbid unexpected fields if strictness is desired

### QueryResponse

The two guide-required fields are mandatory:

    answer: str
    sources: list[str]

Each source string should be stable and human-readable, for example:

    GINA 2026 Strategy Report, p. 23

Optional additive fields may include:

- confidence
- risk
- citation_checks
- source_details
- answer_mode
- generation_attempts
- request_id

Do not remove or rename answer and sources.

If source_details is added, define a typed SourceDetail with chunk ID, document, page range, section, score type/value, and source URL. The frontend must still work when only the two required fields are present.

## Endpoints

### GET /health

Return 200 only when the application is alive. Include readiness detail:

    status
    rag_ready
    collection_name
    indexed_chunks
    ollama_reachable
    model

Do not disclose secrets or filesystem paths.

Decide whether temporary Ollama unavailability makes status degraded or returns 503. Document the behavior.

### POST /query

Flow:

1. Pydantic validates the request.
2. Retrieve candidate chunks.
3. Apply safety and confidence logic.
4. Generate only when allowed.
5. Verify the output.
6. Return typed answer and sources.

The endpoint remains thin. It must not contain embedding, prompt, or citation business logic.

Expected errors:

| Condition | HTTP result |
|---|---|
| Invalid or empty payload | 422 |
| RAG artifacts unavailable | 503 |
| Ollama/model unavailable | 503 or 502, documented consistently |
| Ollama timeout | 504 |
| Controlled insufficient evidence | 200 with grounded refusal and sources, normally empty or diagnostic |
| Unexpected internal error | 500 with safe generic detail |

Never return a Python stack trace to the client.

## CORS

Parse FRONTEND_ORIGINS and pass only those origins to CORSMiddleware. For local development:

    http://localhost:8501
    http://127.0.0.1:8501

The reference main.py demonstrates lifespan and CORS setup, but its any-localhost regex is broader than needed. Use an explicit list unless there is a documented reason otherwise.

## Logging

Use structured, concise logs with:

- request ID
- route
- status
- total latency
- retrieval latency
- generation latency
- answer mode and Ollama attempt count
- retrieved/selected chunk counts
- confidence decision
- verification outcome

Medical questions may contain sensitive information. Do not log raw question or full generated answer by default. If debug logging is enabled, redact or hash content and document the risk.

## Dockerfile

Build a small Python image:

- set a non-root user
- install pinned backend requirements
- copy app and required persisted artifacts
- expose 8000
- start Uvicorn without reload
- add a healthcheck if practical

Ollama may run on the host or another container. Make OLLAMA_BASE_URL configurable. Do not install or pull an Ollama model during image build.

## Tests required in this phase

At minimum, use TestClient and mocked services:

1. GET /health returns a usable readiness payload.
2. POST /query happy path returns 200, answer, and source list.
3. Missing question returns 422.
4. Blank question returns 422.
5. Oversized question returns 422.
6. Retrieval service unavailable returns the chosen controlled status.
7. Ollama timeout maps to 504.
8. Insufficient evidence returns a grounded fallback without calling Ollama.

The graduation guide explicitly requires one happy path and one invalid-input 422 test; the list above makes the backend defensible.

## Reference reuse

Adapt:

- Lifespan and app.state patterns from reference/src/main.py.
- Typed request ideas from routes/schemes/nlp.py.
- Separation of retrieval and generation responsibilities from NLPController.
- Evidence/source construction and verification logic.
- Unit-test categories from src/tests/test_core.py.

Do not adapt:

- Database engine creation.
- Project, asset, and chunk database models.
- Upload, process, and index endpoints.
- Cohere/OpenAI factories.
- SSE streaming.

## Commands

From backend:

    python -m pytest -q
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Manual checks:

    curl http://localhost:8000/health

    curl -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"question":"What is MART therapy?"}'

Open:

    http://localhost:8000/docs

## Exit gate

- The required directory structure exists.
- Lifespan loads the persisted index and models once.
- GET /health and POST /query behave as documented.
- The public response always contains answer and sources.
- Happy-path and 422 tests pass.
- No route rebuilds the vector store.
- No external API key is required.
- CORS permits the configured Streamlit origin.
- The Docker image starts when provided valid artifacts and a reachable Ollama service.

## Deliverables

- Complete backend/ tree.
- Passing backend tests.
- Swagger-visible API.
- backend/.env.example.
- backend/requirements.txt.
- backend/Dockerfile.
