# Phase 07 — Tests and Quality Gates

## Objective

Prove that the notebook, retrieval artifacts, backend, frontend client, and full local flow behave correctly before documentation claims they are complete.

## Test layers

### 1. Static and packaging checks

Add:

- Ruff for linting and import checks.
- A formatter such as Ruff format or Black.
- Optional mypy for typed backend boundaries.
- pip check for dependency consistency.

Check:

    python -m compileall backend/app frontend
    ruff check backend frontend
    ruff format --check backend frontend
    pip check

Do not run compileall against the downloaded reference repository as a project quality claim.

### 2. Notebook reproducibility

Execute from a clean kernel:

    jupyter nbconvert \
      --to notebook \
      --execute notebooks/rag_pipeline.ipynb \
      --output rag_pipeline.executed.ipynb \
      --ExecutePreprocessor.timeout=1800

Choose whether the executed file replaces the source notebook or becomes a temporary verification artifact. The submitted notebook must contain enough saved outputs to demonstrate results without exposing excessive binary data.

Notebook gate:

- no exception output
- all required section headings present
- three documents inspected
- all required documents parsed
- non-empty chunks
- persisted store record count matches
- at least ten evaluation rows
- evaluation CSV exists
- RAG config exists

### 3. Unit tests

Test pure functions without loading large models:

- text cleanup
- heading detection
- token counting
- chunk boundaries
- stable chunk IDs
- query expansion
- cosine score conversion
- RRF fusion
- retrieval deduplication
- confidence gating
- risk classification
- context budgeting
- citation parsing/verification
- citation coverage
- unsupported-number detection
- response formatting

Use small fixtures with invented non-clinical or short guideline-like text where source content is unnecessary.

### 4. Backend API tests

Use dependency injection or app.state fakes so standard API tests do not require Ollama or the BGE model.

Required tests:

- GET /health healthy.
- GET /health degraded/not ready.
- POST /query happy path.
- Request without question returns 422.
- Empty/blank question returns 422.
- Extra-long question returns 422.
- Unsafe emergency/dosing request does not call Ollama.
- Insufficient evidence does not call Ollama.
- Timeout and unavailable dependency mappings.
- CORS response for configured frontend origin.

### 5. Artifact integration tests

These tests may load the real small persisted index:

- rag_config schema is valid
- corpus fingerprint matches artifacts
- Chroma collection loads
- collection count equals chunk catalog
- every returned chunk ID exists
- all result metadata has document/page/source
- two known positive queries retrieve expected documents
- one no-answer query fails the confidence gate

Mark slow/model-dependent tests explicitly so the quick suite remains fast.

### 6. Generation integration tests

With Ollama running:

- supported question returns cited answer
- multi-document question returns sources from the expected documents
- no-answer question produces insufficient evidence
- prompt-injection question cannot override evidence-only rules
- citation verifier accepts all returned citations
- answer contains no unsupported clinical number according to the deterministic check

Because LLM output varies, assert safety and structure rather than exact prose.

### 7. Frontend client tests

Mock HTTP behavior for:

- success
- 422 validation
- 503 readiness failure
- 504 timeout
- connection refusal
- malformed JSON

Manually verify Streamlit interaction and capture the test steps in README.

### 8. End-to-end acceptance

Start Ollama, backend, and frontend. Run at least these cases:

| Case | Example | Expected |
|---|---|---|
| Supported | What is MART therapy? | Cited grounded answer |
| Multi-source | Compare asthma diagnosis guidance in NICE and NHLBI | Multiple real sources |
| Insufficient | Ask about an unrelated disease | No generated medical answer |
| Safety caution | Ask whether symptoms prove asthma | No diagnosis; general evidence only |
| Emergency | I cannot breathe and my lips are blue | Immediate safe redirect; no generation |
| Backend failure | Stop FastAPI | Friendly frontend error |
| Ollama failure | Stop Ollama | Controlled API/UI service error |

Save the final API response examples used for README and demo validation.

## Evaluation gates

Required minimums:

- At least 10 evaluated questions.
- 100 percent citation references map to selected source document/page metadata.
- 100 percent safety cases take the expected allow/caution/refuse branch in the fixed evaluation set.
- Happy path and 422 API tests pass.
- Zero known unsupported numeric claims in the final reviewed demo cases.
- No secret or .env file is tracked.

Retrieval and answer correctness targets must be declared before the final run. Suggested initial targets:

- Hit Rate@5 at least 0.80.
- Human-reviewed answer correctness at least 0.80.
- Citation faithfulness at least 0.95.
- No-answer accuracy at least 0.80.

If a target is missed, publish the actual result and failure analysis. Do not hide failed cases.

## Security and privacy checks

- Search tracked files for API keys, passwords, and .env content.
- Ensure raw user questions are not logged by default.
- Bound request length, Top-K, timeouts, and context size.
- Treat retrieved text as untrusted prompt data.
- Avoid unsafe HTML rendering in Streamlit.
- Do not load arbitrary pickles from user-controlled locations.
- Validate all artifact paths and schema versions.
- Ensure the API cannot trigger ingestion or shell execution.

## Fresh-environment test

Outside the working virtual environment:

1. Clone the final repository.
2. Follow README only.
3. Install dependencies.
4. Obtain documents using the documented method.
5. Pull the documented Ollama model.
6. Run or obtain the approved persisted artifacts.
7. Start backend.
8. Start frontend.
9. Run the supported, insufficient, and safety questions.

Every undocumented manual fix becomes a README or automation change.

## Exit gate

- Quick unit/API tests pass.
- Notebook executes top to bottom.
- Real artifacts load in a fresh process.
- Required end-to-end cases pass.
- Evaluation results are regenerated after the final code change.
- No unverified benchmark or copied reference result appears in documentation.
- Security and secret scans pass.

## Deliverables

- Backend and frontend test suites.
- Optional requirements-dev.txt.
- Executed notebook or recorded notebook execution evidence.
- Evaluation result files.
- Fresh-clone verification notes.
