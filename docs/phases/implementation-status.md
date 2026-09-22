# Implementation Status — 2026-09-22

## Outcome

All locally implementable Core Track requirements in
`graduation_project_guide.md` are complete. The notebook, persisted retrieval
artifacts, FastAPI API, Streamlit UI, evaluation fixtures/results, tests,
screenshots, provenance records, and release documentation are present.

## Requirement evidence

| Guide requirement | Evidence | Status |
|---|---|---|
| Parse and inspect a real corpus | 3 official PDFs, 374 parsed pages, 0 failures, 0 OCR candidates | Complete |
| Clean and chunk | 515 section-aware, cross-page chunks with stable IDs and page provenance | Complete |
| Embed and persist | BGE-base-en-v1.5, 768 dimensions, 515-record Chroma store | Complete |
| Retrieval testing | 12 saved cases; dense+BM25 reciprocal-rank fusion | Complete |
| Local generation | Ollama 0.34.2 with llama3.2:1b | Complete |
| Grounded citations | Exact document/page verification, numeric checks, one retry, verified extractive fallback | Complete |
| Evaluation table | 15 answer/safety rows in the executed notebook and CSV | Complete |
| FastAPI | GET /health and POST /query, lifespan loading, CORS, typed schemas | Complete |
| Streamlit | Chat input, loading state, answer, sources, evidence expander, friendly errors | Complete |
| Automated tests | 64 passed; lint, format, compile, and pip checks passed | Complete |
| End-to-end demo | Cold API query and warm browser query returned grounded HTTP 200 responses | Complete |
| Screenshots | Streamlit grounded response and Swagger 200 response in docs/images | Complete |
| GitHub publication | Local release content is ready; owner account/remote/push required | Owner action |
| Human clinical correctness review | Columns are deliberately blank; instructor/qualified human review required | Owner action |
| Live presentation/video | Script exists in Phase 08; recording and delivery require the student | Owner action |

## Measured results

### Retrieval

| Metric | Value |
|---|---:|
| Precision@3 | 0.242 |
| Precision@5 | 0.182 |
| Hit Rate@3 | 0.727 |
| Hit Rate@5 | 0.909 |
| Hit Rate@10 | 0.909 |
| Mean Reciprocal Rank | 0.556 |

### Answer and safety evaluation

| Metric | Value |
|---|---:|
| Cases | 15 |
| Expected-document context relevance | 0.800 |
| Grounded rate | 1.000 |
| Citation faithfulness | 1.000 |
| Safety accuracy | 1.000 |
| Refusal accuracy | 1.000 |
| Unsupported numeric claims | 0 |

The `correct` and `reviewer_notes` fields are not counted as complete clinical
review. Automated citation validity proves provenance, not clinical entailment.

## Failure analysis

- Retrieval missed the NICE gold page for the every-review question.
- The multi-document diagnosis and action-plan cases did not retrieve every
  expected publisher within Top-K.
- The NHLBI exercise case retrieved GINA instead of NHLBI.
- The 1B model often fails the exact citation format. One correction is tried;
  if it still fails, the application selects a query-relevant sentence verbatim
  from evidence and verifies its page citation and numbers.
- Cold CPU-only inference is slow: the measured MART API request took 488.37
  seconds. Prompt caching reduced the browser flow to 156.44 seconds.

## Verification commands

    .venv/bin/python scripts/validate_corpus.py
    .venv/bin/pytest -q
    .venv/bin/ruff format --check backend frontend scripts
    .venv/bin/ruff check backend frontend scripts
    .venv/bin/python -m compileall -q backend frontend scripts
    .venv/bin/pip check

The executed notebook's final cell confirms 3 documents, 374 pages, 515 chunks,
515 Chroma records, 12 retrieval questions, and 15 answer cases.
