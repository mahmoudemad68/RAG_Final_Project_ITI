# Phase 00 — Current-State Audit

## Objective

Classify every current artifact as usable, needing edits, missing, or reference-only. This phase is an audit; it does not claim that unexecuted notebook code works.

## Workspace inventory

| Artifact | Current state | Decision |
|---|---|---|
| graduation_project_guide.md | Present and readable | Keep unchanged as the requirements source |
| data/raw/GINA_2026_Strategy_Report.pdf | Present, 298 pages, about 17 MB, AES-256 encrypted | Keep, but fix parser dependency and verify extractability |
| data/raw/NICE_NG245_Asthma_Guideline.pdf | Present, 64 pages, text extracts successfully | Keep and add provenance metadata |
| data/raw/NHLBI_Asthma_Care_Quick_Reference.pdf | Present, 12 pages, text extracts successfully | Keep and add provenance metadata |
| notebooks/Rag_pipline.ipynb | Present, 54 cells | Rename and substantially edit |
| backend/ | Empty | Implement |
| frontend/ | Empty | Implement |
| reference/RAG_AI_Hackathon/ | Downloaded successfully | Use as local reference only; never submit the clone |
| Root README.md | Missing | Implement |
| Root .gitignore | Missing | Implement before the first real commit |
| Evaluation fixtures/results | Missing outside notebook | Implement |
| Root Git repository | Not initialized; .git is only an environment placeholder | Initialize during the release phase |

## Data inspection findings

The local corpus contains 374 pages in total:

- GINA: 298 pages.
- NICE: 64 pages.
- NHLBI: 12 pages.

NICE yielded about 119,510 extracted characters and no pages under the notebook's 20-character OCR threshold. NHLBI yielded about 43,740 characters and no low-text pages.

The current virtual environment cannot extract GINA through pypdf because the PDF uses AES encryption and cryptography is missing. The observed error was:

    DependencyError: cryptography>=3.1 is required for AES algorithm

This is a Phase 01 blocker. Add a compatible cryptography package or use a tested PyMuPDF loading path, then record the real GINA extraction and OCR statistics. Do not report 374 successfully parsed pages until this is fixed.

The raw files have these audit hashes:

| File | SHA-256 |
|---|---|
| GINA_2026_Strategy_Report.pdf | 33d0453df2d6d49be525a93ae547f71943e9af4f138ab44b9e5eaa765a04b445 |
| NHLBI_Asthma_Care_Quick_Reference.pdf | a3617538f4634420d7173fd71a4b665ce23417238dc5aade9ce364f007dec29a |
| NICE_NG245_Asthma_Guideline.pdf | 5e0101a680f6c4a74561af802cf5dd0c182283c7f22a27338737e85d37334d1c |

## Notebook findings

### Usable foundations

- Domain and product title are clear: asthma guideline assistant.
- PDF and TXT discovery exists.
- Per-page extraction metadata includes document, file, and page.
- Basic running header/footer cleanup exists.
- Heading detection and section-aware chunking exist.
- A BGE-family encoder and Chroma are appropriate local choices.
- The Chroma collection is configured for cosine distance.
- Query expansion exists for several asthma terms.
- A grounded system prompt exists.
- Risk classification, confidence gating, citation parsing, and unsupported-number checks exist.
- Ten evaluation cases are declared.
- Vector-store and configuration export paths match the intended backend area.

### Blocking defects

1. The required filename is notebooks/rag_pipeline.ipynb; the current name is misspelled as Rag_pipline.ipynb.
2. Only one of 52 code cells has an execution count and only one has output. The notebook is not a demonstrated top-to-bottom run.
3. The embedding-model initialization cell is completely commented out. The next cell calls embedding_model.encode, causing a NameError on a clean run.
4. GINA parsing fails in the current environment because cryptography is missing.
5. The guide requires report-like markdown sections for 2.1 through 2.7. The notebook has only two markdown cells.
6. CHUNK_MIN_TOKENS is configured but never enforced.
7. CHUNK_OVERLAP_TOKENS is passed to a function that slices words, so the configured unit and actual unit disagree.
8. Chunking is performed page by page. A logical section crossing a page boundary cannot remain intact.
9. source_url, publisher, version/date, and file hash are absent from chunk metadata.
10. Chunk IDs depend only on list position. Stable content-derived IDs are safer for reproducible rebuilds.
11. The vector collection is deleted unconditionally on every run. This is acceptable for an explicit rebuild cell, but it must be clearly labeled and guarded.
12. Retrieval is dense-only. There is no measured Top-K comparison, keyword baseline, hybrid fusion, or reranking.
13. MIN_TOP_SCORE is not shown as calibrated against positive and negative evaluation examples.
14. The evaluation uses an expected document plus any expected keyword as a proxy for correctness. It lacks golden page windows, per-rank relevance labels, negative cases, and human answer judgments.
15. No written failure-analysis paragraph is present.
16. The notebook does not prove exported artifacts can be loaded in a fresh process.
17. Some notebook commentary says behavior is adapted from another repository, but the project has no attribution notice.

### Required action

Treat the notebook as a strong draft, not a finished deliverable. Preserve the useful logic while repairing reproducibility, metadata, evaluation, and reporting in Phases 02 through 04.

## Graduation guide gap matrix

| Guide requirement | Status | Work type | Planned phase |
|---|---|---|---|
| Choose a meaningful document domain | Present: asthma guidance | Verify and document | 01 |
| Collect and manually verify documents | Partly present | Edit and record provenance | 01 |
| Notebook at exact required path | Wrong filename | Edit | 01 |
| Load/inspect narrative | Code exists, narrative/results missing | Edit | 02 |
| Cleaning and chunking | Draft exists with correctness gaps | Major edit | 02 |
| Explain chunk size and overlap | Partial | Edit | 02 |
| Embeddings | Broken by undefined model | Fix | 03 |
| Persistent vector store | Draft exists, unverified | Fix and verify | 03 |
| Retrieval tested on at least 10 questions | Cases exist, results absent | Implement and run | 03–04 |
| Grounded prompt and citations | Draft exists | Extract, harden, test | 04 |
| Evaluation table | Generated in code but not executed | Run and preserve outputs | 04 |
| Failure analysis | Missing | Implement | 04 |
| Backend required layout | Missing | Implement | 05 |
| GET /health | Missing | Implement | 05 |
| POST /query | Missing | Implement | 05 |
| Lifespan loading | Missing | Implement | 05 |
| Backend tests | Missing | Implement | 05 and 07 |
| Backend requirements, env example, Dockerfile | Missing | Implement | 05 |
| Streamlit or Gradio chat interface | Missing | Implement | 06 |
| Environment-driven API URL | Missing | Implement | 06 |
| Loading and error states | Missing | Implement | 06 |
| End-to-end verification | Missing | Implement | 07 |
| Root README and architecture diagram | Missing | Implement | 08 |
| Git hygiene and public repository | Missing | Implement | 08 |
| Screenshots, live demo, recorded walkthrough | Missing | Produce | 08 |
| Extended YOLO track | Not selected | Out of scope | Optional after Core Track |

## Guide ambiguities requiring instructor confirmation

The guide itself contains two administrative contradictions:

- The opening table says the deadline is 4 days; the Goal section says 6 days.
- The opening table says the assignment is individual; the Goal section permits individuals or teams of 2–3.

These do not block technical work. Use the stricter four-day, independently authored interpretation until the instructor confirms otherwise.

## Reference repository assessment

Reference location:

    reference/RAG_AI_Hackathon

Pinned audit commit:

    121b5ad80e26e08f0f40c9bd34e23fad27f09d20

### Patterns worth adapting

| Reference area | Useful behavior | Target destination |
|---|---|---|
| src/controllers/ProccesController.py | Cleaning, page normalization, section extraction, small-chunk merging | Notebook helpers, then backend-shared retrieval module where appropriate |
| src/controllers/NLPController.py | Query expansion, evidence gate, prompt budgeting, source construction, citation checks, numeric-claim checks | backend/app/services |
| src/config/safety_config.json | Data-driven medical risk and refusal rules | backend/app/core/safety_rules.json |
| src/stores/templates/locales/en/rag.py | Evidence-only prompt and exact citation format | backend prompt constants |
| src/main.py | FastAPI lifespan and CORS pattern | backend/app/main.py |
| src/tests/test_core.py | Test categories for safety, confidence, citations, query expansion, and claims | backend/tests |
| eval/clinical_questions_v2.json | Golden document/page/keyword retrieval labels | eval/retrieval_questions.json, corrected to local document versions |
| eval/answer_cases_v2.json | Direct, multi-chunk, refusal, ambiguity, and language cases | eval/answer_cases.json |
| scripts/eval_retrieval_v2.py | Precision and hit-rate calculations | Local evaluation helper or notebook section |
| scripts/eval_answers.py | Citation, refusal, and language metrics | Local evaluation helper or notebook section |
| scripts/chunk_diagnostics.py | Metadata completeness and chunk distribution report | Notebook diagnostics |

### Do not import into the required solution

- PostgreSQL, SQLAlchemy, Alembic, and pgvector.
- Qdrant provider abstraction.
- Cohere/OpenAI generation and embedding providers.
- Multi-project upload/process/index APIs.
- SSE streaming.
- TTS.
- The static HTML UI.
- Database migrations and asset/project models.

Those components add operational risk, external credentials, and architecture that the graduation guide does not require. The target must remain runnable with persisted Chroma plus local Ollama.

### Reference inconsistencies to correct

- The reference README says NICE NG80, while the local file is NICE NG245. Do not carry the stale identifier into prompts, labels, or citations.
- Golden page labels must be revalidated against the exact local PDF versions; reference page windows cannot be trusted automatically.
- The reference declares MIT in README but has no standalone LICENSE file in the clone.
- Reference benchmark claims are not results for this workspace. Re-run evaluation locally and publish only locally measured numbers.
- Reference code has its own misspellings and compatibility names. Do not copy names such as ProccesController or ProccessRequest.

## Phase exit gate

This audit is complete when:

- The team agrees to the Core Track scope.
- Each guide item is mapped to a phase.
- Reference reuse is limited to clearly identified patterns.
- The GINA extraction blocker is assigned to Phase 01.
- No benchmark result from the reference repository is presented as a result of this project.
