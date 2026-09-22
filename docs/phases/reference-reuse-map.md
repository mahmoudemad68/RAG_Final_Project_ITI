# Reference Repository Reuse Map

## Reference identity

- URL: https://github.com/mo22zy2/RAG_AI_Hackathon.git
- Local clone: reference/RAG_AI_Hackathon
- Audited commit: 121b5ad80e26e08f0f40c9bd34e23fad27f09d20
- Commit date: 2026-08-26
- README license declaration: MIT
- Standalone license file in clone: not found

The clone is a study input, not a dependency and not a submission artifact.

## Adaptation rules

For every reused pattern:

1. Confirm it is relevant to the graduation guide.
2. Rewrite it against the target's own data models and filenames.
3. Remove PostgreSQL, external provider, and multi-project assumptions.
4. Add focused tests.
5. Verify behavior against the exact local PDFs.
6. Record material adaptation in THIRD_PARTY_NOTICES.md.

Do not copy benchmark values, golden labels, prompts, or implementation text without review. The guide prohibits copied submissions, regardless of license.

## Detailed mapping

### Ingestion and chunking

| Reference symbol | Decision | Target use | Required correction |
|---|---|---|---|
| ProcessController._normalize_metadata | Adapt | Normalize page and file metadata in notebook | Add manifest ID, hash, source URL, publisher, and page range |
| ProcessController._clean_pages | Adapt | Remove running headers/footers | Add removal diagnostics and protect numeric/table content |
| ProcessController._compile_patterns | Adapt | Compile optional cleanup/heading rules | Treat invalid regex as a visible configuration error |
| ProcessController.process_section_splitter | Adapt | Main chunking strategy | Use exact target record types |
| ProcessController._extract_sections | Adapt | Maintain section context across pages | Keep page start/end and document boundaries |
| ProcessController._chunk_sections | Adapt | Token-budget chunks | Make overlap unit explicit and deterministic |
| ProcessController._merge_small_chunks | Adapt carefully | Reduce fragments | Never merge across unrelated section/document boundaries |
| ProcessController._detect_heading | Adapt | Heading candidates | Test against tables and short uppercase labels |
| ProcessController._estimate_tokens | Replace/improve | Token sizing | Use model tokenizer if practical; otherwise name approximation honestly |
| ProcessController.process_semantic_splitter | Defer | Optional experiment | Not required for the Core Track |

### Retrieval

| Reference symbol | Decision | Target use | Required correction |
|---|---|---|---|
| NLPController.expand_query | Adapt | Asthma abbreviation expansion | Move terms to target-owned config and test false positives |
| NLPController._dedupe_documents | Adapt | Merge retrieval candidates | Deduplicate by stable chunk ID |
| NLPController.search_vector_db_collection | Study only | Retrieval orchestration | Reimplement with Chroma and optional local BM25 |
| PGVectorProvider.search_by_vector | Reject implementation | Dense search concept only | Chroma supplies dense retrieval |
| PGVectorProvider.search_by_keyword | Reject implementation | Lexical baseline concept only | Use a local persisted BM25 index |
| PGVectorProvider.search_hybrid | Study formula | Reciprocal-rank fusion | Reimplement provider-neutral pure function |
| CohereRerankProvider | Reject | None | No external API dependency |
| QdrantDBProvider | Reject | None | Guide-selected store is persisted Chroma |

### Grounding and safety

| Reference symbol | Decision | Target use | Required correction |
|---|---|---|---|
| NLPController.classify_input_risk | Adapt | Allow/caution/refuse classifier | Simplify rules and verify English/Arabic claims |
| safety_config.json risk_rules | Adapt | Data-driven safety configuration | Own the wording and reduce false positives |
| NLPController._build_refusal_answer | Adapt | Deterministic refusals | Use careful medical language and local emergency wording |
| NLPController._build_confidence | Adapt | Pre-generation evidence gate | Calibrate against target score semantics |
| NLPController._has_official_evidence_metadata | Adapt | Provenance eligibility | Validate by manifest document ID, not filename keywords alone |
| NLPController._select_documents_for_prompt | Adapt | Context budget | Budget by tokens if possible and preserve complete chunks |
| NLPController._render_document_prompt | Adapt | Evidence blocks | Add exact page range and chunk ID |
| NLPController._build_sources | Adapt | Public source list/details | Match QueryResponse contract |
| NLPController.build_evidence_panel | Optional adapt | Streamlit expander | Keep API's required sources list |
| NLPController.verify_citations | Adapt and test | Citation mapping | Support exact target names and page ranges |
| NLPController.build_answer_quality | Adapt | Evaluation metrics | Add citation coverage and human correctness |
| NLPController.detect_unsupported_claims | Adapt carefully | Post-generation verification | Strip citation page numbers and document known limitations |
| NLPController._numeric_values | Improve | Numeric claim checks | Normalize ranges, percentages, units, durations, and frequencies |
| NLPController._is_numeric_or_dosing_query | Adapt | Stronger evidence gate | Test false positives |
| stores/templates/locales/en/rag.py | Adapt | Evidence-only prompt | Rewrite prompt for Ollama and target schema |

### API and runtime

| Reference area | Decision | Target use | Required correction |
|---|---|---|---|
| src/main.py lifespan | Adapt | One-time service loading | Remove database/provider factories |
| src/main.py CORS | Adapt carefully | Frontend origin support | Prefer an explicit environment list |
| routes/schemes/nlp.py | Study | Pydantic conventions | Expose guide-required question, answer, and sources names |
| routes/nlp.py | Reject structure | None | Implement only /health and /query |
| LLMProviderFactory | Reject | None | Use a small Ollama client/service |
| VectorDBProviderFactory | Reject | None | Use a concrete Chroma retrieval service |
| database models/migrations | Reject | None | Persisted Chroma metadata schema is sufficient |
| answer streaming/SSE | Defer | Optional future work | Not required |
| TTS provider | Reject | None | Not required |

### Evaluation and tests

| Reference artifact | Decision | Target use | Required correction |
|---|---|---|---|
| eval/clinical_questions_v2.json | Adapt labels | Retrieval fixture | Revalidate every page against local GINA/NICE NG245/NHLBI |
| eval/answer_cases_v2.json | Adapt categories | Answer/safety fixture | Keep only cases supported by local corpus and runtime |
| scripts/eval_retrieval_v2.py | Adapt formulas | Precision, hit rate, MRR | Target POST /query or direct retrieval service |
| scripts/eval_answers.py | Adapt reporting | Safety, citation, language checks | Add human correctness and citation coverage |
| scripts/chunk_diagnostics.py | Adapt | Notebook chunk report | Read notebook records rather than PostgreSQL |
| src/tests/test_core.py safety tests | Adapt | Backend unit tests | Rewrite around target services |
| src/tests/test_core.py citation tests | Adapt | Verifier tests | Add page ranges and citation-number exclusion |
| src/tests/test_core.py provider/SQL tests | Reject | None | Providers and SQL are not in target architecture |

## Known traps

- The reference identifies NICE as NG80 in places; the local source is NG245.
- Reference page windows may not match the exact local PDF editions.
- Reference score thresholds were calibrated for different retrieval stacks.
- Reference claims of 59 passing tests and published benchmark values do not prove target behavior.
- The reference's architecture depends on external providers and PostgreSQL; importing it wholesale would violate target simplicity.
- Filename-keyword checks are weaker than a manifest-bound provenance check.
- Citation validity does not automatically prove semantic claim support.

## Completion record

When implementation is complete, extend this document with:

- target commit containing each adaptation
- tests covering it
- material deviations
- attribution decision
- permission/license confirmation
