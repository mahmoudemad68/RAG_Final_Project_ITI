# Phase 02 — Notebook Ingestion and Chunking

## Objective

Turn the current notebook draft into a readable, reproducible report for loading, inspecting, cleaning, and chunking the exact source corpus.

## Required notebook report structure

Add markdown headings and short explanations for:

1. Project objective and domain.
2. Environment and deterministic configuration.
3. Phase 2.1 — Load and inspect.
4. Phase 2.2 — Cleaning.
5. Phase 2.2 — Chunking strategy.
6. Phase 2.3 — Embeddings and vector store.
7. Phase 2.4 — Retrieval and prompting.
8. Phase 2.5 — Vision component.
9. Phase 2.6 — Evaluation.
10. Phase 2.7 — Export.
11. Limitations and failure analysis.

For Phase 2.5, explicitly state that this is the Core Track and the vision component is not applicable. This proves the requirement was considered rather than forgotten.

## Notebook engineering rules

- Kernel → Restart & Run All must work.
- No cell may depend on variables created manually or out of order.
- Configuration appears once near the top.
- Random seeds are set wherever randomness is used.
- Destructive rebuild behavior is explicit.
- The final notebook retains selected outputs that prove inspection, chunking, retrieval, and evaluation.
- Avoid importing backend application code; the notebook must remain a clear, standalone learning artifact.

## Tasks

### 1. Repair model and import initialization

Uncomment and simplify the embedding initialization. The current code defines EMBEDDING_MODEL_NAME but never creates embedding_model.

Use one of these modes:

- Normal mode loads BAAI/bge-base-en-v1.5 through SentenceTransformer.
- Optional offline mode loads a previously downloaded model from an environment-configured local path.

Never automatically download a model into the repository. Document first-run download size and location.

### 2. Make source discovery manifest-driven

Read data/manifest.json and validate every expected file and hash. Do not treat any arbitrary file dropped into data/raw as trusted input.

Create an inspection table containing:

- document_id
- display name
- format
- expected pages
- extracted pages
- extracted characters
- parse status
- OCR-candidate count
- source URL

Follow it with a markdown paragraph that answers the guide's exact questions: number of documents/pages, formats, failures, OCR needs, and notable extraction problems.

### 3. Preserve provenance through cleaning

Each page record must keep:

- document_id
- document_name
- file_name
- publisher
- source_url
- page_number
- file_sha256

Cleaning may remove:

- blank lines
- repeated running headers and footers
- standalone page-number lines
- known copyright/footer artifacts
- obvious extraction noise

Cleaning must not:

- combine different documents
- lose page numbers
- delete numeric recommendations
- rewrite clinical wording
- silently drop tables

Show before/after samples from each document and report the percentage of text removed. A large removal percentage requires inspection.

### 4. Correct section-aware chunking

Adapt the useful design from reference/RAG_AI_Hackathon/src/controllers/ProccesController.py, but rewrite it for notebook records.

Fix the current defects:

- Process pages in document order so a section can continue across a page boundary.
- Carry page-span metadata, such as page_start and page_end.
- Enforce both minimum and maximum token targets.
- Make overlap units real tokens or rename them accurately as words.
- Avoid merging unrelated adjacent sections simply to reach a minimum.
- Preserve the dominant or starting section title.
- Reject empty chunks.
- Create deterministic IDs from document ID, page span, section, and content hash rather than global list position alone.

Recommended starting configuration:

| Parameter | Starting value | Reason |
|---|---:|---|
| Target minimum | 300–400 tokens | Avoid fragments |
| Maximum | 700–800 tokens | Preserve recommendations without oversized prompts |
| Overlap | 60–100 tokens | Preserve local continuity |

These are hypotheses, not final values. Phase 03 must compare alternatives.

### 5. Add chunk diagnostics

Produce tables or plots for:

- chunks per document
- min, median, p90, and max token count
- chunks below minimum
- chunks above maximum
- empty chunks
- metadata completeness
- page coverage
- section-title coverage
- duplicate content hashes

Adapt the checks from reference/scripts/chunk_diagnostics.py. The notebook must fail on empty chunks, missing document/page metadata, duplicate chunk IDs, or chunks above the hard maximum.

Inspect at least three example chunks from each source, including one table-heavy area.

### 6. Keep notebook and backend artifact schemas aligned

Define the exported metadata contract now:

    chunk_id
    text
    document_id
    document_name
    file_name
    publisher
    source_url
    page_start
    page_end
    section_title
    chunk_order
    content_sha256

If Chroma metadata requires scalar values, keep only scalar forms there and export the full records to chunks.jsonl.

## Tests inside the notebook

Add assertions that:

- all manifest documents were parsed
- parsed page counts are plausible
- all chunks are non-empty
- chunk IDs are unique
- every chunk has document and page metadata
- every chunk stays under the hard maximum
- every source contributes at least one chunk
- chunk ordering is monotonic within each source
- source URLs and hashes survive export

## Reference reuse

Adapt:

- Running header/footer removal.
- Section heading detection.
- Cross-page section extraction.
- Small-chunk merge idea.
- Token diagnostics and required metadata list.

Improve before use:

- The reference overlap tail uses characters in one path; the target must use a documented unit.
- Its small-chunk merge can cross semantic boundaries; add section-aware constraints.
- Its page choice uses a majority page; the target should retain a page range for honest citations.

## Exit gate

- Restart & Run All passes through chunk diagnostics without a manual variable.
- Inspection markdown reports exact local results.
- GINA, NICE, and NHLBI all contribute chunks.
- Every chunk has complete provenance.
- The chosen chunk parameters are justified and later evaluation variants are listed.
- There are no empty, duplicate-ID, or oversized chunks.
- The notebook contains visible outputs proving these statements.

## Deliverables

- Updated notebooks/rag_pipeline.ipynb through the chunking section.
- Exportable chunk catalog in backend/data/chunks.jsonl.
- Chunk statistics in notebook output and optionally backend/data/chunk_diagnostics.csv.
