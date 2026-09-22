# Phase 03 — Embeddings, Persistence, and Retrieval

## Objective

Build a persisted, measurable retrieval layer that the backend can load without reprocessing PDFs or regenerating embeddings at request time.

## Baseline and enhanced design

The required baseline is:

    query
      → BGE-base query embedding
      → Chroma cosine search
      → top-K chunks with provenance

The enhanced local design is:

    query
      ├── dense Chroma search
      ├── persisted BM25 lexical search
      └── reciprocal-rank fusion
            → optional local reranker
            → top evidence chunks

The dense baseline must work before hybrid features are added. No external API or paid reranker is required.

## Tasks

### 1. Generate embeddings deterministically

Use BAAI/bge-base-en-v1.5 with normalized embeddings. The corpus is English,
and this model is strong enough for the domain while remaining reproducible on
the target CPU-only machine. The larger BGE-M3 model exceeded a two-hour cell
limit during the measured build and is therefore not the default. Create
checkpointed embeddings in batches and record:

- exact model identifier
- resolved model revision if available
- vector dimension
- normalization setting
- batch size
- device used
- corpus chunk count
- generation timestamp
- corpus fingerprint

The corpus fingerprint should hash the ordered source hashes plus chunking configuration. It lets the backend reject stale artifacts.

Guard against:

- an empty chunk list
- embedding count not matching chunk count
- NaN or infinite vector values
- a changed model dimension
- a partial write after interruption

Write artifacts to a temporary build directory and move them into the final artifact directory only after validation.

### 2. Persist Chroma correctly

Use:

    backend/data/vector_store/chroma/

Store:

- chunk ID
- chunk text
- scalar provenance metadata
- explicit cosine collection metadata

Use an explicit rebuild flag instead of silently deleting the collection on every notebook run. A normal read/evaluation run should reuse a valid collection. A rebuild should be intentional and should validate the corpus fingerprint.

After persistence, open a new Chroma client and collection in a later cell, query it, and assert that the record count equals the exported chunk count. This tests real persistence rather than the in-memory object.

### 3. Export a versioned RAG configuration

Write backend/data/rag_config.json with at least:

    schema_version
    built_at
    collection_name
    embedding_model
    embedding_dimension
    normalize_embeddings
    chunk_min_tokens
    chunk_max_tokens
    chunk_overlap_tokens
    dense_candidate_k
    lexical_candidate_k
    final_top_k
    fusion_method
    fusion_rrf_k
    score_semantics
    ollama_model
    corpus_fingerprint
    source_hashes

The backend must refuse startup if required fields are absent, the collection is empty, or configured model/dimension differs from the index.

### 4. Implement a measured dense baseline

Return a consistent internal record:

    chunk_id
    text
    metadata
    dense_score
    lexical_score
    fused_score
    rank

For Chroma cosine distance, convert distance to similarity in one well-tested function. Do not apply a threshold copied from the reference repository because score distributions differ by model and index.

Evaluate Top-3, Top-5, and Top-10 against locally verified golden labels.

### 5. Add local lexical retrieval

For a stronger system, add bm25s as a local, persisted lexical index:

- tokenize normalized chunk text
- preserve original chunk IDs
- save the BM25 index beneath backend/data/vector_store/bm25/
- load it once during backend startup
- never rebuild it inside POST /query

If bm25s is not implemented, keep dense retrieval as the required path and document hybrid retrieval as future work. Do not claim hybrid retrieval in README without the persisted lexical artifact and tests.

### 6. Fuse dense and lexical ranks

Use Reciprocal Rank Fusion:

    RRF score = sum of 1 / (rrf_k + rank) across retrieval lists

Do not combine raw cosine and BM25 scores directly because their scales are unrelated.

Deduplicate by chunk_id, preserve both component ranks/scores, and choose final evidence by fused rank. Start with:

- dense candidates: 15–20
- lexical candidates: 15–20
- rrf_k: 60
- final context chunks: 5

Tune these through evaluation.

### 7. Add query expansion cautiously

Adapt the asthma abbreviation patterns from the reference NLPController into a data-driven mapping. Keep the original query and append expansions only for retrieval.

Required terms include:

- MART
- ICS
- SABA
- AIR
- bronchodilator reversibility
- step-up and step-down

Tests must prove:

- matched terms append the expected concepts
- unmatched queries remain unchanged
- expansion never replaces the user's original wording
- expansion text is not presented as retrieved evidence

Avoid broad expansion that forces an expected answer into retrieval.

### 8. Optional local reranking

Add reranking only after hybrid retrieval is measured. If used:

- load the model once
- rerank no more than the fused candidate pool
- make it configurable
- provide a deterministic fallback when unavailable
- measure latency and retrieval improvement

Do not introduce Cohere because the graduation project requires a local Ollama path and should not require external credentials.

## Retrieval evaluation design

Adapt reference/eval/clinical_questions_v2.json into eval/retrieval_questions.json, but manually revalidate every label against the local PDFs.

Include at least:

- ten in-scope questions
- one multi-document comparison
- two out-of-scope or no-answer questions
- abbreviations and paraphrases
- at least one question where lexical matching is useful

Each positive label contains:

- document ID
- acceptable PDF page range
- supporting keywords or passage note

Report:

- Precision@3
- Precision@5
- Hit Rate@3, @5, and @10
- Mean Reciprocal Rank
- no-answer false-positive rate
- average and p95 retrieval latency

Compare:

1. dense baseline
2. dense plus query expansion
3. hybrid RRF, if implemented
4. hybrid plus reranking, if implemented

Select the simplest configuration that materially improves the evaluation. Keep a table of all tested configurations in the notebook.

## Reference reuse

Adapt:

- Query expansion patterns from src/controllers/NLPController.py.
- Deduplication and metadata preservation concepts.
- Evaluation label schema and page-window matching.
- Precision and hit-rate formulas from scripts/eval_retrieval_v2.py.

Do not reuse:

- PGVectorProvider or QdrantDBProvider.
- Cohere embedding or reranking calls.
- Project-specific collection names.
- Reference score thresholds or benchmark numbers.

## Tests

- Unit: distance-to-similarity conversion.
- Unit: query expansion.
- Unit: RRF calculation and stable tie-breaking.
- Unit: duplicate chunk removal.
- Unit: invalid/empty collection handling.
- Integration: persisted Chroma reload.
- Integration: same query produces source IDs present in chunks.jsonl.
- Regression: golden questions meet the chosen local target.

Suggested acceptance targets, subject to honest calibration:

- Hit Rate@5 at least 0.80 on the verified positive set.
- No-answer false-positive rate no more than 0.20.
- Every result has complete citation metadata.
- Warm retrieval p95 below 2 seconds on the documented test machine, excluding first model load.

Targets may be revised with a written reason. Never alter golden labels solely to make metrics pass.

## Exit gate

- A fresh process loads the Chroma collection without rebuilding.
- The collection count matches the chunk catalog.
- rag_config.json accurately describes the index.
- At least ten questions have measured results.
- Top-K choices are based on the comparison table.
- The selected retrieval path returns only real indexed chunk IDs with page provenance.
- Any claimed hybrid or reranked feature has a persisted artifact, test, and metric.

## Deliverables

- backend/data/vector_store/chroma/
- backend/data/vector_store/bm25/ if hybrid retrieval is selected
- backend/data/rag_config.json
- backend/data/chunks.jsonl
- eval/retrieval_questions.json
- notebook retrieval comparison and selected configuration
