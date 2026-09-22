# Vector Store Schema

## Storage

- Engine: Chroma persistent local store
- Collection: asthma_guidelines
- Distance: cosine
- Embedding dimension: 768
- Embedding model: read from backend/data/rag_config.json
- Artifact schema version: 1

The notebook is the only component that builds these artifacts. FastAPI opens
them read-only for retrieval during application startup.

## Chroma record

| Field | Type | Meaning |
|---|---|---|
| id | string | Stable 24-character content-derived chunk ID |
| document | string | Original cleaned chunk text |
| embedding | float vector | Normalized dense embedding |
| document_id | string | Stable manifest document identifier |
| document_name | string | Exact display name used in citations |
| file_name | string | Local source filename |
| publisher | string | Source organization |
| source_url | string | Official publisher URL |
| file_sha256 | string | Source-file content hash |
| page_start | integer | First one-based PDF page represented |
| page_end | integer | Last one-based PDF page represented |
| section_title | string | Detected section heading |
| content_sha256 | string | Full chunk-text hash |
| chunk_order | integer | One-based order inside the source |

## Full chunk catalog

backend/data/chunks.jsonl stores each chunk as:

    {
      "chunk_id": "4e93f3c61f92f38f0f11f810",
      "text": "Short example text...",
      "metadata": {
        "document_id": "nice-ng245-asthma-guideline",
        "document_name": "NICE NG245 Asthma Guideline",
        "page_start": 9,
        "page_end": 10,
        "section_title": "Objective tests"
      }
    }

The real catalog includes every metadata field in the table above.

## BM25

backend/data/vector_store/bm25 contains the bm25s sparse index. Its stored
corpus contains chunk IDs, not copied chunk objects. Retrieval joins IDs back to
chunks.jsonl.

## Configuration and consistency

backend/data/rag_config.json contains:

- collection and model identifiers
- vector dimension and normalization
- chunk sizes and overlap
- dense and lexical candidate counts
- final Top-K and RRF constant
- source hashes
- corpus fingerprint

The corpus fingerprint hashes ordered source hashes, the embedding model, and
the chunking configuration. Changing any of those inputs requires a rebuild.
