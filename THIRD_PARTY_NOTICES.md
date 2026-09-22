# Third-Party Notices and Implementation References

## RAG_AI_Hackathon

- Repository: https://github.com/mo22zy2/RAG_AI_Hackathon
- Audited commit: 121b5ad80e26e08f0f40c9bd34e23fad27f09d20
- Local audit date: 2026-09-21
- License statement observed: the repository README states MIT
- Standalone LICENSE file observed: no

The repository is used as an implementation reference for:

- section-aware document chunking
- medical-input safety classification
- evidence-confidence gating
- citation verification
- query expansion
- retrieval and answer evaluation design

This project does not depend on or distribute the cloned repository. Adapted
logic is rewritten for a local Chroma and Ollama architecture and receives
project-specific tests. PostgreSQL, pgvector, Qdrant, Cohere, OpenAI, SSE, TTS,
and multi-project upload features from the reference are intentionally excluded.

Before publishing code copied verbatim from the reference, obtain or preserve
the complete applicable license notice. Academic originality requirements apply
independently of software licensing.

## Source documents

The asthma guidelines remain the property of their respective publishers.
They are used locally for educational retrieval experiments. See
data/manifest.json and data/raw/README.md for provenance and acquisition links.
