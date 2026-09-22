import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.schemas.query import RetrievedChunk


class ArtifactConfigurationError(RuntimeError):
    """Raised when persisted RAG artifacts are absent or inconsistent."""


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: int = 1
    collection_name: str
    embedding_model: str
    embedding_dimension: int = Field(gt=0)
    normalize_embeddings: bool = True
    query_embedding_prefix: str = ""
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: str = "dense"
    dense_candidate_k: int = Field(default=20, ge=1, le=100)
    lexical_candidate_k: int = Field(default=20, ge=1, le=100)
    fusion_rrf_k: int = Field(default=60, ge=1, le=1000)
    distance: str = "cosine"
    corpus_fingerprint: str = ""
    min_top_score: float = 0.25
    max_context_chars: int = 12000


QUERY_EXPANSIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\bmart\b", re.IGNORECASE),
        "maintenance and reliever therapy ICS formoterol",
    ),
    (
        re.compile(r"\bair\b", re.IGNORECASE),
        "anti-inflammatory reliever low-dose ICS formoterol",
    ),
    (
        re.compile(r"\bics\b", re.IGNORECASE),
        "inhaled corticosteroid controller preventer",
    ),
    (
        re.compile(r"\bsaba\b", re.IGNORECASE),
        "short-acting beta agonist salbutamol albuterol",
    ),
    (
        re.compile(r"\bdiagnos\w*", re.IGNORECASE),
        "asthma diagnosis spirometry bronchodilator reversibility FeNO",
    ),
    (
        re.compile(r"\bstep(?:ping)?\s*down\b", re.IGNORECASE),
        "decrease maintenance therapy well-controlled asthma",
    ),
    (
        re.compile(r"\breview\w*\b", re.IGNORECASE),
        "monitor asthma control routine review inhaler technique adherence reliever",
    ),
)


def expand_query(question: str) -> str:
    expansions = [
        expansion for pattern, expansion in QUERY_EXPANSIONS if pattern.search(question)
    ]
    if not expansions:
        return question
    return f"{question}\nExpanded retrieval terms: {' '.join(expansions)}"


def cosine_distance_to_similarity(distance: float) -> float:
    return max(-1.0, min(1.0, 1.0 - float(distance)))


def persisted_bm25_chunk_id(value: Any) -> str:
    """Read a chunk ID from bm25s' scalar or persisted corpus representation."""
    if isinstance(value, dict):
        return str(value.get("text") or value.get("id"))
    return str(value)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    rrf_k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class RetrievalService:
    """Loads and queries the notebook-produced Chroma artifacts."""

    def __init__(
        self,
        *,
        config: RagConfig,
        collection: Any,
        embedding_model: Any,
        chunk_catalog: dict[str, dict[str, Any]] | None = None,
        bm25_retriever: Any | None = None,
    ) -> None:
        self.config = config
        self.collection = collection
        self.embedding_model = embedding_model
        self.chunk_catalog = chunk_catalog or {}
        self.bm25_retriever = bm25_retriever
        self.collection_name = config.collection_name
        self.indexed_chunks = int(collection.count())
        if self.indexed_chunks <= 0:
            raise ArtifactConfigurationError("The Chroma collection is empty")

        dimension_getter = getattr(embedding_model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = embedding_model.get_sentence_embedding_dimension
        actual_dimension = int(dimension_getter())
        if actual_dimension != config.embedding_dimension:
            raise ArtifactConfigurationError(
                "Embedding dimension mismatch: "
                f"config={config.embedding_dimension}, model={actual_dimension}"
            )

    @classmethod
    def load(cls, settings: Settings) -> "RetrievalService":
        if not settings.rag_config_path.is_file():
            raise ArtifactConfigurationError(
                f"RAG configuration not found: {settings.rag_config_path}"
            )
        if not settings.vector_store_path.is_dir():
            raise ArtifactConfigurationError(
                f"Vector store not found: {settings.vector_store_path}"
            )

        try:
            raw_config = json.loads(
                settings.rag_config_path.read_text(encoding="utf-8")
            )
            config = RagConfig.model_validate(raw_config)
        except (OSError, ValueError) as exc:
            raise ArtifactConfigurationError(
                f"Invalid RAG configuration: {exc}"
            ) from exc

        import chromadb
        from sentence_transformers import SentenceTransformer

        client = chromadb.PersistentClient(path=str(settings.vector_store_path))
        try:
            collection = client.get_collection(config.collection_name)
        except Exception as exc:
            raise ArtifactConfigurationError(
                f"Chroma collection {config.collection_name!r} is unavailable"
            ) from exc

        embedding_model = SentenceTransformer(config.embedding_model)
        chunk_catalog: dict[str, dict[str, Any]] = {}
        if settings.chunks_path.is_file():
            try:
                for line in settings.chunks_path.read_text(
                    encoding="utf-8"
                ).splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    chunk_catalog[str(record["chunk_id"])] = record
            except (OSError, ValueError, KeyError) as exc:
                raise ArtifactConfigurationError(
                    f"Invalid chunk catalog: {exc}"
                ) from exc

        bm25_retriever = None
        bm25_path = settings.vector_store_path.parent / "bm25"
        if config.retrieval_mode == "hybrid" and bm25_path.is_dir():
            try:
                import bm25s

                bm25_retriever = bm25s.BM25.load(
                    str(bm25_path),
                    load_corpus=True,
                    show_progress=False,
                )
            except Exception as exc:
                raise ArtifactConfigurationError(
                    f"Cannot load persisted BM25 index: {type(exc).__name__}"
                ) from exc

        if config.retrieval_mode == "hybrid" and (
            bm25_retriever is None or not chunk_catalog
        ):
            raise ArtifactConfigurationError(
                "Hybrid retrieval requires the BM25 index and chunks.jsonl"
            )

        return cls(
            config=config,
            collection=collection,
            embedding_model=embedding_model,
            chunk_catalog=chunk_catalog,
            bm25_retriever=bm25_retriever,
        )

    def _dense_retrieve(self, question: str, limit: int) -> list[RetrievedChunk]:
        retrieval_query = self.config.query_embedding_prefix + expand_query(question)
        query_embedding = self.embedding_model.encode(
            [retrieval_query],
            normalize_embeddings=self.config.normalize_embeddings,
        )[0]

        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] or {}
            distance = distances[index]
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=str(documents[index] or ""),
                    metadata=dict(metadata),
                    score=cosine_distance_to_similarity(distance),
                    score_type="dense_cosine",
                    rank=index + 1,
                )
            )
        return chunks

    def _lexical_ids(self, question: str, limit: int) -> list[str]:
        if self.bm25_retriever is None:
            return []
        import bm25s

        query_tokens = bm25s.tokenize(
            expand_query(question),
            show_progress=False,
        )
        results = self.bm25_retriever.retrieve(
            query_tokens,
            k=min(limit, self.indexed_chunks),
            show_progress=False,
        )
        documents = results.documents
        if len(documents) == 0:
            return []
        return [persisted_bm25_chunk_id(value) for value in documents[0]]

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        requested_k = min(
            max(1, top_k or self.config.retrieval_top_k),
            self.indexed_chunks,
        )
        if self.config.retrieval_mode != "hybrid":
            return self._dense_retrieve(question, requested_k)

        dense = self._dense_retrieve(
            question,
            min(self.config.dense_candidate_k, self.indexed_chunks),
        )
        lexical_ids = self._lexical_ids(
            question,
            min(self.config.lexical_candidate_k, self.indexed_chunks),
        )
        dense_by_id = {chunk.chunk_id: chunk for chunk in dense}
        fused = reciprocal_rank_fusion(
            [[chunk.chunk_id for chunk in dense], lexical_ids],
            rrf_k=self.config.fusion_rrf_k,
        )

        results: list[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(fused[:requested_k], start=1):
            dense_chunk = dense_by_id.get(chunk_id)
            if dense_chunk is not None:
                text = dense_chunk.text
                metadata = dense_chunk.metadata
            else:
                record = self.chunk_catalog.get(chunk_id)
                if record is None:
                    continue
                text = str(record.get("text", ""))
                metadata = dict(record.get("metadata", {}))
            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=text,
                    metadata=metadata,
                    score=score,
                    score_type="hybrid_rrf",
                    rank=rank,
                )
            )
        return results

    def status(self) -> dict[str, Any]:
        return {
            "collection_name": self.collection_name,
            "indexed_chunks": self.indexed_chunks,
            "corpus_fingerprint": self.config.corpus_fingerprint,
        }
