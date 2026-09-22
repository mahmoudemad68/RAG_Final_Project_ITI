from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)

    @field_validator("question", mode="before")
    @classmethod
    def strip_question(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: dict
    score: float
    score_type: str = "dense_cosine"
    rank: int


class RiskAssessment(BaseModel):
    risk_level: Literal["allowed", "needs_caution", "refuse_redirect"]
    reason: str


class ConfidenceInfo(BaseModel):
    generation_allowed: bool
    confidence_level: Literal["high", "medium", "low", "insufficient", "blocked"]
    top_score: float | None = None
    score_type: str | None = None
    evidence_count: int = 0
    reason: str


class CitationCheck(BaseModel):
    citation: str
    document: str
    page_number: int
    supported: bool
    source_chunk_id: str | None = None


class SourceDetail(BaseModel):
    chunk_id: str
    document: str
    page_start: int
    page_end: int
    section: str = ""
    score: float
    score_type: str
    source_url: str = ""
    text_preview: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    source_details: list[SourceDetail] = Field(default_factory=list)
    risk: RiskAssessment
    confidence: ConfidenceInfo
    citation_checks: list[CitationCheck] = Field(default_factory=list)
    citation_faithfulness: float = 1.0
    citation_coverage: float = 1.0
    unsupported_numbers: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    rag_ready: bool
    collection_name: str | None = None
    indexed_chunks: int = 0
    ollama_reachable: bool
    model: str
    detail: str | None = None
