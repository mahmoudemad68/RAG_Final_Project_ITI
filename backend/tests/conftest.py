from types import SimpleNamespace

import pytest
from app.core.config import Settings
from app.main import create_app
from app.schemas.query import (
    ConfidenceInfo,
    QueryResponse,
    RetrievedChunk,
    RiskAssessment,
)
from fastapi.testclient import TestClient


class FakeRetrievalService:
    collection_name = "test_asthma_guidelines"
    indexed_chunks = 1
    config = SimpleNamespace(min_top_score=0.25, max_context_chars=12000)

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, question: str, top_k: int | None = None):
        self.calls += 1
        return [
            RetrievedChunk(
                chunk_id="chunk-test",
                text="MART uses ICS-formoterol as maintenance and reliever therapy.",
                metadata={
                    "document_id": "gina-2026-strategy-report",
                    "document_name": "GINA 2026 Strategy Report",
                    "page_start": 23,
                    "page_end": 23,
                    "section_title": "Treatment",
                    "source_url": "https://ginasthma.org/",
                },
                score=0.8,
                rank=1,
            )
        ]


class FakeGenerationService:
    def __init__(self) -> None:
        self.answer_calls = 0

    def check_ready(self) -> bool:
        return True

    def classify_risk(self, question: str) -> RiskAssessment:
        if "cannot breathe" in question.lower():
            return RiskAssessment(
                risk_level="refuse_redirect",
                reason="possible_emergency",
            )
        return RiskAssessment(
            risk_level="allowed",
            reason="within_asthma_guideline_scope",
        )

    def answer(self, question: str, chunks, **kwargs) -> QueryResponse:
        self.answer_calls += 1
        if "cannot breathe" in question.lower():
            return QueryResponse(
                answer="Seek immediate professional emergency care.",
                sources=[],
                risk=self.classify_risk(question),
                confidence=ConfidenceInfo(
                    generation_allowed=False,
                    confidence_level="blocked",
                    reason="blocked_by_safety_classifier",
                ),
            )
        return QueryResponse(
            answer=(
                "MART combines maintenance and reliever treatment "
                "[GINA 2026 Strategy Report, p. 23]."
            ),
            sources=["GINA 2026 Strategy Report, p. 23"],
            risk=self.classify_risk(question),
            confidence=ConfidenceInfo(
                generation_allowed=True,
                confidence_level="high",
                top_score=0.8,
                score_type="dense_cosine",
                evidence_count=1,
                reason="official_evidence_available",
            ),
        )


@pytest.fixture
def fake_retrieval():
    return FakeRetrievalService()


@pytest.fixture
def fake_generation():
    return FakeGenerationService()


@pytest.fixture
def client(fake_retrieval, fake_generation):
    settings = Settings()
    app = create_app(
        settings=settings,
        retrieval_service=fake_retrieval,
        generation_service=fake_generation,
    )
    with TestClient(app) as test_client:
        yield test_client
