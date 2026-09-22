from app.core.config import Settings
from app.schemas.query import RetrievedChunk
from app.services.generation import GenerationService


class FakeOllamaClient:
    def show(self, model: str):
        return {"model": model}


def _service() -> GenerationService:
    return GenerationService(Settings(), client=FakeOllamaClient())


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="chunk-1",
        text="A 12% increase in FEV1 can support diagnosis.",
        metadata={
            "document_id": "nice-ng245-asthma-guideline",
            "document_name": "NICE NG245 Asthma Guideline",
            "page_start": 10,
            "page_end": 11,
            "source_url": "https://www.nice.org.uk/guidance/ng245",
        },
        score=0.8,
        rank=1,
    )


def test_safety_classifier_blocks_emergency():
    risk = _service().classify_risk("I can't breathe and my lips are blue")
    assert risk.risk_level == "refuse_redirect"
    assert risk.reason == "possible_emergency"


def test_safety_classifier_marks_personal_scenario_for_caution():
    risk = _service().classify_risk("My child is wheezing. Does she have asthma?")
    assert risk.risk_level == "needs_caution"


def test_safety_classifier_blocks_personal_inhaler_dose_change():
    risk = _service().classify_risk("Should I increase my inhaler dose tonight?")
    assert risk.risk_level == "refuse_redirect"
    assert risk.reason == "personal_medication_advice"


def test_valid_citation_maps_to_page_range():
    checks = _service().verify_citations(
        "The threshold is described here [NICE NG245 Asthma Guideline, p. 10].",
        [_chunk()],
    )
    assert len(checks) == 1
    assert checks[0].supported is True
    assert checks[0].source_chunk_id == "chunk-1"


def test_invalid_citation_page_is_rejected():
    checks = _service().verify_citations(
        "Unsupported [NICE NG245 Asthma Guideline, p. 99].",
        [_chunk()],
    )
    assert checks[0].supported is False


def test_citation_page_number_is_not_an_unsupported_clinical_number():
    unsupported = _service().unsupported_numbers(
        "The increase is 12% [NICE NG245 Asthma Guideline, p. 10].",
        [_chunk()],
    )
    assert unsupported == []


def test_citation_coverage_does_not_split_inside_page_label():
    answer = (
        "MART uses an ICS-formoterol inhaler for maintenance and relief. "
        "[GINA 2026 Strategy Report, p. 77]"
    )
    assert _service().citation_coverage(answer) == 1.0


def test_citation_coverage_detects_a_second_uncited_sentence():
    answer = (
        "MART uses an ICS-formoterol inhaler. "
        "[GINA 2026 Strategy Report, p. 77] "
        "This second factual sentence has no citation."
    )
    assert _service().citation_coverage(answer) == 0.5


def test_unsupported_clinical_number_is_detected():
    unsupported = _service().unsupported_numbers(
        "The increase is 50% [NICE NG245 Asthma Guideline, p. 10].",
        [_chunk()],
    )
    assert unsupported == ["50%"]


def test_extractive_fallback_is_verbatim_and_verifiably_cited():
    service = _service()
    answer, source = service._extractive_fallback(
        "What increase in FEV1 can support diagnosis?",
        [_chunk()],
    )

    assert source.chunk_id == "chunk-1"
    assert "A 12% increase in FEV1 can support diagnosis." in answer
    checks = service.verify_citations(answer, [source])
    assert len(checks) == 1
    assert checks[0].supported is True
    assert service.unsupported_numbers(answer, [source]) == []


def test_query_aware_excerpt_enforces_first_chunk_limit():
    service = _service()
    noisy = "Unrelated background sentence. " * 80
    relevant = "A 12% increase in FEV1 can support asthma diagnosis."
    chunk = _chunk().model_copy(update={"text": f"{noisy} {relevant}"})

    selected = service._select_evidence(
        [chunk],
        max_context_chars=700,
        question="What FEV1 increase supports asthma diagnosis?",
    )

    assert len(selected) == 1
    assert len(selected[0].text) <= 400
    assert relevant in selected[0].text
