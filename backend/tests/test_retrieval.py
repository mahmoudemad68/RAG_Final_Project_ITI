from app.services.retrieval import (
    cosine_distance_to_similarity,
    expand_query,
    persisted_bm25_chunk_id,
    reciprocal_rank_fusion,
)


def test_cosine_distance_conversion_is_bounded():
    assert cosine_distance_to_similarity(0.2) == 0.8
    assert cosine_distance_to_similarity(-1.0) == 1.0
    assert cosine_distance_to_similarity(3.0) == -1.0


def test_query_expansion_keeps_original_question():
    question = "What is MART?"
    expanded = expand_query(question)
    assert expanded.startswith(question)
    assert "maintenance and reliever" in expanded


def test_unmatched_query_is_unchanged():
    question = "Which environmental factors matter?"
    assert expand_query(question) == question


def test_review_expansion_adds_monitoring_terms():
    expanded = expand_query("What should be checked at every asthma review?")
    assert "monitor asthma control" in expanded
    assert expanded.startswith("What should be checked")


def test_arabic_asthma_query_expands_to_english_retrieval_terms():
    expanded = expand_query("كيف يتم تشخيص الربو باستخدام قياس التنفس؟")

    assert "asthma airway disease" in expanded
    assert "asthma diagnosis spirometry" in expanded


def test_persisted_bm25_records_return_their_text_chunk_id():
    assert persisted_bm25_chunk_id("chunk-1") == "chunk-1"
    assert persisted_bm25_chunk_id({"id": 4, "text": "chunk-2"}) == "chunk-2"


def test_rrf_combines_and_deduplicates_ranked_lists():
    fused = reciprocal_rank_fusion(
        [["a", "b", "c"], ["b", "d", "a"]],
        rrf_k=60,
    )
    ids = [item[0] for item in fused]
    assert ids[0] in {"a", "b"}
    assert len(ids) == len(set(ids)) == 4
