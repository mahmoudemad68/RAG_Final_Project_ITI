import pytest
from app.core.config import Settings
from app.main import create_app
from app.services.generation import GenerationService
from fastapi.testclient import TestClient


class ReadyOllamaClient:
    def show(self, model: str):
        return {"model": model}


def test_health_reports_ready(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "rag_ready": True,
        "collection_name": "test_asthma_guidelines",
        "indexed_chunks": 1,
        "ollama_reachable": True,
        "model": "llama3.2:1b",
        "detail": None,
    }


def test_query_happy_path(client):
    response = client.post("/query", json={"question": "What is MART therapy?"})

    assert response.status_code == 200
    body = response.json()
    assert "MART" in body["answer"]
    assert body["sources"] == ["GINA 2026 Strategy Report, p. 23"]
    assert body["confidence"]["generation_allowed"] is True


def test_missing_question_returns_422(client):
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_blank_question_returns_422(client):
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422


def test_extra_field_returns_422(client):
    response = client.post(
        "/query",
        json={"question": "What is MART?", "unexpected": True},
    )
    assert response.status_code == 422


def test_emergency_refusal_skips_retrieval(client, fake_retrieval):
    response = client.post(
        "/query",
        json={"question": "I cannot breathe and my lips are blue"},
    )

    assert response.status_code == 200
    assert response.json()["risk"]["risk_level"] == "refuse_redirect"
    assert response.json()["confidence"]["generation_allowed"] is False
    assert fake_retrieval.calls == 0


@pytest.mark.parametrize(
    ("question", "apology_prefix"),
    [
        ("What is a dog?", "I'm sorry"),
        ("How are you?", "I'm sorry"),
        ("ما هو الكلب؟", "عذرًا"),
        ("كيف حالك؟", "عذرًا"),
    ],
)
def test_out_of_domain_question_is_blocked_before_retrieval(
    fake_retrieval,
    question,
    apology_prefix,
):
    generation = GenerationService(Settings(), client=ReadyOllamaClient())
    app = create_app(
        settings=Settings(),
        retrieval_service=fake_retrieval,
        generation_service=generation,
    )

    with TestClient(app) as scoped_client:
        response = scoped_client.post("/query", json={"question": question})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith(apology_prefix)
    assert body["risk"] == {
        "risk_level": "refuse_redirect",
        "reason": "outside_asthma_scope",
    }
    assert body["confidence"]["generation_allowed"] is False
    assert body["answer_mode"] == "safety_refusal"
    assert body["generation_attempts"] == 0
    assert body["sources"] == []
    assert fake_retrieval.calls == 0


def test_cors_preflight_for_configured_frontend(client):
    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
