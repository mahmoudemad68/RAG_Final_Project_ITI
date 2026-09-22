import pytest
import requests

from frontend.api_client import (
    APIConnectionError,
    APIResponseError,
    APITimeoutError,
    ask_question,
)


class FakeResponse:
    def __init__(self, status_code=200, payload=None, json_error=False):
        self.status_code = status_code
        self.ok = 200 <= status_code < 400
        self._payload = payload
        self._json_error = json_error

    def json(self):
        if self._json_error:
            raise ValueError("bad JSON")
        return self._payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def request(self, method, url, json=None, timeout=None):
        self.calls.append(
            {"method": method, "url": url, "json": json, "timeout": timeout}
        )
        if self.error:
            raise self.error
        return self.response


def test_ask_question_success(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://example.test/")
    session = FakeSession(
        FakeResponse(
            payload={
                "answer": "Grounded answer",
                "sources": ["Document, p. 1"],
            }
        )
    )

    result = ask_question("What is asthma?", session=session)

    assert result["answer"] == "Grounded answer"
    assert session.calls[0]["url"] == "http://example.test/query"
    assert session.calls[0]["json"] == {"question": "What is asthma?"}


def test_validation_error_is_friendly():
    session = FakeSession(FakeResponse(status_code=422, payload={"detail": []}))
    with pytest.raises(APIResponseError, match="valid"):
        ask_question("", session=session)


def test_timeout_is_friendly():
    session = FakeSession(error=requests.Timeout())
    with pytest.raises(APITimeoutError, match="too long"):
        ask_question("Question", session=session)


def test_connection_error_is_friendly():
    session = FakeSession(error=requests.ConnectionError())
    with pytest.raises(APIConnectionError, match="not reachable"):
        ask_question("Question", session=session)


def test_invalid_json_is_rejected():
    session = FakeSession(FakeResponse(json_error=True))
    with pytest.raises(APIResponseError, match="invalid response"):
        ask_question("Question", session=session)


def test_missing_answer_is_rejected():
    session = FakeSession(FakeResponse(payload={"sources": []}))
    with pytest.raises(APIResponseError, match="missing answer"):
        ask_question("Question", session=session)
