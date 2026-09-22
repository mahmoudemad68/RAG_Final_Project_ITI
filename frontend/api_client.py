import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


class APIClientError(RuntimeError):
    """Base error displayed safely by the frontend."""


class APIConnectionError(APIClientError):
    pass


class APITimeoutError(APIClientError):
    pass


class APIResponseError(APIClientError):
    pass


def _settings() -> tuple[str, float]:
    base_url = os.getenv("API_BASE_URL", "http://localhost:8000").strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise APIClientError("API_BASE_URL must be an HTTP(S) URL.")
    try:
        timeout = float(os.getenv("API_TIMEOUT_SECONDS", "660"))
    except ValueError as exc:
        raise APIClientError("API_TIMEOUT_SECONDS must be numeric.") from exc
    if timeout <= 0:
        raise APIClientError("API_TIMEOUT_SECONDS must be positive.")
    return base_url, timeout


def _request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    session: Any = requests,
) -> dict[str, Any]:
    base_url, timeout = _settings()
    try:
        response = session.request(
            method,
            f"{base_url}{path}",
            json=json_body,
            timeout=(5, timeout),
        )
    except requests.Timeout as exc:
        raise APITimeoutError(
            "The assistant took too long to respond. Please try again."
        ) from exc
    except requests.ConnectionError as exc:
        raise APIConnectionError(
            "The assistant service is not reachable. Start the backend and try again."
        ) from exc
    except requests.RequestException as exc:
        raise APIClientError("The request could not be completed.") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise APIResponseError("The backend returned an invalid response.") from exc

    if not response.ok:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if response.status_code == 422:
            raise APIResponseError("Please enter a valid, non-empty question.")
        if response.status_code in {502, 503}:
            raise APIResponseError(
                detail or "The RAG index or local language model is not ready."
            )
        if response.status_code == 504:
            raise APITimeoutError(detail or "The local language model timed out.")
        raise APIResponseError(detail or f"Backend error ({response.status_code}).")

    if not isinstance(payload, dict):
        raise APIResponseError("The backend response has an unexpected shape.")
    return payload


def ask_question(question: str, *, session: Any = requests) -> dict[str, Any]:
    payload = _request(
        "POST",
        "/query",
        json_body={"question": question},
        session=session,
    )
    if not isinstance(payload.get("answer"), str) or not isinstance(
        payload.get("sources"), list
    ):
        raise APIResponseError("The backend response is missing answer or sources.")
    return payload


def get_health(*, session: Any = requests) -> dict[str, Any]:
    return _request("GET", "/health", session=session)
