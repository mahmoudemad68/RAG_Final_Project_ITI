# Phase 06 — Streamlit Frontend

## Objective

Provide a simple chat-style UI that calls the FastAPI backend, displays grounded answers and citations, and handles waiting and failures clearly.

## Required structure

    frontend/
    ├── app.py
    ├── api_client.py
    ├── requirements.txt
    └── .env.example

Use Streamlit because it provides the required chat experience with minimal code and no Node build pipeline.

## api_client.py

Implement a focused API wrapper:

    ask_question(question: str) -> dict
    get_health() -> dict

Requirements:

- Read API_BASE_URL from the environment.
- Normalize trailing slashes.
- Use explicit connect/read timeouts.
- Send the exact QueryRequest payload.
- Validate status and response shape.
- Convert connection, timeout, validation, and server failures into small typed or well-defined client exceptions.
- Never hard-code http://localhost:8000 inside request functions.

The .env.example may contain the local default; frontend/.env must be ignored.

## app.py

### Required UI

- Page title and short asthma-guideline scope statement.
- Visible educational/medical disclaimer.
- Chat history in Streamlit session state.
- st.chat_input for the question.
- User and assistant message rendering.
- Spinner or status text while the API request runs.
- Answer text.
- Clearly separated cited sources.
- Friendly error messages.
- Optional expandable evidence details when source_details are returned.

### Source presentation

Show sources beneath the answer as a list. Never invent a source on the frontend. Render only backend-returned source strings/details.

If source URLs are available and belong to the manifest, make them clickable. Treat all returned values as untrusted text and do not enable unsafe HTML.

### Failure states

Handle:

- backend offline
- backend not ready
- Ollama offline or model absent
- request timeout
- validation error
- malformed response
- unexpected server error

Examples of friendly messages:

- The assistant service is not reachable. Start the FastAPI backend and try again.
- The local language model is not ready. Check Ollama and the configured model.
- The request took too long. Try a shorter question or retry after the model has warmed up.

Do not expose raw stack traces.

### Session behavior

Store displayed turns in session state. The required API can remain stateless. Do not send prior answers back to the model unless conversation history becomes a documented, tested API feature.

Provide a Clear conversation button that clears only UI state.

## Optional quality features

Add these only after the required flow works:

- example questions
- response confidence badge
- retrieval score/evidence expander
- copy answer button
- health status indicator
- feedback control stored locally

Do not add image upload because this plan selects the Core Track.

## Tests

Test api_client independently with mocked HTTP responses:

- successful query
- validation response
- timeout
- connection error
- malformed JSON

For the Streamlit layer, at minimum perform a manual smoke test. If time permits, use Streamlit's app testing utilities for initial render and a mocked successful question.

## Run and verify

Terminal 1:

    cd backend
    uvicorn app.main:app --reload --port 8000

Terminal 2:

    cd frontend
    streamlit run app.py --server.port 8501

Verify:

1. Open http://localhost:8501.
2. Ask a real question supported by the corpus.
3. Observe a loading state.
4. Confirm the answer has page-level citations.
5. Confirm the sources shown match the API response.
6. Stop the backend and confirm the UI shows a friendly error.
7. Restart the backend and confirm recovery.

## Reference reuse

The reference repository's static HTML page and SSE path are not suitable for this guide. Reuse only response-display concepts such as an evidence panel. Build the Streamlit frontend specifically for the target POST /query contract.

## Exit gate

- API_BASE_URL is read from the environment.
- The page behaves like a chat.
- Loading, success, and failure states are visible.
- Answers and sources are both shown.
- The medical disclaimer is visible.
- A backend outage does not crash the page.
- The complete browser → FastAPI → retrieval → Ollama → cited answer flow works.

## Deliverables

- frontend/app.py.
- frontend/api_client.py.
- frontend/requirements.txt.
- frontend/.env.example.
- Screenshot-ready working UI.
