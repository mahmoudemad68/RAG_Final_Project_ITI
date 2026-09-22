# Start the Asthma Guideline RAG Project

Run all commands from the project root:

```bash
cd /path/to/rag-assistant-project
```

The first model download needs internet access. The included Chroma, BM25,
chunk-catalog, and evaluation artifacts let the backend start without rebuilding
the notebook.

## Option A — Run everything locally

### 1. Check prerequisites

```bash
python3 --version
git --version
ollama --version
```

Python 3.12 is recommended. Python 3.10 or newer is required.

### 2. Create the Python environment

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt -r frontend/requirements.txt
pip install -r requirements-dev.txt
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt -r frontend/requirements.txt
pip install -r requirements-dev.txt
```

The first install supplies the backend and frontend runtime packages. The
second supplies the notebook, corpus validation, and test tools used later in
this guide.

### 3. Start Ollama and obtain the model

If Ollama already runs as a system service, only pull the model:

```bash
ollama pull llama3.2:1b
ollama list
```

If Ollama is not running, start it in its own terminal:

```bash
ollama serve
```

Then use a second terminal:

```bash
ollama pull llama3.2:1b
```

Verify Ollama:

```bash
curl http://127.0.0.1:11434/api/tags
```

### 4. Configure and start FastAPI

Linux or macOS:

```bash
cp backend/.env.example backend/.env
source .venv/bin/activate
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Windows PowerShell:

```powershell
Copy-Item backend\.env.example backend\.env
.venv\Scripts\Activate.ps1
Set-Location backend
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Keep this terminal open. Model/index startup can take a minute on a CPU-only
machine.

Verify the backend from another terminal:

```bash
curl http://127.0.0.1:8000/health
```

Swagger is available at <http://127.0.0.1:8000/docs>.

### 5. Configure and start Streamlit

Open another terminal at the project root.

Linux or macOS:

```bash
source .venv/bin/activate
cp frontend/.env.example frontend/.env
cd frontend
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
Copy-Item frontend\.env.example frontend\.env
Set-Location frontend
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open <http://127.0.0.1:8501> and ask:

```text
What is MART therapy?
```

CPU-only generation can take several minutes. The configured API timeouts are
intentionally long enough for slow local inference.

### 6. Test the API directly

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is MART therapy?"}'
```

Fast safety-path test:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"I cannot breathe and my lips are blue."}'
```

### 7. Stop the local services

Press `Ctrl+C` in the Streamlit, FastAPI, and manually started Ollama terminals.
If Ollama runs as an operating-system service, it can remain running.

## Option B — Run Ollama, FastAPI, and Streamlit with Docker

This path does not require the local Python virtual environment. Allow at least
8 GB of free disk space for Docker images, Python dependencies, the embedding
model cache, and the Ollama model.

### 1. Create a Docker network and persistent volumes

```bash
docker network create rag-assistant-net
docker volume create rag-ollama-data
docker volume create rag-huggingface-cache
```

If Docker says the network or volumes already exist, continue to the next step.

### 2. Start Ollama in Docker

```bash
docker run -d \
  --name rag-ollama \
  --network rag-assistant-net \
  -p 11434:11434 \
  -v rag-ollama-data:/root/.ollama \
  ollama/ollama:latest
```

Pull the generation model into the Ollama container:

```bash
docker exec rag-ollama ollama pull llama3.2:1b
docker exec rag-ollama ollama list
```

### 3. Build the FastAPI image

```bash
docker build -t asthma-rag-backend:local ./backend
```

Give the non-root backend user ownership of its persistent model cache. This
one-time initialization is also safe to repeat after rebuilding the image:

```bash
docker run --rm \
  --user root \
  -v rag-huggingface-cache:/cache \
  asthma-rag-backend:local \
  chown -R app:app /cache
```

### 4. Start FastAPI in Docker

```bash
docker run -d \
  --name rag-backend \
  --network rag-assistant-net \
  -p 8000:8000 \
  -e OLLAMA_BASE_URL=http://rag-ollama:11434 \
  -e OLLAMA_MODEL=llama3.2:1b \
  -e REQUEST_TIMEOUT_SECONDS=600 \
  -e FRONTEND_ORIGINS=http://localhost:8501,http://127.0.0.1:8501 \
  -e HF_HOME=/tmp/huggingface \
  -v rag-huggingface-cache:/tmp/huggingface \
  asthma-rag-backend:local
```

Watch startup logs:

```bash
docker logs -f rag-backend
```

Press `Ctrl+C` to leave the log view without stopping the container. Verify the
backend:

```bash
curl http://127.0.0.1:8000/health
```

Wait until `rag_ready` and `ollama_reachable` are both `true`.

### 5. Start Streamlit in Docker

The following command uses the official Python image and mounts only the
frontend source. It installs the small frontend dependency set on container
startup:

Linux or macOS:

```bash
docker run -d \
  --name rag-frontend \
  --network rag-assistant-net \
  -p 8501:8501 \
  -e API_BASE_URL=http://rag-backend:8000 \
  -e API_TIMEOUT_SECONDS=660 \
  -v "$(pwd)/frontend:/app" \
  -w /app \
  python:3.12-slim \
  sh -c "pip install --no-cache-dir -r requirements.txt && streamlit run app.py --server.address 0.0.0.0 --server.port 8501"
```

Windows PowerShell:

```powershell
docker run -d `
  --name rag-frontend `
  --network rag-assistant-net `
  -p 8501:8501 `
  -e API_BASE_URL=http://rag-backend:8000 `
  -e API_TIMEOUT_SECONDS=660 `
  -v "${PWD}/frontend:/app" `
  -w /app `
  python:3.12-slim `
  sh -c "pip install --no-cache-dir -r requirements.txt && streamlit run app.py --server.address 0.0.0.0 --server.port 8501"
```

Watch frontend startup:

```bash
docker logs -f rag-frontend
```

Open <http://127.0.0.1:8501> after Streamlit reports that it is ready.

### 6. Inspect the Docker services

```bash
docker ps --filter name=rag-
docker logs --tail 100 rag-ollama
docker logs --tail 100 rag-backend
docker logs --tail 100 rag-frontend
```

### 7. Stop and remove the Docker containers

This keeps the downloaded Ollama and Hugging Face models in named volumes:

```bash
docker stop rag-frontend rag-backend rag-ollama
docker rm rag-frontend rag-backend rag-ollama
docker network rm rag-assistant-net
```

To start again, recreate the network and repeat steps 2, 4, and 5. The named
volumes avoid downloading the models again.

Optional: remove the model caches only when they are no longer needed:

```bash
docker volume rm rag-ollama-data rag-huggingface-cache
```

## Port conflicts

If port 8000 is already used, run FastAPI on 8001.

Local backend:

```bash
cd backend
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

Set the frontend URL before starting Streamlit:

Linux or macOS:

```bash
export API_BASE_URL=http://127.0.0.1:8001
```

Windows PowerShell:

```powershell
$env:API_BASE_URL="http://127.0.0.1:8001"
```

For Docker, change the backend publishing option from `-p 8000:8000` to
`-p 8001:8000`. The frontend container still uses
`API_BASE_URL=http://rag-backend:8000` because containers communicate through
the Docker network.

If port 8501 is already used, change the host side only, for example
`-p 8502:8501`, and add `http://localhost:8502` to `FRONTEND_ORIGINS` for the
backend container.

## Rebuild the RAG index (optional)

The checked-in runtime index is ready to use. Rebuild only after changing the
PDF corpus, embedding model, or chunking configuration.

1. Download the exact PDFs described in `data/raw/README.md`.
2. Put them under `data/raw/` using the documented filenames.
3. Activate `.venv`.
4. Run:

```bash
REBUILD_INDEX=1 .venv/bin/python scripts/execute_notebook.py --timeout 7200
```

Validate the corpus and tests afterward:

```bash
.venv/bin/python scripts/validate_corpus.py
.venv/bin/pytest -q
```

## Common troubleshooting commands

```bash
# Confirm the required Ollama model exists
ollama list

# Check local services
curl http://127.0.0.1:11434/api/tags
curl http://127.0.0.1:8000/health

# Check whether ports are already occupied on Linux
ss -ltnp | grep -E ':8000|:8501|:11434'

# Inspect Docker container state
docker ps -a --filter name=rag-
```

If `/health` reports that the RAG artifacts are unavailable, confirm these
paths exist:

```text
backend/data/rag_config.json
backend/data/chunks.jsonl
backend/data/vector_store/chroma/
backend/data/vector_store/bm25/
```
