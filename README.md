# Local RAG

Answers questions using only the documentation stored in PostgreSQL (pgvector),
with Ollama running `nomic-embed-text` and `llama3.2:1b` locally.

## 1. Start the project

```bash
cp .env.example .env
docker compose up -d --build
```

## 2. Download the models

```bash
#docker exec -it ollama ollama pull llama3.2:3b
docker exec -it ollama ollama pull nomic-embed-text
```

Models are stored in the `ollama` Docker volume, so this is done only once.

## 3. Add documents

Copy `.txt`, `.md` or `.pdf` files into the `./documents` folder.

## 4. Run the ingestion

```bash
docker compose exec api python ingest.py /documents
```

Re-running replaces the chunks of files that were already ingested.

## 5. Make a request

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the procedure for creating a user?"}'
```

(On Windows PowerShell, use `curl.exe` instead of `curl`.)

## 6. Verify everything is working

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ok","chunks":<number greater than 0>}`.

## Optional: similarity threshold

By default `MIN_SIMILARITY=0` (no threshold). The API logs the similarity of the
retrieved chunks on every question (`docker compose logs api`). Ask a few questions
that the documentation does NOT answer, note their top similarity, then set
`MIN_SIMILARITY` in `.env` slightly above it and run `docker compose up -d`.
If no chunk reaches the threshold, the model is not called.

## Others Commands
```bash
docker compose logs --tail=40 api
docker compose logs --tail=40 ollama
docker compose up -d --force-recreate api
docker compose build 
```