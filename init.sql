CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id         BIGSERIAL PRIMARY KEY,
    source     TEXT NOT NULL,
    content    TEXT NOT NULL,
    embedding  vector(768) NOT NULL,   -- nomic-embed-text = 768 dimensions
    metadata   JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Optional, only worth it for large collections:
-- CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);