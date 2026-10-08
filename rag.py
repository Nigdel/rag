import os

import psycopg
import requests

DATABASE_URL = os.environ["DATABASE_URL"]
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
GEN_MODEL = os.getenv("GEN_MODEL", "llama3.2:1b")
TOP_K = int(os.getenv("TOP_K", "5"))
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0"))

NOT_FOUND = "I could not find enough information in the documentation to answer this question."

SYSTEM_PROMPT = f"""Answer the user's question using the DOCUMENTATION CONTEXT.

The context is retrieved from the documentation and may contain the answer.
When the context contains relevant instructions or facts, answer directly,
clearly, and step by step if appropriate.

Do not add information that is absent from the context.

Return exactly this sentence only when the context contains no information
related to the question:

{NOT_FOUND}
"""



def embed(texts, prefix):
    """Embed a list of texts with Ollama. prefix: 'search_document: ' or 'search_query: '."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": [prefix + t for t in texts]},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()["embeddings"]


def to_vector(v):
    return "[" + ",".join(map(str, v)) + "]"


def search(question):
    """Return up to TOP_K (source, content, similarity) rows, best first."""
    vec = to_vector(embed([question], "search_query: ")[0])
    with psycopg.connect(DATABASE_URL) as conn:
        rows = conn.execute(
            """SELECT source, content, 1 - (embedding <=> %s::vector) AS similarity
               FROM documents
               ORDER BY embedding <=> %s::vector
               LIMIT %s""",
            (vec, vec, TOP_K),
        ).fetchall()
    print("similarities:", [round(r[2], 3) for r in rows], flush=True)  # helps tune MIN_SIMILARITY
    return [r for r in rows if r[2] >= MIN_SIMILARITY]


def answer(question):
    hits = search(question)
    if not hits:
        return NOT_FOUND

    parts, seen = [], set()
    for source, content, _ in hits:
        if content in seen:  # skip duplicated chunks
            continue
        seen.add(content)
        parts.append(f"[Document: {source}]\n[Chunk {len(parts) + 1}]\n{content}")

    prompt = (
        "=== DOCUMENTATION CONTEXT ===\n\n"
        + "\n\n".join(parts)
        + "\n\n=== END DOCUMENTATION CONTEXT ===\n\n"
        + f"QUESTION:\n{question}\n\n"
        + "Answer using only the CONTEXT above. If it contains the answer, give it."
    )
    GEN_URL = os.getenv("GEN_URL", f"{OLLAMA_URL}/api/chat")
    GEN_MODEL = os.getenv("GEN_MODEL", "llama3.2:3b")
    OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

    headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
    r = requests.post(
        GEN_URL,
        headers=headers,
        json={
            "model": GEN_MODEL,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096},
            "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
        },
        timeout=600,
    )

    r.raise_for_status()
    return r.json()["message"]["content"].strip()