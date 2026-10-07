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

SYSTEM_PROMPT = f"""You answer questions using ONLY the DOCUMENTATION CONTEXT provided by the user.

If the CONTEXT contains information that answers the question, answer using that information.

Rules:
- Answer exclusively using the information contained in the provided CONTEXT.
- Do not use knowledge acquired during your training.
- Do not invent information.
- Do not make assumptions.
- Do not complete missing information.
- Do not make inferences that are not explicitly supported by the CONTEXT.
- Do not mention information that does not explicitly appear in the CONTEXT.
- If only part of the question can be answered using the CONTEXT, answer only that part and clearly indicate which information is not available in the documentation.
- Only if the CONTEXT has nothing relevant to the question, answer exactly: "{NOT_FOUND}" """


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

    r = requests.post(
        f"{OLLAMA_URL}/api/chat",
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