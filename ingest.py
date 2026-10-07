import pathlib
import sys

import psycopg
from psycopg.types.json import Jsonb
from pypdf import PdfReader

import rag

EXTENSIONS = {".txt", ".md", ".pdf"}
CHUNK_SIZE = 800   # characters
OVERLAP = 100


def read_text(path):
    if path.suffix.lower() == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text):
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            cut = text.rfind("\n", start, end)  # prefer breaking at a line end
            if cut > start + CHUNK_SIZE // 2:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - OVERLAP
    return chunks


def main(folder):
    folder = pathlib.Path(folder)
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in EXTENSIONS]
    if not files:
        sys.exit(f"No .txt, .md or .pdf files found in {folder}")

    with psycopg.connect(rag.DATABASE_URL) as conn:
        for path in files:
            source = str(path.relative_to(folder))
            chunks = chunk(read_text(path))
            if not chunks:
                print(f"{source}: no text extracted, skipped")
                continue

            vectors = []
            for i in range(0, len(chunks), 16):
                vectors += rag.embed(chunks[i:i + 16], "search_document: ")

            conn.execute("DELETE FROM documents WHERE source = %s", (source,))
            for n, (text, vec) in enumerate(zip(chunks, vectors), start=1):
                conn.execute(
                    """INSERT INTO documents (source, content, embedding, metadata)
                       VALUES (%s, %s, %s::vector, %s)""",
                    (source, text, rag.to_vector(vec), Jsonb({"chunk": n})),
                )
            conn.commit()
            print(f"{source}: {len(chunks)} chunks")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python ingest.py <folder>")
    main(sys.argv[1])