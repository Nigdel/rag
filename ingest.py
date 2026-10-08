import hashlib
import pathlib
import sys

import psycopg
from psycopg.types.json import Jsonb

import rag

try:
    import pymupdf  # mucho más rápido que pypdf (requiere pymupdf>=1.24.3)
    HAS_PYMUPDF = True
except ImportError:
    from pypdf import PdfReader
    HAS_PYMUPDF = False

EXTENSIONS = {".txt", ".md", ".pdf"}
CHUNK_SIZE = 800    # caracteres
OVERLAP = 50        # antes 100: ~12% menos texto a embeber
MIN_CHUNK = 40      # descarta chunks muy cortos (encabezados, números de página)
BATCH = 16          # chunks por petición de embeddings


def read_text(path):
    if path.suffix.lower() == ".pdf":
        if HAS_PYMUPDF:
            with pymupdf.open(path) as doc:
                return "\n".join(page.get_text() for page in doc)
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text):
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            cut = text.rfind("\n", start, end)  # preferir cortar al final de línea
            if cut > start + CHUNK_SIZE // 2:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = end - OVERLAP
    return chunks


def clean_chunks(chunks):
    """Quita chunks demasiado cortos y duplicados exactos (conserva el orden)."""
    seen, result = set(), []
    for c in chunks:
        if len(c) < MIN_CHUNK or c in seen:
            continue
        seen.add(c)
        result.append(c)
    return result


def file_hash(path):
    # Incluye los parámetros de chunking: si los cambias, se reingesta el archivo.
    h = hashlib.sha256(f"{CHUNK_SIZE}:{OVERLAP}:{MIN_CHUNK}".encode())
    h.update(path.read_bytes())
    return h.hexdigest()


def main(folder):
    folder = pathlib.Path(folder)
    files = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in EXTENSIONS]
    if not files:
        sys.exit(f"No .txt, .md or .pdf files found in {folder}")

    with psycopg.connect(rag.DATABASE_URL) as conn:
        for path in files:
            source = str(path.relative_to(folder))
            h = file_hash(path)

            unchanged = conn.execute(
                "SELECT 1 FROM documents WHERE source = %s AND metadata->>'hash' = %s LIMIT 1",
                (source, h),
            ).fetchone()
            if unchanged:
                print(f"{source}: sin cambios, omitido")
                continue

            chunks = clean_chunks(chunk(read_text(path)))
            if not chunks:
                print(f"{source}: no text extracted, skipped")
                continue

            vectors = []
            for i in range(0, len(chunks), BATCH):
                vectors += rag.embed(chunks[i:i + BATCH], "search_document: ")

            conn.execute("DELETE FROM documents WHERE source = %s", (source,))
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO documents (source, content, embedding, metadata)
                       VALUES (%s, %s, %s::vector, %s)""",
                    [
                        (source, text, rag.to_vector(vec), Jsonb({"chunk": n, "hash": h}))
                        for n, (text, vec) in enumerate(zip(chunks, vectors), start=1)
                    ],
                )
            conn.commit()
            print(f"{source}: {len(chunks)} chunks")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python ingest.py <folder>")
    main(sys.argv[1])