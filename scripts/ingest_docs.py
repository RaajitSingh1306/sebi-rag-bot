"""Ingestion pipeline for SEBI, RBI, and DPDPA regulatory documents.
Extracts text from data/*.pdf, chunks text, generates 384-dim embeddings,
and indexes into Qdrant collection 'sebi_rbi_docs'.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import pymupdf as fitz
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "sebi_rbi_docs"
VECTOR_DIM = 384
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def get_qdrant_client() -> QdrantClient:
    """Return QdrantClient instance (Cloud if configured, else local disk store)."""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if url and api_key:
        print(f"Connecting to Qdrant Cloud at {url}")
        return QdrantClient(url=url, api_key=api_key)
    print("QDRANT_URL/KEY not found; using local persistence at ./qdrant_data")
    return QdrantClient(path="./qdrant_data")

def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract pages from a PDF file preserving paragraphs and page numbers."""
    pages = []
    doc = fitz.open(str(pdf_path))
    for page_idx, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({
                "page_num": page_idx + 1,
                "text": text.strip(),
                "source": pdf_path.name
            })
    doc.close()
    return pages

def _word_safe_tail(text: str, chunk_overlap: int) -> str:
    """Return the trailing `chunk_overlap` characters of text, trimmed back to a whole-word
    boundary. Without this, a raw character slice can land mid-word (e.g. cutting
    'Regulation' into 'ulation'), which then gets glued onto the front of the next chunk."""
    if len(text) <= chunk_overlap:
        return text.strip()
    tail = text[-chunk_overlap:]
    space_idx = tail.find(" ")
    if space_idx != -1:
        tail = tail[space_idx + 1:]  # drop the partial word before the first space
    return tail.strip()


def split_text_into_chunks(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[str]:
    """Split text into overlapping chunks using paragraph, sentence, and word boundaries."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    paragraphs = re.split(r'(\n\n+)', text)
    chunks = []
    current_chunk = ""

    for segment in paragraphs:
        if not segment:
            continue
        if len(current_chunk) + len(segment) <= chunk_size:
            current_chunk += segment
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = _word_safe_tail(current_chunk, chunk_overlap)
                current_chunk = (overlap_text + " " + segment.strip()) if overlap_text else segment
            else:
                # If a single segment is too long, split by sentences or space
                words = segment.split()
                temp = ""
                for w in words:
                    if len(temp) + len(w) + 1 <= chunk_size:
                        temp += (" " if temp else "") + w
                    else:
                        if temp:
                            chunks.append(temp.strip())
                            overlap_text = _word_safe_tail(temp, chunk_overlap)
                            temp = (overlap_text + " " + w) if overlap_text else w
                        else:
                            temp = w
                if temp:
                    current_chunk = temp

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) > 20]

def ingest_directory(data_dir: Path = Path("data")):
    """Process all PDFs in data_dir, embed chunks, and upload to Qdrant."""
    pdf_files = sorted(list(data_dir.glob("*.pdf")))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir.resolve()}")

    print(f"Found {len(pdf_files)} PDFs: {[f.name for f in pdf_files]}")

    all_chunks = []
    for pdf_path in pdf_files:
        pages = extract_text_from_pdf(pdf_path)
        doc_chunks = []
        for page in pages:
            chunks = split_text_into_chunks(page["text"])
            for idx, c in enumerate(chunks):
                doc_chunks.append({
                    "text": c,
                    "source": pdf_path.name,
                    "doc_type": "regulation",
                    "page": page["page_num"],
                    "chunk_id": f"{pdf_path.stem}_p{page['page_num']}_c{idx}"
                })
        print(f"{pdf_path.name}: {len(doc_chunks)} chunks extracted")
        all_chunks.extend(doc_chunks)

    print(f"Total chunks across all documents: {len(all_chunks)}")

    print(f"Loading embedding model: {DEFAULT_MODEL} ...")
    embedder = SentenceTransformer(DEFAULT_MODEL)

    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        print(f"Recreating collection '{COLLECTION_NAME}' ...")
        client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
    else:
        print(f"Creating collection '{COLLECTION_NAME}' ...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )

    print("Encoding chunks and uploading to Qdrant ...")
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        texts = [item["text"] for item in batch]
        embeddings = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        points = [
            PointStruct(
                id=i + idx,
                vector=embeddings[idx].tolist(),
                payload={
                    "text": item["text"],
                    "source": item["source"],
                    "doc_type": item["doc_type"],
                    "page": item["page"],
                    "chunk_id": item["chunk_id"]
                }
            )
            for idx, item in enumerate(batch)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Ingested {len(all_chunks)} chunks into '{COLLECTION_NAME}'")
    print(f"Collection '{COLLECTION_NAME}' points count: {info.points_count}")
    return len(all_chunks)

if __name__ == "__main__":
    ingest_directory()
