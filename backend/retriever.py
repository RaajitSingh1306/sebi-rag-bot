"""Hybrid Retriever combining Dense Semantic Search (Qdrant),
Sparse Keyword Search (BM25Okapi), and CrossEncoder / Semantic Reranking.
"""
import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

COLLECTION_NAME = "sebi_rbi_docs"
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric words for BM25 matching."""
    return re.findall(r'[a-zA-Z0-9_%]+', text.lower())

_shared_client: Optional[QdrantClient] = None

def get_shared_client(url: Optional[str] = None, api_key: Optional[str] = None) -> QdrantClient:
    """Return a process-wide shared QdrantClient to avoid local file locking conflicts."""
    global _shared_client
    if _shared_client is not None:
        return _shared_client
    if url and api_key:
        _shared_client = QdrantClient(url=url, api_key=api_key)
    elif os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY"):
        _shared_client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))
    else:
        _shared_client = QdrantClient(path="./qdrant_data")
    return _shared_client

class HybridRetriever:
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        all_chunks: Optional[List[Any]] = None,
        client: Optional[QdrantClient] = None,
        model_name: Optional[str] = None
    ):
        # 1. Setup Qdrant client
        if client is not None:
            self.client = client
        else:
            self.client = get_shared_client(url=url, api_key=api_key)

        # 2. Setup Embedding Model
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self.embedder = SentenceTransformer(self.model_name)

        # 3. Load or accept all chunks
        if all_chunks is not None:
            if all_chunks and isinstance(all_chunks[0], str):
                self.chunks = [{"text": text, "source": "unknown", "doc_type": "regulation"} for text in all_chunks]
            else:
                self.chunks = all_chunks
        else:
            try:
                points = self.client.scroll(collection_name=COLLECTION_NAME, limit=10_000)[0]
                self.chunks = [p.payload for p in points]
            except Exception:
                self.chunks = []

        # 4. Setup BM25 index
        if self.chunks:
            tokenized_corpus = [tokenize_for_bm25(c.get("text", "")) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

        # 5. Setup CrossEncoder / Reranker
        self.cross_encoder = None
        try:
            from sentence_transformers import CrossEncoder
            # Only load if cached or fast
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            self.cross_encoder = None

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Execute hybrid search: BM25 + Qdrant Dense -> Deduplicate -> Rerank -> Top-K."""
        if not self.chunks:
            return []

        candidates: Dict[str, Dict[str, Any]] = {}

        # 1. Dense retrieval from Qdrant
        query_vector = self.embedder.encode(query, normalize_embeddings=True).tolist()
        try:
            dense_results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=15
            )
            for res in dense_results:
                payload = dict(res.payload)
                text = payload.get("text", "")
                candidates[text] = {
                    "text": text,
                    "source": payload.get("source", "sebi_rbi_docs"),
                    "page": payload.get("page", 1),
                    "dense_score": float(res.score),
                    "sparse_score": 0.0
                }
        except Exception:
            # Fallback if search fails or in-memory array search
            corpus_texts = [c.get("text", "") for c in self.chunks]
            doc_embeddings = self.embedder.encode(corpus_texts, normalize_embeddings=True)
            similarities = np.dot(doc_embeddings, query_vector)
            top_dense_idx = np.argsort(similarities)[::-1][:15]
            for idx in top_dense_idx:
                c = self.chunks[idx]
                text = c.get("text", "")
                candidates[text] = {
                    "text": text,
                    "source": c.get("source", "sebi_rbi_docs"),
                    "page": c.get("page", 1),
                    "dense_score": float(similarities[idx]),
                    "sparse_score": 0.0
                }

        # 2. Sparse retrieval via BM25
        if self.bm25:
            tokens = tokenize_for_bm25(query)
            bm25_scores = self.bm25.get_scores(tokens)
            top_sparse_idx = np.argsort(bm25_scores)[::-1][:15]
            for idx in top_sparse_idx:
                score = float(bm25_scores[idx])
                if score <= 0 and len(candidates) >= 5:
                    continue
                c = self.chunks[idx]
                text = c.get("text", "")
                if text in candidates:
                    candidates[text]["sparse_score"] = score
                else:
                    candidates[text] = {
                        "text": text,
                        "source": c.get("source", "sebi_rbi_docs"),
                        "page": c.get("page", 1),
                        "dense_score": 0.0,
                        "sparse_score": score
                    }

        candidate_list = list(candidates.values())[:20]

        # 3. Reranking (CrossEncoder or Semantic Dot-Product + BM25 Fusion)
        if self.cross_encoder is not None:
            try:
                pairs = [(query, item["text"]) for item in candidate_list]
                scores = self.cross_encoder.predict(pairs)
                for item, score in zip(candidate_list, scores):
                    item["rerank_score"] = float(score)
            except Exception:
                self._fallback_score(candidate_list, query_vector)
        else:
            self._fallback_score(candidate_list, query_vector)

        # Sort descending by rerank score
        candidate_list.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidate_list[:k]

    def _fallback_score(self, candidate_list: List[Dict[str, Any]], query_vector: List[float]):
        """Compute normalized hybrid score when CrossEncoder is not active."""
        max_sparse = max([item["sparse_score"] for item in candidate_list] + [1e-5])
        for item in candidate_list:
            norm_sparse = item["sparse_score"] / max_sparse if max_sparse > 0 else 0.0
            dense_s = item["dense_score"]
            if dense_s == 0.0:
                emb = self.embedder.encode(item["text"], normalize_embeddings=True)
                dense_s = float(np.dot(emb, query_vector))
                item["dense_score"] = dense_s
            # Hybrid combined score: 60% semantic + 40% keyword
            item["rerank_score"] = float(0.60 * dense_s + 0.40 * norm_sparse)
