"""Hybrid Retriever combining Dense Semantic Search (Qdrant Cloud),
Sparse Keyword Search (BM25Okapi), and lightweight Hybrid Scoring.
Optimized for low memory footprint (fastembed / ONNX, no PyTorch required).
"""
import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient

COLLECTION_NAME = "sebi_rbi_docs"
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "true").lower() != "false"

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
        _shared_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
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

        # 2. Embedding model metadata & lazy holder
        self.model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self._embedder = None
        self._embedder_type = None  # "fastembed", "sentence_transformers", or None

        # 3. Load or accept all chunks
        if all_chunks is not None:
            if all_chunks and isinstance(all_chunks[0], str):
                self.chunks = [{"text": text, "source": "unknown", "doc_type": "regulation"} for text in all_chunks]
            else:
                self.chunks = all_chunks
        else:
            try:
                points = self.client.scroll(collection_name=COLLECTION_NAME, limit=10_000)[0]
                self.chunks = [p.payload for p in points if p.payload]
            except Exception as e:
                print(f"Warning: could not scroll chunks from Qdrant: {e}")
                self.chunks = []

        # 4. Setup BM25 index
        if self.chunks:
            tokenized_corpus = [tokenize_for_bm25(c.get("text", "")) for c in self.chunks]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            self.bm25 = None

        self.cross_encoder = None

    def _get_embedder(self):
        """Lazy load lightweight embedder on first demand."""
        if not USE_LOCAL_MODELS:
            return None

        if self._embedder is not None:
            return self._embedder

        # Priority 1: fastembed (ONNX runtime, ~50MB RAM, no PyTorch)
        try:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding(model_name=self.model_name)
            self._embedder_type = "fastembed"
            return self._embedder
        except Exception as e:
            print(f"fastembed not available ({e}), trying sentence-transformers...")

        # Priority 2: sentence-transformers (if installed)
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name)
            self._embedder_type = "sentence_transformers"
            return self._embedder
        except Exception as e:
            print(f"sentence-transformers not available: {e}")

        self._embedder_type = None
        return None

    def encode_text(self, text: str) -> Optional[List[float]]:
        """Encode a single text string into a normalized embedding vector."""
        embedder = self._get_embedder()
        if embedder is None:
            return None

        try:
            if self._embedder_type == "fastembed":
                embeddings = list(embedder.embed([text]))
                vec = np.array(embeddings[0], dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                return vec.tolist()
            elif self._embedder_type == "sentence_transformers":
                vec = embedder.encode(text, normalize_embeddings=True)
                return vec.tolist()
        except Exception as e:
            print(f"Error encoding text: {e}")
        return None

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Execute hybrid search: BM25 + Qdrant Dense -> Deduplicate -> Hybrid Score -> Top-K."""
        if not self.chunks and not self.client:
            return []

        candidates: Dict[str, Dict[str, Any]] = {}
        query_vector = self.encode_text(query)

        # 1. Dense retrieval from Qdrant if query vector is available
        if query_vector is not None:
            try:
                dense_points = []
                if hasattr(self.client, "query_points"):
                    res = self.client.query_points(
                        collection_name=COLLECTION_NAME,
                        query=query_vector,
                        limit=15
                    )
                    dense_points = res.points
                elif hasattr(self.client, "search"):
                    dense_points = self.client.search(
                        collection_name=COLLECTION_NAME,
                        query_vector=query_vector,
                        limit=15
                    )

                for p in dense_points:
                    payload = dict(p.payload or {})
                    text = payload.get("text", "")
                    if not text:
                        continue
                    candidates[text] = {
                        "text": text,
                        "source": payload.get("source", "sebi_rbi_docs"),
                        "page": payload.get("page", 1),
                        "dense_score": float(p.score),
                        "sparse_score": 0.0
                    }
            except Exception as e:
                print(f"Dense Qdrant search failed, falling back: {e}")

        # 2. Sparse retrieval via BM25
        if self.bm25 and self.chunks:
            tokens = tokenize_for_bm25(query)
            bm25_scores = self.bm25.get_scores(tokens)
            top_sparse_idx = np.argsort(bm25_scores)[::-1][:15]
            for idx in top_sparse_idx:
                score = float(bm25_scores[idx])
                if score <= 0 and len(candidates) >= 5:
                    continue
                c = self.chunks[idx]
                text = c.get("text", "")
                if not text:
                    continue
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
        if not candidate_list:
            return []

        # 3. Hybrid scoring (60% dense + 40% sparse)
        max_sparse = max([item["sparse_score"] for item in candidate_list] + [1e-5])
        for item in candidate_list:
            norm_sparse = item["sparse_score"] / max_sparse if max_sparse > 0 else 0.0
            dense_s = item["dense_score"]
            if query_vector is not None and dense_s > 0:
                item["rerank_score"] = float(0.60 * dense_s + 0.40 * norm_sparse)
            else:
                item["rerank_score"] = float(norm_sparse)

        # Sort descending by rerank score
        candidate_list.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidate_list[:k]
