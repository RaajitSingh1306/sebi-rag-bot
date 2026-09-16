"""Tests for HybridRetriever (Dense Qdrant + Sparse BM25 + CrossEncoder / Rerank)."""
import os
import pytest
from backend.retriever import HybridRetriever

@pytest.fixture(scope="module")
def retriever():
    """Fixture initializing HybridRetriever with local qdrant_data."""
    return HybridRetriever()

def test_retrieve_returns_five_results(retriever):
    """Verify that retrieval returns top 5 results by default."""
    results = retriever.retrieve("SEBI minimum public shareholding", k=5)
    assert isinstance(results, list)
    assert len(results) <= 5
    if len(retriever.chunks) >= 5:
        assert len(results) == 5
    for r in results:
        assert "text" in r
        assert "source" in r
        assert "rerank_score" in r

def test_rerank_scores_are_sorted(retriever):
    """Verify results are sorted by rerank_score in descending order."""
    results = retriever.retrieve("SEBI LODR Regulation 38", k=5)
    assert len(results) > 0
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)

def test_regulatory_keyword_retrieval(retriever):
    """Verify keyword retrieval surfaces matching regulatory terms."""
    results = retriever.retrieve("SEBI LODR Regulation 38 public shareholding", k=5)
    assert len(results) > 0
    combined_text = " ".join(r["text"].lower() for r in results)
    # Check that at least one key regulatory concept is retrieved
    assert any(term in combined_text for term in ["sebi", "lodr", "shareholding", "regulation", "percent", "25"])
