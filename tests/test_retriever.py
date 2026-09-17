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

def test_word_safe_tail():
    """Verify that _word_safe_tail never cuts words mid-word."""
    from scripts.ingest_docs import _word_safe_tail
    text = "Regulation 38: Minimum Public Shareholding"
    # Slicing at 20 chars lands inside 'Minimum' -> should trim cleanly to 'Public Shareholding'
    tail = _word_safe_tail(text, 20)
    assert not tail.startswith("inimum")
    assert tail == "Public Shareholding"

def test_split_text_into_chunks_no_broken_words():
    """Verify chunking splits cleanly without mid-word fragments."""
    from scripts.ingest_docs import split_text_into_chunks
    text = "Regulation 38 specifies Minimum Public Shareholding. Every listed company shall maintain at least 25 percent public shareholding."
    chunks = split_text_into_chunks(text, chunk_size=60, chunk_overlap=25)
    assert len(chunks) >= 2
    for c in chunks:
        # No chunk should start with a partial word fragment like 'ulation' or 'inimum'
        words = c.split()
        assert words[0] not in ["ulation", "inimum", "pecifies", "hareholding"]

