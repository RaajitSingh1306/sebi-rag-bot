"""Tests for FastAPI endpoints (/health, /query, /eval-summary, /)."""
import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
has_groq_key = bool(os.getenv("GROQ_API_KEY") and not os.getenv("GROQ_API_KEY").startswith("gsk_your_"))

def test_root_endpoint():
    """Verify GET / returns service information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data.get("service") == "SEBI RAG Bot"
    assert "endpoints" in data

def test_health_endpoint():
    """Verify GET /health returns 200 and status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert data.get("version") == "1.0.0"

def test_query_compliance_endpoint():
    """Verify POST /query returns an answer with sources and agent used."""
    payload = {"question": "What is minimum public shareholding under SEBI LODR?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 20
    assert "sources" in data
    assert "agent_used" in data
    assert data["agent_used"] in ["rag_agent", "quant_agent", "__end__"]

def test_query_quant_endpoint():
    """Verify POST /query routes quant question properly."""
    payload = {"question": "What is the current market volatility regime and Sharpe ratio?"}
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data.get("agent_used") in ["quant_agent", "__end__"]

def test_eval_summary_endpoint():
    """Verify GET /eval-summary responds with either report or pending status."""
    response = client.get("/eval-summary")
    assert response.status_code in [200, 404]
    data = response.json()
    assert isinstance(data, dict)

def test_invalid_query_validation():
    """Verify short or missing questions trigger 422 Unprocessable Entity."""
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
