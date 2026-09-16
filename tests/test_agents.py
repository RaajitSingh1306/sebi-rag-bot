"""Tests for LangGraph multi-agent system (supervisor, rag_agent, quant_agent)."""
import os
import pytest
from langchain_core.messages import HumanMessage
from backend.agents import build_graph, supervisor, AgentState

has_groq_key = bool(os.getenv("GROQ_API_KEY") and not os.getenv("GROQ_API_KEY").startswith("gsk_your_"))

def test_build_graph_compiles():
    """Verify that the LangGraph StateGraph builds and compiles without errors."""
    graph = build_graph()
    assert graph is not None

def test_supervisor_routes_regulatory_query():
    """Verify that a SEBI compliance query routes to rag_agent."""
    state: AgentState = {
        "messages": [HumanMessage(content="What are SEBI's minimum public shareholding requirements under LODR?")],
        "next_agent": "",
        "sources": []
    }
    decision = supervisor(state)
    assert decision.get("next_agent") == "rag_agent"

def test_supervisor_routes_quant_query():
    """Verify that a market volatility / Sharpe query routes to quant_agent."""
    state: AgentState = {
        "messages": [HumanMessage(content="What is the current Nifty volatility regime and Sharpe ratio?")],
        "next_agent": "",
        "sources": []
    }
    decision = supervisor(state)
    assert decision.get("next_agent") == "quant_agent"

def test_graph_rag_execution():
    """Verify graph execution for compliance query returns answer and sources."""
    graph = build_graph()
    result = graph.invoke({
        "messages": [HumanMessage(content="What is SEBI LODR Regulation 38?")],
        "next_agent": "",
        "sources": []
    })
    messages = result.get("messages", [])
    assert len(messages) >= 2
    answer = messages[-1].content
    assert len(answer) > 20
    assert "sources" in result

def test_graph_quant_execution():
    """Verify graph execution for quant query returns answer gracefully even if P1 is offline."""
    graph = build_graph()
    result = graph.invoke({
        "messages": [HumanMessage(content="Show me current volatility regime statistics.")],
        "next_agent": "",
        "sources": []
    })
    messages = result.get("messages", [])
    assert len(messages) >= 2
    answer = messages[-1].content
    assert len(answer) > 20

@pytest.mark.skipif(not has_groq_key, reason="GROQ_API_KEY not configured for live LLM tests")
def test_groq_live_rag_answer():
    """Test live generation with Groq Llama 3.3 70B."""
    graph = build_graph()
    result = graph.invoke({
        "messages": [HumanMessage(content="What percentage of public shareholding must listed companies maintain under SEBI rules?")],
        "next_agent": "",
        "sources": []
    })
    answer = result["messages"][-1].content
    assert len(answer) > 30
    assert any(term in answer.lower() for term in ["25", "shareholding", "sebi", "lodr"])
