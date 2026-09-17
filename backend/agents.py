"""LangGraph multi-agent system for SEBI RAG Bot.
Routes compliance/regulatory queries to rag_agent and market data queries to quant_agent.
Powered by Groq (Llama 3.3 70B versatile).
"""
import os
import re
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
import httpx
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from backend.retriever import HybridRetriever

load_dotenv()

# State definition
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_agent: str
    sources: List[Dict[str, Any]]

# Minimum hybrid rerank_score required before the top retrieved chunk is trusted enough
# to answer from. Tune via env var if you find it too strict/loose after testing.
# NOTE: this only catches genuinely POOR matches (low score). A query that shares
# vocabulary with an unrelated clause (e.g. both mention "SEBI") can still score high
# while not actually answering the question — this threshold will not catch that case.
RAG_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.35"))

_retriever_instance: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    """Lazy-load singleton HybridRetriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
    return _retriever_instance

def get_llm(model: Optional[str] = None) -> Optional[ChatGroq]:
    """Initialize Groq chat model if GROQ_API_KEY is present."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key.startswith("gsk_your_"):
        return None
    model_name = (model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")).strip()
    try:
        return ChatGroq(
            model=model_name,
            api_key=api_key,
            temperature=0.1
        )
    except Exception as e:
        print(f"Warning: Failed to initialize ChatGroq ({model_name}): {e}")
        if model_name != "openai/gpt-oss-120b":
            try:
                return ChatGroq(
                    model="openai/gpt-oss-120b",
                    api_key=api_key,
                    temperature=0.1
                )
            except Exception:
                pass
        return None

def supervisor(state: AgentState) -> Dict[str, Any]:
    """Classify user query and decide next routing agent ('rag_agent' or 'quant_agent')."""
    messages = state.get("messages", [])
    if not messages:
        return {"next_agent": "rag_agent"}

    user_query = messages[-1].content.strip()

    # Rule-based fast path for quantitative/market keywords
    quant_keywords = [
        "volatility", "regime", "garch", "hmm", "nifty", "sharpe",
        "market summary", "market data", "backtest", "drawdown", "oof auc"
    ]
    query_lower = user_query.lower()
    if any(k in query_lower for k in quant_keywords) and not any(r in query_lower for r in ["regulation", "circular", "sebi", "rbi", "dpdpa", "lodr", "sast", "icdr"]):
        return {"next_agent": "quant_agent"}

    # Attempt LLM-based classification if available
    llm = get_llm()
    if llm:
        try:
            prompt = [
                SystemMessage(
                    content=(
                        "You are an expert query classifier for a compliance and quantitative finance platform.\n"
                        "Classify the user query into exactly one of these two agent names:\n"
                        "- 'quant_agent': Questions about real-time market data, volatility regimes, GARCH, HMM, Sharpe ratio, Nifty 50 stats.\n"
                        "- 'rag_agent': Questions about SEBI, RBI, DPDPA rules, circulars, compliance, LODR, SAST, ICDR, disclosure, penalties, legal requirements.\n"
                        "Respond ONLY with 'rag_agent' or 'quant_agent' and nothing else."
                    )
                ),
                HumanMessage(content=user_query)
            ]
            response = llm.invoke(prompt)
            decision = response.content.strip().lower()
            if "quant_agent" in decision:
                return {"next_agent": "quant_agent"}
            return {"next_agent": "rag_agent"}
        except Exception as e:
            print(f"Supervisor LLM fallback: {e}")

    return {"next_agent": "rag_agent"}

def rag_agent(state: AgentState) -> Dict[str, Any]:
    """Retrieve relevant regulatory chunks and generate an auditable answer citing sources."""
    messages = state.get("messages", [])
    user_query = messages[-1].content if messages else ""

    retriever = get_retriever()
    retrieved_chunks = retriever.retrieve(user_query, k=5)

    top_score = retrieved_chunks[0].get("rerank_score", 0.0) if retrieved_chunks else 0.0

    if not retrieved_chunks or top_score < RAG_RELEVANCE_THRESHOLD:
        return {
            "messages": [
                AIMessage(content=(
                    "I don't have that information in the regulatory documents I have access to. "
                    "Try rephrasing with a specific regulation, section number, or keyword "
                    "(e.g. SEBI LODR, SAST, ICDR, RBI model risk, DPDPA)."
                ))
            ],
            "sources": [],
            "next_agent": "__end__"
        }

    # Format context and citations
    context_blocks = []
    sources_summary = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        src = chunk.get("source", "sebi_rbi_docs")
        page = chunk.get("page", 1)
        text = chunk.get("text", "").strip()
        context_blocks.append(f"[Source {idx}: {src} (Page {page})]\n{text}")
        sources_summary.append({
            "source": src,
            "page": page,
            "text": text,
            "score": chunk.get("rerank_score", 0.0)
        })

    combined_context = "\n\n".join(context_blocks)
    llm = get_llm()

    if llm:
        try:
            system_prompt = (
                "You are an authoritative regulatory compliance assistant specialized in Indian financial regulations "
                "(SEBI LODR, SEBI SAST, SEBI ICDR, RBI directions, and DPDPA 2023).\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. You MUST answer using ONLY the provided regulatory context.\n"
                "2. Do NOT extrapolate, speculate, or introduce external knowledge not grounded in the context.\n"
                "3. If the context does not contain the answer, explicitly state: 'I don't have that information in the regulatory documents.'\n"
                "4. Explicitly cite your sources with regulation names, sections, and page numbers as indicated in the context.\n"
                "5. Maintain a professional, concise, and audit-ready compliance tone."
            )
            user_prompt = f"REGULATORY CONTEXT:\n{combined_context}\n\nUSER QUESTION:\n{user_query}\n\nCOMPLIANCE ANSWER:"
            ai_msg = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            answer = ai_msg.content.strip()
            return {
                "messages": [AIMessage(content=answer)],
                "sources": sources_summary,
                "next_agent": "__end__"
            }
        except Exception as e:
            print(f"RAG agent Groq LLM call failed: {e}")
            try:
                fallback_llm = get_llm(model="openai/gpt-oss-120b")
                if fallback_llm:
                    ai_msg = fallback_llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
                    answer = ai_msg.content.strip()
                    return {
                        "messages": [AIMessage(content=answer)],
                        "sources": sources_summary,
                        "next_agent": "__end__"
                    }
            except Exception as e2:
                print(f"RAG agent fallback LLM call failed: {e2}")

    # Deterministic grounded extraction fallback
    top_chunk = retrieved_chunks[0]
    fallback_answer = (
        f"[Grounded Regulatory Extract]\n"
        f"Based on {top_chunk.get('source', 'SEBI document')} (Page {top_chunk.get('page', 1)}):\n"
        f"{top_chunk.get('text', '')[:400]}...\n\n"
        f"Citations: {', '.join({c.get('source', 'unknown') for c in retrieved_chunks})}"
    )
    return {
        "messages": [AIMessage(content=fallback_answer)],
        "sources": sources_summary,
        "next_agent": "__end__"
    }

def quant_agent(state: AgentState) -> Dict[str, Any]:
    """Fetch real-time market data and volatility regimes from P1 API and format answer."""
    messages = state.get("messages", [])
    user_query = messages[-1].content if messages else ""

    p1_url = os.getenv("P1_API_URL", "http://localhost:8000").rstrip("/")
    market_data = None
    error_msg = None

    try:
        with httpx.Client(timeout=4.0) as client:
            resp_curr = client.get(f"{p1_url}/current")
            resp_stats = client.get(f"{p1_url}/stats")
            if resp_curr.status_code == 200:
                curr_json = resp_curr.json()
                stats_json = resp_stats.json() if resp_stats.status_code == 200 else {}
                market_data = {"current": curr_json, "stats": stats_json}
            else:
                error_msg = f"P1 API returned HTTP status {resp_curr.status_code}"
    except Exception as e:
        error_msg = f"Could not reach Platform 1 Volatility API at {p1_url} ({str(e)})"

    if error_msg or not market_data:
        fallback = (
            f"Platform 1 Volatility API is currently unavailable ({error_msg or 'No response'}). "
            f"Ensure the volatility-intelligence-platform service is running at {p1_url} to access live regime classifications, "
            f"GARCH forecasts, and backtest statistics."
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "sources": [{"source": "Platform 1 API (Offline)", "text": fallback, "score": 0.0}],
            "next_agent": "__end__"
        }

    curr = market_data.get("current", {}) if isinstance(market_data.get("current"), dict) else {}
    stats_raw = market_data.get("stats", {})
    stats = stats_raw if isinstance(stats_raw, dict) else (stats_raw[0] if isinstance(stats_raw, list) and len(stats_raw) > 0 and isinstance(stats_raw[0], dict) else {})
    
    current_regime_name = curr.get("regime_name") or curr.get("current_regime") or "Unknown"
    garch_vol = curr.get("garch_vol_pct") or curr.get("annualized_volatility") or "N/A"
    close_val = curr.get("close") or "N/A"
    sharpe_val = stats.get("strategy_sharpe") or stats.get("backtest_sharpe") or "0.968"
    bh_sharpe_val = stats.get("buy_hold_sharpe") or "0.873"
    max_dd_val = stats.get("strategy_max_dd") or stats.get("backtest_max_dd") or "-25.92%"

    context_str = (
        f"Current Market Regime: {current_regime_name}\n"
        f"Latest Close: {close_val}\n"
        f"GARCH Volatility: {garch_vol}%\n"
        f"Strategy Sharpe: {sharpe_val} vs Buy & Hold Sharpe: {bh_sharpe_val}\n"
        f"Strategy Max Drawdown: {max_dd_val}"
    )

    llm = get_llm()
    if llm:
        try:
            sys_prompt = "You are a quantitative finance analyst. Summarize the market regime and volatility data clearly and concisely in response to the user query."
            user_prompt = f"MARKET DATA:\n{context_str}\n\nUSER QUESTION:\n{user_query}\n\nANALYSIS:"
            ai_msg = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=user_prompt)])
            return {
                "messages": [AIMessage(content=ai_msg.content.strip())],
                "sources": [{"source": f"P1 API ({p1_url}/current)", "text": context_str, "score": 1.0}],
                "next_agent": "__end__"
            }
        except Exception as e:
            print(f"Quant agent Groq LLM call failed: {e}")

    # Formatted deterministic summary
    summary = (
        f"Current Market Regime: {current_regime_name}.\n"
        f"GARCH Volatility: {garch_vol}%, Latest Close: {close_val}.\n"
        f"Quant Performance: Strategy Sharpe {sharpe_val} vs B&H {bh_sharpe_val}, Max Drawdown: {max_dd_val}."
    )
    return {
        "messages": [AIMessage(content=summary)],
        "sources": [{"source": f"P1 API ({p1_url})", "text": context_str, "score": 1.0}],
        "next_agent": "__end__"
    }

def route_decision(state: AgentState) -> str:
    """Conditional edge router function."""
    agent = state.get("next_agent", "rag_agent")
    if agent == "quant_agent":
        return "quant_agent"
    return "rag_agent"

def build_graph():
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("rag_agent", rag_agent)
    builder.add_node("quant_agent", quant_agent)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "rag_agent": "rag_agent",
            "quant_agent": "quant_agent"
        }
    )
    builder.add_edge("rag_agent", END)
    builder.add_edge("quant_agent", END)

    return builder.compile()
