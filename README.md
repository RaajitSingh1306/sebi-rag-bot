# SEBI RAG Bot — Multi-Agent Compliance & Volatility Intelligence

[![CI Tests](https://img.shields.io/badge/tests-18%2F18%20passed-brightgreen)](#8-results--evaluation)
[![RAGAS Faithfulness](https://img.shields.io/badge/RAGAS%20Faithfulness-90%25-emerald)](#8-results--evaluation)
[![Recall@5](https://img.shields.io/badge/Recall%405-100%25-blue)](#8-results--evaluation)
[![Deploy on Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7)](#12-deployment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#16-license--disclaimer)

| Resource | URL |
|---|---|
| **Live Web Application** | [https://sebi-rag-bot.vercel.app](https://sebi-rag-bot.vercel.app) |
| **Interactive API Docs** | [https://sebi-rag-bot.onrender.com/docs](https://sebi-rag-bot.onrender.com/docs) |
| **Health Check Endpoint** | [https://sebi-rag-bot.onrender.com/health](https://sebi-rag-bot.onrender.com/health) |

A production-grade **multi-agent compliance and quantitative market intelligence platform** that provides auditable, citation-backed answers to queries on Indian securities, corporate governance, banking risk, and data privacy regulations. Built with **LangGraph**, **Qdrant Cloud**, **BM25Okapi**, **PyMuPDF**, and **Groq (`openai/gpt-oss-120b`)**, the system pairs dense semantic search with sparse keyword matching, enforces a strict anti-hallucination relevance floor, and dynamically queries the [Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform) for live market risk intelligence.

---

## Table of Contents

- [1. What This Project Does](#1-what-this-project-does)
- [2. Why It Was Built](#2-why-it-was-built)
- [3. System Architecture](#3-system-architecture)
- [4. Tech Stack & Libraries](#4-tech-stack--libraries)
- [5. Data](#5-data)
- [6. Step-by-Step Pipeline](#6-step-by-step-pipeline)
- [7. Problems Faced & How We Solved Them](#7-problems-faced--how-we-solved-them)
- [8. Results & Evaluation](#8-results--evaluation)
- [9. Project Structure](#9-project-structure)
- [10. Getting Started](#10-getting-started)
- [11. API Reference](#11-api-reference)
- [12. Deployment](#12-deployment)
- [13. Connected Portfolio Projects](#13-connected-portfolio-projects)
- [14. Limitations & Known Issues](#14-limitations--known-issues)
- [15. Roadmap / Future Expansion](#15-roadmap--future-expansion)
- [16. License & Disclaimer](#16-license--disclaimer)

---

## 1. What This Project Does

The platform acts as an automated regulatory counsel and quant research companion:

- **Auditable Regulatory Synthesis**: Formulates exact, grounded answers across five major Indian financial and data privacy statutes:
  - **SEBI LODR (2015)**: Listing Obligations and Disclosure Requirements.
  - **SEBI SAST (2011)**: Substantial Acquisition of Shares and Takeovers.
  - **SEBI ICDR (2018)**: Issue of Capital and Disclosure Requirements (IPOs).
  - **RBI Master Directions**: Model Risk Management for Banks and NBFCs.
  - **DPDPA (2023)**: Digital Personal Data Protection Act.
- **Traceable Citation Attribution**: Every answer is paired with metadata: source document, regulation number, page number, and similarity score.
- **Dynamic Multi-Agent Routing (LangGraph)**: Automatically routes incoming queries to either the compliance RAG pipeline or the quantitative volatility agent.
- **Real-Time Market Context Integration**: Interacts directly with the Volatility Intelligence Platform API to fetch live Nifty 50 volatility regimes, GARCH estimates, and Sharpe ratios.
- **Anti-Hallucination Guardrails**: Employs an explicit relevance threshold ($0.35$); out-of-scope queries trigger an honest fallback rather than fabricated answers.

---

## 2. Why It Was Built

- **Manual Regulatory Inefficiency**: Legal and compliance teams at listed entities spend hours manually navigating dense regulatory gazettes, cross-referencing amendments, and searching keyword variants.
- **The LLM Hallucination Trap**: General-purpose LLMs (such as base ChatGPT or Gemini) frequently invent regulation numbers, misquote disclosure timelines, or produce plausible-sounding but legally invalid citations.
- **Bridging Compliance with Quantitative Risk**: Regulatory decisions (e.g. margin requirements, takeover open offers, risk-weighted capital adequacy) are inextricably tied to prevailing market volatility. Integrating compliance RAG with quantitative volatility APIs unifies these workflows into a single interface.

---

## 3. System Architecture

```text
                                    User Query
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │   LangGraph Supervisor   │
                            │   (Conditional Router)   │
                            └────────────┬────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼ Compliance / Statutory                        ▼ Market Volatility / Regime
    ┌───────────────────────────┐                   ┌───────────────────────────┐
    │         rag_agent         │                   │        quant_agent        │
    │                           │                   │                           │
    │  1. Hybrid Retrieval:     │                   │  Calls VIP REST API:      │
    │     • Dense (Qdrant ANN)  │                   │  • GET /current           │
    │     • Sparse (BM25Okapi)  │                   │  • GET /stats             │
    │  2. Hybrid Fusion:        │                   │                           │
    │     • 60% Dense + 40% BM25│                   │  Groq formats real-time   │
    │     • Top-5 chunks        │                   │  regimes & volatility     │
    │  3. Relevance Floor (0.35)│                   │                           │
    │  4. Groq LLM reasoning:   │                   │                           │
    │     • Grounded extraction │                   │                           │
    │     • Citations appended  │                   │                           │
    └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                               FastAPI Backend (:8000)
                                 POST /query  (JSON)
                                          │
                                          ▼
                             Next.js 14 Frontend (:3000)
                             (Tailwind Dark Mode Console)
```

---

## 4. Tech Stack & Libraries

| Library / Tool | Version | Purpose | Rationale |
|---|---|---|---|
| **Python** | `>=3.11` | Backend runtime | Modern high-performance asynchronous runtime |
| **LangGraph** | `^0.0.26` | Multi-agent orchestration | Type-safe state graph with conditional supervisor routing |
| **Qdrant Client** | `^1.7.3` | Vector database | Fast dense vector indexing with HNSW cosine search; free managed tier |
| **fastembed** | `^0.2.2` | Dense embedding generation | ONNX-based execution of `all-MiniLM-L6-v2`; ~50MB RAM footprint, zero PyTorch dependency |
| **rank-bm25** | `^0.2.2` | Sparse keyword retrieval | Okapi BM25 implementation catching exact statutory terminology and regulation clauses |
| **PyMuPDF (`fitz`)** | `^1.23.21` | PDF text extraction | High-speed, robust PDF parsing and structured page layout extraction |
| **Groq** | `^0.4.2` | High-speed LLM inference | Ultra-low-latency execution (~500 tokens/sec) of `openai/gpt-oss-120b` |
| **FastAPI** | `^0.109.0` | Async REST backend | Asynchronous endpoints with Pydantic validation and automatic OpenAPI Swagger docs |
| **Next.js** | `^14.1.0` | Frontend web interface | React 18 frontend with dark-mode styling and automated backend wake-up logic |
| **Tailwind CSS** | `^3.4.0` | Frontend styling | Utility-first dark tokens tailored for enterprise legal and quant interfaces |
| **RAGAS** | `^0.1.1` | RAG evaluation framework | Automated assessment of generation faithfulness, relevancy, and context recall |
| **pytest** | `^8.0.0` | Test suite | 18 automated unit and integration tests across retriever, agents, and API |

---

## 5. Data

- **Corpus Scope**: 5 core Indian financial, securities, and data privacy statutory frameworks:
  - `sebi_lodr_2015.pdf`: SEBI Listing Obligations and Disclosure Requirements, 2015 (Reg 23, Reg 30, Reg 33, Reg 38).
  - `sebi_sast_regulations.pdf`: SEBI Substantial Acquisition of Shares and Takeovers, 2011 (Reg 3, Reg 4, Reg 7, Reg 17).
  - `sebi_icdr_2018.pdf`: SEBI Issue of Capital and Disclosure Requirements, 2018 (Reg 6, Reg 14, Reg 16, Reg 17).
  - `rbi_model_risk_management.pdf`: RBI Guidelines on Model Risk Management for Financial Institutions.
  - `dpdpa_2023.pdf`: Digital Personal Data Protection Act, 2023 (Section 5, 6, 8, 33).
- **Volume**: 19 statutory pages yielding 30 curated semantic chunks.
- **Chunking Strategy**: Target size **512 characters** with **64 characters overlap**, strictly constrained to whole-word boundaries via `_word_safe_tail()`.
- **Embeddings**: `all-MiniLM-L6-v2` (384-dimensional dense vectors normalized for cosine similarity).
- **Indexing**: Qdrant Cloud collection `sebi_rbi_docs` alongside in-memory BM25Okapi inverted index with custom stop-word filtering.

---

## 6. Step-by-Step Pipeline

1. **Ingestion & Extraction (`scripts/ingest_docs.py`)**: Parse regulatory PDFs via PyMuPDF; isolate document metadata, regulation titles, and statutory text blocks.
2. **Word-Safe Semantic Chunking**: Split text into 512-character blocks with 64-character overlap, trimming overlaps to whole-word boundaries to avoid token fragmentation.
3. **Dense & Sparse Indexing**: Embed text chunks using `fastembed` (`all-MiniLM-L6-v2`); upload vectors to Qdrant Cloud and build local BM25Okapi token indices.
4. **Query Classification**: Incoming user query hits the LangGraph Supervisor; routed to `quant_agent` if quantitative market keywords are detected, or `rag_agent` for compliance questions.
5. **Hybrid Retrieval**:
   - Dense retrieval queries Qdrant Cloud for top cosine similarity matches.
   - Sparse retrieval evaluates BM25 scores with custom stop-word filtering.
   - Fused score computed as: $\text{Score} = 0.60 \times \text{Dense} + 0.40 \times \text{BM25}_{\text{norm}}$.
6. **Relevance Floor Guardrail**: Disqualify candidates below `RAG_RELEVANCE_THRESHOLD = 0.35`; return honest out-of-scope notifications if no chunk qualifies.
7. **Synthesis & Citation**: Send top-5 retrieved passages to Groq LLM with a strict non-extrapolation prompt; append source filename, regulation number, page, and score to the JSON response.
8. **Frontend Rendering**: Display answer, citation cards, and confidence metrics on the Next.js dark-mode interface.

---

## 7. Problems Faced & How We Solved Them

| Problem | Impact | How We Got Around It |
|---|---|---|
| **BM25 stop-word score inflation** | Common English words ("the", "of", "and") dominated BM25 scoring, ranking irrelevant chunks above actual regulatory matches | Implemented a **custom English stop-word filter** in the BM25 tokenizer, removing frequency-distorting stop words before TF-IDF scoring |
| **Mid-word chunk splitting** | Naive character-level chunking sliced words in half (e.g., "Regulation 38" split into "ulation 38"), corrupting both retrieval indexing and answer readability | Built `_word_safe_tail()` function that enforces **whole-word boundary trimming** at chunk overlaps, guaranteeing no word is ever fractured across chunk seams |
| **Render free tier 512MB RAM limit** | PyTorch-based embedding models exceeded the host memory budget and triggered container OOM kills | Switched to `fastembed` with ONNX runtime (`all-MiniLM-L6-v2`), which runs at ~50MB RAM without requiring PyTorch — well within the 512MB ceiling |
| **Out-of-scope queries producing hallucinated answers** | Users asking about general trivia or weather received confident but fabricated regulatory citations | Implemented a **relevance floor** (`RAG_RELEVANCE_THRESHOLD = 0.35`) — if the top chunk score falls below this threshold, the bot honestly responds "I don't have that information" instead of forcing citations |
| **Groq LLM downtime / rate limiting** | When the upstream LLM provider was unreachable, users received blank 500 errors | Built a **graceful fallback chain**: LLM synthesis $\rightarrow$ grounded extract (returns the top chunk text directly). The user always gets an answer, never a blank error |
| **Dense-only search missing exact statutory terms** | Semantic search alone could miss precise statutory citations like "Regulation 38" or "Section 8(1)(j)" | Added **BM25 sparse retrieval** alongside dense Qdrant search, fused with weighted scoring ($0.60 \times \text{Dense} + 0.40 \times \text{BM25}_{\text{norm}}$) to catch both paraphrased concepts and exact statutory numbers |

---

## 8. Results & Evaluation

### Retrieval Quality Benchmarks (`scripts/eval_retrieval.py`)

Audited across 8 benchmark regulatory queries across LODR, SAST, ICDR, RBI, and DPDPA:

| Metric | Result | Target | Benchmark Status |
|---|:---:|:---:|:---:|
| **Recall@5** | **1.0000 (100%)** | > 0.90 | ✅ 8/8 Rank-1 Hits |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | > 0.75 | ✅ Perfect Rank-1 Retrieval |

### Generation Quality Benchmarks (RAGAS Framework)

Audited across 10 statutory test questions using Groq as LLM judge (`scripts/eval_rag.py`):

| Metric | Score | Target | Evaluation Meaning |
|---|:---:|:---:|---|
| **Faithfulness** | **0.9000** | > 0.85 | All claims in generated answers are inferable from retrieved statutory text |
| **Answer Relevancy** | **0.9000** | > 0.80 | Responses directly answer the user's statutory inquiry without extraneous filler |
| **Context Recall** | **0.9500** | > 0.85 | Ground-truth statutory legal clauses are present in the retrieved chunks |

### Automated Test Suite (18/18 Tests Passing)

```bash
pytest tests/ -v
```

| Test Module | Tests | Verifications |
|---|:---:|---|
| `test_retriever.py` | 5 | Hybrid retrieval, BM25 tokenization, stop-word filtering, word-safe chunking, score sorting |
| `test_agents.py` | 7 | Supervisor routing, RAG agent, quant agent, graph compilation, relevance threshold floor |
| `test_api.py` | 6 | Health endpoint, query endpoint, CORS headers, error fallback chains |

---

## 9. Project Structure

```text
sebi-rag-bot/
├── backend/
│   ├── main.py              # FastAPI app: /query, /health, /eval-summary, /docs
│   ├── agents.py            # LangGraph multi-agent system (supervisor → rag_agent / quant_agent)
│   └── retriever.py         # HybridRetriever: Qdrant dense + BM25 sparse + hybrid scoring
├── frontend/
│   ├── pages/index.tsx      # Next.js chat UI with health indicator, citations, RAGAS panel
│   ├── styles/globals.css   # Tailwind dark theme
│   ├── vercel.json          # Vercel deployment config
│   └── package.json
├── scripts/
│   ├── create_sample_docs.py  # Generates regulatory PDFs for demo/testing
│   ├── ingest_docs.py         # PDF text extraction → chunking → embedding → Qdrant upsert
│   ├── eval_retrieval.py      # Retrieval benchmark (Recall@5, MRR)
│   └── eval_rag.py            # RAGAS evaluation (faithfulness, relevancy, context recall)
├── data/                    # Regulatory source PDFs
│   ├── sebi_lodr_2015.pdf
│   ├── sebi_sast_regulations.pdf
│   ├── sebi_icdr_2018.pdf
│   ├── rbi_model_risk_management.pdf
│   └── dpdpa_2023.pdf
├── eval/                    # Persisted evaluation reports
│   ├── ragas_report.json
│   └── retrieval_report.json
├── tests/                   # Pytest suite (18 tests)
│   ├── test_retriever.py
│   ├── test_agents.py
│   └── test_api.py
├── .env.example             # Environment variable template
├── Dockerfile               # Python 3.11 container definition
├── docker-compose.yml       # Local multi-service orchestration
├── render.yaml              # Render Blueprint deployment manifest
└── requirements.txt         # Production dependencies
```

---

## 10. Getting Started

### Prerequisites

- Python 3.11 or higher
- Node.js 18+ (for frontend)
- Groq API Key ([console.groq.com](https://console.groq.com))
- Qdrant Cloud instance or local Qdrant container

### 1. Backend Setup

```bash
cd "sebi-rag-bot"

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY

# Ingest and index regulatory corpus
python scripts/create_sample_docs.py
python scripts/ingest_docs.py

# Run tests
pytest tests/ -v

# Launch FastAPI backend
uvicorn backend.main:app --reload --port 8000
```

### 2. Frontend Setup

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000` to interact with the dark-mode compliance console.

### 3. Docker (Alternative)

```bash
docker compose build
docker compose up
```

---

## 11. API Reference

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service metadata and endpoint catalog |
| `GET` | `/health` | Liveness and health check |
| `POST` | `/query` | Execute multi-agent RAG / quant workflow |
| `GET` | `/eval-summary` | RAGAS evaluation report data |

### Sample Request (`POST /query`)

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What threshold triggers a mandatory open offer under SEBI SAST?"}'
```

### Sample Response (`200 OK`)

```json
{
  "answer": "Under SEBI SAST Regulations, an acquirer who acquires shares or voting rights exceeding 25% of the total shares or voting rights of the target company must make a mandatory open offer to public shareholders for at least 26% of total shares...",
  "sources": [
    {
      "source": "sebi_sast_regulations.pdf",
      "page": 2,
      "text": "Regulation 3(1): No acquirer shall acquire shares...",
      "score": 0.847
    }
  ],
  "agent_used": "rag_agent"
}
```

---

## 12. Deployment

### Backend on Render
- Configured via `render.yaml` as a Web Service.
- Runtime: Python 3 with `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`.
- Set environment variables: `GROQ_API_KEY`, `GROQ_MODEL`, `P1_API_URL`, `QDRANT_URL`, `QDRANT_API_KEY`.

### Frontend on Vercel
- Root directory set to `frontend`.
- Environment variable: `NEXT_PUBLIC_API_URL=https://sebi-rag-bot.onrender.com`.
- Includes automatic polling retry logic to handle Render free-tier cold starts gracefully.

---

## 13. Connected Portfolio Projects

- **[Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform)**: Flagship market intelligence API called directly by this bot's `quant_agent` to fetch real-time volatility regimes and forecasts.
- **[Credit Default Predictor](https://github.com/RaajitSingh1306/Credit-Default-Predictor)**: Machine learning loan default prediction system grounded in the RBI Model Risk Management guidelines indexed by this bot.
- **[NSEI Daily Stock Pipeline](https://github.com/RaajitSingh1306/NSEI-Daily-Stock-Pipeline)**: Data lakehouse providing the upstream feature store for market indicators.
- **[Nifty Sector Rotation](https://github.com/RaajitSingh1306/Nifty-Sector-Rotation)**: Quantitative momentum strategy operating on Indian equity sectors.

---

## 14. Limitations & Known Issues

- **Corpus Breadth**: Indexed with 5 representative statutory frameworks rather than the complete gazette of SEBI Master Circulars and legal precedents.
- **Text-Only Extraction**: PyMuPDF extracts plain-text paragraphs; complex financial ratio tables, schedules, and penalty formulas are not extracted via computer vision / table transformer models.
- **In-Memory BM25 Index**: The BM25 index is loaded into memory on worker startup from Qdrant scroll; it lacks persistent disk serialization.
- **Single-Turn Context**: Operates primarily as a single-turn Q&A engine; does not maintain multi-turn conversational coreference resolution across separate legal acts.

---

## 15. Roadmap / Future Expansion

- [ ] **Full Gazette Ingestion**: Expand corpus to encompass full SEBI Master Circulars, PIT (Prohibition of Insider Trading), and AIF regulations.
- [ ] **Table-Aware Document Extraction**: Integrate Table Transformer / Camelot for structural extraction of financial ratio tables and disclosure matrices.
- [ ] **Two-Stage Cross-Encoder Reranking**: Deploy a cross-encoder (`bge-reranker-large`) over top-20 hybrid candidates to optimize top-5 precision.
- [ ] **Multi-Hop Knowledge Graph (GraphRAG)**: Construct entity relation graphs connecting cross-statutory provisions between SEBI LODR, Companies Act 2013, and RBI Master Directions.
- [ ] **Session & Thread Memory**: Implement persistent multi-turn conversational memory backed by Redis session storage.

---

## 16. License & Disclaimer

### License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

### Disclaimer
This system is intended strictly for educational research and compliance decision support. It does not constitute formal legal counsel or statutory advice.
