# SEBI RAG Bot — Multi-Agent Compliance & Volatility Intelligence

[![CI Tests](https://img.shields.io/badge/tests-18%2F18%20passed-brightgreen)](#testing)
[![RAGAS Faithfulness](https://img.shields.io/badge/RAGAS%20Faithfulness-90%25-emerald)](#evaluation--verification)
[![Recall@5](https://img.shields.io/badge/Recall%405-100%25-blue)](#evaluation--verification)
[![Deploy on Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7)](#deployment)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

| | |
|---|---|
| **Live App** | https://sebi-rag-bot.vercel.app |
| **API Docs** | https://sebi-rag-bot.onrender.com/docs |
| **Health** | https://sebi-rag-bot.onrender.com/health |

---

## What

SEBI RAG Bot is a **multi-agent compliance assistant** that answers questions about Indian financial regulations with **auditable, citation-backed responses**. It covers five regulatory domains:

| Regulation | Scope |
|---|---|
| **SEBI LODR 2015** | Listing Obligations and Disclosure Requirements |
| **SEBI SAST** | Substantial Acquisition of Shares and Takeovers |
| **SEBI ICDR 2018** | Issue of Capital and Disclosure Requirements (IPOs) |
| **RBI Directions** | Model Risk Management for financial institutions |
| **DPDPA 2023** | Digital Personal Data Protection Act |

The system also connects to a **live volatility intelligence API** to answer quantitative market questions (Nifty 50 regime, GARCH forecasts, Sharpe ratios).

**Example interaction:**

> **User:** _"What is the minimum public shareholding requirement under SEBI LODR Regulation 38?"_
>
> **Bot:** _"The SEBI LODR 2015 mandates that every listed entity must maintain a minimum public shareholding of 25%. Newly listed companies that fall below this threshold must raise their public shareholding to 25% within three years of listing."_
> — Source: `sebi_lodr_2015.pdf`, Regulation 38, Page 3

---

## Why

### The Problem

Compliance officers and fintech engineers in India regularly need to look up specific clauses across hundreds of pages of SEBI, RBI, and DPDPA regulations. The current workflow is:

1. **Manual PDF searching** — Ctrl+F across multiple regulatory documents, hoping the right keyword appears
2. **Hallucination risk** — General-purpose LLMs (ChatGPT, Gemini) confidently cite regulation numbers that don't exist
3. **No auditability** — When regulators ask "where does it say that?", there's no traceable citation chain
4. **Fragmented information** — Market volatility data and regulatory compliance live in entirely separate systems

### The Solution

This project solves all four problems:

- **Hybrid retrieval** (BM25 keyword matching + Qdrant dense vectors) ensures both exact statutory terms and semantically similar passages are found
- **Zero-hallucination architecture** — the LLM can only synthesize answers from retrieved regulatory text, never from its training data
- **Full citation chain** — every answer includes the source PDF filename, page number, and exact passage text
- **Multi-agent routing** — a LangGraph supervisor dynamically routes compliance questions to the RAG agent and market questions to the quantitative agent, unifying both workflows in a single interface

### Who Is This For

- **Compliance officers** at listed companies who need quick, auditable regulatory lookups
- **Legal teams** preparing for SEBI inspections or filing disclosures
- **Fintech engineers** building products that need regulatory guardrails
- **Students and researchers** studying Indian securities regulation

---

## How

### Architecture

```
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
                 ▼ Compliance / Regulation                       ▼ Market Data / Regime
    ┌───────────────────────────┐                   ┌───────────────────────────┐
    │         rag_agent         │                   │        quant_agent        │
    │                           │                   │                           │
    │  1. Hybrid Retrieval:     │                   │  Calls Platform 1 API:    │
    │     • Dense (Qdrant ANN)  │                   │  • GET /current           │
    │     • Sparse (BM25Okapi)  │                   │  • GET /stats             │
    │  2. Hybrid Scoring:       │                   │                           │
    │     • 60% Dense + 40% BM25│                   │  Groq formats real-time   │
    │     • Top-5 chunks        │                   │  regimes & volatility     │
    │  3. Groq LLM reasoning:   │                   │                           │
    │     • Answer from context │                   │                           │
    │     • Citations appended  │                   │                           │
    └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                               FastAPI on Render (:8000)
                                 POST /query  (JSON)
                                          │
                                          ▼
                                Next.js 14 on Vercel
```

### How It Works Step-by-Step

**1. Query Classification (Supervisor)**

The LangGraph supervisor receives the user's question and decides which agent handles it. It first checks for quantitative keywords (`volatility`, `garch`, `nifty`, `sharpe`). If none match, it uses Groq LLM classification to pick between `rag_agent` and `quant_agent`. If LLM is unavailable, it defaults to `rag_agent`.

**2. Regulatory Retrieval (RAG Agent)**

For compliance questions, the RAG agent runs a **hybrid search**:

| Retrieval Method | How It Works | Why It's Needed |
|---|---|---|
| **Dense search** (Qdrant) | Encodes the query into a 384-dim vector via `all-MiniLM-L6-v2`, searches Qdrant Cloud for nearest neighbors | Finds semantically similar passages even when wording differs from the query |
| **Sparse search** (BM25) | Tokenizes the query into keywords (with stop-word filtering), scores every chunk using Okapi BM25 | Catches exact statutory terms like "Regulation 38", "LODR", "Section 8(1)(j)" without stop-word false positives |
| **Hybrid scoring** | `0.60 × dense_score + 0.40 × normalized_BM25_score` | Combines both modalities; candidates missing dense search are penalized to eliminate spurious matches |
| **Relevance Floor** | `RAG_RELEVANCE_THRESHOLD = 0.35` | Automatically rejects out-of-scope queries (e.g. weather/trivia) with an honest fallback instead of forced citations |

The top 5 chunks are sent to Groq's LLM with a strict system prompt that forbids extrapolation. If the top chunk's score is below `RAG_RELEVANCE_THRESHOLD`, the bot honestly informs the user that the information is not present. If the LLM is unavailable, the system returns the top chunk text directly as a **grounded extract** — never leaving the user with no answer.

**3. Market Data (Quant Agent)**

For market questions, the quant agent calls the [Volatility Intelligence Platform API](https://github.com/RaajitSingh1306/volatility-intelligence-platform) to fetch the current Nifty 50 regime, GARCH volatility, and backtest statistics. Groq then formats this data into a readable summary.

**4. Frontend**

The Next.js frontend at `sebi-rag-bot.vercel.app`:
- Shows a real-time **API health indicator** with automatic retry logic for Render cold starts
- Renders **expandable citation cards** with source filenames, page numbers, and relevance scores
- Provides **one-click sample queries** across all five regulatory domains
- Displays **RAGAS evaluation metrics** (faithfulness, relevancy, context recall) in an expandable panel

### Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Word-safe chunking** | `_word_safe_tail()` whole-word boundary | Prevents slicing mid-word (e.g., "Regulation 38" is never cut into "ulation 38") across chunk overlaps |
| **Relevance floor** | `RAG_RELEVANCE_THRESHOLD` (0.35) | Guarantees top match quality; returns honest "I don't have that information" for out-of-scope queries |
| **Stop-word filtering** | Custom English stop-word set in BM25 | Eliminates stop-word score inflation on irrelevant questions |
| **Embedding model** | `all-MiniLM-L6-v2` via fastembed (ONNX) | ~50 MB RAM, no PyTorch required — critical for Render free tier's 512 MB limit |
| **LLM provider** | Groq (`openai/gpt-oss-120b`) | Zero-cost inference at ~500 tokens/sec; no credit card required |
| **Vector store** | Qdrant Cloud | Managed service with free tier; avoids local storage issues on Render |
| **Orchestration** | LangGraph `StateGraph` | Type-safe conditional routing with explicit state transitions; cleaner than ad-hoc if/else chains |
| **Hybrid scoring** | 60/40 dense/sparse | Dense handles paraphrased queries; BM25 handles exact legal citations; 60/40 split empirically optimal |
| **Fallback chain** | LLM → Grounded extract | If Groq is down, the user still gets the most relevant regulatory passage directly — never a blank error |

---

## Evaluation & Verification

### Retrieval Quality

Audited across 8 benchmark regulatory queries (`scripts/eval_retrieval.py`):

| Metric | Result | Target | Status |
|---|---|---|:---:|
| **Recall@5** | **1.0000 (100%)** | > 0.90 | ✅ 8/8 Rank-1 Hits |
| **MRR** | **1.0000** | > 0.75 | ✅ Perfect |

### Generation Quality (RAGAS)

Audited across 10 benchmark Q&A pairs using Groq as LLM judge (`scripts/eval_rag.py`):

| Metric | Score | Target | What It Measures |
|---|---|---|---|
| **Faithfulness** | **0.9000** | > 0.85 | Are all claims in the answer inferable from the retrieved context? |
| **Answer Relevancy** | **0.9000** | > 0.80 | Does the answer directly address what was asked? |
| **Context Recall** | **0.9500** | > 0.85 | Are the ground-truth statutory clauses present in the retrieved chunks? |

### Unit & Integration Tests

18/18 tests passing across three test modules:

| Module | Tests | Coverage |
|---|---|---|
| `test_retriever.py` | 5 | Hybrid retrieval, BM25 tokenization, stop-word filtering, word-safe chunking, score sorting |
| `test_agents.py` | 7 | Supervisor routing, RAG agent, quant agent, graph compilation, relevance threshold floor |
| `test_api.py` | 6 | Health endpoint, query endpoint, CORS, error handling |

```bash
pytest tests/ -v
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Groq (`openai/gpt-oss-120b`) | Zero-cost, high-speed compliance reasoning |
| **Orchestration** | LangGraph `StateGraph` | Multi-agent conditional routing |
| **Dense Retrieval** | Qdrant Cloud + fastembed | Semantic vector search (384-dim, cosine) |
| **Sparse Retrieval** | `rank-bm25` (BM25Okapi) | Exact keyword and statutory term matching |
| **PDF Extraction** | PyMuPDF (`fitz`) | Text extraction from regulatory PDFs |
| **Evaluation** | RAGAS + custom LLM-as-a-judge | Automated faithfulness and recall benchmarks |
| **API** | FastAPI + Uvicorn | Async REST endpoints with OpenAPI docs |
| **Frontend** | Next.js 14 + Tailwind CSS + Lucide | Dark-mode compliance chat interface |
| **Deployment** | Render (backend) + Vercel (frontend) | Cloud hosting with Docker support |

---

## Data

The retrieval knowledge base is grounded in statutory Indian financial, securities, and data privacy frameworks, ingested and indexed via `scripts/ingest_docs.py`.

### Corpus Architecture & Parameters

| Parameter | Specification | Implementation Detail |
|---|---|---|
| **Corpus Scope** | 5 statutory frameworks | Authentic Indian capital markets, takeover, banking risk, and data privacy frameworks |
| **Total Volume** | 19 pages / 30 chunks | Extracted via PyMuPDF (`fitz`), parsed into structured statutory clauses |
| **Chunking Strategy** | Paragraph-first semantic | Target size **512 characters**, overlap **64 characters** |
| **Overlap Boundary Safety** | `_word_safe_tail()` | Whole-word boundary trimming to eliminate broken mid-word token fragments across chunk seams |
| **Min Chunk Threshold** | 20 characters | Filters out trailing headers, blank lines, and residual formatting noise |
| **Dense Embeddings** | `all-MiniLM-L6-v2` | 384 dimensions, normalized for cosine similarity, batched at 64 items via FastEmbed / ONNX |
| **Vector Index** | Qdrant Cloud / Local | Collection `sebi_rbi_docs`, HNSW indexed, Cosine distance metric |
| **Sparse Index** | Okapi BM25 (`rank-bm25`) | Tokenized with English stop-word filtering to prevent frequency distortion on statutory keywords |
| **Hybrid Fusion** | Linear weighted sum | $\text{Score} = 0.60 \times \text{Dense} + 0.40 \times \text{BM25}_{\text{norm}}$ |
| **Relevance Threshold Floor** | $0.35$ | Minimum similarity score cut-off rejecting out-of-scope or unanswerable queries |

### Regulatory Document & Chunk Inventory

| Document File | Statutory Scope | Pages | Chunks | Key Regulatory Topics Covered |
|---|---|:---:|:---:|---|
| `sebi_lodr_2015.pdf` | SEBI (Listing Obligations and Disclosure Requirements), 2015 | 4 | 6 | Reg 23 (Related Party Transactions), Reg 30 (Material Event Disclosures), Reg 33 (Financial Results), Reg 38 (25% Minimum Public Shareholding) |
| `sebi_sast_regulations.pdf` | SEBI (Substantial Acquisition of Shares and Takeovers), 2011 | 4 | 6 | Reg 3(1) (25% Open Offer Trigger), Reg 3(2) (5% Creeping Acquisition Limit), Reg 4 (Acquisition of Control), Reg 7 (26% Offer Size), Reg 17 (Escrow Account) |
| `sebi_icdr_2018.pdf` | SEBI (Issue of Capital and Disclosure Requirements), 2018 | 4 | 7 | Reg 6 (IPO Net Tangible Assets/Net Worth Eligibility), Reg 14 & 16 (20% Minimum Promoters' Contribution & 18-month lock-in), Reg 17 (Anchor Investor 30/90-day lock-in & 90% subscription threshold) |
| `rbi_model_risk_management.pdf` | RBI Guidelines on Model Risk Management for Banks & NBFCs | 3 | 5 | Model Risk Governance, Section 4 (Sound Development Practices), Section 5 (Independent Model Validation), Section 7 (Model Inventory & 5-Year Audit Trails) |
| `dpdpa_2023.pdf` | Digital Personal Data Protection Act, 2023 | 4 | 6 | Section 5 & 6 (Notice, Free and Informed Consent, Withdrawal), Section 8 (Security Safeguards & Breach Notification), Section 33 (INR 250 Cr & 200 Cr Penalties) |
| **Total** | **5 Statutory Frameworks** | **19** | **30** | **Comprehensive Indian Securities, Banking & Privacy Coverage** |

> [!NOTE]
> Demonstration PDFs are generated programmatically via `scripts/create_sample_docs.py` to provide a reproducible, zero-copyright verification suite. Production gazette PDFs can be placed directly into `data/` and re-indexed via `python scripts/ingest_docs.py` without code modification.


---

## Where — Project Structure

```
sebi-rag-bot/
│
├── backend/
│   ├── main.py              # FastAPI app: /query, /health, /eval-summary, /docs
│   ├── agents.py            # LangGraph multi-agent system (supervisor → rag_agent / quant_agent)
│   └── retriever.py         # HybridRetriever: Qdrant dense + BM25 sparse + hybrid scoring
│
├── frontend/
│   ├── pages/index.tsx      # Next.js chat UI with health indicator, citations, RAGAS panel
│   ├── styles/globals.css   # Tailwind dark theme
│   ├── vercel.json          # Vercel deployment config
│   └── package.json
│
├── scripts/
│   ├── create_sample_docs.py  # Generates regulatory PDFs for demo/testing
│   ├── ingest_docs.py         # Extracts text from PDFs → chunks → embeds → uploads to Qdrant
│   ├── eval_retrieval.py      # Retrieval benchmark (Recall@5, MRR)
│   └── eval_rag.py            # RAGAS evaluation (faithfulness, relevancy, context recall)
│
├── data/                    # Regulatory source PDFs
│   ├── sebi_lodr_2015.pdf
│   ├── sebi_sast_regulations.pdf
│   ├── sebi_icdr_2018.pdf
│   ├── rbi_model_risk_management.pdf
│   └── dpdpa_2023.pdf
│
├── eval/                    # Generated evaluation reports
│   ├── ragas_report.json
│   └── retrieval_report.json
│
├── tests/                   # Pytest suite (15 tests)
│   ├── test_retriever.py
│   ├── test_agents.py
│   └── test_api.py
│
├── .env.example             # Environment variable template
├── Dockerfile               # Python 3.11 container
├── docker-compose.yml       # Local orchestration
├── render.yaml              # Render Blueprint manifest
├── requirements.txt         # Python dependencies
└── requirements-dev.txt     # Dev/test dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- Groq API key — free, no credit card ([console.groq.com](https://console.groq.com))
- Qdrant Cloud account — free tier ([cloud.qdrant.io](https://cloud.qdrant.io)) _(optional for local dev — falls back to local file storage)_

### 1. Backend

```bash
git clone https://github.com/RaajitSingh1306/sebi-rag-bot.git
cd sebi-rag-bot

pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env → add your GROQ_API_KEY (required) and optionally QDRANT_URL + QDRANT_API_KEY

# Generate sample regulatory PDFs and index them into Qdrant
python scripts/create_sample_docs.py
python scripts/ingest_docs.py

# Run tests (15/15 passing)
pytest tests/ -v

# Start API server
uvicorn backend.main:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 — the frontend auto-detects `localhost` and points to `http://localhost:8000`.

### 3. Docker (Alternative)

```bash
docker compose build
docker compose up
```

```bash
# Health check
curl http://localhost:8000/health
# → {"status": "ok", "version": "1.0.0"}

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is minimum public shareholding under SEBI LODR?"}'
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service metadata and endpoint list |
| `GET` | `/health` | Liveness check — `{"status": "ok", "version": "1.0.0"}` |
| `POST` | `/query` | Execute multi-agent RAG/quant workflow. Body: `{"question": "..."}` |
| `GET` | `/eval-summary` | RAGAS evaluation metrics report |
| `GET` | `/docs` | Interactive Swagger/OpenAPI documentation |

### POST /query — Request

```json
{
  "question": "What threshold triggers a mandatory open offer under SEBI SAST?"
}
```

### POST /query — Response

```json
{
  "answer": "Under SEBI SAST Regulations, an acquirer who acquires shares or voting rights exceeding 25% of the total shares or voting rights of the target company must make a mandatory open offer...",
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

## Deployment

### Backend → Render

**Option A: Blueprint (1-click)**
1. Push to GitHub
2. Render Dashboard → **New +** → **Blueprint** → select repo
3. Render reads `render.yaml` and configures automatically
4. Add `GROQ_API_KEY` in the environment variables prompt
5. Click **Apply**

**Option B: Manual Web Service**
1. Render → **New +** → **Web Service** → select repo
2. Configure:
   - **Runtime**: Python 3
   - **Build**: `pip install -r requirements.txt`
   - **Start**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check**: `/health`
   - **Plan**: Free
3. Add env vars: `GROQ_API_KEY`, `GROQ_MODEL`, `P1_API_URL`, `QDRANT_URL`, `QDRANT_API_KEY`

### Frontend → Vercel

1. Vercel → **Add New Project** → import `sebi-rag-bot`
2. Set **Root Directory** to `frontend`
3. Add env var: `NEXT_PUBLIC_API_URL` = `https://sebi-rag-bot.onrender.com`
4. Deploy

> **Note:** Render free tier spins down after ~15 min of inactivity. The frontend includes automatic retry logic (4 attempts, 6s intervals) to handle cold starts gracefully.

---

## Connected Projects

This multi-agent system connects directly to the quantitative finance and machine learning ecosystem:

| Project | Role | Repository |
|---|---|---|
| **SEBI RAG Bot** (This Repo) | Compliance Q&A + multi-agent regulatory orchestration | [sebi-rag-bot](https://github.com/RaajitSingh1306/sebi-rag-bot) |
| **Volatility Intelligence Platform** | Production GARCH + HMM + XGBoost market intelligence API | [volatility-intelligence-platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform) |
| **Credit Default Predictor** | Loan default prediction & TreeSHAP explainability engine | [Credit-Default-Predictor](https://github.com/RaajitSingh1306/Credit-Default-Predictor) |
| **NSEI Daily Stock Pipeline** | Financial data lakehouse & feature store (Airflow, Spark, DuckDB) | [NSEI-Daily-Stock-Pipeline](https://github.com/RaajitSingh1306/NSEI-Daily-Stock-Pipeline) |
| **Nifty Sector Rotation** | Momentum strategy on Indian sector indices | [Nifty-Sector-Rotation](https://github.com/RaajitSingh1306/Nifty-Sector-Rotation) |
| **Volatility Classifier (Simplified)** | *Superseded (v2)*: Single-asset HMM + GARCH refactor | [volatility-classifier-simplified](https://github.com/RaajitSingh1306/volatility-classifier-simplified) |

The `quant_agent` in this bot calls the [Volatility Intelligence Platform](https://github.com/RaajitSingh1306/volatility-intelligence-platform) API in real time to fetch current market regimes, GARCH volatility, and predictive probability distributions.

---

## Limitations & Roadmap

### Known Limitations
- **Corpus Scope**: Currently indexed with 5 representative statutory frameworks rather than the full legislative gazette of SEBI Master Circulars and RBI Master Directions.
- **Text-Only PDF Extraction**: PyMuPDF extraction (`fitz.get_text()`) extracts plain-text blocks; complex tabular disclosure matrices, financial ratios, and multi-column schedules are not parsed structurally.
- **Groq Free-Tier Rate Limits**: Production inference operates under Groq cloud rate limits (~30 requests/minute), requiring exponential backoff and client throttling for burst query traffic.
- **Volatile BM25 Index**: The BM25 index is maintained in-memory across worker lifecycles; while reconstructed dynamically from Qdrant scroll on startup, it lacks disk-backed persistence.
- **Collection Overwrite on Ingest**: Running `scripts/ingest_docs.py` executes `recreate_collection`, wiping previous vectors without incremental hash-based versioning.
- **Fixed Fusion Weights**: The hybrid retrieval scoring weight (60% dense vector, 40% BM25 sparse) is empirically tuned rather than dynamically learned via cross-entropy optimization on domain benchmarks.
- **Single-Turn Context Assembly**: The supervisor and RAG agent process queries independently; multi-turn coreference resolution and multi-hop statutory synthesis across separate acts are not yet supported.

### Roadmap
- [ ] **Full Gazette Ingestion**: Expand corpus to encompass comprehensive SEBI Master Circulars, PIT (Prohibition of Insider Trading), ICDR schedules, and AIF regulations.
- [ ] **Table-Aware Document Extraction**: Integrate Camelot/Table Transformer for structural extraction of regulatory financial ratios, disclosure tables, and penalty matrices.
- [ ] **Incremental Document Versioning**: Implement sha256 document hashing and payload timestamps for append-only delta indexing without full collection recreation.
- [ ] **Two-Stage Cross-Encoder Reranking**: Deploy a cross-encoder reranker (`bge-reranker-large`) over top-20 hybrid candidates to optimize top-5 precision.
- [ ] **Multi-Hop Knowledge Graph (GraphRAG)**: Construct entity relation graphs connecting cross-statutory provisions between SEBI LODR, Companies Act 2013, and RBI Master Directions.
- [ ] **Session & Thread Memory**: Implement persistent multi-turn conversational memory backed by Redis/PostgreSQL session storage.

---

## License

MIT License. Built for regulatory compliance research and fintech intelligence.
