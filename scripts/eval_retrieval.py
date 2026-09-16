"""Evaluation script for Hybrid Retriever.
Computes Recall@K and MRR across regulatory benchmark test cases.
Saves results to eval/retrieval_report.json.
"""
import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.retriever import HybridRetriever

TEST_CASES = [
    {
        "id": "TC-01",
        "query": "What is the minimum public shareholding requirement under SEBI LODR?",
        "expected_source": "sebi_lodr_2015.pdf",
        "target_keywords": ["25%", "twenty-five", "regulation 38", "public shareholding"]
    },
    {
        "id": "TC-02",
        "query": "What triggers an open offer under SEBI SAST regulations?",
        "expected_source": "sebi_sast_regulations.pdf",
        "target_keywords": ["25%", "voting rights", "open offer", "regulation 3"]
    },
    {
        "id": "TC-03",
        "query": "What is the minimum promoters contribution and lock-in period for IPO under SEBI ICDR?",
        "expected_source": "sebi_icdr_2018.pdf",
        "target_keywords": ["20%", "promoter", "lock-in", "18 months", "eighteen"]
    },
    {
        "id": "TC-04",
        "query": "What are RBI guidelines on independent model validation and audit trails?",
        "expected_source": "rbi_model_risk_management.pdf",
        "target_keywords": ["model risk", "independent", "validation", "audit", "five"]
    },
    {
        "id": "TC-05",
        "query": "What is the maximum penalty for failure to prevent personal data breach under DPDPA 2023?",
        "expected_source": "dpdpa_2023.pdf",
        "target_keywords": ["250 crore", "two hundred and fifty", "penalty", "section 33"]
    },
    {
        "id": "TC-06",
        "query": "What is the notice period and approval requirement for related party transactions under LODR?",
        "expected_source": "sebi_lodr_2015.pdf",
        "target_keywords": ["related party", "21 days", "audit committee", "regulation 23"]
    },
    {
        "id": "TC-07",
        "query": "What is the creeping acquisition limit under SEBI SAST?",
        "expected_source": "sebi_sast_regulations.pdf",
        "target_keywords": ["creeping", "5%", "five per cent", "financial year"]
    },
    {
        "id": "TC-08",
        "query": "What are the lock-in periods for anchor investors under ICDR?",
        "expected_source": "sebi_icdr_2018.pdf",
        "target_keywords": ["anchor", "30 days", "90 days", "lock-in"]
    }
]

def evaluate_retrieval(k: int = 5) -> Dict[str, Any]:
    print(f"Initializing HybridRetriever for Recall@{k} and MRR evaluation...")
    retriever = HybridRetriever()

    results_detail = []
    hits_at_k = 0
    reciprocal_ranks = []

    print("-" * 75)
    print(f"{'ID':<6} | {'Query':<45} | {'Hit?':<5} | {'Rank':<5} | {'Top Source':<20}")
    print("-" * 75)

    for case in TEST_CASES:
        cid = case["id"]
        query = case["query"]
        expected_src = case["expected_source"].lower()
        keywords = [kw.lower() for kw in case["target_keywords"]]

        retrieved = retriever.retrieve(query, k=k)

        first_hit_rank = None
        for rank_idx, item in enumerate(retrieved, start=1):
            text_lower = item.get("text", "").lower()
            item_src = item.get("source", "").lower()

            # A hit matches the source document and at least one target keyword
            has_keyword = any(kw in text_lower for kw in keywords)
            source_matches = expected_src in item_src or not item_src

            if has_keyword and source_matches:
                first_hit_rank = rank_idx
                break

        if first_hit_rank is not None and first_hit_rank <= k:
            hits_at_k += 1
            rr = 1.0 / first_hit_rank
            reciprocal_ranks.append(rr)
            hit_str = "YES"
            rank_str = str(first_hit_rank)
        else:
            reciprocal_ranks.append(0.0)
            hit_str = "NO"
            rank_str = "N/A"

        top_src = retrieved[0].get("source", "none") if retrieved else "none"
        short_query = query[:42] + "..." if len(query) > 45 else query
        print(f"{cid:<6} | {short_query:<45} | {hit_str:<5} | {rank_str:<5} | {top_src:<20}")

        results_detail.append({
            "id": cid,
            "query": query,
            "hit": hit_str == "YES",
            "rank": first_hit_rank,
            "expected_source": case["expected_source"],
            "top_retrieved": [
                {"source": r.get("source"), "score": r.get("rerank_score"), "snippet": r.get("text")[:100]}
                for r in retrieved[:3]
            ]
        })

    recall_at_k = hits_at_k / len(TEST_CASES)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    print("-" * 75)
    print(f"Total Test Cases : {len(TEST_CASES)}")
    print(f"Recall@{k}         : {recall_at_k:.4f} ({hits_at_k}/{len(TEST_CASES)})")
    print(f"MRR (Mean RR)    : {mrr:.4f}")
    print("-" * 75)

    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    report_data = {
        "metric": f"Recall@{k} & MRR",
        "num_test_cases": len(TEST_CASES),
        "recall_at_k": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
        "k": k,
        "details": results_detail
    }

    report_path = eval_dir / "retrieval_report.json"
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"Saved retrieval evaluation report to {report_path}")

    return report_data

if __name__ == "__main__":
    evaluate_retrieval()
