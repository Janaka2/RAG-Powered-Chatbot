
"""
Hybrid scorer utilities to fuse dense (FAISS) and sparse (BM25) results.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import math

def rrf_fuse(results: List[List[Tuple[str, float]]], k: int = 10, k_rrf: int = 60) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF).
    results: list of ranked lists; each is [(doc_id, score), ...] in rank order (best first)
    k_rrf: RRF constant (>0)
    """
    scores: Dict[str, float] = {}
    for ranked in results:
        for rank, (doc_id, _score) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    # return top-k by fused score
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
