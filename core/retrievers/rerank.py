
"""
Cross-Encoder re-ranking retriever.

Pipeline:
1) Get a candidate set (dense, sparse, or hybrid) — we reuse the hybrid scorer here.
2) Score (query, chunk) pairs with a Cross-Encoder.
3) Return top-K by CE score.
"""
from __future__ import annotations
from typing import List, Dict
from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, base_retriever, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", pool_factor: int = 5):
        self.base = base_retriever
        self.model = CrossEncoder(model_name)
        self.pool_factor = pool_factor

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        # Pull a slightly larger candidate pool from base
        pool_k = max(top_k * self.pool_factor, top_k)
        cands = self.base.retrieve(query, pool_k)
        if not cands:
            return []

        pairs = [(query, c["text"]) for c in cands]
        scores = self.model.predict(pairs)
        for it, s in zip(cands, scores):
            it["_score_ce"] = float(s)

        ranked = sorted(cands, key=lambda x: x.get("_score_ce", 0.0), reverse=True)[:top_k]
        return ranked
