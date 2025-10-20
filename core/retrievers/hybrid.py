
"""
Hybrid retriever: combines FAISS (dense) and BM25 (sparse) using simple weighted score fusion.
You can swap this for RRF if you prefer rank-based fusion.
"""
from __future__ import annotations
from typing import List, Dict, Tuple

class HybridRetriever:
    def __init__(self, faiss_store, bm25_store, embedder, alpha: float = 0.6):
        self.faiss = faiss_store
        self.bm25 = bm25_store
        self.embedder = embedder
        self.alpha = alpha

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        q = self.embedder.encode([query], normalize=True)
        dense_items, dense_scores = self.faiss.search(q, top_k)
        sparse_items, sparse_scores = self.bm25.search(query, top_k)

        # Build dicts by id(text hash) or by stable index; here we use (source, id) pair
        def key(d: Dict) -> str:
            return f"{d.get('source','')}#{d.get('id','')}"

        scored: Dict[str, Dict] = {}

        for it, s in zip(dense_items, dense_scores):
            k = key(it)
            it = dict(it)  # shallow copy
            it["_score_dense"] = float(s)
            it["_score_sparse"] = 0.0
            scored[k] = it

        for it, s in zip(sparse_items, sparse_scores):
            k = key(it)
            if k not in scored:
                it = dict(it)
                it["_score_dense"] = 0.0
                it["_score_sparse"] = float(s)
                scored[k] = it
            else:
                scored[k]["_score_sparse"] = float(s)

        # Normalize scores per channel (avoid domination)
        if scored:
            dmax = max((v["_score_dense"] for v in scored.values()), default=1.0) or 1.0
            smax = max((v["_score_sparse"] for v in scored.values()), default=1.0) or 1.0
            for v in scored.values():
                v["_score"] = self.alpha * (v["_score_dense"]/dmax) + (1 - self.alpha) * (v["_score_sparse"]/smax)

        ranked = sorted(scored.values(), key=lambda x: x.get("_score", 0.0), reverse=True)[:top_k]
        return ranked
