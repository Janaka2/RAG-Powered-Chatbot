
"""
RRF-based fusion retriever combining dense FAISS and BM25 using rank-level fusion.
"""
from __future__ import annotations
from typing import List, Dict, Tuple

from core.stores.hybrid import rrf_fuse

class RRFHybridRetriever:
    def __init__(self, faiss_store, bm25_store, embedder, k_rrf: int = 60):
        self.faiss = faiss_store
        self.bm25 = bm25_store
        self.embedder = embedder
        self.k_rrf = k_rrf

    def _key(self, d: Dict) -> str:
        return f"{d.get('source','')}#{d.get('id','')}"

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        qv = self.embedder.encode([query], normalize=True)
        # get generous lists from each channel for better fusion
        cand_k = max(top_k * 5, 20)
        d_items, d_scores = self.faiss.search(qv, cand_k)
        s_items, s_scores = self.bm25.search(query, cand_k)

        d_ranked = [(self._key(it), float(sc)) for it, sc in zip(d_items, d_scores)]
        s_ranked = [(self._key(it), float(sc)) for it, sc in zip(s_items, s_scores)]
        fused = rrf_fuse([d_ranked, s_ranked], k=top_k, k_rrf=self.k_rrf)

        # Map back to chunk dicts
        by_key = {self._key(it): it for it in (d_items + s_items)}
        out = []
        for kid, score in fused:
            it = dict(by_key[kid])
            it["_score_rrf"] = float(score)
            out.append(it)
        return out
