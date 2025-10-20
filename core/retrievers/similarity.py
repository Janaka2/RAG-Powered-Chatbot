
"""
Pure similarity top-k retriever from FAISS scores.
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np

class SimilarityRetriever:
    def __init__(self, faiss_store, embedder):
        self.faiss = faiss_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        q = self.embedder.encode([query], normalize=True)
        items, scores = self.faiss.search(q, top_k)
        # attach scores for potential downstream use
        for it, s in zip(items, scores):
            it["_score_dense"] = float(s)
        return items
