
"""
MMR (Maximal Marginal Relevance) retriever built on top of FAISS candidates.
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np

def mmr_select(query_vec: np.ndarray, cand_embs: np.ndarray, lambda_: float, k: int) -> List[int]:
    sims = (cand_embs @ query_vec.reshape(-1,1)).flatten()  # cosine sims, embeddings normalized
    selected: List[int] = []
    remaining = list(range(len(cand_embs)))

    while len(selected) < k and remaining:
        if not selected:
            best = int(np.argmax(sims[remaining]))
            selected.append(remaining.pop(best))
            continue
        # compute max similarity to already selected
        sel_embs = cand_embs[selected]
        div = []
        for ridx in remaining:
            div.append(float((cand_embs[ridx:ridx+1] @ sel_embs.T).max()))
        div = np.asarray(div)
        mmr = lambda_ * sims[remaining] - (1 - lambda_) * div
        pick = int(np.argmax(mmr))
        selected.append(remaining.pop(pick))
    return selected

class MMRRetriever:
    def __init__(self, faiss_store, embedder, lambda_: float = 0.5, pool_factor: int = 3):
        self.faiss = faiss_store
        self.embedder = embedder
        self.lambda_ = lambda_
        self.pool_factor = pool_factor  # how many candidates to pull before MMR

    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        q = self.embedder.encode([query], normalize=True)
        # get a larger candidate pool
        cand_items, cand_scores = self.faiss.search(q, max(top_k * self.pool_factor, top_k))
        cand_texts = [c["text"] for c in cand_items]
        cand_embs = self.embedder.encode(cand_texts, normalize=True)
        idxs = mmr_select(q[0], cand_embs, self.lambda_, top_k)
        picked = [cand_items[i] for i in idxs]
        for it in picked:
            it["_score_dense"] = 1.0  # placeholder; order already diversified
        return picked
