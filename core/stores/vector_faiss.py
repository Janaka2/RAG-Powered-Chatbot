
"""
FAISS-backed dense vector store.
- Persists the FAISS index and JSONL chunk store.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Iterable
from pathlib import Path
import json
import numpy as np
import faiss

class FaissStore:
    def __init__(self, index_path: Path, chunk_store_path: Path, dim: int):
        self.index_path = index_path
        self.chunk_store_path = chunk_store_path
        self.dim = dim
        self._faiss = None
        self._chunks: List[Dict] = []
        self._load_or_init()

    # ---------- Persistence ----------
    def _load_or_init(self):
        if self.index_path.exists() and self.chunk_store_path.exists():
            self._faiss = faiss.read_index(str(self.index_path))
            # chunks in JSONL (one JSON per line)
            self._chunks = [json.loads(l) for l in self.chunk_store_path.read_text(encoding="utf-8").splitlines()]
        else:
            self._faiss = faiss.IndexFlatIP(self.dim)
            self._chunks = []

    def save(self):
        faiss.write_index(self._faiss, str(self.index_path))
        with self.chunk_store_path.open("w", encoding="utf-8") as f:
            for rec in self._chunks:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------- CRUD ----------
    def add(self, embeddings: np.ndarray, records: List[Dict]):
        self._faiss.add(embeddings.astype("float32"))
        self._chunks.extend(records)

    def reset(self, embeddings: np.ndarray, records: List[Dict]):
        self._faiss = faiss.IndexFlatIP(self.dim)
        if embeddings.size:
            self._faiss.add(embeddings.astype("float32"))
        self._chunks = list(records)

    # ---------- Query ----------
    def search(self, query_vec: np.ndarray, k: int) -> Tuple[List[Dict], np.ndarray]:
        scores, idxs = self._faiss.search(query_vec, k)
        idxs = idxs[0].tolist()
        items = [self._chunks[i] for i in idxs if i >= 0]
        return items, scores[0]
