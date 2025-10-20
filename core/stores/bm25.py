
"""
BM25 (sparse) index using rank_bm25.
We store tokenized chunks and use BM25Okapi for classic bag-of-words search.
"""
from __future__ import annotations
from typing import List, Dict, Tuple
from pathlib import Path
import json
import re
from rank_bm25 import BM25Okapi

WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9_]+")

def simple_tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]

class BM25Store:
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self._corpus_tokens: List[List[str]] = []
        self._chunks: List[Dict] = []
        self._bm25: BM25Okapi | None = None
        self._load_or_init()

    def _load_or_init(self):
        if self.index_path.exists():
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._chunks = data["chunks"]
            self._corpus_tokens = data["corpus_tokens"]
            self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        else:
            self._chunks = []
            self._corpus_tokens = []
            self._bm25 = None

    def save(self):
        payload = {"chunks": self._chunks, "corpus_tokens": self._corpus_tokens}
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def build(self, chunks: List[Dict]):
        self._chunks = chunks
        self._corpus_tokens = [simple_tokenize(c["text"]) for c in chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    def add(self, new_chunks: List[Dict]):
        new_tokens = [simple_tokenize(c["text"]) for c in new_chunks]
        self._chunks.extend(new_chunks)
        self._corpus_tokens.extend(new_tokens)
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, k: int) -> Tuple[List[Dict], List[float]]:
        if not self._bm25:
            return [], []
        q_tokens = simple_tokenize(query)
        scores = self._bm25.get_scores(q_tokens)  # length = corpus size
        # top-k indices
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._chunks[i] for i in top_idxs], [scores[i] for i in top_idxs]
