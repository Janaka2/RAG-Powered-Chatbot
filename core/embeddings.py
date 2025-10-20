
"""
Embedding wrapper. Centralizing here lets you switch models easily.
"""
from __future__ import annotations
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        X = self.model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
        return np.asarray(X, dtype="float32")
