
from __future__ import annotations
from typing import Protocol, List, Dict, Tuple

class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> List[Dict]:
        ...
