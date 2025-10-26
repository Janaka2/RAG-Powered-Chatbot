
"""
Evaluation metrics for Q/A over RAG.
- exact_match
- token_f1
- jaccard_overlap (between predicted answer and reference)
- citation_recall (optional heuristic using provided contexts)
"""
from __future__ import annotations
import re
from typing import List, Dict, Tuple
from collections import Counter

_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")

def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def _tokens(s: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD.finditer(s)]

def exact_match(pred: str, ref: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(ref) else 0.0

def token_f1(pred: str, ref: str) -> float:
    ptoks, rtoks = _tokens(pred), _tokens(ref)
    if not ptoks and not rtoks:
        return 1.0
    if not ptoks or not rtoks:
        return 0.0
    pc, rc = Counter(ptoks), Counter(rtoks)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(pc.values()))
    recall = overlap / max(1, sum(rc.values()))
    return 2 * precision * recall / (precision + recall)

def jaccard_overlap(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def contains_any(text: str, needles: List[str]) -> bool:
    t = _normalize(text)
    return any(_normalize(n) in t for n in needles if n)

def citation_recall(answer: str, contexts: List[Dict], reference_spans: List[str] | None) -> float:
    """
    Heuristic: fraction of reference spans that appear in any retrieved context if present.
    If reference_spans is None/empty, returns -1.0 to signal N/A.
    """
    if not reference_spans:
        return -1.0
    ctx = " ".join(c.get("text", "") for c in contexts)
    hit = sum(1 for span in reference_spans if span and span.lower() in ctx.lower())
    return hit / max(1, len(reference_spans))
