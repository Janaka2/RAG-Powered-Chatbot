
"""
Simple paragraph-aware chunker with overlap.
You can swap this for character/token-based chunkers later.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import re

@dataclass
class ChunkRecord:
    id: str
    text: str
    source: str
    title: str

def chunk_text(text: str, *, source: str, title: str, chunk_size: int = 800, overlap: int = 120) -> List[ChunkRecord]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    pieces: List[str] = []
    buf: List[str] = []
    cur_len = 0

    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if cur_len + len(para) <= chunk_size:
            buf.append(para)
            cur_len += len(para)
        else:
            if buf:
                pieces.append("\n\n".join(buf))
            carry = pieces[-1][-overlap:] if pieces else ""
            buf = ([carry, para] if carry else [para])
            cur_len = len(" ".join(buf))

    if buf:
        pieces.append("\n\n".join(buf))

    recs: List[ChunkRecord] = []
    for i, p in enumerate(pieces):
        recs.append(ChunkRecord(id=f"{title}_{i}", text=p, source=source, title=title))
    return recs
