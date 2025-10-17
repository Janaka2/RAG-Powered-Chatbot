import os, re
from typing import List, Dict
from pathlib import Path
from pypdf import PdfReader

# --------- Loading ----------
def load_paths(folder: str):
    paths = []
    for p in Path(folder).glob("*"):
        if p.suffix.lower() in [".pdf", ".txt", ".md"]:
            paths.append(str(p))
    return paths

def load_text_from_paths(paths: List[str]):
    out = {}
    for p in paths:
        if p.lower().endswith(".pdf"):
            out[p] = pdf_to_text(p)
        else:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                out[p] = f.read()
    return out

def pdf_to_text(path: str) -> str:
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)

def extract_title(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

# --------- Chunking ----------
def chunk_texts(text: str, source: str, title: str, chunk_size: int = 800, overlap: int = 120):
    # simple sentence-aware-ish chunker by splitting on paragraph breaks first
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    pieces = []
    buf = []
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
            # start new buffer; carry overlap from end of previous
            carry = pieces[-1][-overlap:] if pieces else ""
            buf = [carry, para] if carry else [para]
            cur_len = len(" ".join(buf))
    if buf:
        pieces.append("\n\n".join(buf))
    records = []
    for i, p in enumerate(pieces):
        records.append({"id": f"{title}_{i}", "text": p, "source": source, "title": title})
    return records

# --------- MMR (diversification) ----------
import numpy as np

def mmr_select(query_vec, candidate_texts: List[str], lambda_: float, k: int, embed_fn):
    if len(candidate_texts) <= k:
        return list(range(len(candidate_texts)))
    cand_embs = embed_fn(candidate_texts)
    # cosine sims since embeddings assumed unit-normalized
    query = query_vec.reshape(1, -1)
    sims = (cand_embs @ query.T).flatten()
    selected = []
    remaining = list(range(len(candidate_texts)))
    while len(selected) < k and remaining:
        if not selected:
            next_idx = int(np.argmax(sims[remaining]))
            selected.append(remaining.pop(next_idx))
            continue
        # diversity term = max similarity to already selected
        div = []
        for ridx in remaining:
            sim_to_sel = np.max(cand_embs[ridx:ridx+1] @ cand_embs[selected].T)
            div.append(sim_to_sel)
        div = np.array(div)
        mmr = lambda_ * sims[remaining] - (1 - lambda_) * div
        pick = int(np.argmax(mmr))
        selected.append(remaining.pop(pick))
    return selected