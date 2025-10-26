
"""
Compatibility utilities mapping your previous function names to the new modular ones.
If your original code imported: load_paths, load_text_from_paths, chunk_texts, mmr_select, extract_title
they will work and delegate to the new modules.
"""
from __future__ import annotations
from typing import List, Dict, Tuple
from pathlib import Path

from core.utils import list_doc_paths as _list_doc_paths, read_texts as _read_texts, title_from_path as _title_from_path
from core.chunking import chunk_text as _chunk_text
from core.retrievers.mmr import mmr_select as _mmr_select

def load_paths(folder: str | Path) -> List[Path]:
    """Old name → new implementation."""
    return _list_doc_paths(folder)

def load_text_from_paths(paths: List[str | Path]) -> Dict[str, str]:
    """Old name → new implementation."""
    return _read_texts(paths)

def chunk_texts(text: str, *, source: str, title: str, chunk_size: int = 800, overlap: int = 120) -> List[Dict]:
    """Return plain dicts for older call sites that expected a list of dicts."""
    recs = _chunk_text(text, source=source, title=title, chunk_size=chunk_size, overlap=overlap)
    return [{"id": r.id, "text": r.text, "source": r.source, "title": r.title} for r in recs]

def mmr_select(query_vec, cand_embs, lambda_: float, k: int):
    """Directly proxy to new mmr implementation."""
    return _mmr_select(query_vec, cand_embs, lambda_, k)

def extract_title(path: str | Path) -> str:
    """Old name → new implementation."""
    return _title_from_path(path)
