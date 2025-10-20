
"""
I/O utilities: load file paths, read text from PDFs/TXT/MD.
"""
from pathlib import Path
from typing import Dict, List
from pypdf import PdfReader

ALLOWED_EXTS = {".pdf", ".txt", ".md"}

def list_doc_paths(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    return [p for p in folder.glob("*") if p.suffix.lower() in ALLOWED_EXTS]

def read_texts(paths: List[str | Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".pdf":
            out[str(p)] = pdf_to_text(p)
        else:
            out[str(p)] = p.read_text(encoding="utf-8", errors="ignore")
    return out

def pdf_to_text(path: str | Path) -> str:
    reader = PdfReader(str(path))
    texts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        texts.append(t)
    return "\n".join(texts)

def title_from_path(path: str | Path) -> str:
    p = Path(path)
    return p.stem
