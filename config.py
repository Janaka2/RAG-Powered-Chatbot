from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Paths:
    root: Path = Path(".")
    docs_dir: Path = Path("docs")
    index_dir: Path = Path("storage")
    faiss_index_path: Path = Path("storage") / "faiss.index"
    chunk_store_path: Path = Path("storage") / "chunks.jsonl"
    bm25_index_path: Path = Path("storage") / "bm25.json"

@dataclass
class Models:
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_chat_model: str = "gpt-4o-mini"

@dataclass
class Retrieval:
    top_k: int = 5
    mmr_lambda: float = 0.5
    hybrid_alpha: float = 0.6
    ce_pool_factor: int = 5

@dataclass
class Chunking:
    chunk_size: int = 800
    overlap: int = 120

@dataclass
class Settings:
    paths: Paths = field(default_factory=Paths)
    models: Models = field(default_factory=Models)
    retrieval: Retrieval = field(default_factory=Retrieval)
    chunking: Chunking = field(default_factory=Chunking)