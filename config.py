# config.py
from dataclasses import dataclass, field
from pathlib import Path

# -------- Paths --------
@dataclass
class Paths:
    root: Path = Path(".")
    docs_dir: Path = Path("docs")
    index_dir: Path = Path("storage")

    faiss_index_path: Path = field(default_factory=lambda: Path("storage") / "faiss.index")
    chunk_store_path: Path = field(default_factory=lambda: Path("storage") / "chunks.json")
    bm25_index_path: Path = field(default_factory=lambda: Path("storage") / "bm25.json")

    def __post_init__(self):
        # normalize
        self.root = Path(self.root)
        self.docs_dir = Path(self.docs_dir)
        self.index_dir = Path(self.index_dir)
        # keep artifacts under index_dir if not absolute
        if not self.faiss_index_path.is_absolute():
            self.faiss_index_path = self.index_dir / self.faiss_index_path.name
        if not self.chunk_store_path.is_absolute():
            self.chunk_store_path = self.index_dir / self.chunk_store_path.name
        if not self.bm25_index_path.is_absolute():
            self.bm25_index_path = self.index_dir / self.bm25_index_path.name

# -------- Models (add openai_chat_model so core/llm.py can use it) --------
@dataclass
class Models:
    # sentence-transformers model for embeddings
    embed_model_name: str = "all-MiniLM-L6-v2"

    # LLM settings expected by core/llm.py / pipeline.py
    openai_chat_model: str = "gpt-4o-mini"   # change if you like
    temperature: float = 0.2
    max_tokens: int = 800

# -------- Retrieval knobs --------
@dataclass
class Retrieval:
    top_k: int = 5
    mmr_lambda: float = 0.5     # 0→more diversity, 1→more relevance
    hybrid_alpha: float = 0.6    # weight for dense score in hybrid fusion

# -------- Chunking --------
@dataclass
class Chunking:
    chunk_size: int = 800
    overlap: int = 120

# -------- Root settings --------
@dataclass
class Settings:
    paths: Paths = field(default_factory=Paths)
    models: Models = field(default_factory=Models)
    retrieval: Retrieval = field(default_factory=Retrieval)
    chunking: Chunking = field(default_factory=Chunking)