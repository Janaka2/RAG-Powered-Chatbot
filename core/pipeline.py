
"""
End-to-end RAG Pipeline:
- Ingest: load -> chunk -> embed -> index (FAISS + optional BM25)
- Retrieve: choose retriever (similarity / MMR / hybrid)
- Generate: build prompt + call LLM
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import shutil

from config import Settings
from core.utils import list_doc_paths, read_texts, title_from_path
from core.chunking import chunk_text, ChunkRecord
from core.embeddings import Embedder
from core.stores.vector_faiss import FaissStore
from core.stores.bm25 import BM25Store
from core.retrievers.similarity import SimilarityRetriever
from core.retrievers.mmr import MMRRetriever
from core.retrievers.hybrid import HybridRetriever
from core.llm import LLM

class RAGPipeline:
    def __init__(self, settings: Settings = Settings()):
        self.cfg = settings
        self.cfg.paths.index_dir.mkdir(parents=True, exist_ok=True)
        self.cfg.paths.docs_dir.mkdir(parents=True, exist_ok=True)

        # Core services
        self.embedder = Embedder(self.cfg.models.embed_model_name)
        self.faiss = FaissStore(self.cfg.paths.faiss_index_path, self.cfg.paths.chunk_store_path, self.embedder.dim)
        self.bm25 = BM25Store(self.cfg.paths.bm25_index_path)
        self.llm = LLM(self.cfg.models.openai_chat_model)

        # Retrievers (you can switch at runtime)
        self.sim_retriever = SimilarityRetriever(self.faiss, self.embedder)
        self.mmr_retriever = MMRRetriever(self.faiss, self.embedder, lambda_=self.cfg.retrieval.mmr_lambda)
        self.hybrid_retriever = HybridRetriever(self.faiss, self.bm25, self.embedder, alpha=self.cfg.retrieval.hybrid_alpha)

    # --------------- Ingestion ---------------
    def rebuild_from_folder(self) -> int:
        paths = list_doc_paths(self.cfg.paths.docs_dir)
        texts = read_texts(paths)
        records: List[Dict] = []
        for p, text in texts.items():
            title = title_from_path(p)
            for rec in chunk_text(text, source=p, title=title, chunk_size=self.cfg.chunking.chunk_size, overlap=self.cfg.chunking.overlap):
                records.append({"id": rec.id, "text": rec.text, "source": rec.source, "title": rec.title})

        # dense
        if records:
            embs = self.embedder.encode([r["text"] for r in records], normalize=True)
        else:
            import numpy as np
            embs = np.empty((0, self.embedder.dim), dtype="float32")
        self.faiss.reset(embs, records)

        # sparse
        self.bm25.build(records)

        self.save()
        return len(records)

    def add_files(self, files: List[str | Path]) -> List[str]:
        saved: List[str] = []
        docs_dir = self.cfg.paths.docs_dir
        docs_dir.mkdir(parents=True, exist_ok=True)
        for f in files or []:
            f = Path(f)
            if not f.exists() or not f.is_file():
                continue
            dst = docs_dir / f.name
            if f.resolve() != dst.resolve():
                shutil.copy2(str(f), str(dst))
            else:
                dst = f
            saved.append(str(dst))

        # index incrementally
        if saved:
            texts = read_texts(saved)
            new_records: List[Dict] = []
            for p, text in texts.items():
                title = title_from_path(p)
                for rec in chunk_text(text, source=p, title=title, chunk_size=self.cfg.chunking.chunk_size, overlap=self.cfg.chunking.overlap):
                    new_records.append({"id": rec.id, "text": rec.text, "source": rec.source, "title": rec.title})

            if new_records:
                embs = self.embedder.encode([r["text"] for r in new_records], normalize=True)
                self.faiss.add(embs, new_records)
                self.bm25.add(new_records)
                self.save()
        return saved

    def save(self):
        self.faiss.save()
        self.bm25.save()

    # --------------- Retrieval ---------------
    def _build_prompt(self, question: str, contexts: List[Dict], system_hint: Optional[str]) -> str:
        numbered = [f"[{i}] {c['text']}" for i, c in enumerate(contexts, start=1)]
        context_block = "\n\n".join(numbered)
        preface = system_hint or ""
        return (
            f"{preface}\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            f"Answer (cite sources like [1], [2]; say \"I don't know\" if not in context):"
        )

    def _citations(self, contexts: List[Dict]) -> List[Dict]:
        return [{"title": c.get("title",""), "source": c.get("source","")} for c in contexts]

    def answer(self, question: str, *, retriever: str = "mmr", system_hint: Optional[str] = None,
               chat_history: Optional[List[tuple[str,str]]] = None) -> tuple[str, List[Dict]]:
        if chat_history:
            last_turns = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in chat_history[-2:]])
            question = f"{question}\n\nConversation context:\n{last_turns}\n"

        top_k = self.cfg.retrieval.top_k
        if retriever == "similarity":
            contexts = self.sim_retriever.retrieve(question, top_k)
        elif retriever == "hybrid":
            contexts = self.hybrid_retriever.retrieve(question, top_k)
        else:
            contexts = self.mmr_retriever.retrieve(question, top_k)

        prompt = self._build_prompt(question, contexts, system_hint)
        answer = self.llm.generate(prompt, temperature=0.0, max_tokens=600)
        return answer, self._citations(contexts)
