import os, json, pickle, faiss, numpy as np
import shutil
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from utils import load_paths, load_text_from_paths, chunk_texts, mmr_select, extract_title

# OpenAI SDK (v1.x)
import openai
openai.api_key = os.environ.get("OPENAI_API_KEY")

class RAGPipeline:
    def __init__(self, index_dir, docs_dir, embed_model_name,
                 openai_model="gpt-4o-mini", top_k=5, mmr_lambda=0.5):
        self.index_dir = index_dir
        self.docs_dir = docs_dir
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(self.docs_dir, exist_ok=True)

        self.embed = SentenceTransformer(embed_model_name)
        self.dim = self.embed.get_sentence_embedding_dimension()
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda

        self.docstore_path = os.path.join(self.index_dir, "chunks.pkl")
        self.meta_path     = os.path.join(self.index_dir, "meta.json")
        self.faiss_path    = os.path.join(self.index_dir, "faiss.index")

        self.openai_model = openai_model
        self._load_or_init()

    # ---------- Index I/O ----------
    def _load_or_init(self):
        if all(os.path.exists(p) for p in [self.faiss_path, self.docstore_path, self.meta_path]):
            self.index = faiss.read_index(self.faiss_path)
            with open(self.docstore_path, "rb") as f:
                self.chunks = pickle.load(f)
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.chunks = []
            self.meta = {"sources": {}}

    def save(self):
        faiss.write_index(self.index, self.faiss_path)
        with open(self.docstore_path, "wb") as f:
            pickle.dump(self.chunks, f)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    # ---------- Ingestion ----------
    def rebuild_from_folder(self) -> int:
        paths = load_paths(self.docs_dir)
        texts_by_path = load_text_from_paths(paths)
        all_records = []
        for p, text in texts_by_path.items():
            title = extract_title(p)
            for rec in chunk_texts(text, source=p, title=title):
                all_records.append(rec)
        self._build_index_from_records(all_records)
        return len(self.chunks)

    def add_files(self, files) -> List[str]:
        """
        Accepts list of file paths (str) or file-like objects.
        Copies into self.docs_dir and indexes incrementally.
        """
        os.makedirs(self.docs_dir, exist_ok=True)
        saved = []

        for f in files or []:
            # Case 1: already a file path (string)
            if isinstance(f, str):
                src = f
                if not os.path.isfile(src):
                    continue
                dst = os.path.join(self.docs_dir, os.path.basename(src))
                if os.path.abspath(src) != os.path.abspath(dst):
                    shutil.copy2(src, dst)
                else:
                    dst = src
                saved.append(dst)
                continue

            # Case 2: file-like (Gradio TemporaryUploadedFile, etc.)
            name = getattr(f, "name", None) or getattr(f, "orig_name", None) or "upload.bin"
            dst = os.path.join(self.docs_dir, os.path.basename(name))
            if hasattr(f, "read"):
                with open(dst, "wb") as out:
                    out.write(f.read())
            else:
                # last resort: try to treat it as a path
                src = str(f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                else:
                    continue
            saved.append(dst)

        # index new files
        texts_by_path = load_text_from_paths(saved)
        new_records = []
        for p, text in texts_by_path.items():
            title = extract_title(p)
            for rec in chunk_texts(text, source=p, title=title):
                new_records.append(rec)
        if new_records:
            self._append_to_index(new_records)
        return saved

    def _build_index_from_records(self, records: List[Dict]):
        texts = [r["text"] for r in records]
        embs = self._embed_norm(texts)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(embs.astype("float32"))
        self.chunks = records
        self._refresh_meta(records)

    def _append_to_index(self, records: List[Dict]):
        if self.index.ntotal == 0:
            return self._build_index_from_records(records)
        texts = [r["text"] for r in records]
        embs = self._embed_norm(texts)
        self.index.add(embs.astype("float32"))
        self.chunks.extend(records)
        self._refresh_meta(records)

    def _embed_norm(self, texts: List[str]) -> np.ndarray:
        X = self.embed.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(X, dtype="float32")

    def _refresh_meta(self, records: List[Dict]):
        for r in records:
            src = r["source"]
            self.meta["sources"].setdefault(src, {"count": 0, "title": r.get("title")})
            self.meta["sources"][src]["count"] += 1

    # ---------- Retrieval + Generation ----------
    def retrieve(self, query: str) -> Tuple[List[Dict], List[int]]:
        q = self._embed_norm([query])
        scores, idxs = self.index.search(q, max(self.top_k * 3, self.top_k))
        idxs = idxs[0].tolist()
        cands = [self.chunks[i] for i in idxs if i >= 0]
        selected = mmr_select(
            query_vec=q[0],
            candidate_texts=[c["text"] for c in cands],
            lambda_=self.mmr_lambda,
            k=self.top_k,
            embed_fn=lambda x: self._embed_norm(x),
        )
        chosen = [cands[i] for i in selected]
        return chosen, selected

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

    def _gen_openai(self, prompt: str) -> str:
        resp = openai.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system", "content": "You are a concise, citation-friendly assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()

    def answer(self, question: str, chat_history: Optional[List[List[str]]] = None, system_hint: Optional[str] = None):
        if chat_history:
            last_turns = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in chat_history[-2:]])
            question = f"{question}\n\nConversation context:\n{last_turns}\n"

        contexts, _ = self.retrieve(question)
        prompt = self._build_prompt(question, contexts, system_hint)
        answer = self._gen_openai(prompt)
        citations = [{"title": c["title"], "source": c["source"]} for c in contexts]
        return answer, citations