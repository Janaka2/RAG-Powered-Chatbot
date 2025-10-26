
# RAG • Modular Skeleton (Python)

Production-ready *skeleton* you can grow:
- **Embeddings**: SentenceTransformers (MiniLM-v6) – easy to swap.
- **Vector store**: FAISS (cosine/IP).
- **Sparse**: BM25 (rank_bm25).
- **Retrievers**: Similarity, **MMR**, **Hybrid** (dense+sparse).
- **UI**: Gradio (messages type).
- **LLM**: OpenAI Chat Completions (pluggable).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
mkdir -p docs storage
# add PDFs/TXT/MD into docs/
python ingest.py
python app.py
```

## Why this structure?
- **Separation of concerns**: `core/` has small focused modules (embedding, chunking, stores, retrievers).
- **Multiple search modes**: switch at runtime: similarity / MMR / hybrid.
- **Scales later**: you can replace FAISS with a service (e.g., Milvus, Qdrant), or BM25 with Elasticsearch/Opensearch.
- **Testability**: pure-Python functions are easy to unit test.

## Where to customize
- `config.py` → models, paths, retrieval params.
- `core/chunking.py` → chunk policy (size/overlap).
- `core/embeddings.py` → switch embedding model.
- `core/stores/*` → swap vector/sparse backends.
- `core/retrievers/*` → add re-rankers (Cross-Encoder), add RRF, A/B testers.
- `core/llm.py` → pick another LLM provider (Azure OpenAI etc.).

## Roadmap ideas
- Add **cross-encoder re-ranking** (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- Add **RRF** fusion as option in UI.
- Add **evaluations**: answer correctness, faithfulness, context recall.
- Add **async batching** for embedding calls.
- Add **token-based chunker** (tiktoken) and PDF layout-aware parsing.
```



---
### Migration notes
- Your original files are preserved under `/legacy`.
- Keep using `from rag_core import RAGPipeline` and `import utils` — the compat shims map to the new modules.
- Prefer importing new modules directly for future work (see `core/*`).


### New: Cross-Encoder Re-ranking & RRF
- **hybrid+rerank** uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-score top candidates.
- **rrf** performs Reciprocal Rank Fusion between FAISS and BM25 lists.


### Evaluation (accuracy • faithfulness • latency)
Dataset format: **JSONL** with fields:
```json
{"question": "...", "answer": "...", "spans": ["optional", "phrases"]}
```
- `answer`: reference answer for EM/F1/Jaccard.
- `spans` (optional): phrases expected in supporting context; used for **citation recall** heuristic.

Run:
```bash
python eval.py --dataset data/sample.jsonl --retriever hybrid --top_k 5 --ce_pool 5
```
Outputs go to `reports/` (CSV + JSON summary).

### Cross-Encoder pool size
In the UI, tweak **"CE pool factor (×Top-K)"** to control how many candidates are re-ranked for `hybrid+rerank`.
