
# RAG Test Pack (FAISS + BM25 + MMR/Hybrid)

This pack includes:
- `docs.jsonl` — small, diverse corpus with overlapping facts, distractors, and numeric reasoning.
- `qas.jsonl` — questions with expected answers and gold document IDs.
- `evaluate_retrieval.py` — compute Recall@k, MRR, and nDCG@k from your system's ranked results.
- `evaluate_generation.py` — compute simple Exact Match/F1 and optional embedding similarity.

## File formats

**docs.jsonl**
```json
{"id": "D01", "title": "...", "text": "...", "tags": ["..."]}
```

**qas.jsonl**
```json
{"id": "Q01", "question": "...", "answers": ["..."], "expected_context_doc_ids": ["D01","D02"], "type":"extractive"}
```

## How to evaluate

1. Run your retriever/generator and write a `predictions.jsonl` with one line per question:
```json
{"id":"Q01","ranked_doc_ids":["D01","D05","D02","D03","D04"],"answer":"25 days; up to 5 days can be carried over."}
```

2. Retrieval metrics
```bash
python evaluate_retrieval.py --gold qas.jsonl --pred predictions.jsonl --k 1 3 5
```

3. Generation metrics (optional; EM/F1; cosine if you plug an embedding model)
```bash
python evaluate_generation.py --gold qas.jsonl --pred predictions.jsonl
```

### Tips for your hybrid stack

- Normalize text (diacritics, case, punctuation). Keep both raw and normalized fields for BM25.
- Tune BM25 (k1, b) on `qas.jsonl`. Start k1=1.5, b=0.6.
- Blend BM25 + dense with α∈[0.3,0.7]. Re-rank top-50 with MMR or cross-encoder.
- For small corpora (<10k), FAISS IndexFlatIP is often enough; switch to IVF/HNSW as you scale.
- Use answerable/unanswerable checks (e.g., threshold on max score) and abstain when uncertain.

