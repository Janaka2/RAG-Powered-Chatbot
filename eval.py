
"""
Simple evaluation runner.

Dataset format: JSONL with fields:
{
  "question": "...",
  "answer": "...",                 # reference answer
  "spans": ["optional", "phrases"] # optional: phrases that should exist in supporting context
}

Usage:
python eval.py --dataset data/dev.jsonl --retriever mmr --top_k 5 --ce_pool 5 --judge none
"""
from __future__ import annotations
import argparse, json, os, time, csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from config import Settings
from core.pipeline import RAGPipeline
from core.eval.metrics import exact_match, token_f1, jaccard_overlap, citation_recall

def load_jsonl(path: str | Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    ap.add_argument("--retriever", default="mmr", choices=["similarity","mmr","hybrid","rrf","hybrid+rerank"])
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--ce_pool", type=int, default=5)
    ap.add_argument("--report_dir", default="reports")
    ap.add_argument("--judge", default="none", choices=["none"])  # placeholder for future LLM-judge
    args = ap.parse_args()

    settings = Settings()
    settings.retrieval.top_k = args.top_k
    settings.retrieval.ce_pool_factor = args.ce_pool
    rag = RAGPipeline(settings)

    data = load_jsonl(args.dataset)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    out_csv = report_dir / f"eval-{args.retriever}-k{args.top_k}-pool{args.ce_pool}-{ts}.csv"
    out_json = report_dir / f"eval-{args.retriever}-k{args.top_k}-pool{args.ce_pool}-{ts}.json"

    rows_csv = []
    agg = {"em":0.0, "f1":0.0, "jac":0.0, "cit":0.0, "count":0, "latency_ms":0.0}

    for i, sample in enumerate(data, 1):
        q = sample["question"]
        ref = sample.get("answer","").strip()
        spans = sample.get("spans") or []

        t0 = time.perf_counter()
        ans, ctxs = rag.answer(q, retriever=args.retriever)
        t1 = time.perf_counter()

        em = exact_match(ans, ref)
        f1 = token_f1(ans, ref)
        jac = jaccard_overlap(ans, ref)
        cit = citation_recall(ans, ctxs, spans)
        lat_ms = (t1 - t0)*1000.0

        rows_csv.append({
            "idx": i, "retriever": args.retriever, "top_k": args.top_k, "ce_pool": args.ce_pool,
            "em": f"{em:.3f}", "f1": f"{f1:.3f}", "jaccard": f"{jac:.3f}", "citation_recall": f"{cit:.3f}",
            "latency_ms": f"{lat_ms:.1f}", "question": q, "ref": ref, "pred": ans
        })

        agg["em"] += em; agg["f1"] += f1; agg["jac"] += jac; agg["latency_ms"] += lat_ms; agg["count"] += 1
        if cit >= 0: agg["cit"] += cit

    # Write CSV
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()) if rows_csv else [])
        w.writeheader()
        for r in rows_csv: w.writerow(r)

    # Write JSON summary
    n = max(1, agg["count"])
    summary = {
        "retriever": args.retriever, "top_k": args.top_k, "ce_pool": args.ce_pool,
        "examples": len(rows_csv),
        "avg_em": round(agg["em"]/n, 3),
        "avg_f1": round(agg["f1"]/n, 3),
        "avg_jaccard": round(agg["jac"]/n, 3),
        "avg_citation_recall": round((agg["cit"]/n) if agg["cit"]>0 else -1, 3),
        "avg_latency_ms": round(agg["latency_ms"]/n, 1),
        "csv_path": str(out_csv)
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows_csv}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
