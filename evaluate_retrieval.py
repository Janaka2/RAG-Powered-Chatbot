
import json, argparse, math
from collections import defaultdict

def dcg(scores):
    return sum((s / math.log2(i+2) for i, s in enumerate(scores)))

def ndcg_at_k(ranked, gold_set, k):
    gains = [1.0 if d in gold_set else 0.0 for d in ranked[:k]]
    ideal = sorted([1.0]*len(gold_set) + [0.0]*(k-len(gold_set)))[:k]
    return (dcg(gains) / dcg(ideal)) if dcg(ideal) > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--k", nargs="+", type=int, default=[1,3,5])
    args = ap.parse_args()

    gold = {}
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            gold[q["id"]] = set(q["expected_context_doc_ids"])

    preds = {}
    with open(args.pred, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            preds[p["id"]] = p.get("ranked_doc_ids", [])

    total = len(gold)
    hits_at_k = defaultdict(int)
    mrr_sum = 0.0
    ndcg_sums = defaultdict(float)

    for qid, gold_set in gold.items():
        ranked = preds.get(qid, [])
        # Hits@k and MRR
        first_hit = None
        for i, d in enumerate(ranked):
            if d in gold_set:
                if first_hit is None:
                    first_hit = i+1
                for k in args.k:
                    if i < k:
                        hits_at_k[k] += 1
        if first_hit:
            mrr_sum += 1.0 / first_hit
        # nDCG
        for k in args.k:
            ndcg_sums[k] += ndcg_at_k(ranked, gold_set, k)

    print(f"Questions: {total}")
    for k in args.k:
        print(f"Recall@{k}: {hits_at_k[k]/total:.3f}")
    print(f"MRR: {mrr_sum/total:.3f}")
    for k in args.k:
        print(f"nDCG@{k}: {ndcg_sums[k]/total:.3f}")

if __name__ == "__main__":
    main()
