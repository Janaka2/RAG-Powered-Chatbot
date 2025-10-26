
import json, argparse, re
from collections import Counter

def normalize(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\.\-:/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def f1_score(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0
    precision = common / max(1, len(pred_tokens))
    recall = common / max(1, len(gold_tokens))
    return 2 * precision * recall / (precision + recall)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    gold = {}
    with open(args.gold, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            gold[q["id"]] = q["answers"]

    total = len(gold)
    em_hits = 0
    f1_sum = 0.0

    with open(args.pred, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            qid = p["id"]
            pred = p.get("answer","")
            # Compare against any of the acceptable gold answers
            ganswers = gold.get(qid, [""])
            em = any(normalize(pred) == normalize(ga) for ga in ganswers)
            if em:
                em_hits += 1
            f1 = max(f1_score(pred, ga) for ga in ganswers)
            f1_sum += f1

    print(f"Questions: {total}")
    print(f"Exact Match: {em_hits/total:.3f}")
    print(f"Token F1: {f1_sum/total:.3f}")

if __name__ == "__main__":
    main()
