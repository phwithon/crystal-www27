"""K-fold 교차 예측(roberta_veracity/kfold_predictions.csv, train_kfold.py)을 읽어 전체 기사에 pred_label, pred_prob_fake(보정),
veracity_source, final_label, fold를 붙여 articles.jsonl을 같은 순서로 덮어쓴다 (split은 전건 "test").
crystal/data/veracity/factcheck.json(factcheck.py 산출)이 있으면, 공개 팩트체크가 매칭된 기사는 그 rating을 verdict로 채택해
final_label을 덮고 veracity_source="factcheck", factcheck={publisher,url,rating}로 둔다 (논문 2.2). 매칭 없으면 RoBERTa 판정.
실행: python -X utf8 crystal/veracity/predict.py"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATH = os.path.join(ROOT, "crystal", "data", "articles.jsonl")
PRED = os.path.join(ROOT, "roberta_veracity", "kfold_predictions.csv")
FACTCHECK = os.path.join(ROOT, "crystal", "data", "veracity", "factcheck.json")


def main():
    with open(PRED, encoding="utf-8", newline="") as f:
        pred = {r["news_id"]: r for r in csv.DictReader(f)}
    fc = {}
    if os.path.exists(FACTCHECK):
        with open(FACTCHECK, encoding="utf-8") as f:
            fc = json.load(f)
    rows = [json.loads(l) for l in open(PATH, encoding="utf-8")]
    n_fc = 0
    for r in rows:
        p = pred[r["article_id"]]
        r.update({"split": "test", "fold": int(p["fold"]), "pred_label": int(p["pred"]), "pred_prob_fake": round(float(p["prob_fake"]), 6),
                  "factcheck": None, "veracity_source": "model", "final_label": int(p["pred"])})
        hit = fc.get(r["article_id"])
        if hit and hit.get("label") in (0, 1):  # 공개 팩트체크가 있으면 그 판정을 verdict로 (논문 2.2)
            r["final_label"] = int(hit["label"])
            r["veracity_source"] = "factcheck"
            r["factcheck"] = {"publisher": hit.get("publisher", ""), "url": hit.get("url", ""), "rating": hit.get("rating", "")}
            n_fc += 1
    tmp = PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, PATH)
    acc = sum(r["pred_label"] == r["label"] for r in rows) / len(rows)
    print(f"done {len(rows)} rows, pooled acc {acc:.4f} (RoBERTa), factcheck verdicts {n_fc}")


if __name__ == "__main__":
    main()
