"""Google Fact Check Tools API (claims:search)로 기사 핵심 주장의 공개 팩트체크를 조회해 캐시한다.
매칭되면 그 textualRating을 true/fake로 매핑해 verdict로 쓰고(veracity/predict.py가 veracity_source=factcheck로 반영),
출처(기관명·url)를 evidence로 남긴다. 매칭이 없으면 RoBERTa 판정을 그대로 쓴다.

캐시: crystal/data/veracity/factcheck.json  { "<article_id>": {"label": 0|1, "rating": str, "publisher": str, "url": str}, ... }
      (매칭·판정 가능한 기사만. 이 파일은 .gitignore로 저장소에 올리지 않는다.)
실행: set FACTCHECK_API_KEY=...  # 또는 .env (GOOGLE_API_KEY도 허용)
      python -X utf8 crystal/veracity/factcheck.py [--limit N] [--all] [--interval 0.2]
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from crystal.common import is_displayable  # noqa: E402

ARTICLES = os.path.join(ROOT, "crystal", "data", "articles.jsonl")
OUT = os.path.join(ROOT, "crystal", "data", "veracity", "factcheck.json")
ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

AMBIGUOUS = ("half", "mixture", "mixed", "partly", "partially", "barely", "unproven", "unverified",
             "outdated", "disputed", "no consensus", "needs context", "missing context", "lacks context",
             "research in progress", "in dispute")
FAKE_TERMS = ("false", "fake", "incorrect", "misleading", "miscaption", "misattribut", "hoax", "fabricat",
              "debunk", "pants on fire", "not true", "untrue", "inaccurate", "no evidence", "scam", "distort",
              "doctored", "manipulat", "baseless", "satire", "altered", "spoof", "fiction", "conspiracy")
TRUE_TERMS = ("true", "correct", "accurate", "confirmed", "legit", "authentic", "verified")


def map_rating(text):
    """자유 텍스트 등급 → 1(fake) / 0(true) / None(판정 불가). 애매·부정 등급을 먼저 걸러 오판을 줄인다."""
    if not text:
        return None
    r = text.strip().lower()
    if any(a in r for a in AMBIGUOUS):
        return None
    if any(f in r for f in FAKE_TERMS):
        return 1
    if any(t in r for t in TRUE_TERMS):
        return 0
    return None


def load_key():
    for name in ("FACTCHECK_API_KEY", "GOOGLE_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    env = os.path.join(ROOT, ".env")
    if os.path.exists(env):
        with open(env, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                for name in ("FACTCHECK_API_KEY", "GOOGLE_API_KEY"):
                    if line.startswith(name + "="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("FACTCHECK_API_KEY (or GOOGLE_API_KEY) not found in environment or .env")


def first_sentence(body):
    m = re.search(r".+?[.!?](?:\s|$)", (body or "").strip())
    return (m.group(0) if m else (body or ""))[:200].strip()


def search(query, key, lang="en", max_retry=4):
    """claims:search 호출 → claims 리스트. 429/5xx는 지수 백오프로 재시도."""
    params = urllib.parse.urlencode({"query": query[:500], "key": key, "languageCode": lang, "pageSize": 5})
    url = f"{ENDPOINT}?{params}"
    for attempt in range(max_retry):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode("utf-8")).get("claims", [])
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 503) and attempt < max_retry - 1:
                time.sleep(2 * (attempt + 1))
                continue
            if e.code == 400:  # 쿼리 문제: 매칭 없음으로 취급
                return []
            raise
        except urllib.error.URLError:
            if attempt < max_retry - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    return []


def lookup(article, key, use_first_sentence=True):
    """기사 제목(없으면 첫 문장)으로 조회해 판정 가능한 첫 claimReview를 반환. 매칭 없으면 None."""
    queries = [article.get("title") or ""]
    if use_first_sentence:
        fs = first_sentence(article.get("body"))
        if fs and fs.lower() != (article.get("title") or "").strip().lower():
            queries.append(fs)
    for q in queries:
        if not q.strip():
            continue
        for claim in search(q, key):
            for rev in claim.get("claimReview") or []:
                label = map_rating(rev.get("textualRating"))
                if label is None:
                    continue
                pub = (rev.get("publisher") or {})
                return {"label": label, "rating": rev.get("textualRating"),
                        "publisher": pub.get("name") or pub.get("site") or "", "url": rev.get("url") or "",
                        "query": q, "claim": claim.get("text", "")[:200]}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="처음 N건만 (0=전체)")
    ap.add_argument("--all", action="store_true", help="is_displayable 외 전체 기사도 조회")
    ap.add_argument("--interval", type=float, default=0.2, help="호출 간 최소 간격(초)")
    ap.add_argument("--no-first-sentence", action="store_true")
    a = ap.parse_args()
    key = load_key()
    rows = [json.loads(l) for l in open(ARTICLES, encoding="utf-8")]
    targets = [r for r in rows if a.all or is_displayable(r.get("title"), r.get("body"))]
    if a.limit:
        targets = targets[:a.limit]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cache = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            cache = json.load(f)
    done = set(cache)  # 이미 조회해 매칭된 기사. (미매칭은 재조회되므로 전량 재개 시 중복 호출 주의)
    todo = [r for r in targets if r["article_id"] not in done]
    print(f"targets {len(targets)}  cached {len(cache)}  todo {len(todo)}", flush=True)

    t0, last, n_hit = time.time(), 0.0, 0
    for i, r in enumerate(todo, 1):
        wait = a.interval - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.time()
        hit = lookup(r, key, use_first_sentence=not a.no_first_sentence)
        if hit:
            cache[r["article_id"]] = hit
            n_hit += 1
        if i % 50 == 0 or i == len(todo):
            with open(OUT + ".tmp", "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=1)
            os.replace(OUT + ".tmp", OUT)
            print(f"  {i}/{len(todo)}  matched {n_hit}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"done. cache entries {len(cache)} (+{n_hit} this run) -> {OUT}")


if __name__ == "__main__":
    main()
