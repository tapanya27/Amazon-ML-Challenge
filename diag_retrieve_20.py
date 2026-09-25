"""20-S1 timing diagnostic vs existing full train SQLite index. Does not modify the DB."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.blocking.sqlite_index import LEGALISH, SqliteReferenceIndex, _tokens
from src.features.pair_features import PairFeatureGenerator
from src.preprocessing import preprocess_dataframe


def rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return -1.0


def retrieve_timed(index, name, addr, country, retrieve_cap=350, max_tokens=6, require_country=True):
    scores = {}
    country = (country or "").lower()

    t0 = time.perf_counter()
    exact_ids = index.exact_name_rows(name, country if require_country else "")
    t_exact = time.perf_counter() - t0
    for rid in exact_ids:
        scores[rid] = scores.get(rid, 0) + 50

    def add_tokens(text, which, min_len, stop, weight):
        toks = list(_tokens(text, min_len, stop))
        toks.sort(key=lambda t: index.token_df(t, which) or 10**9)
        used = 0
        for t in toks:
            posting = index.posting(t, which)
            if posting.size == 0:
                continue
            used += 1
            for rid in posting:
                scores[int(rid)] = scores.get(int(rid), 0) + weight
            if used >= max_tokens or len(scores) >= retrieve_cap * 3:
                break

    t0 = time.perf_counter()
    add_tokens(name, "name", 3, LEGALISH, 3)
    add_tokens(addr, "addr", 2, set(), 2)
    t_tok = time.perf_counter() - t0

    t0 = time.perf_counter()
    if require_country and country and scores:
        recs = index.fetch_rows(list(scores.keys())[: retrieve_cap * 3])
        scores = {rid: scores[rid] for rid, _e, _n, _a, co in recs if co == country or not co}
    t_filter = time.perf_counter() - t0

    t0 = time.perf_counter()
    if not scores:
        t_cap = time.perf_counter() - t0
        empty = np.empty(0, dtype=np.int32)
        return empty, 0, False, t_exact, t_tok, t_filter, t_cap
    raw_n = len(scores)
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:retrieve_cap]
    rows = np.asarray([r for r, _ in ranked], dtype=np.int32)
    t_cap = time.perf_counter() - t0
    return rows, raw_n, raw_n >= retrieve_cap, t_exact, t_tok, t_filter, t_cap


def main():
    n = 20
    seed = 2026
    retrieve_cap = 350
    top_k = 30
    db = "data/indexes/s2s3.sqlite"
    assert os.path.isfile(db), db

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_baseline.json")

    s1_ids_all = pd.read_csv(cfg["data"]["train"]["source1"], sep="\t", usecols=["entity_id"])
    rng = np.random.RandomState(seed)
    chosen = rng.choice(s1_ids_all["entity_id"].values, size=n, replace=False)
    full_s1 = pd.read_csv(cfg["data"]["train"]["source1"], sep="\t")
    sample = full_s1[full_s1.entity_id.isin(chosen)]
    s1_p = preprocess_dataframe(sample, "diag20")

    peak = rss_mb()
    print(f"RSS open={peak:.1f} MB", flush=True)
    index = SqliteReferenceIndex(db)
    peak = max(peak, rss_mb())
    print(f"index.n={index.n} RSS after_open={rss_mb():.1f} MB", flush=True)
    if index.n != 10320219:
        print(f"WARNING unexpected refs count {index.n}", flush=True)

    feat_gen = PairFeatureGenerator({})
    names = s1_p["name_normalized"].fillna("").astype(str).tolist()
    addrs = s1_p["address_normalized"].fillna("").astype(str).tolist()
    countries = s1_p["country_normalized"].fillna("").astype(str).str.lower().tolist()
    s1_ids = s1_p["entity_id"].tolist()

    rows_out = []
    zero = 0
    for i, (s1_id, nm, ad, co) in enumerate(zip(s1_ids, names, addrs, countries)):
        nm = "" if nm == "nan" else nm
        ad = "" if ad == "nan" else ad
        t_s1 = time.perf_counter()

        rows, raw_n, hit, t_exact, t_tok, t_filter, t_cap = retrieve_timed(
            index, nm, ad, co, retrieve_cap=retrieve_cap
        )
        t_retrieve = t_exact + t_tok + t_filter + t_cap
        if rows.size == 0:
            zero += 1

        t0 = time.perf_counter()
        top = index.cheap_prerank(nm, ad, co, rows, top_k)
        t_prerank = time.perf_counter() - t0
        n_exp = int(len(top))

        t0 = time.perf_counter()
        feat_df = feat_gen.generate_from_index(index, [s1_id], [nm], [ad], [co], [top])
        t_feat = time.perf_counter() - t0

        t0 = time.perf_counter()
        if len(feat_df):
            _ = model.predict_proba(feat_df[feature_cols])[:, 1]
        t_xgb = time.perf_counter() - t0

        total = time.perf_counter() - t_s1
        cur = rss_mb()
        peak = max(peak, cur)
        rec = {
            "i": i,
            "s1_id": s1_id,
            "t_exact_name_s": round(t_exact, 4),
            "t_token_posting_s": round(t_tok, 4),
            "t_country_filter_s": round(t_filter, 4),
            "t_sqlite_retrieve_s": round(t_retrieve, 4),
            "raw_n_before_cap": int(raw_n),
            "cap_hit": bool(hit),
            "t_cap_s": round(t_cap, 4),
            "t_prerank_s": round(t_prerank, 4),
            "n_expensive": n_exp,
            "t_features_s": round(t_feat, 4),
            "t_xgb_s": round(t_xgb, 4),
            "t_total_s": round(total, 4),
            "rss_mb": round(cur, 1),
        }
        rows_out.append(rec)
        print(
            f"[{i+1:02d}/20] {s1_id} retrieve={t_retrieve:.3f}s "
            f"(exact={t_exact:.3f} tok={t_tok:.3f} filter={t_filter:.3f}) "
            f"raw={raw_n} cap={t_cap:.4f}s prerank={t_prerank:.3f}s "
            f"exp={n_exp} feat={t_feat:.3f}s xgb={t_xgb:.3f}s total={total:.3f}s rss={cur:.1f}",
            flush=True,
        )

    df = pd.DataFrame(rows_out)
    print("\n=== 20-S1 DIAGNOSTIC (full train index, production retrieve logic) ===", flush=True)
    print(f"index_rows={index.n} db={db}", flush=True)
    print(f"zero_candidate_s1={zero}/20", flush=True)
    print(f"RSS current={rss_mb():.1f} MB peak={peak:.1f} MB", flush=True)
    means = {
        "avg_sqlite_retrieve_s": float(df["t_sqlite_retrieve_s"].mean()),
        "avg_exact_name_s": float(df["t_exact_name_s"].mean()),
        "avg_token_posting_s": float(df["t_token_posting_s"].mean()),
        "avg_country_filter_s": float(df["t_country_filter_s"].mean()),
        "avg_raw_n_before_cap": float(df["raw_n_before_cap"].mean()),
        "avg_cap_s": float(df["t_cap_s"].mean()),
        "avg_prerank_s": float(df["t_prerank_s"].mean()),
        "avg_n_expensive": float(df["n_expensive"].mean()),
        "avg_features_s": float(df["t_features_s"].mean()),
        "avg_xgb_s": float(df["t_xgb_s"].mean()),
        "avg_total_s_per_s1": float(df["t_total_s"].mean()),
        "sum_total_s": float(df["t_total_s"].sum()),
        "zero_candidate_s1": zero,
        "peak_rss_mb": peak,
    }
    for k, v in means.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) and "n_" not in k and "zero" not in k and "peak" not in k else f"{k}: {v}", flush=True)
    os.makedirs("eval_train", exist_ok=True)
    with open("eval_train/diag20_timing.json", "w") as f:
        json.dump({"means": means, "rows": rows_out}, f, indent=2)
    print("wrote eval_train/diag20_timing.json", flush=True)


if __name__ == "__main__":
    main()
