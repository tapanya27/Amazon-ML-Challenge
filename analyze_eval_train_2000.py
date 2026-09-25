"""Error analysis of eval_train 2000-S1 vs GT. Read-only on SQLite. No production changes."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yaml
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.blocking.sqlite_index import LEGALISH, SqliteReferenceIndex, _tokens
from src.features.pair_features import PairFeatureGenerator
from src.models.evaluation import compute_entity_f05
from src.preprocessing import preprocess_dataframe


def parse_ids(s):
    if pd.isna(s) or not str(s).strip():
        return set()
    return {x.strip() for x in str(s).split(",") if x.strip()}


def load_gt(path):
    gt_df = pd.read_csv(path, sep="\t")
    gt = {}
    for s1, ms in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        gt[s1] = parse_ids(ms)
    return gt


def blocking_fail_reason(index, qn, qa, qc, tn, ta, tc):
    reasons = []
    qn_tok = _tokens(qn, 3, LEGALISH)
    tn_tok = _tokens(tn, 3, LEGALISH)
    qa_tok = _tokens(qa, 2, set())
    ta_tok = _tokens(ta, 2, set())
    name_ov = qn_tok & tn_tok
    addr_ov = qa_tok & ta_tok
    usable_name = []
    for t in name_ov:
        df = index.token_df(t, "name")
        if 2 <= df <= 800:
            usable_name.append((t, df))
    usable_addr = []
    for t in addr_ov:
        df = index.token_df(t, "addr")
        if 2 <= df <= 800:
            usable_addr.append((t, df))
    if qn != tn:
        reasons.append("exact_name_mismatch")
    else:
        reasons.append("exact_name_match")
    if qc and tc and qc != tc:
        reasons.append("country_mismatch")
    if not name_ov:
        reasons.append("no_name_token_overlap")
    elif not usable_name:
        reasons.append("name_overlap_only_stop_or_extreme_df")
    if not addr_ov:
        reasons.append("no_addr_token_overlap")
    elif not usable_addr:
        reasons.append("addr_overlap_only_extreme_df")
    if not usable_name and not usable_addr and qn != tn:
        reasons.append("no_indexable_token_and_no_exact")
    return {
        "reasons": reasons,
        "q_name": qn,
        "t_name": tn,
        "q_addr": qa,
        "t_addr": ta,
        "q_country": qc,
        "t_country": tc,
        "name_token_overlap": sorted(name_ov)[:12],
        "addr_token_overlap": sorted(addr_ov)[:12],
        "usable_name_df": usable_name[:8],
        "usable_addr_df": usable_addr[:8],
    }


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    with open("models/model_metadata.json") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    threshold = float(meta.get("best_threshold", 0.92))
    model = xgb.XGBClassifier()
    model.load_model("models/xgb_baseline.json")

    sample = pd.read_csv("eval_train/_sample_s1.tsv", sep="\t")
    pred_df = pd.read_csv("eval_train/matching_results.tsv", sep="\t")
    cand_df = pd.read_csv("eval_train/candidate_pairs.tsv", sep="\t")
    gt_all = load_gt(cfg["data"]["train"]["ground_truth"])

    s1_ids = sample["entity_id"].tolist()
    assert len(s1_ids) == 2000, len(s1_ids)
    preds = {r.source1_entity_id: parse_ids(r.matched_entity_ids) for r in pred_df.itertuples()}
    cands_topk = {r.source1_entity_id: list(parse_ids(r.candidate_entity_ids)) if not pd.isna(r.candidate_entity_ids) else [] for r in cand_df.itertuples()}
    # preserve order from TSV
    cands_ordered = {}
    for r in cand_df.itertuples():
        raw = "" if pd.isna(r.candidate_entity_ids) else str(r.candidate_entity_ids)
        cands_ordered[r.source1_entity_id] = [x.strip() for x in raw.split(",") if x.strip()]

    s1_p = preprocess_dataframe(sample, "err")
    s1_p = s1_p.set_index("entity_id")

    index = SqliteReferenceIndex("data/indexes/s2s3.sqlite")
    print(f"index.n={index.n} threshold={threshold}", flush=True)

    needed_eids = set()
    for s1_id in s1_ids:
        needed_eids |= gt_all.get(s1_id, set())
        needed_eids |= preds.get(s1_id, set())
    index.conn.execute("CREATE TEMP TABLE need_eid (eid TEXT PRIMARY KEY)")
    index.conn.executemany("INSERT OR IGNORE INTO need_eid VALUES (?)", [(e,) for e in needed_eids])
    rec_by_eid = {}
    for eid, nm, ad, co in index.conn.execute(
        "SELECT r.entity_id, r.name, r.addr, r.country FROM refs r JOIN need_eid n ON n.eid = r.entity_id"
    ):
        rec_by_eid[eid] = (nm or "", ad or "", (co or "").lower())
    print(f"prefetched {len(rec_by_eid)} / {len(needed_eids)} true/pred ref rows", flush=True)

    feat_gen = PairFeatureGenerator({})
    cats = Counter()
    a_examples = []
    a_reason_counts = Counter()
    b_ranks = []
    c_examples = []
    d_scores = []
    e_examples = []

    def rec_for_eid(eid: str):
        return rec_by_eid.get(eid, ("", "", ""))

    need_retrieve = []
    for s1_id in s1_ids:
        true = gt_all.get(s1_id, set())
        pred = preds.get(s1_id, set())
        topk = cands_ordered.get(s1_id, [])
        topk_set = set(topk)
        f05 = compute_entity_f05(pred, true)
        if f05 == 1.0:
            cats["correct"] += 1
            continue
        if not topk and true:
            cats["E"] += 1
            q = s1_p.loc[s1_id]
            qn = str(q["name_normalized"] or "")
            qa = str(q["address_normalized"] or "")
            qc = str(q["country_normalized"] or "").lower()
            tinfo = []
            for tid in list(true)[:2]:
                tn, ta, tc = rec_for_eid(tid)
                tinfo.append({"true_id": tid, "t_name": tn, "t_addr": ta, "t_country": tc})
            if len(e_examples) < 12:
                e_examples.append({"s1_id": s1_id, "q_name": qn, "q_addr": qa, "q_country": qc, "true": tinfo})
            continue
        if not true and pred:
            cats["D"] += 1
            q = s1_p.loc[s1_id]
            qn = str(q.get("name_normalized") or "")
            qa = str(q.get("address_normalized") or "")
            qc = str(q.get("country_normalized") or "").lower()
            need_retrieve.append(("D", s1_id, true, pred, topk, qn, qa, qc))
            continue
        need_retrieve.append(("?", s1_id, true, pred, topk, None, None, None))

    print(f"pre-retrieve cats={dict(cats)} pending={len(need_retrieve)}", flush=True)

    for i, item in enumerate(need_retrieve):
        kind, s1_id, true, pred, topk, qn, qa, qc = item
        q = s1_p.loc[s1_id]
        qn = str(q["name_normalized"] or "") if qn is None else qn
        qa = str(q["address_normalized"] or "") if qa is None else qa
        qc = str(q["country_normalized"] or "").lower() if qc is None else qc
        qn = "" if qn == "nan" else qn
        qa = "" if qa == "nan" else qa

        rows, raw_n, hit = index.retrieve(qn, qa, qc, retrieve_cap=350)
        retrieve_eids = []
        eid_to_rank = {}
        if rows.size:
            recs = index.fetch_rows(rows.tolist())
            idmap = {r[0]: r[1] for r in recs}
            for rank, rid in enumerate(rows.tolist(), start=1):
                eid = idmap.get(int(rid))
                if eid:
                    retrieve_eids.append(eid)
                    eid_to_rank.setdefault(eid, rank)
        retrieve_set = set(retrieve_eids)
        topk_set = set(topk)
        prerank = index.cheap_prerank(qn, qa, qc, rows, 30)
        prerank_eids = []
        prerank_rank = {}
        if prerank.size:
            recs2 = {r[0]: r[1] for r in index.fetch_rows(prerank.tolist())}
            for rank, rid in enumerate(prerank.tolist(), start=1):
                eid = recs2.get(int(rid))
                if eid:
                    prerank_eids.append(eid)
                    prerank_rank.setdefault(eid, rank)

        if kind == "D":
            # scores of selected FPs
            row_lists = [prerank if prerank.size else rows[:30]]
            feat_df = feat_gen.generate_from_index(index, [s1_id], [qn], [qa], [qc], [row_lists[0]])
            if len(feat_df):
                feat_df["score"] = model.predict_proba(feat_df[feature_cols])[:, 1]
                for pid in pred:
                    sub = feat_df[feat_df["candidate_id"] == pid]
                    if len(sub):
                        d_scores.append(float(sub.iloc[0]["score"]))
            continue

        missed = true - pred
        missing_from_retrieve = [t for t in true if t not in retrieve_set]
        missing_from_topk = [t for t in true if t not in topk_set]

        if missing_from_retrieve:
            cats["A"] += 1
            for tid in missing_from_retrieve[:2]:
                tn, ta, tc = rec_for_eid(tid)
                info = blocking_fail_reason(index, qn, qa, qc, tn or "", ta or "", (tc or "").lower())
                info["s1_id"] = s1_id
                info["true_id"] = tid
                for r in info["reasons"]:
                    a_reason_counts[r] += 1
                if len(a_examples) < 15:
                    a_examples.append(info)
            continue

        if missing_from_topk:
            cats["B"] += 1
            for tid in missing_from_topk:
                b_ranks.append(
                    {
                        "s1_id": s1_id,
                        "true_id": tid,
                        "retrieve_rank": eid_to_rank.get(tid),
                        "prerank_rank": prerank_rank.get(tid),
                        "raw_n": int(raw_n),
                        "cap_hit": bool(hit),
                    }
                )
            continue

        cats["C"] += 1
        row_lists = [prerank]
        feat_df = feat_gen.generate_from_index(index, [s1_id], [qn], [qa], [qc], [row_lists[0]])
        if len(feat_df):
            feat_df["score"] = model.predict_proba(feat_df[feature_cols])[:, 1]
            true_rows = feat_df[feat_df["candidate_id"].isin(true)]
            fp_rows = feat_df[feat_df["candidate_id"].isin(pred - true)]
            fn_rows = feat_df[feat_df["candidate_id"].isin(true - pred)]
            if len(c_examples) < 20:
                def pack(df):
                    if df is None or df.empty:
                        return []
                    cols = ["candidate_id", "score"] + [c for c in ["name_exact", "name_token_set", "addr_exact", "addr_token_set", "country_same", "composite_score"] if c in df.columns]
                    return df[cols].head(3).to_dict(orient="records")
                c_examples.append(
                    {
                        "s1_id": s1_id,
                        "true": sorted(true),
                        "pred": sorted(pred),
                        "true_scores": pack(true_rows),
                        "fp_scores": pack(fp_rows),
                        "fn_in_topk_scores": pack(fn_rows),
                    }
                )

        if (i + 1) % 200 == 0:
            print(f"classified {i+1}/{len(need_retrieve)} {dict(cats)}", flush=True)

    n_err = 2000 - cats["correct"]
    report = {
        "n_s1": 2000,
        "n_correct_f05_1": cats["correct"],
        "n_incorrect": n_err,
        "categories": {
            "A_true_not_in_retrieve": cats["A"],
            "B_in_retrieve_below_topk": cats["B"],
            "C_in_topk_model_wrong": cats["C"],
            "D_fp_should_be_singleton": cats["D"],
            "E_zero_candidate": cats["E"],
        },
        "pct_of_2000": {
            k: round(100.0 * v / 2000, 2) for k, v in {
                "A": cats["A"], "B": cats["B"], "C": cats["C"], "D": cats["D"], "E": cats["E"], "correct": cats["correct"]
            }.items()
        },
        "pct_of_incorrect": {
            k: round(100.0 * cats[k] / n_err, 2) if n_err else 0 for k in "ABCDE"
        },
        "A_reason_counts": dict(a_reason_counts.most_common()),
        "A_examples": a_examples,
        "B_n_true_links": len(b_ranks),
        "B_retrieve_rank_p50": float(np.nanmedian([x["retrieve_rank"] if x["retrieve_rank"] else np.nan for x in b_ranks])) if b_ranks else None,
        "B_retrieve_rank_p90": float(np.nanpercentile([x["retrieve_rank"] for x in b_ranks if x["retrieve_rank"]], 90)) if any(x["retrieve_rank"] for x in b_ranks) else None,
        "B_n_beyond_350": sum(1 for x in b_ranks if x["retrieve_rank"] is None or x["retrieve_rank"] > 350),
        "B_n_prerank_none": sum(1 for x in b_ranks if x["prerank_rank"] is None),
        "B_examples": b_ranks[:20],
        "C_examples": c_examples,
        "D_n": cats["D"],
        "D_fp_score_mean": float(np.mean(d_scores)) if d_scores else None,
        "D_fp_score_min": float(np.min(d_scores)) if d_scores else None,
        "D_n_scores": len(d_scores),
        "D_n_score_ge_0.92": sum(1 for s in d_scores if s >= 0.92),
        "E_examples": e_examples,
        "threshold": threshold,
    }
    os.makedirs("eval_train", exist_ok=True)
    with open("eval_train/error_analysis_2000.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({k: report[k] for k in ["n_s1", "n_correct_f05_1", "n_incorrect", "categories", "pct_of_2000", "pct_of_incorrect", "A_reason_counts", "B_n_true_links", "B_retrieve_rank_p50", "B_retrieve_rank_p90", "D_fp_score_mean", "D_n_score_ge_0.92"]}, indent=2), flush=True)
    print("wrote eval_train/error_analysis_2000.json", flush=True)


if __name__ == "__main__":
    main()
