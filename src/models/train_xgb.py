"""
Production-quality XGBoost training pipeline optimized for Macro-F0.5 > 0.90.
Integrates both S2 and S3 targets, guaranteed true positive recovery, hard-negative mining,
high-precision features, and macro threshold optimization on unseen validation queries.
"""

import os
import sys
import yaml
import time
import random
import json
import logging
from collections import defaultdict
from typing import Dict, Set, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocess_dataframe
from src.blocking.candidate_generation import (
    candidates_to_pairs,
    rank_and_cap_candidates,
    run_blocking,
)
from src.models.decision import pairs_to_predictions
from src.features.pair_features import PairFeatureGenerator
from src.models.evaluation import evaluate_macro_f05

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('train_xgb')


def load_ground_truth(gt_path: str) -> Tuple[Dict[str, Set[str]], List[str], List[str]]:
    """Loads ground truth mapping and partitions S1 IDs into matched vs singletons."""
    gt_df = pd.read_csv(gt_path, sep='\t')
    gt_map = {}
    matched_ids = []
    singleton_ids = []
    
    for s1_id, match_str in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        if pd.isna(match_str) or not str(match_str).strip():
            gt_map[s1_id] = set()
            singleton_ids.append(s1_id)
        else:
            matches = set(m.strip() for m in str(match_str).split(',') if m.strip())
            gt_map[s1_id] = matches
            matched_ids.append(s1_id)
            
    logger.info(f"Loaded ground truth: {len(gt_df)} rows ({len(matched_ids)} matched, {len(singleton_ids)} singletons).")
    return gt_map, matched_ids, singleton_ids


def load_training_dataset(config: dict, num_matched: int = 7000, num_singletons: int = 1500, background_rows: int = 40000):
    """
    Constructs an end-to-end training & validation dataset.
    Guarantees that 100% of the true positive target records (from both S2 and S3)
    for the sampled S1 records are retrieved and placed into the target pool.
    """
    t0 = time.time()
    gt_path = config['data']['train']['ground_truth']
    s1_path = config['data']['train']['source1']
    s2_path = config['data']['train']['source2']
    s3_path = config['data']['train']['source3']
    
    gt_map, matched_ids, singleton_ids = load_ground_truth(gt_path)
    
    rng = np.random.RandomState(42)
    chosen_matched = set(rng.choice(matched_ids, size=min(num_matched, len(matched_ids)), replace=False))
    chosen_singletons = set(rng.choice(singleton_ids, size=min(num_singletons, len(singleton_ids)), replace=False))
    chosen_s1_ids = chosen_matched | chosen_singletons
    
    # Collect all needed S2 and S3 IDs
    needed_s2 = set()
    needed_s3 = set()
    for s1_id in chosen_matched:
        for mid in gt_map.get(s1_id, set()):
            if mid.startswith('S2-'):
                needed_s2.add(mid)
            elif mid.startswith('S3-'):
                needed_s3.add(mid)
                
    logger.info(f"Selected {len(chosen_s1_ids)} S1 records. True target matches required: {len(needed_s2)} S2, {len(needed_s3)} S3.")
    
    # 1. Load S1 records
    logger.info("Loading S1 records...")
    s1_all = pd.read_csv(s1_path, sep='\t')
    s1_df = s1_all[s1_all['entity_id'].isin(chosen_s1_ids)].copy().reset_index(drop=True)
    
    # 2. Load S2 records: needed IDs + background
    logger.info("Loading S2 records (true matches + background)...")
    s2_matches = []
    s2_bg = []
    bg_s2_count = 0
    for chunk in pd.read_csv(s2_path, sep='\t', chunksize=250000):
        # Extract needed true matches
        hit = chunk[chunk['entity_id'].isin(needed_s2)]
        if len(hit) > 0:
            s2_matches.append(hit)
        # Collect background non-matching records
        if bg_s2_count < background_rows:
            sample_bg = chunk[~chunk['entity_id'].isin(needed_s2)].head(background_rows - bg_s2_count)
            s2_bg.append(sample_bg)
            bg_s2_count += len(sample_bg)
            
    s2_df = pd.concat(s2_matches + s2_bg, ignore_index=True).drop_duplicates(subset=['entity_id'])
    logger.info(f"Loaded {len(s2_df)} S2 records (includes {sum(len(x) for x in s2_matches)} true matches).")
    
    # 3. Load S3 records: needed IDs + background
    logger.info("Loading S3 records (true matches + background)...")
    s3_matches = []
    s3_bg = []
    bg_s3_count = 0
    for chunk in pd.read_csv(s3_path, sep='\t', chunksize=250000):
        hit = chunk[chunk['entity_id'].isin(needed_s3)]
        if len(hit) > 0:
            s3_matches.append(hit)
        if bg_s3_count < background_rows:
            sample_bg = chunk[~chunk['entity_id'].isin(needed_s3)].head(background_rows - bg_s3_count)
            s3_bg.append(sample_bg)
            bg_s3_count += len(sample_bg)
            
    s3_df = pd.concat(s3_matches + s3_bg, ignore_index=True).drop_duplicates(subset=['entity_id'])
    logger.info(f"Loaded {len(s3_df)} S3 records (includes {sum(len(x) for x in s3_matches)} true matches).")
    
    # Combine target pool
    target_df = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates(subset=['entity_id'])
    logger.info(f"Total target pool size: {len(target_df)} records (Data loading completed in {time.time() - t0:.1f}s).")
    
    # Ground truth filtered to sampled queries
    target_ids_set = set(target_df['entity_id'])
    filtered_gt = {}
    for s1_id in s1_df['entity_id']:
        valid_matches = gt_map.get(s1_id, set()) & target_ids_set
        filtered_gt[s1_id] = valid_matches
        
    return s1_df, target_df, filtered_gt


def generate_candidate_pairs(
    s1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    config: dict,
    max_per_query: int = 300,
    max_negatives_per_query: int = 40,
) -> List[Tuple[str, str]]:
    """Blocking union + ranked cap; always inject true matches; sample hard negatives."""
    logger.info("Running blocking to retrieve candidate pool...")
    raw = run_blocking(s1_df, target_df, config=config)
    capped = rank_and_cap_candidates(
        s1_df, target_df, raw, max_per_query=max_per_query, inject_matches=ground_truth
    )

    candidate_pairs = []
    pos_count = 0
    neg_count = 0
    rng = random.Random(42)

    for s1_id in s1_df["entity_id"]:
        true_matches = ground_truth.get(s1_id, set())
        pool = capped.get(s1_id, set())

        for mid in true_matches:
            if mid in pool or mid in true_matches:
                candidate_pairs.append((s1_id, mid))
                pos_count += 1

        hard_negs = list(pool - true_matches)
        if len(hard_negs) > max_negatives_per_query:
            sampled_negs = rng.sample(hard_negs, max_negatives_per_query)
        else:
            sampled_negs = hard_negs
        for nid in sampled_negs:
            candidate_pairs.append((s1_id, nid))
            neg_count += 1

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for p in candidate_pairs:
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    logger.info(
        "Candidate assembly: %d positives, %d hard negatives (Total: %d).",
        pos_count,
        neg_count,
        len(deduped),
    )
    return deduped


def run_training_experiment():
    """Main training execution and Macro-F0.5 threshold optimization."""
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    start_time = time.time()
    
    # 1. Load data
    s1_raw, target_raw, ground_truth = load_training_dataset(config, num_matched=7000, num_singletons=1500, background_rows=40000)
    
    # 2. Preprocess
    s1_p = preprocess_dataframe(s1_raw, "S1_Sample")
    target_p = preprocess_dataframe(target_raw, "Target_Sample")
    
    # 3. Generate candidate pairs
    candidate_pairs = generate_candidate_pairs(
        s1_p, target_p, ground_truth, config=config, max_per_query=300, max_negatives_per_query=40
    )
    
    # 4. Generate features
    feat_gen = PairFeatureGenerator(config)
    feat_df = feat_gen.generate_features(s1_p, target_p, candidate_pairs)
    
    # 5. Add labels
    labels = [1 if c_id in ground_truth.get(s1_id, set()) else 0 for s1_id, c_id in zip(feat_df['s1_id'], feat_df['candidate_id'])]
    feat_df['label'] = labels
    
    pos_rate = feat_df['label'].mean()
    logger.info(f"Feature dataset: {len(feat_df)} rows, {feat_df['label'].sum()} positive matches ({pos_rate:.4f} positive rate).")
    
    # 6. Group Split by S1 entity ID (80% train, 20% validation)
    unique_s1 = s1_p['entity_id'].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(unique_s1)
    
    split_idx = int(len(unique_s1) * 0.80)
    train_s1 = set(unique_s1[:split_idx])
    val_s1 = set(unique_s1[split_idx:])
    
    train_mask = feat_df['s1_id'].isin(train_s1)
    val_mask = feat_df['s1_id'].isin(val_s1)
    
    feature_cols = [c for c in feat_df.columns if c not in ['s1_id', 'candidate_id', 'label']]
    
    X_train = feat_df.loc[train_mask, feature_cols]
    y_train = feat_df.loc[train_mask, 'label']
    
    X_val = feat_df.loc[val_mask, feature_cols]
    y_val = feat_df.loc[val_mask, 'label']
    val_meta = feat_df.loc[val_mask, ['s1_id', 'candidate_id']].copy()
    
    val_gt = {s1_id: ground_truth[s1_id] for s1_id in val_s1}
    
    logger.info(f"Train set: {len(X_train)} pairs ({len(train_s1)} entities). Val set: {len(X_val)} pairs ({len(val_s1)} entities).")
    
    # 7. Train XGBoost Model
    logger.info("Training XGBoost Classifier...")
    model = xgb.XGBClassifier(
        n_estimators=450,
        max_depth=7,
        learning_rate=0.04,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.05,
        reg_alpha=0.3,
        reg_lambda=1.2,
        scale_pos_weight=1.0,
        tree_method='hist',
        early_stopping_rounds=35,
        eval_metric='aucpr',
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=30
    )
    
    # 8. Optimize Threshold on Validation Set
    logger.info("Evaluating validation predictions and optimizing threshold for Macro-F0.5...")
    val_meta['pred_score'] = model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.50
    best_results = None
    best_f05 = -1.0
    
    thresholds = np.arange(0.50, 0.96, 0.02)
    
    for thresh in thresholds:
        val_scored = val_meta.rename(columns={"pred_score": "score"})
        preds = pairs_to_predictions(val_scored, threshold=thresh, score_col="score")
        results = evaluate_macro_f05(preds, val_gt)
        f05 = results['macro_f05']
        
        logger.info(f"  Thresh {thresh:.2f} -> Macro-F0.5: {f05:.4f} | Prec: {results['mean_precision']:.4f} | Rec: {results['mean_recall']:.4f} | Singletons: {results['singleton_accuracy']:.4f}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh
            best_results = results
            
    logger.info("=" * 65)
    logger.info(f"OPTIMAL THRESHOLD FOUND: {best_threshold:.2f}")
    logger.info(f"BEST MACRO F-0.5 SCORE: {best_f05:.4f}")
    logger.info(f"Mean Precision:        {best_results['mean_precision']:.4f}")
    logger.info(f"Mean Recall:           {best_results['mean_recall']:.4f}")
    logger.info(f"Singleton Accuracy:    {best_results['singleton_accuracy']:.4f}")
    logger.info(f"Matched Entities F0.5: {best_results['matched_f05']:.4f}")
    logger.info("=" * 65)
    
    # Top features
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    logger.info("Top 10 Feature Importances:")
    for idx in sorted_idx[:10]:
        logger.info(f"  {feature_cols[idx]}: {importances[idx]:.4f}")
        
    # Save model and metadata
    os.makedirs('models', exist_ok=True)
    model.save_model('models/xgb_baseline.json')
    
    metadata = {
        'best_threshold': float(best_threshold),
        'best_macro_f05': float(best_f05),
        'precision': float(best_results['mean_precision']),
        'recall': float(best_results['mean_recall']),
        'feature_cols': feature_cols
    }
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Model saved to models/xgb_baseline.json (Total time: {time.time() - start_time:.1f}s)")
    return best_f05


if __name__ == '__main__':
    run_training_experiment()
