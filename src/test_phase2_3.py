"""
PHASE 2 & 3: Traditional Blocking & Candidate Generation Test Script

Loads a sample of preprocessed data, runs the blockers, computes candidate union,
and evaluates Candidate Recall against the ground truth.
"""

import sys
import os
import logging
import time
from collections import defaultdict

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure output encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('phase2_3_test')

from src.preprocessing import preprocess_dataframe
from src.blocking import TokenBlocker, NgramBlocker


def run_blocking_test(sample_size=10000):
    """Run blockers on a sample of data to evaluate recall and reduction."""
    
    print("\n" + "=" * 80)
    print(f"PHASE 2 & 3: BLOCKING TEST (sample_size={sample_size})")
    print("=" * 80)
    
    # 1. Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # 2. Load ground truth and get a sample of S1 entities that have matches in S2
    logger.info("Loading ground truth...")
    gt_df = pd.read_csv(config['data']['train']['ground_truth'], sep='\t')
    
    # Filter to only entities with matches, for recall evaluation
    gt_matches = gt_df.dropna(subset=['matched_entity_ids']).copy()
    
    # Take a sample of S1 IDs
    s1_sample_ids = gt_matches['source1_entity_id'].head(sample_size).values
    
    # Build ground truth dictionary for these IDs (only S2 matches for this test)
    # The gt contains both S2 and S3, we only want to test S2 recall right now
    true_s2_matches = defaultdict(set)
    total_true_s2 = 0
    
    for s1_id, match_str in zip(gt_matches['source1_entity_id'].head(sample_size), gt_matches['matched_entity_ids'].head(sample_size)):
        matches = [m for m in match_str.split(',') if m.startswith('S2-')]
        if matches:
            true_s2_matches[s1_id] = set(matches)
            total_true_s2 += len(matches)
            
    logger.info(f"Loaded ground truth: {len(true_s2_matches)} S1 entities have {total_true_s2} total S2 matches.")
    
    # 3. Load S1 data for the sample
    logger.info("Loading S1 data...")
    s1_df = pd.read_csv(config['data']['train']['source1'], sep='\t')
    s1_sample = s1_df[s1_df['entity_id'].isin(true_s2_matches.keys())].copy()
    logger.info(f"S1 sample size: {len(s1_sample)}")
    
    # 4. Load S2 data (we need all of it, but maybe just a 500k chunk for testing speed)
    s2_limit = 500000
    logger.info(f"Loading first {s2_limit} rows of S2 data for test speed...")
    s2_df = pd.read_csv(config['data']['train']['source2'], sep='\t', nrows=s2_limit)
    
    # Filter the ground truth to ONLY include S2 matches that are actually in our 500k sample!
    s2_valid_ids = set(s2_df['entity_id'].values)
    filtered_true_s2 = defaultdict(set)
    total_filtered_true_s2 = 0
    for s1_id, matches in true_s2_matches.items():
        valid = matches.intersection(s2_valid_ids)
        if valid:
            filtered_true_s2[s1_id] = valid
            total_filtered_true_s2 += len(valid)
            
    logger.info(f"After filtering to S2 sample: {len(filtered_true_s2)} S1 entities have {total_filtered_true_s2} total true matches.")
    
    if total_filtered_true_s2 == 0:
        logger.error("No true matches in the S2 sample. Cannot evaluate recall. Increase sample size or randomize.")
        return
        
    s1_sample = s1_sample[s1_sample['entity_id'].isin(filtered_true_s2.keys())]
    
    # 5. Preprocess
    s1_processed = preprocess_dataframe(s1_sample, "S1 Sample")
    s2_processed = preprocess_dataframe(s2_df, "S2 Sample")
    
    # 6. Initialize Blockers
    blockers = [
        ("Name Token", TokenBlocker(
            config['blocking']['name_token'], 
            column='name_normalized', 
            min_token_len=config['blocking']['name_token']['min_token_length'],
            max_freq_ratio=config['blocking']['name_token']['max_token_frequency_ratio'],
            stopwords=set(config['blocking']['name_token']['stopwords'])
        )),
        ("Address Token", TokenBlocker(
            config['blocking']['address_token'], 
            column='address_normalized',
            min_token_len=config['blocking']['address_token']['min_token_length'],
            max_freq_ratio=config['blocking']['address_token']['max_token_frequency_ratio']
        )),
        ("Name N-gram", NgramBlocker(
            config['blocking']['ngram'],
            column='name_normalized',
            n=config['blocking']['ngram']['n'],
            top_k=config['blocking']['ngram']['top_k']
        )),
        ("Address N-gram", NgramBlocker(
            config['blocking']['ngram'],
            column='address_normalized',
            n=config['blocking']['ngram']['n'],
            top_k=config['blocking']['ngram']['top_k']
        ))
    ]
    
    # 7. Run Blockers
    final_candidates = defaultdict(set)
    
    for b_name, blocker in blockers:
        print(f"\n--- Running {b_name} Blocker ---")
        t0 = time.time()
        blocker.fit(s2_processed)
        candidates = blocker.transform(s1_processed)
        elapsed = time.time() - t0
        
        # Evaluate individual blocker
        found_matches = 0
        total_cands = 0
        for s1_id, true_matches in filtered_true_s2.items():
            b_cands = candidates.get(s1_id, set())
            found_matches += len(true_matches.intersection(b_cands))
            total_cands += len(b_cands)
            
            # Add to union
            final_candidates[s1_id].update(b_cands)
            
        recall = found_matches / total_filtered_true_s2 if total_filtered_true_s2 else 0
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Candidates generated: {total_cands}")
        print(f"  Avg candidates per query: {total_cands / len(s1_processed):.1f}")
        print(f"  Recall: {found_matches}/{total_filtered_true_s2} ({recall:.4f})")
    
    # 8. Evaluate Union
    print("\n" + "=" * 40)
    print("UNION CANDIDATE EVALUATION")
    print("=" * 40)
    
    total_union_cands = sum(len(c) for c in final_candidates.values())
    found_matches = 0
    
    for s1_id, true_matches in filtered_true_s2.items():
        b_cands = final_candidates.get(s1_id, set())
        found_matches += len(true_matches.intersection(b_cands))
        
    recall = found_matches / total_filtered_true_s2 if total_filtered_true_s2 else 0
    
    print(f"Total union candidates: {total_union_cands}")
    print(f"Avg candidates per query: {total_union_cands / len(s1_processed):.1f}")
    print(f"Final Recall: {found_matches}/{total_filtered_true_s2} ({recall:.4f})")


if __name__ == '__main__':
    run_blocking_test()
