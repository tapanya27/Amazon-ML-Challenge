import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

# Missing values and country distributions
for label, fname in [
    ('S1-train', 'dataset/train/train_source1.tsv'),
    ('S2-train', 'dataset/train/train_source2.tsv'),
    ('S3-train', 'dataset/train/train_source3.tsv'),
]:
    df = pd.read_csv(fname, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'])
    print(f'\n=== {label} ({len(df)} rows) ===')
    print(f'Missing values:\n{df.isnull().sum()}')
    print(f'Country distribution:\n{df["country"].value_counts()}')

# Ground truth statistics
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
print(f'\n=== Ground Truth ({len(gt)} rows) ===')
print(f'Missing matched_entity_ids: {gt["matched_entity_ids"].isnull().sum()}')

singletons = gt['matched_entity_ids'].isnull().sum()
matched = len(gt) - singletons
print(f'Singletons (no match): {singletons}')
print(f'Matched (has matches): {matched}')
print(f'Singleton ratio: {singletons/len(gt):.4f}')

match_counts = gt[gt['matched_entity_ids'].notna()]['matched_entity_ids'].str.split(',').str.len()
print(f'\nMatch count distribution for matched entities:')
print(match_counts.value_counts().sort_index().head(20))

# Test data stats
for label, fname in [
    ('S1-test', 'dataset/test/test_source1.tsv'),
    ('S2-test', 'dataset/test/test_source2.tsv'),
    ('S3-test', 'dataset/test/test_source3.tsv'),
]:
    df = pd.read_csv(fname, sep='\t', usecols=['entity_id', 'business_name', 'business_address', 'country'])
    print(f'\n=== {label} ({len(df)} rows) ===')
    print(f'Missing values:\n{df.isnull().sum()}')
    print(f'Country distribution:\n{df["country"].value_counts()}')
