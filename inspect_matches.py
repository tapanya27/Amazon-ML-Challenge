import pandas as pd

s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=5)
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt_map = gt.set_index('source1_entity_id')['matched_entity_ids'].to_dict()

s2_reader = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=100000)
s3_reader = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=100000)

target_ids = set()
for s1_id in s1['entity_id']:
    m = gt_map.get(s1_id)
    if pd.notna(m):
        for mid in m.split(','):
            target_ids.add(mid)

s2_matches = {}
for chunk in s2_reader:
    sub = chunk[chunk['entity_id'].isin(target_ids)]
    for _, r in sub.iterrows():
        s2_matches[r['entity_id']] = r.to_dict()
    if len(s2_matches) == len([x for x in target_ids if x.startswith('S2-')]):
        break

s3_matches = {}
for chunk in s3_reader:
    sub = chunk[chunk['entity_id'].isin(target_ids)]
    for _, r in sub.iterrows():
        s3_matches[r['entity_id']] = r.to_dict()
    if len(s3_matches) == len([x for x in target_ids if x.startswith('S3-')]):
        break

for _, r in s1.iterrows():
    s1_id = r['entity_id']
    print("="*60)
    print(f"S1 ID: {s1_id}")
    print(f"  Name:    {r['business_name']}")
    print(f"  Address: {r['business_address']}")
    print(f"  Country: {r['country']}")
    m = gt_map.get(s1_id)
    if pd.notna(m):
        for mid in m.split(','):
            rec = s2_matches.get(mid) or s3_matches.get(mid)
            if rec:
                print(f"  -> MATCH {mid}:")
                print(f"       Name:    {rec['business_name']}")
                print(f"       Address: {rec['business_address']}")
                print(f"       Country: {rec['country']}")
            else:
                print(f"  -> MATCH {mid}: (not in first chunks)")
    else:
        print("  -> SINGLETON")
