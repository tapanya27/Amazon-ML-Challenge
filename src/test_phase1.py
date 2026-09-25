"""
PHASE 1: Preprocessing & Normalization Test Script

Loads sample data from each source, applies normalization, and reports:
- Resulting schema
- Normalization examples (before/after)
- Missing values
- Issues found
"""

import sys
import os
import logging
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# Configure output encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

pd.set_option('display.max_colwidth', 100)
pd.set_option('display.width', 250)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('phase1_test')

from src.preprocessing import NameNormalizer, AddressNormalizer, CountryNormalizer, preprocess_dataframe


def test_name_normalizer():
    """Test the name normalizer with various edge cases."""
    norm = NameNormalizer()
    
    test_cases = [
        # (input, expected_description)
        ("Orelee's Barbershop", "apostrophe handling"),
        ("B+ Retail Inc", "punctuation + legal suffix"),
        ("Custom Wealth Services LLC", "LLC suffix"),
        ("Consulting Nyasa Nursing Private Limited", "full legal suffix"),
        ("SHIVSHAKTI VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com", "URL + uppercase"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", "Hindi/Devanagari"),
        ("-- Holloway Peak Inc Seafood", "leading punctuation"),
        ("Delta Tetlecommunication Inc", "typo preservation"),
        ("Lee and Lawson", "'and' keyword"),
        ("Smith & Wesson Enterprises", "ampersand + suffix"),
        ("Pvt. EFS Print Ventures Ltd.", "abbreviation with dots"),
        ("LLC Moncada Léarning Center", "accent + LLC prefix"),
        ("Ectolumdrex dba X+ Madison Inc", "DBA handling"),
        ("   Spaced   Out   Name   ", "whitespace"),
        (None, "None input"),
        ("", "empty string"),
        (float('nan'), "NaN input"),
        ("Société Générale S.A.", "French with accents"),
        ("L'Oréal International", "French apostrophe"),
        ("Bouygues Construction", "French construction"),
    ]
    
    print("\n" + "=" * 80)
    print("NAME NORMALIZER TEST RESULTS")
    print("=" * 80)
    
    for raw, desc in test_cases:
        result = norm.normalize(raw)
        print(f"\n  [{desc}]")
        print(f"    Input:  {repr(raw)}")
        print(f"    Output: {repr(result)}")


def test_address_normalizer():
    """Test the address normalizer with various edge cases."""
    norm = AddressNormalizer()
    
    test_cases = [
        ("1795 Westchester Drive, High Point, NC", "US standard"),
        ("2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "unit/apartment"),
        ("797, Lake Town Block A, Kolkata, Howrah, West Bengal", "Indian city/state"),
        ("2505, Tower 1, Oakwood, Runwal Greens, Mulund Goreagon Link Road, Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra", "Indian complex address"),
        ("OH, Columbus, 5559 Orville Avenue", "US reversed format"),
        ("KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi", "Indian KH number"),
        ("G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh", "Indian colony"),
        ("H.NO 204 C ROAD HOSHIARPUR, PUNJAB, Punjab", "Indian house number"),
        ("63/2275/7, ALHIND TOWER, FIRST FLOOR, JAFFERKHAN COLONY, KOZHIKODE, Kerala", "Indian floor"),
        ("282 SAXONY DRIVE, FTT MITCHELL, KY", "US typo"),
        ("Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ", "Kannada script"),
        ("2260- Housecreek Trail, Unit 407, Raleigh, North Carolina", "US with dash"),
        ("##8 Willow Oak Lane, Fl. 0, Saint Louis, Missouri", "double hash + floor"),
        ("P.O. Box 6009, Cincinnati, Ohio", "PO Box"),
        ("5780 Fawn Ct, Fort Worth, Texas", "court abbreviation"),
        (None, "None input"),
        ("", "empty string"),
        ("15 Rue de la Paix, 75002 Paris", "French address"),
        ("42 Boulevard Haussmann, Paris, Île-de-France", "French with region"),
    ]
    
    print("\n" + "=" * 80)
    print("ADDRESS NORMALIZER TEST RESULTS")
    print("=" * 80)
    
    for raw, desc in test_cases:
        result = norm.normalize(raw)
        print(f"\n  [{desc}]")
        print(f"    Input:  {repr(raw)}")
        print(f"    Output: {repr(result)}")


def test_country_normalizer():
    """Test the country normalizer."""
    norm = CountryNormalizer()
    
    test_cases = [
        ("US", "standard US"),
        ("India", "standard India"),
        ("France", "standard France"),
        ("usa", "lowercase variant"),
        ("United States", "full name"),
        ("United States of America", "very full name"),
        (" India ", "whitespace"),
        ("FR", "ISO code"),
        ("Deutschland", "non-English name"),
        ("SomeNewCountry", "open-set unknown"),
        (None, "None"),
        ("", "empty"),
    ]
    
    print("\n" + "=" * 80)
    print("COUNTRY NORMALIZER TEST RESULTS")
    print("=" * 80)
    
    for raw, desc in test_cases:
        result = norm.normalize(raw)
        print(f"  [{desc}]  {repr(raw)} → {repr(result)}")


def test_full_preprocessing(sample_size=1000):
    """Test full preprocessing pipeline on sample data."""
    
    print("\n" + "=" * 80)
    print(f"FULL PREPROCESSING TEST (sample_size={sample_size})")
    print("=" * 80)
    
    datasets = {
        'S1-train': 'dataset/train/train_source1.tsv',
        'S2-train': 'dataset/train/train_source2.tsv',
        'S3-train': 'dataset/train/train_source3.tsv',
    }
    
    for label, path in datasets.items():
        if not os.path.exists(path):
            print(f"\n  SKIP {label}: {path} not found")
            continue
            
        print(f"\n{'─' * 60}")
        print(f"  Processing {label}...")
        
        # Read sample
        df = pd.read_csv(path, sep='\t', nrows=sample_size)
        
        # Apply preprocessing
        t0 = time.time()
        df_processed = preprocess_dataframe(df, source_label=label)
        elapsed = time.time() - t0
        
        # Report schema
        print(f"\n  Schema after preprocessing:")
        for col in df_processed.columns:
            dtype = df_processed[col].dtype
            n_null = df_processed[col].isnull().sum()
            n_empty = (df_processed[col] == '').sum() if dtype == 'object' else 0
            print(f"    {col:30s}  dtype={dtype}  null={n_null}  empty={n_empty}")
        
        # Show before/after examples
        print(f"\n  Sample normalizations (first 5 rows):")
        for i in range(min(5, len(df_processed))):
            row = df_processed.iloc[i]
            print(f"\n    Row {i}:")
            print(f"      entity_id:          {row['entity_id']}")
            print(f"      name_orig:          {repr(row['business_name'][:80])}")
            print(f"      name_normalized:    {repr(row['name_normalized'][:80])}")
            print(f"      address_orig:       {repr(str(row['business_address'])[:80])}")
            print(f"      address_normalized: {repr(row['address_normalized'][:80])}")
            print(f"      country_orig:       {repr(row['country'])}")
            print(f"      country_normalized: {repr(row['country_normalized'])}")
        
        print(f"\n  Preprocessing time: {elapsed:.2f}s for {sample_size} rows")
        print(f"  Throughput: {sample_size/elapsed:.0f} rows/sec")


def test_performance_benchmark(n_rows=10000):
    """Benchmark preprocessing speed on a larger sample."""
    
    print("\n" + "=" * 80)
    print(f"PERFORMANCE BENCHMARK (n_rows={n_rows})")
    print("=" * 80)
    
    path = 'dataset/train/train_source1.tsv'
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return
    
    df = pd.read_csv(path, sep='\t', nrows=n_rows)
    
    t0 = time.time()
    df_processed = preprocess_dataframe(df, source_label='benchmark')
    elapsed = time.time() - t0
    
    print(f"  Rows processed: {n_rows}")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Throughput: {n_rows/elapsed:.0f} rows/sec")
    print(f"  Estimated time for 2.2M S1 rows: {2_206_821/max(1, n_rows/elapsed)/60:.1f} min")
    print(f"  Estimated time for 5M S2 rows: {5_034_616/max(1, n_rows/elapsed)/60:.1f} min")
    print(f"  Estimated time for 5.3M S3 rows: {5_285_603/max(1, n_rows/elapsed)/60:.1f} min")
    
    # Check for NaN/inf issues
    for col in ['name_normalized', 'address_normalized', 'country_normalized']:
        n_nan = df_processed[col].isnull().sum()
        n_inf = 0  # Strings can't be inf, but check anyway
        if n_nan > 0:
            print(f"  WARNING: {col} has {n_nan} NaN values!")
        else:
            print(f"  OK: {col} has no NaN values")


if __name__ == '__main__':
    print("=" * 80)
    print("PHASE 1: PREPROCESSING & NORMALIZATION TEST")
    print("=" * 80)
    
    test_name_normalizer()
    test_address_normalizer()
    test_country_normalizer()
    test_full_preprocessing(sample_size=1000)
    test_performance_benchmark(n_rows=10000)
    
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)
