#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data profiling script for Amazon ML Challenge 2026."""

import sys
import io
import os
import pandas as pd
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

BASE = 'student_resource/dataset'

def profile_source_file(path, nrows=None):
    """Profile a source TSV file."""
    df = pd.read_csv(path, sep='\t', nrows=nrows)
    info = {
        'path': path,
        'shape': list(df.shape),
        'columns': list(df.columns),
        'dtypes': {c: str(df[c].dtype) for c in df.columns},
    }
    for col in df.columns:
        info[f'{col}_null_pct'] = round(df[col].isnull().mean() * 100, 2)
        info[f'{col}_nunique'] = int(df[col].nunique())
        if df[col].dtype == 'object' or str(df[col].dtype) == 'str':
            lengths = df[col].dropna().str.len()
            info[f'{col}_len_mean'] = round(lengths.mean(), 1) if len(lengths) > 0 else 0
            info[f'{col}_len_max'] = int(lengths.max()) if len(lengths) > 0 else 0
            info[f'{col}_len_min'] = int(lengths.min()) if len(lengths) > 0 else 0
    return info, df

# Profile ground truth
gt = pd.read_csv(os.path.join(BASE, 'train/train_ground_truth.tsv'), sep='\t')
print(f"Ground truth shape: {gt.shape}")
print(f"Columns: {list(gt.columns)}")

# Count matches
def count_matches(x):
    if pd.isna(x) or str(x).strip() == '':
        return 0
    return len(str(x).split(','))

gt['match_count'] = gt['matched_entity_ids'].apply(count_matches)
print(f"\nMatch count distribution:")
vc = gt['match_count'].value_counts().sort_index()
for k, v in vc.items():
    print(f"  {k} matches: {v} entities ({v/len(gt)*100:.1f}%)")

singleton_count = (gt['match_count'] == 0).sum()
matched_count = (gt['match_count'] > 0).sum()
print(f"\nSingletons (0 matches): {singleton_count} ({singleton_count/len(gt)*100:.1f}%)")
print(f"Has matches (>0): {matched_count} ({matched_count/len(gt)*100:.1f}%)")
print(f"Mean matches per entity: {gt['match_count'].mean():.2f}")
print(f"Max matches: {gt['match_count'].max()}")

# Analyze match sources
def analyze_match_sources(x):
    if pd.isna(x) or str(x).strip() == '':
        return {'s2': 0, 's3': 0}
    ids = str(x).split(',')
    s2 = sum(1 for i in ids if i.startswith('S2-'))
    s3 = sum(1 for i in ids if i.startswith('S3-'))
    return {'s2': s2, 's3': s3}

source_counts = gt['matched_entity_ids'].apply(analyze_match_sources)
s2_counts = [s['s2'] for s in source_counts]
s3_counts = [s['s3'] for s in source_counts]
print(f"\nS2 matches per entity - mean: {sum(s2_counts)/len(s2_counts):.2f}, max: {max(s2_counts)}")
print(f"S3 matches per entity - mean: {sum(s3_counts)/len(s3_counts):.2f}, max: {max(s3_counts)}")

# Country distributions
for fname, label in [('train/train_source1.tsv', 'Train S1'), 
                     ('train/train_source2.tsv', 'Train S2'),
                     ('train/train_source3.tsv', 'Train S3'),
                     ('test/test_source1.tsv', 'Test S1'),
                     ('test/test_source2.tsv', 'Test S2'),
                     ('test/test_source3.tsv', 'Test S3')]:
    df = pd.read_csv(os.path.join(BASE, fname), sep='\t', usecols=['country'])
    print(f"\n{label} country distribution (total={len(df)}):")
    for country, cnt in df['country'].value_counts().items():
        print(f"  {country}: {cnt} ({cnt/len(df)*100:.1f}%)")

# Missing values analysis (sample of first 100k rows)
print("\n=== MISSING VALUES (first 100k rows) ===")
for fname in ['train/train_source1.tsv', 'train/train_source2.tsv', 'train/train_source3.tsv']:
    df = pd.read_csv(os.path.join(BASE, fname), sep='\t', nrows=100000)
    print(f"\n{fname}:")
    for col in df.columns:
        null_pct = df[col].isnull().mean() * 100
        if null_pct > 0:
            print(f"  {col}: {null_pct:.2f}% missing")
    if df.isnull().sum().sum() == 0:
        print("  No missing values in sample")

# String length statistics
print("\n=== STRING LENGTH STATS (first 100k rows) ===")
for fname in ['train/train_source1.tsv', 'train/train_source2.tsv', 'train/train_source3.tsv']:
    df = pd.read_csv(os.path.join(BASE, fname), sep='\t', nrows=100000)
    print(f"\n{fname}:")
    for col in ['business_name', 'business_address']:
        lengths = df[col].dropna().str.len()
        print(f"  {col}: min={lengths.min()}, max={lengths.max()}, mean={lengths.mean():.1f}, median={lengths.median():.0f}")

print("\nProfiling complete.")
