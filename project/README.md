# Amazon ML Challenge 2026 — Business Entity Resolution

## Overview

This project implements a high-performance **Business Entity Resolution** system for the Amazon ML Challenge 2026. Given business records from 3 independent noisy data sources, it determines which records across sources refer to the same real-world business entity.

## Architecture

```
Data → Normalization → Country-Partitioned Blocking → Feature Engineering → XGBoost Classification → F₀.₅ Threshold Optimization → Submission
```

### Key Components

1. **Normalization** — Unicode-aware business name & address normalization handling legal suffixes (Inc/Ltd/Pvt/SARL/SAS), abbreviations, punctuation, and cross-country variations (US/India/France)
2. **Blocking** — IDF-weighted inverted index with word tokens + character 3-grams + exact name matching, partitioned by country for memory efficiency
3. **Features** — 34 pairwise similarity features including Levenshtein, Jaro-Winkler, token Jaccard, character n-gram similarity, address component matching, and cross-field signals
4. **Model** — XGBoost binary classifier with class imbalance handling
5. **Threshold** — Systematic F₀.₅ optimization via threshold sweep

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Full Pipeline (Train + Test + Submission)

```bash
python run_pipeline.py --mode all
```

### Run Training Only

```bash
python run_pipeline.py --mode train --train-sample 200000 --val-sample 50000
```

### Run Test Inference Only (requires trained model)

```bash
python run_pipeline.py --mode predict
```

### Validate Submission

```bash
python run_pipeline.py --mode validate
```

Or use the official validator directly:

```bash
python student_resource/utils/validate_submission.py \
    --matching project/output/matching_results.tsv \
    --candidate project/output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```

## Output Files

- `project/output/matching_results.tsv` — Final matches (scored on leaderboard)
- `project/output/candidate_pairs.tsv` — Blocking candidate set
- `project/models/xgb_model.pkl` — Trained model + threshold
- `project/reports/` — Data profiles, experiment logs
- `project/logs/pipeline.log` — Execution log

## Dataset

- **Train**: S1=2.2M, S2=5.0M, S3=5.3M entities with ground truth
- **Test**: S1=1.7M, S2=4.9M, S3=5.1M entities (no ground truth)
- **Countries**: US, India (train) + France (test only)
- **Format**: Tab-separated (TSV)

## Evaluation

- **Metric**: Macro F₀.₅ (precision-heavy, β=0.5)
- **Singletons**: Correctly predicting no-match earns 1.0; false merges score 0.0
- **Target**: ≥ 0.980473

## Restrictions

- No external data, APIs, or business databases
- Model must be MIT/Apache 2.0 licensed, ≤8B parameters
- Only provided competition data + standard ML libraries
