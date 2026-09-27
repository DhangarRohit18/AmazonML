#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Amazon ML Challenge 2026 — Business Entity Resolution Pipeline
==============================================================
Complete, self-contained, end-to-end solution.

Usage:
    python run_pipeline.py --mode all        # Full pipeline: train + test + submission
    python run_pipeline.py --mode train      # Train & validate only
    python run_pipeline.py --mode predict    # Test inference only (requires trained model)
    python run_pipeline.py --mode validate   # Run official validator on outputs
"""

import os, sys, gc, time, re, math, json, pickle, argparse, logging, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "student_resource" / "dataset"
    TRAIN_DIR = DATA_DIR / "train"
    TEST_DIR = DATA_DIR / "test"
    OUTPUT_DIR = BASE_DIR / "project" / "output"
    MODEL_DIR = BASE_DIR / "project" / "models"
    REPORT_DIR = BASE_DIR / "project" / "reports"
    LOG_DIR = BASE_DIR / "project" / "logs"

    TRAIN_SAMPLE = 200_000       # S1 entities for model training
    VAL_SAMPLE = 50_000          # S1 entities for validation
    TOP_K_CANDIDATES = 50        # candidates per S1 entity from blocking
    RANDOM_SEED = 42
    XGBOOST_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 7,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0,
    }

for d in [Config.OUTPUT_DIR, Config.MODEL_DIR, Config.REPORT_DIR, Config.LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Config.LOG_DIR / "pipeline.log", encoding='utf-8'),
    ]
)
log = logging.getLogger(__name__)

# ============================================================
# NORMALIZATION
# ============================================================
LEGAL_SUFFIX_MAP = {
    'incorporated': 'inc', 'inc.': 'inc', 'incorporation': 'inc',
    'corporation': 'corp', 'corp.': 'corp',
    'company': 'co', 'co.': 'co',
    'limited': 'ltd', 'ltd.': 'ltd', 'lmtd': 'ltd',
    'llc': 'llc', 'l.l.c.': 'llc', 'l.l.c': 'llc',
    'llp': 'llp', 'l.l.p.': 'llp', 'l.l.p': 'llp',
    'pvt': 'pvt', 'private': 'pvt', 'pvt.': 'pvt',
    'plc': 'plc', 'p.l.c.': 'plc',
    'associates': 'assoc', 'association': 'assoc',
    'international': 'intl', 'intl.': 'intl',
    'group': 'grp', 'grp.': 'grp',
    'services': 'svcs', 'service': 'svc',
    'enterprises': 'ent', 'enterprise': 'ent',
    'industries': 'ind', 'industry': 'ind',
    'technologies': 'tech', 'technology': 'tech',
    'consultants': 'consult', 'consulting': 'consult', 'consultancy': 'consult',
    'partners': 'partners', 'partnership': 'partners',
    'foundation': 'fdn', 'holdings': 'holdings', 'holding': 'holdings',
    'sarl': 'sarl', 's.a.r.l.': 'sarl', 's.a.r.l': 'sarl',
    'sas': 'sas', 's.a.s.': 'sas', 's.a.s': 'sas',
    'sa': 'sa', 's.a.': 'sa',
    'sci': 'sci', 's.c.i.': 'sci',
    'eurl': 'eurl', 'sasu': 'sasu',
}

ADDR_ABBREV = {
    'street': 'st', 'st.': 'st', 'road': 'rd', 'rd.': 'rd',
    'avenue': 'ave', 'ave.': 'ave', 'boulevard': 'blvd', 'blvd.': 'blvd',
    'drive': 'dr', 'dr.': 'dr', 'lane': 'ln', 'ln.': 'ln',
    'court': 'ct', 'ct.': 'ct', 'place': 'pl', 'pl.': 'pl',
    'circle': 'cir', 'highway': 'hwy', 'hwy.': 'hwy',
    'parkway': 'pkwy', 'terrace': 'ter', 'trail': 'trl',
    'north': 'n', 'south': 's', 'east': 'e', 'west': 'w',
    'apartment': 'apt', 'apt.': 'apt', 'suite': 'ste', 'ste.': 'ste',
    'building': 'bldg', 'bldg.': 'bldg', 'floor': 'fl',
    'number': 'no', 'no.': 'no', 'mount': 'mt',
    'rue': 'rue', 'r.': 'rue', 'route': 'rte',
}

def _strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))

def normalize_name(name):
    """Normalize a business name for matching."""
    if not isinstance(name, str) or not name.strip():
        return ''
    s = unicodedata.normalize('NFC', name).lower().strip()
    s = re.sub(r'[''`]', "'", s)
    s = re.sub(r'[""«»]', '"', s)
    s = re.sub(r'[–—−]', '-', s)
    s = re.sub(r'^[<>«»\-–—\*#@!]+\s*', '', s)  # strip leading junk
    s = re.sub(r'\s*[<>«»\-–—\*#@!]+$', '', s)  # strip trailing junk
    s = s.replace(' & ', ' and ').replace('&', ' and ')
    s = re.sub(r'(?<![0-9])\.(?![0-9])', '', s)  # remove periods (not in numbers)
    s = re.sub(r'^(the|shri|shree|sri|sree|m/s|ms|messrs)\s+', '', s)
    tokens = s.split()
    out = []
    for t in tokens:
        ct = t.strip('.,;:')
        if ct in LEGAL_SUFFIX_MAP:
            out.append(LEGAL_SUFFIX_MAP[ct])
        else:
            out.append(t)
    s = ' '.join(out)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_name_alpha(name):
    """Aggressive alphanumeric-only normalization for blocking."""
    s = normalize_name(name)
    s = _strip_accents(s)
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize_address(addr):
    """Normalize an address for matching."""
    if not isinstance(addr, str) or not addr.strip():
        return ''
    s = unicodedata.normalize('NFC', addr).lower().strip()
    s = re.sub(r'[–—−]', '-', s)
    s = re.sub(r'(?<![0-9])\.(?![0-9])', '', s)
    tokens = s.split()
    out = []
    for t in tokens:
        ct = t.strip('.,;:')
        if ct in ADDR_ABBREV:
            out.append(ADDR_ABBREV[ct])
        else:
            out.append(t)
    s = ' '.join(out)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize(s):
    """Extract word tokens from normalized text."""
    if not s:
        return []
    return [t for t in s.split() if len(t) > 1]

def get_char_ngrams(s, n=3):
    """Extract character n-grams."""
    if not s or len(s) < n:
        return []
    s = re.sub(r'\s+', '', s)
    return [s[i:i+n] for i in range(len(s) - n + 1)]

# ============================================================
# BLOCKING — IDF-WEIGHTED INVERTED INDEX
# ============================================================
class BlockingIndex:
    """Memory-efficient IDF-weighted inverted index for candidate retrieval."""

    def __init__(self):
        self.token_to_indices = defaultdict(list)
        self.idf = {}
        self.n_docs = 0
        self.entity_ids = []

    def build(self, entity_ids, norm_names, norm_names_alpha, norm_addrs=None, use_ngrams=True):
        """Build inverted index from entity data."""
        self.entity_ids = list(entity_ids)
        self.n_docs = len(self.entity_ids)
        df_counter = Counter()
        doc_tokens = []

        log.info(f"  Building index for {self.n_docs:,} entities...")
        for i in range(self.n_docs):
            name = norm_names[i] if i < len(norm_names) else ''
            toks = set(tokenize(name))
            if use_ngrams and name:
                alpha = norm_names_alpha[i]
                toks.update(get_char_ngrams(alpha, 3))
            if norm_addrs is not None:
                addr = norm_addrs[i] if i < len(norm_addrs) else ''
                addr_toks = tokenize(addr)
                # Add address tokens with prefix to avoid name/addr collision
                toks.update('A_' + t for t in addr_toks if len(t) > 2)
            doc_tokens.append(toks)
            for t in toks:
                df_counter[t] += 1

        # Compute IDF
        for token, df in df_counter.items():
            self.idf[token] = math.log(self.n_docs / (df + 1))

        # Build posting lists (only keep tokens with reasonable IDF)
        min_idf = 0.5  # filter very common tokens
        for i, toks in enumerate(doc_tokens):
            for t in toks:
                if self.idf.get(t, 0) >= min_idf:
                    self.token_to_indices[t].append(i)

        log.info(f"  Index built: {len(self.token_to_indices):,} unique tokens")

    def query(self, norm_name, norm_name_alpha, norm_addr=None, top_k=50):
        """Retrieve top-k candidates for a query entity."""
        query_toks = set(tokenize(norm_name))
        alpha = norm_name_alpha if norm_name_alpha else ''
        query_toks.update(get_char_ngrams(alpha, 3))
        if norm_addr:
            addr_toks = tokenize(norm_addr)
            query_toks.update('A_' + t for t in addr_toks if len(t) > 2)

        scores = Counter()
        for t in query_toks:
            idf = self.idf.get(t, 0)
            if idf < 0.5:
                continue
            for idx in self.token_to_indices.get(t, []):
                scores[idx] += idf

        if not scores:
            return []

        top = scores.most_common(top_k)
        return [(self.entity_ids[idx], score) for idx, score in top]

    def query_batch(self, norm_names, norm_addrs, top_k=50, desc="Blocking"):
        """Query multiple entities efficiently."""
        results = []
        for i in tqdm(range(len(norm_names)), desc=desc, mininterval=5):
            addr = norm_addrs[i] if norm_addrs is not None and i < len(norm_addrs) else None
            candidates = self.query(norm_names[i], addr, top_k=top_k)
            results.append(candidates)
        return results


# ============================================================
# FEATURE ENGINEERING
# ============================================================
try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import JaroWinkler
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    from difflib import SequenceMatcher

def _sim(s1, s2, func='lev'):
    """Compute string similarity. Returns 0-1."""
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    if func == 'lev':
        return rfuzz.ratio(s1, s2) / 100.0 if HAS_RAPIDFUZZ else SequenceMatcher(None, s1, s2).ratio()
    elif func == 'jw':
        return JaroWinkler.similarity(s1, s2) if HAS_RAPIDFUZZ else SequenceMatcher(None, s1, s2).ratio()
    elif func == 'tok_sort':
        return rfuzz.token_sort_ratio(s1, s2) / 100.0 if HAS_RAPIDFUZZ else SequenceMatcher(None, ' '.join(sorted(s1.split())), ' '.join(sorted(s2.split()))).ratio()
    elif func == 'tok_set':
        return rfuzz.token_set_ratio(s1, s2) / 100.0 if HAS_RAPIDFUZZ else 0.0
    elif func == 'partial':
        return rfuzz.partial_ratio(s1, s2) / 100.0 if HAS_RAPIDFUZZ else SequenceMatcher(None, s1, s2).ratio()
    return 0.0

def _jaccard(s1, s2):
    t1, t2 = set(s1.split()), set(s2.split())
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / len(t1 | t2)

def _overlap_ratio(s1, s2):
    t1, t2 = set(s1.split()), set(s2.split())
    if not t1 or not t2: return 0.0
    return len(t1 & t2) / min(len(t1), len(t2))

def _overlap_count(s1, s2):
    return len(set(s1.split()) & set(s2.split())) if s1 and s2 else 0

def _numeric_jaccard(s1, s2):
    n1 = set(re.findall(r'\b\d+\b', s1)) if s1 else set()
    n2 = set(re.findall(r'\b\d+\b', s2)) if s2 else set()
    if not n1 and not n2: return 1.0
    if not n1 or not n2: return 0.0
    return len(n1 & n2) / len(n1 | n2)

def _ngram_sim(s1, s2, n=3):
    if not s1 and not s2: return 1.0
    if not s1 or not s2: return 0.0
    if len(s1) < n or len(s2) < n: return _sim(s1, s2, 'lev')
    g1 = set(s1[i:i+n] for i in range(len(s1)-n+1))
    g2 = set(s2[i:i+n] for i in range(len(s2)-n+1))
    return len(g1 & g2) / len(g1 | g2) if g1 | g2 else 0.0

def _len_ratio(s1, s2):
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0: return 1.0
    if l1 == 0 or l2 == 0: return 0.0
    return min(l1, l2) / max(l1, l2)

FEATURE_NAMES = [
    'name_lev', 'name_jw', 'name_tok_sort', 'name_tok_set', 'name_partial',
    'name_jaccard', 'name_overlap', 'name_overlap_ratio', 'name_ngram3',
    'name_len_ratio', 'name_len_diff', 'name_exact',
    'name_first_tok', 'name_prefix5',
    'addr_lev', 'addr_jw', 'addr_tok_sort', 'addr_tok_set',
    'addr_jaccard', 'addr_overlap', 'addr_overlap_ratio', 'addr_ngram3',
    'addr_len_ratio', 'addr_num_jaccard', 'addr_num_overlap', 'addr_exact',
    'country_match',
    'name_addr_avg', 'both_strong', 'name_strong_addr_weak',
    'addr1_miss', 'addr2_miss', 'any_addr_miss',
    'blocking_score',
    'same_name_diff_addr', 'branch_indicator', 'name_perfect'
]

def compute_features(n1, n2, a1, a2, c1, c2, block_score=0.0):
    """Compute pairwise features for one candidate pair. Returns np.float32 array."""
    n1 = n1 or ''; n2 = n2 or ''; a1 = a1 or ''; a2 = a2 or ''
    c1 = c1 or ''; c2 = c2 or ''

    f = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    # Name features
    f[0] = _sim(n1, n2, 'lev')
    f[1] = _sim(n1, n2, 'jw')
    f[2] = _sim(n1, n2, 'tok_sort')
    f[3] = _sim(n1, n2, 'tok_set')
    f[4] = _sim(n1, n2, 'partial')
    f[5] = _jaccard(n1, n2)
    f[6] = _overlap_count(n1, n2)
    f[7] = _overlap_ratio(n1, n2)
    f[8] = _ngram_sim(n1, n2, 3)
    f[9] = _len_ratio(n1, n2)
    f[10] = abs(len(n1) - len(n2))
    f[11] = 1.0 if n1 and n2 and n1 == n2 else 0.0
    t1 = n1.split(); t2 = n2.split()
    f[12] = 1.0 if t1 and t2 and t1[0] == t2[0] else 0.0
    f[13] = 1.0 if n1 and n2 and n1[:5] == n2[:5] else 0.0

    # Address features
    f[14] = _sim(a1, a2, 'lev')
    f[15] = _sim(a1, a2, 'jw')
    f[16] = _sim(a1, a2, 'tok_sort')
    f[17] = _sim(a1, a2, 'tok_set')
    f[18] = _jaccard(a1, a2)
    f[19] = _overlap_count(a1, a2)
    f[20] = _overlap_ratio(a1, a2)
    f[21] = _ngram_sim(a1, a2, 3)
    f[22] = _len_ratio(a1, a2)
    f[23] = _numeric_jaccard(a1, a2)
    f[24] = len(set(re.findall(r'\b\d+\b', a1)) & set(re.findall(r'\b\d+\b', a2))) if a1 and a2 else 0
    f[25] = 1.0 if a1 and a2 and a1 == a2 else 0.0

    # Country
    f[26] = 1.0 if c1 and c2 and c1.lower() == c2.lower() else 0.0

    # Cross features
    f[27] = (f[0] + f[14]) / 2.0
    f[28] = 1.0 if f[0] > 0.7 and f[14] > 0.5 else 0.0
    f[29] = 1.0 if f[0] > 0.7 and f[14] < 0.3 else 0.0

    # Missing indicators
    f[30] = 1.0 if not a1.strip() else 0.0
    f[31] = 1.0 if not a2.strip() else 0.0
    f[32] = 1.0 if f[30] or f[31] else 0.0

    # Blocking score
    f[33] = block_score
    
    # NEW: Branch / Entity matching logic
    # Name is almost identical, but address is different or missing
    f[34] = 1.0 if f[0] >= 0.90 and f[14] <= 0.4 else 0.0
    f[35] = 1.0 if (f[0] >= 0.85 and (f[32] == 1.0 or f[14] < 0.5)) else 0.0
    f[36] = 1.0 if f[0] >= 0.95 or f[3] >= 0.95 else 0.0

    return f


# ============================================================
# F0.5 METRIC (Official)
# ============================================================
def compute_f05_per_entity(pred_set, true_set):
    """F0.5 for a single S1 entity."""
    if not true_set and not pred_set:
        return 1.0  # Correct singleton
    if not true_set and pred_set:
        return 0.0  # False merge on singleton
    if true_set and not pred_set:
        return 0.0  # Missed all matches
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec + rec == 0:
        return 0.0
    return (1.25 * prec * rec) / (0.25 * prec + rec)

def compute_macro_f05(predictions, ground_truth):
    """
    Macro F0.5 across all S1 entities.
    predictions: dict {s1_id: set of matched IDs}
    ground_truth: dict {s1_id: set of matched IDs}
    """
    scores = []
    for s1_id in ground_truth:
        pred = predictions.get(s1_id, set())
        true = ground_truth[s1_id]
        scores.append(compute_f05_per_entity(pred, true))
    return np.mean(scores) if scores else 0.0

def evaluate_predictions(predictions, ground_truth):
    """Full evaluation including detailed metrics."""
    f05_scores = []
    total_tp = total_fp = total_fn = 0
    singleton_correct = singleton_total = 0
    matched_correct = matched_total = 0

    for s1_id in ground_truth:
        pred = predictions.get(s1_id, set())
        true = ground_truth[s1_id]

        f05_scores.append(compute_f05_per_entity(pred, true))

        tp = len(pred & true)
        fp = len(pred - true)
        fn = len(true - pred)
        total_tp += tp; total_fp += fp; total_fn += fn

        if not true:
            singleton_total += 1
            if not pred:
                singleton_correct += 1
        else:
            matched_total += 1
            if pred & true:
                matched_correct += 1

    macro_f05 = np.mean(f05_scores)
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    micro_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    micro_f05 = (1.25 * micro_prec * micro_rec) / (0.25 * micro_prec + micro_rec) if (micro_prec + micro_rec) else 0

    return {
        'macro_f05': macro_f05,
        'micro_f05': micro_f05,
        'micro_precision': micro_prec,
        'micro_recall': micro_rec,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'singleton_accuracy': singleton_correct / singleton_total if singleton_total else 1.0,
        'singleton_total': singleton_total,
        'matched_recall': matched_correct / matched_total if matched_total else 0,
        'matched_total': matched_total,
    }


# ============================================================
# DATA LOADING
# ============================================================
def load_source(path):
    """Load a TSV source file."""
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, na_values=[])
    # Ensure no actual NaN values - replace with empty string
    df = df.fillna('')
    return df

def load_ground_truth(path):
    """Load ground truth and return dict {s1_id: set(matched_ids)}."""
    log.info(f"  Loading ground truth from {path}...")
    df = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, na_values=[])
    df['matched_entity_ids'] = df['matched_entity_ids'].fillna('')
    gt = {}
    s1_ids = df['source1_entity_id'].values
    matched_vals = df['matched_entity_ids'].values
    for i in range(len(df)):
        s1_id = s1_ids[i]
        matched = matched_vals[i]
        if matched and matched.strip():
            gt[s1_id] = set(matched.split(','))
        else:
            gt[s1_id] = set()
    return gt

def add_normalized_columns(df):
    """Add normalized name and address columns."""
    log.info(f"  Normalizing {len(df):,} rows...")
    # Use numpy arrays for faster iteration
    names = df['business_name'].values
    addrs = df['business_address'].values
    norm_names = np.empty(len(df), dtype=object)
    norm_addrs = np.empty(len(df), dtype=object)
    norm_names_alpha = np.empty(len(df), dtype=object)
    for i in range(len(df)):
        nn = normalize_name(names[i])
        norm_names[i] = nn
        norm_addrs[i] = normalize_address(addrs[i])
        norm_names_alpha[i] = _strip_accents(re.sub(r'[^a-z0-9\s]', '', nn)) if nn else ''
    df['norm_name'] = norm_names
    df['norm_addr'] = norm_addrs
    df['norm_name_alpha'] = norm_names_alpha
    return df


# ============================================================
# PIPELINE CORE
# ============================================================
class EntityResolutionPipeline:
    def __init__(self, config=None):
        self.cfg = config or Config()
        self.model = None
        self.threshold = 0.5
        self.experiments = []

    # ----------------------------------------------------------
    # PHASE 1: Load & Normalize
    # ----------------------------------------------------------
    def load_data(self, split='train'):
        """Load source files for a given split."""
        if split == 'train':
            d = self.cfg.TRAIN_DIR
            prefix = 'train'
        else:
            d = self.cfg.TEST_DIR
            prefix = 'test'

        cache_s1 = self.cfg.OUTPUT_DIR / f"{prefix}_s1_norm.pkl"
        cache_s23 = self.cfg.OUTPUT_DIR / f"{prefix}_s23_norm.pkl"

        if cache_s1.exists() and cache_s23.exists():
            log.info(f"Loading cached {split} normalized data...")
            s1 = pd.read_pickle(cache_s1)
            s23 = pd.read_pickle(cache_s23)
            return s1, s23

        log.info(f"Loading {split} data...")
        s1 = load_source(d / f"{prefix}_source1.tsv")
        s2 = load_source(d / f"{prefix}_source2.tsv")
        s3 = load_source(d / f"{prefix}_source3.tsv")
        log.info(f"  S1: {len(s1):,}, S2: {len(s2):,}, S3: {len(s3):,}")

        s1 = add_normalized_columns(s1)
        s2 = add_normalized_columns(s2)
        s3 = add_normalized_columns(s3)

        # Combine S2+S3 for candidate pool
        s23 = pd.concat([s2, s3], ignore_index=True)
        log.info(f"  Combined S2+S3: {len(s23):,} entities")
        
        self.cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        s1.to_pickle(cache_s1)
        s23.to_pickle(cache_s23)
        return s1, s23

    # ----------------------------------------------------------
    # PHASE 2: Validation Split
    # ----------------------------------------------------------
    def create_val_split(self, s1, ground_truth):
        """Entity-level stratified split."""
        np.random.seed(self.cfg.RANDOM_SEED)
        all_ids = list(ground_truth.keys())
        np.random.shuffle(all_ids)

        # Stratify by country and match count
        s1_country = dict(zip(s1['entity_id'], s1['country']))

        total_sample = min(self.cfg.TRAIN_SAMPLE + self.cfg.VAL_SAMPLE, len(all_ids))
        sampled_ids = all_ids[:total_sample]

        val_size = min(self.cfg.VAL_SAMPLE, total_sample // 5)
        val_ids = set(sampled_ids[:val_size])
        train_ids = set(sampled_ids[val_size:])

        log.info(f"  Train S1 entities: {len(train_ids):,}")
        log.info(f"  Val S1 entities: {len(val_ids):,}")
        return train_ids, val_ids

    # ----------------------------------------------------------
    # PHASE 3: Blocking (per country)
    # ----------------------------------------------------------
    def generate_candidates_for_country(self, s1_df, s23_df, country, top_k=50):
        """Generate candidates for S1 entities in a specific country."""
        s1_c = s1_df[s1_df['country'] == country].copy()
        s23_c = s23_df[s23_df['country'] == country].copy()

        if len(s1_c) == 0 or len(s23_c) == 0:
            return {}

        log.info(f"  Country={country}: S1={len(s1_c):,}, S23={len(s23_c):,}")

        # 1. Prepare texts for TF-IDF (vectorized)
        s23_texts = s23_c['norm_name'].fillna('') + ' ' + s23_c['norm_name_alpha'].fillna('') + ' ' + s23_c['norm_addr'].fillna('')
        s1_texts = s1_c['norm_name'].fillna('') + ' ' + s1_c['norm_name_alpha'].fillna('') + ' ' + s1_c['norm_addr'].fillna('')

        log.info(f"  Training TF-IDF for {len(s23_c):,} entities...")
        # Fast TF-IDF with min_df=3 and max_features=50000 for fuzzy matching
        vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=3, max_df=0.1, max_features=50000, dtype=np.float32)
        s23_tfidf = vec.fit_transform(s23_texts)
        
        # Build exact match dictionaries for blazing fast recall of rare names!
        log.info(f"  Building exact match dictionaries...")
        s23_name_dict = defaultdict(list)
        s23_alpha_dict = defaultdict(list)
        
        s23_eids = s23_c['entity_id'].values
        s23_nn = s23_c['norm_name'].fillna('').values
        s23_na = s23_c['norm_name_alpha'].fillna('').values
        
        for i in range(len(s23_eids)):
            eid = s23_eids[i]
            nn = s23_nn[i]
            na = s23_na[i]
            if nn:
                s23_name_dict[nn].append(eid)
            if na:
                s23_alpha_dict[na].append(eid)
        
        log.info(f"  Transforming {len(s1_c):,} S1 queries...")
        s1_tfidf = vec.transform(s1_texts)

        # 2. Exact name lookup
        exact_map = defaultdict(list)
        s23_eids = s23_c['entity_id'].values
        s23_nna = s23_c['norm_name_alpha'].values
        for i in range(len(s23_c)):
            nna = s23_nna[i]
            if nna:
                exact_map[nna].append(s23_eids[i])

        # 3. Batch Sparse Dot Product
        candidates = {}
        s1_ids = s1_c['entity_id'].values
        s1_nna = s1_c['norm_name_alpha'].values
        s1_nn = s1_c['norm_name'].values
        
        batch_size = 1000
        n_batches = math.ceil(len(s1_c) / batch_size)
        log.info(f"  Querying in {n_batches} batches...")

        for b in tqdm(range(n_batches), desc=f"  Blocking {country}", mininterval=5):
            start = b * batch_size
            end = min((b + 1) * batch_size, len(s1_c))
            
            # Sparse matrix multiplication: (batch, vocab) * (vocab, N_S23) = (batch, N_S23)
            sim = s1_tfidf[start:end].dot(s23_tfidf.T)
            
            for i in range(start, end):
                row_idx = i - start
                s1_id = s1_ids[i]
                nna = s1_nna[i]
                
                row = sim.getrow(row_idx)
                cand_dict = {}
                if row.nnz > 0:
                    if row.nnz <= top_k:
                        top_indices = row.indices
                        top_scores = row.data
                    else:
                        top_idx_rel = np.argpartition(row.data, -top_k)[-top_k:]
                        top_indices = row.indices[top_idx_rel]
                        top_scores = row.data[top_idx_rel]
                    
                    for idx, score in zip(top_indices, top_scores):
                        cand_dict[s23_eids[idx]] = float(score)
                    
                if s1_nn[i] and s1_nn[i] in s23_name_dict:
                    for cid in s23_name_dict[s1_nn[i]]:
                        if cid not in cand_dict: cand_dict[cid] = 1.0
                if nna and nna in s23_alpha_dict:
                    for cid in s23_alpha_dict[nna]:
                        if cid not in cand_dict: cand_dict[cid] = 0.9
                            
                candidates[s1_id] = cand_dict

        return candidates

    def generate_all_candidates(self, s1_df, s23_df, top_k=50, s1_filter=None):
        """Generate candidates for all S1 entities, country by country."""
        countries = sorted(s1_df['country'].unique())
        all_candidates = {}

        for country in countries:
            if s1_filter is not None:
                s1_subset = s1_df[s1_df['entity_id'].isin(s1_filter) & (s1_df['country'] == country)]
            else:
                s1_subset = s1_df[s1_df['country'] == country]

            if len(s1_subset) == 0:
                continue

            cands = self.generate_candidates_for_country(
                s1_subset if s1_filter is not None else s1_df,
                s23_df, country, top_k=top_k
            )
            all_candidates.update(cands)
            gc.collect()

        log.info(f"  Total S1 entities with candidates: {len(all_candidates):,}")
        total_cands = sum(len(v) for v in all_candidates.values())
        avg_cands = total_cands / len(all_candidates) if all_candidates else 0
        log.info(f"  Total candidate pairs: {total_cands:,} (avg {avg_cands:.1f} per S1)")
        return all_candidates

    # ----------------------------------------------------------
    # PHASE 4: Feature Computation
    # ----------------------------------------------------------
    def compute_pair_features(self, candidates, s1_df, s23_df):
        """Compute features for all candidate pairs."""
        # Build lookup dicts for fast access using numpy arrays
        s1_lookup = {}
        s1_eids = s1_df['entity_id'].values
        s1_nn = s1_df['norm_name'].values
        s1_na = s1_df['norm_addr'].values
        s1_co = s1_df['country'].values
        for i in range(len(s1_df)):
            s1_lookup[s1_eids[i]] = {
                'norm_name': s1_nn[i],
                'norm_addr': s1_na[i],
                'country': s1_co[i],
            }

        s23_lookup = {}
        s23_eids = s23_df['entity_id'].values
        s23_nn = s23_df['norm_name'].values
        s23_na = s23_df['norm_addr'].values
        s23_co = s23_df['country'].values
        for i in range(len(s23_df)):
            s23_lookup[s23_eids[i]] = {
                'norm_name': s23_nn[i],
                'norm_addr': s23_na[i],
                'country': s23_co[i],
            }

        # Generate feature matrix
        pairs = []
        for s1_id, cand_dict in candidates.items():
            for cand_id, block_score in cand_dict.items():
                pairs.append((s1_id, cand_id, block_score))

        log.info(f"  Computing features for {len(pairs):,} pairs...")
        n_features = len(FEATURE_NAMES)
        X = np.zeros((len(pairs), n_features), dtype=np.float32)
        pair_ids = []

        for i in tqdm(range(len(pairs)), desc="  Features", mininterval=5):
            s1_id, cand_id, block_score = pairs[i]
            s1_data = s1_lookup.get(s1_id, {})
            cand_data = s23_lookup.get(cand_id, {})

            X[i] = compute_features(
                s1_data.get('norm_name', ''),
                cand_data.get('norm_name', ''),
                s1_data.get('norm_addr', ''),
                cand_data.get('norm_addr', ''),
                s1_data.get('country', ''),
                cand_data.get('country', ''),
                block_score
            )
            pair_ids.append((s1_id, cand_id))

        return X, pair_ids

    # ----------------------------------------------------------
    # PHASE 5: Labeling
    # ----------------------------------------------------------
    def create_labels(self, pair_ids, ground_truth):
        """Create binary labels from ground truth."""
        y = np.zeros(len(pair_ids), dtype=np.float32)
        for i, (s1_id, cand_id) in enumerate(pair_ids):
            true_matches = ground_truth.get(s1_id, set())
            if cand_id in true_matches:
                y[i] = 1.0
        pos = int(y.sum())
        neg = len(y) - pos
        log.info(f"  Labels: {pos:,} positive, {neg:,} negative ({pos/(pos+neg)*100:.1f}% positive)")
        return y

    # ----------------------------------------------------------
    # PHASE 6: Model Training
    # ----------------------------------------------------------
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train XGBoost classifier."""
        try:
            import xgboost as xgb

            # Compute scale_pos_weight for imbalanced classes
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

            params = dict(self.cfg.XGBOOST_PARAMS)
            params['scale_pos_weight'] = min(scale_pos_weight, 10.0)
            n_est = params.pop('n_estimators', 300)

            log.info(f"  Training XGBoost: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
            log.info(f"  scale_pos_weight={params['scale_pos_weight']:.2f}")

            model = xgb.XGBClassifier(n_estimators=n_est, **params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Feature importance
            imp = model.feature_importances_
            imp_sorted = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
            log.info("  Top 10 features:")
            for name, score in imp_sorted[:10]:
                log.info(f"    {name}: {score:.4f}")

            self.model = model
            return model

        except ImportError:
            log.error("XGBoost not available!")
            raise

    # ----------------------------------------------------------
    # PHASE 7: Threshold Optimization
    # ----------------------------------------------------------
    def optimize_threshold(self, X_val, pair_ids_val, ground_truth):
        """Find optimal threshold maximizing macro F0.5."""
        probs = self.model.predict_proba(X_val)[:, 1]

        best_f05 = 0
        best_threshold = 0.5
        results = []

        for thresh in np.arange(0.10, 0.96, 0.02):
            predictions = defaultdict(set)
            for i, (s1_id, cand_id) in enumerate(pair_ids_val):
                if probs[i] >= thresh:
                    predictions[s1_id].add(cand_id)

            # Add empty predictions for S1 entities not in predictions
            val_s1_ids = set(s1_id for s1_id, _ in pair_ids_val)
            for s1_id in val_s1_ids:
                if s1_id not in predictions:
                    predictions[s1_id] = set()

            # Filter ground truth to validation entities
            val_gt = {s1_id: ground_truth.get(s1_id, set()) for s1_id in val_s1_ids}
            metrics = evaluate_predictions(dict(predictions), val_gt)

            results.append({
                'threshold': thresh,
                'macro_f05': metrics['macro_f05'],
                'precision': metrics['micro_precision'],
                'recall': metrics['micro_recall'],
                'tp': metrics['total_tp'],
                'fp': metrics['total_fp'],
                'fn': metrics['total_fn'],
                'singleton_acc': metrics['singleton_accuracy'],
            })

            if metrics['macro_f05'] > best_f05:
                best_f05 = metrics['macro_f05']
                best_threshold = thresh

        self.threshold = best_threshold
        log.info(f"\n  === THRESHOLD OPTIMIZATION ===")
        log.info(f"  Best threshold: {best_threshold:.2f}")
        log.info(f"  Best macro F0.5: {best_f05:.6f}")

        # Print top results
        results.sort(key=lambda x: -x['macro_f05'])
        log.info(f"  Top thresholds:")
        for r in results[:5]:
            log.info(f"    t={r['threshold']:.2f}: F0.5={r['macro_f05']:.6f} P={r['precision']:.4f} R={r['recall']:.4f} TP={r['tp']} FP={r['fp']} FN={r['fn']} Sing={r['singleton_acc']:.4f}")

        return best_threshold, best_f05, results

    # ----------------------------------------------------------
    # PHASE 8: Candidate Recall Measurement
    # ----------------------------------------------------------
    def measure_candidate_recall(self, candidates, ground_truth, label=""):
        """Measure what fraction of true matches appear in candidate set."""
        total_true = 0
        found_true = 0
        s1_perfect = 0
        s1_total = 0

        for s1_id, true_matches in ground_truth.items():
            if s1_id not in candidates:
                total_true += len(true_matches)
                if not true_matches:
                    s1_perfect += 1
                s1_total += 1
                continue

            cand_set = set(candidates[s1_id].keys())
            s1_total += 1

            if not true_matches:
                s1_perfect += 1
                continue

            total_true += len(true_matches)
            found = len(true_matches & cand_set)
            found_true += found

            if found == len(true_matches):
                s1_perfect += 1

        recall = found_true / total_true if total_true > 0 else 1.0
        entity_recall = s1_perfect / s1_total if s1_total > 0 else 1.0

        log.info(f"  [{label}] Candidate Recall: {recall:.6f} ({found_true}/{total_true})")
        log.info(f"  [{label}] Entity-level Perfect Recall: {entity_recall:.4f} ({s1_perfect}/{s1_total})")
        return recall

    # ----------------------------------------------------------
    # TRAINING PIPELINE
    # ----------------------------------------------------------
    def run_training(self):
        """Execute the full training pipeline."""
        t0 = time.time()
        log.info("=" * 60)
        log.info("PHASE 1: Loading & Normalizing Training Data")
        log.info("=" * 60)
        s1, s23 = self.load_data('train')
        gt = load_ground_truth(self.cfg.TRAIN_DIR / "train_ground_truth.tsv")
        log.info(f"  Ground truth: {len(gt):,} S1 entities")

        log.info("\n" + "=" * 60)
        log.info("PHASE 2: Creating Validation Split")
        log.info("=" * 60)
        train_ids, val_ids = self.create_val_split(s1, gt)

        log.info("\n" + "=" * 60)
        log.info("PHASE 3: Blocking / Candidate Generation (Training)")
        log.info("=" * 60)
        train_candidates = self.generate_all_candidates(
            s1, s23, top_k=self.cfg.TOP_K_CANDIDATES, s1_filter=train_ids
        )
        train_gt = {k: v for k, v in gt.items() if k in train_ids}
        self.measure_candidate_recall(train_candidates, train_gt, "TRAIN")

        log.info("\n" + "=" * 60)
        log.info("PHASE 3b: Blocking / Candidate Generation (Validation)")
        log.info("=" * 60)
        val_candidates = self.generate_all_candidates(
            s1, s23, top_k=self.cfg.TOP_K_CANDIDATES, s1_filter=val_ids
        )
        val_gt = {k: v for k, v in gt.items() if k in val_ids}
        self.measure_candidate_recall(val_candidates, val_gt, "VAL")

        log.info("\n" + "=" * 60)
        log.info("PHASE 4: Feature Engineering (Training)")
        log.info("=" * 60)
        X_train, pairs_train = self.compute_pair_features(train_candidates, s1, s23)

        log.info("\nPHASE 4b: Feature Engineering (Validation)")
        X_val, pairs_val = self.compute_pair_features(val_candidates, s1, s23)

        log.info("\n" + "=" * 60)
        log.info("PHASE 5: Labeling")
        log.info("=" * 60)
        y_train = self.create_labels(pairs_train, gt)
        y_val = self.create_labels(pairs_val, gt)

        # Free memory
        del train_candidates, val_candidates
        gc.collect()

        log.info("\n" + "=" * 60)
        log.info("PHASE 6: Model Training")
        log.info("=" * 60)
        self.train_model(X_train, y_train, X_val, y_val)

        log.info("\n" + "=" * 60)
        log.info("PHASE 7: Threshold Optimization")
        log.info("=" * 60)
        best_thresh, best_f05, _ = self.optimize_threshold(X_val, pairs_val, gt)

        # Save model
        model_path = self.cfg.MODEL_DIR / "xgb_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({'model': self.model, 'threshold': self.threshold}, f)
        log.info(f"\n  Model saved to {model_path}")

        elapsed = time.time() - t0
        log.info(f"\n  Training completed in {elapsed/60:.1f} minutes")
        log.info(f"  BEST VALIDATION F0.5: {best_f05:.6f}")
        log.info(f"  BEST THRESHOLD: {best_thresh:.2f}")
        log.info(f"  TARGET BENCHMARK: 0.980473")
        log.info(f"  GAP: {0.980473 - best_f05:+.6f}")

        return best_f05

    # ----------------------------------------------------------
    # TEST INFERENCE PIPELINE
    # ----------------------------------------------------------
    def run_test_inference(self):
        """Generate predictions for the test set streaming directly to disk."""
        t0 = time.time()

        # Load model if not already loaded
        if self.model is None:
            model_path = self.cfg.MODEL_DIR / "xgb_model.pkl"
            if not model_path.exists():
                log.error(f"No trained model found at {model_path}. Run training first.")
                return None, None
            with open(model_path, 'rb') as f:
                saved = pickle.load(f)
                self.model = saved['model']
                self.threshold = saved['threshold']
            log.info(f"  Loaded model from {model_path}, threshold={self.threshold:.2f}")

        log.info("=" * 60)
        log.info("TEST INFERENCE: Loading Test Data")
        log.info("=" * 60)
        s1_test, s23_test = self.load_data('test')

        log.info("\n" + "=" * 60)
        log.info("TEST INFERENCE: Streaming Blocking & Prediction")
        log.info("=" * 60)

        matching_path = self.cfg.OUTPUT_DIR / "matching_results.tsv"
        candidate_path = self.cfg.OUTPUT_DIR / "candidate_pairs.tsv"
        
        # Check for existing progress to enable seamless resumption
        processed_ids = set()
        if matching_path.exists() and os.path.getsize(matching_path) > 0:
            with open(matching_path, 'r', encoding='utf-8') as f:
                header = next(f, None)
                for line in f:
                    parts = line.strip().split('\t')
                    if parts and parts[0]:
                        processed_ids.add(parts[0])
            log.info(f"  Found existing progress: {len(processed_ids):,} entities already processed.")

        if processed_ids:
            f_matching = open(matching_path, 'a', encoding='utf-8')
            f_candidate = open(candidate_path, 'a', encoding='utf-8')
        else:
            f_matching = open(matching_path, 'w', encoding='utf-8')
            f_candidate = open(candidate_path, 'w', encoding='utf-8')
            f_matching.write("source1_entity_id\tmatched_entity_ids\n")
            f_candidate.write("source1_entity_id\tcandidate_entity_ids\n")
            f_matching.flush()
            f_candidate.flush()

        countries = sorted(s1_test['country'].unique())
        matched_count = 0
        empty_count = 0
        total_matches = 0
        total_entities_processed = len(processed_ids)

        for country in countries:
            log.info(f"\n--- Processing country: {country} ---")
            s1_c_full = s1_test[s1_test['country'] == country].copy()
            s23_c = s23_test[s23_test['country'] == country].copy()
            
            # Filter out already processed entities for this country
            if processed_ids:
                s1_c = s1_c_full[~s1_c_full['entity_id'].isin(processed_ids)].copy().reset_index(drop=True)
            else:
                s1_c = s1_c_full.reset_index(drop=True)

            log.info(f"  Total S1 in {country}: {len(s1_c_full):,}, Remaining to process: {len(s1_c):,}, S23: {len(s23_c):,}")

            if len(s1_c) == 0:
                log.info(f"  Country {country} already 100% completed. Skipping.")
                continue

            if len(s23_c) == 0:
                for eid in s1_c['entity_id']:
                    f_matching.write(f"{eid}\t\n")
                    f_candidate.write(f"{eid}\t\n")
                    empty_count += 1
                    total_entities_processed += 1
                f_matching.flush()
                f_candidate.flush()
                continue

            # 1. Prepare texts for TF-IDF
            s23_texts = s23_c['norm_name'].fillna('') + ' ' + s23_c['norm_name_alpha'].fillna('') + ' ' + s23_c['norm_addr'].fillna('')
            s1_texts = s1_c['norm_name'].fillna('') + ' ' + s1_c['norm_name_alpha'].fillna('') + ' ' + s1_c['norm_addr'].fillna('')

            log.info(f"  Training TF-IDF for {len(s23_c):,} S23 entities...")
            vec = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=3, max_df=0.1, max_features=50000, dtype=np.float32)
            s23_tfidf = vec.fit_transform(s23_texts)
            
            # Exact dictionaries for fast recall
            s23_name_dict = defaultdict(list)
            s23_alpha_dict = defaultdict(list)
            s23_eids = s23_c['entity_id'].values
            s23_nn = s23_c['norm_name'].fillna('').values
            s23_na = s23_c['norm_name_alpha'].fillna('').values
            for i in range(len(s23_eids)):
                if s23_nn[i]: s23_name_dict[s23_nn[i]].append(s23_eids[i])
                if s23_na[i]: s23_alpha_dict[s23_na[i]].append(s23_eids[i])
            
            log.info(f"  Transforming {len(s1_c):,} S1 queries...")
            s1_tfidf = vec.transform(s1_texts)

            # Build S23 lookup for feature computation
            s23_lookup = {}
            s23_norm_addr = s23_c['norm_addr'].values
            s23_co = s23_c['country'].values
            for i in range(len(s23_c)):
                s23_lookup[s23_eids[i]] = {
                    'norm_name': s23_nn[i],
                    'norm_addr': s23_norm_addr[i],
                    'country': s23_co[i],
                }

            # Process S1 entities in small, memory-safe chunks (200 entities)
            s1_ids = s1_c['entity_id'].values
            s1_nnames = s1_c['norm_name'].values
            s1_naddrs = s1_c['norm_addr'].values
            s1_nnames_alpha = s1_c['norm_name_alpha'].values
            s1_countries = s1_c['country'].values

            CHUNK_SIZE = 200
            n_chunks = math.ceil(len(s1_c) / CHUNK_SIZE)

            for chunk_i in range(n_chunks):
                start = chunk_i * CHUNK_SIZE
                end = min(start + CHUNK_SIZE, len(s1_c))
                chunk_len = end - start

                if chunk_i % 50 == 0:
                    log.info(f"  [{country}] Chunk {chunk_i+1}/{n_chunks}: rows {start:,}-{end:,} ({(start/len(s1_c))*100:.1f}%)")

                # Sparse matrix dot product for small chunk
                sim = s1_tfidf[start:end].dot(s23_tfidf.T)
                chunk_pairs = []
                chunk_entity_candidates = {}

                for i in range(start, end):
                    row_idx = i - start
                    s1_id = s1_ids[i]
                    nna = s1_nnames_alpha[i]
                    
                    row = sim.getrow(row_idx)
                    cand_dict = {}
                    if row.nnz > 0:
                        top_k_req = self.cfg.TOP_K_CANDIDATES
                        if row.nnz <= top_k_req:
                            top_indices = row.indices
                            top_scores = row.data
                        else:
                            top_idx_rel = np.argpartition(row.data, -top_k_req)[-top_k_req:]
                            top_indices = row.indices[top_idx_rel]
                            top_scores = row.data[top_idx_rel]
                            
                        for idx, score in zip(top_indices, top_scores):
                            cand_dict[s23_eids[idx]] = float(score)

                    if s1_nnames[i] and s1_nnames[i] in s23_name_dict:
                        for cid in s23_name_dict[s1_nnames[i]]:
                            if cid not in cand_dict: cand_dict[cid] = 1.0
                    if nna and nna in s23_alpha_dict:
                        for cid in s23_alpha_dict[nna]:
                            if cid not in cand_dict: cand_dict[cid] = 0.9

                    chunk_entity_candidates[s1_id] = cand_dict

                    for cand_id, block_score in cand_dict.items():
                        chunk_pairs.append((s1_id, cand_id, block_score))

                del sim

                chunk_entity_preds = {s1_ids[i]: set() for i in range(start, end)}

                if chunk_pairs:
                    # Compute features
                    X_chunk = np.zeros((len(chunk_pairs), len(FEATURE_NAMES)), dtype=np.float32)

                    s1_chunk_lookup = {}
                    for i in range(start, end):
                        s1_chunk_lookup[s1_ids[i]] = {
                            'norm_name': s1_nnames[i],
                            'norm_addr': s1_naddrs[i],
                            'country': s1_countries[i],
                        }

                    for j in range(len(chunk_pairs)):
                        s1_id, cand_id, block_score = chunk_pairs[j]
                        s1d = s1_chunk_lookup[s1_id]
                        cd = s23_lookup.get(cand_id, {})
                        X_chunk[j] = compute_features(
                            s1d.get('norm_name', ''), cd.get('norm_name', ''),
                            s1d.get('norm_addr', ''), cd.get('norm_addr', ''),
                            s1d.get('country', ''), cd.get('country', ''),
                            block_score
                        )

                    # Predict
                    probs = self.model.predict_proba(X_chunk)[:, 1]

                    # Apply threshold
                    for j in range(len(chunk_pairs)):
                        if probs[j] >= self.threshold:
                            s1_id, cand_id, _ = chunk_pairs[j]
                            chunk_entity_preds[s1_id].add(cand_id)

                    del X_chunk, chunk_pairs

                # Stream out chunk results to disk immediately
                for i in range(start, end):
                    s1_id = s1_ids[i]
                    preds = chunk_entity_preds.get(s1_id, set())
                    cands = set(chunk_entity_candidates.get(s1_id, {}).keys()) | preds
                    
                    matched_str = ','.join(sorted(preds))
                    cand_str = ','.join(sorted(cands))
                    
                    f_matching.write(f"{s1_id}\t{matched_str}\n")
                    f_candidate.write(f"{s1_id}\t{cand_str}\n")

                    if preds:
                        matched_count += 1
                        total_matches += len(preds)
                    else:
                        empty_count += 1
                    total_entities_processed += 1

                del chunk_entity_candidates, chunk_entity_preds
                f_matching.flush()
                f_candidate.flush()

            del s23_lookup, s23_tfidf, s1_tfidf, vec, s23_name_dict, s23_alpha_dict
            gc.collect()

        f_matching.close()
        f_candidate.close()

        elapsed = time.time() - t0
        log.info(f"\n  Test inference completed in {elapsed/60:.1f} minutes")
        log.info(f"  Total processed entities: {total_entities_processed:,}")
        log.info(f"  S1 entities with matches: {matched_count:,}")
        log.info(f"  S1 entities without matches: {empty_count:,}")
        log.info(f"  Total predicted matches: {total_matches:,}")
        if total_entities_processed > 0:
            log.info(f"  Avg matches per entity: {total_matches/total_entities_processed:.2f}")

        return matching_path, candidate_path

    # ----------------------------------------------------------
    # SUBMISSION GENERATION (Legacy Wrapper)
    # ----------------------------------------------------------
    def generate_submission(self, predictions=None, candidates=None, s1_test_path=None):
        """Files are already streamed to disk during inference."""
        matching_path = self.cfg.OUTPUT_DIR / "matching_results.tsv"
        candidate_path = self.cfg.OUTPUT_DIR / "candidate_pairs.tsv"
        log.info(f"  Submission files confirmed at {matching_path} and {candidate_path}")
        return matching_path, candidate_path

    # ----------------------------------------------------------
    # VALIDATE SUBMISSION
    # ----------------------------------------------------------
    def validate_submission(self):
        """Run official validator."""
        log.info("\n" + "=" * 60)
        log.info("RUNNING OFFICIAL VALIDATOR")
        log.info("=" * 60)

        validator = self.cfg.BASE_DIR / "student_resource" / "utils" / "validate_submission.py"
        matching = self.cfg.OUTPUT_DIR / "matching_results.tsv"
        candidate = self.cfg.OUTPUT_DIR / "candidate_pairs.tsv"
        test_dir = self.cfg.TEST_DIR

        cmd = (
            f'python "{validator}" '
            f'--matching "{matching}" '
            f'--candidate "{candidate}" '
            f'--test-dir "{test_dir}"'
        )
        log.info(f"  Command: {cmd}")
        ret = os.system(cmd)
        return ret == 0

    # ----------------------------------------------------------
    # FULL PIPELINE
    # ----------------------------------------------------------
    def run_all(self):
        """Run the complete pipeline: train + test + submission + validate."""
        log.info("=" * 60)
        log.info("AMAZON ML CHALLENGE 2026 — ENTITY RESOLUTION PIPELINE")
        log.info("=" * 60)
        log.info(f"Target benchmark: F0.5 = 0.980473")
        log.info(f"Config: TRAIN_SAMPLE={self.cfg.TRAIN_SAMPLE}, VAL_SAMPLE={self.cfg.VAL_SAMPLE}, TOP_K={self.cfg.TOP_K_CANDIDATES}")

        # Training
        val_f05 = self.run_training()

        # Test inference
        predictions, candidates = self.run_test_inference()

        # Generate submission
        self.generate_submission(predictions, candidates)

        # Validate
        passed = self.validate_submission()

        log.info("\n" + "=" * 60)
        log.info("FINAL SUMMARY")
        log.info("=" * 60)
        log.info(f"  Validation F0.5: {val_f05:.6f}")
        log.info(f"  Threshold: {self.threshold:.2f}")
        log.info(f"  Benchmark: 0.980473")
        log.info(f"  Validator: {'PASS' if passed else 'FAIL'}")
        log.info(f"  Output: {self.cfg.OUTPUT_DIR}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 Pipeline")
    parser.add_argument('--mode', choices=['all', 'train', 'predict', 'validate'],
                        default='all', help='Pipeline mode')
    parser.add_argument('--top-k', type=int, default=50, help='Top-K candidates')
    parser.add_argument('--train-sample', type=int, default=200_000, help='Training sample size')
    parser.add_argument('--val-sample', type=int, default=50_000, help='Validation sample size')
    args = parser.parse_args()

    Config.TOP_K_CANDIDATES = args.top_k
    Config.TRAIN_SAMPLE = args.train_sample
    Config.VAL_SAMPLE = args.val_sample

    pipeline = EntityResolutionPipeline()

    if args.mode == 'all':
        pipeline.run_all()
    elif args.mode == 'train':
        pipeline.run_training()
    elif args.mode == 'predict':
        predictions, candidates = pipeline.run_test_inference()
        pipeline.generate_submission(predictions, candidates)
        pipeline.validate_submission()
    elif args.mode == 'validate':
        pipeline.validate_submission()

if __name__ == '__main__':
    main()
