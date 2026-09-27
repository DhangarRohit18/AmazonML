#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Feature computation module for business entity resolution.
Computes similarity features between candidate entity pairs.
"""

import re
import numpy as np

# Try to import rapidfuzz, fall back to difflib
try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz.distance import Levenshtein, JaroWinkler, Jaro
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    from difflib import SequenceMatcher


def safe_str(x):
    """Convert to string safely, return empty string for NaN/None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    return str(x)


# ============================================================
# STRING SIMILARITY FUNCTIONS
# ============================================================

def levenshtein_similarity(s1, s2):
    """Normalized Levenshtein similarity (0-1)."""
    s1, s2 = safe_str(s1), safe_str(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return rfuzz.ratio(s1, s2) / 100.0
    else:
        return SequenceMatcher(None, s1, s2).ratio()


def jaro_similarity(s1, s2):
    """Jaro similarity (0-1)."""
    s1, s2 = safe_str(s1), safe_str(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return Jaro.similarity(s1, s2)
    else:
        return SequenceMatcher(None, s1, s2).ratio()


def jaro_winkler_similarity(s1, s2):
    """Jaro-Winkler similarity (0-1)."""
    s1, s2 = safe_str(s1), safe_str(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return JaroWinkler.similarity(s1, s2)
    else:
        return SequenceMatcher(None, s1, s2).ratio()


def token_sort_similarity(s1, s2):
    """Token sort ratio - sorts tokens before comparing."""
    s1, s2 = safe_str(s1), safe_str(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return rfuzz.token_sort_ratio(s1, s2) / 100.0
    else:
        t1 = ' '.join(sorted(s1.split()))
        t2 = ' '.join(sorted(s2.split()))
        return SequenceMatcher(None, t1, t2).ratio()


def token_set_similarity(s1, s2):
    """Token set ratio - handles partial and reordered matches."""
    s1, s2 = safe_str(s1), safe_str(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return rfuzz.token_set_ratio(s1, s2) / 100.0
    else:
        t1 = set(s1.lower().split())
        t2 = set(s2.lower().split())
        if not t1 or not t2:
            return 0.0
        intersection = t1 & t2
        return len(intersection) / max(len(t1), len(t2))


def partial_ratio(s1, s2):
    """Partial ratio - best substring match."""
    s1, s2 = safe_str(s1), safe_str(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return rfuzz.partial_ratio(s1, s2) / 100.0
    else:
        return SequenceMatcher(None, s1, s2).ratio()


# ============================================================
# TOKEN-BASED FEATURES
# ============================================================

def token_jaccard(s1, s2):
    """Jaccard similarity of token sets."""
    s1, s2 = safe_str(s1), safe_str(s2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    intersection = t1 & t2
    union = t1 | t2
    return len(intersection) / len(union)


def token_overlap_count(s1, s2):
    """Count of common tokens."""
    s1, s2 = safe_str(s1), safe_str(s2)
    if not s1 or not s2:
        return 0
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    return len(t1 & t2)


def token_overlap_ratio(s1, s2):
    """Ratio of common tokens to minimum token count."""
    s1, s2 = safe_str(s1), safe_str(s2)
    if not s1 or not s2:
        return 0.0
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / min(len(t1), len(t2))


def common_numeric_tokens(s1, s2):
    """Count of common numeric tokens (street numbers, postal codes)."""
    s1, s2 = safe_str(s1), safe_str(s2)
    nums1 = set(re.findall(r'\b\d+\b', s1))
    nums2 = set(re.findall(r'\b\d+\b', s2))
    if not nums1 or not nums2:
        return 0
    return len(nums1 & nums2)


def numeric_token_jaccard(s1, s2):
    """Jaccard of numeric tokens."""
    s1, s2 = safe_str(s1), safe_str(s2)
    nums1 = set(re.findall(r'\b\d+\b', s1))
    nums2 = set(re.findall(r'\b\d+\b', s2))
    if not nums1 and not nums2:
        return 1.0
    if not nums1 or not nums2:
        return 0.0
    return len(nums1 & nums2) / len(nums1 | nums2)


# ============================================================
# CHARACTER N-GRAM FEATURES
# ============================================================

def char_ngram_similarity(s1, s2, n=3):
    """Character n-gram Jaccard similarity."""
    s1, s2 = safe_str(s1).lower(), safe_str(s2).lower()
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    if len(s1) < n or len(s2) < n:
        return levenshtein_similarity(s1, s2)
    grams1 = set(s1[i:i+n] for i in range(len(s1) - n + 1))
    grams2 = set(s2[i:i+n] for i in range(len(s2) - n + 1))
    if not grams1 or not grams2:
        return 0.0
    return len(grams1 & grams2) / len(grams1 | grams2)


# ============================================================
# LENGTH-BASED FEATURES
# ============================================================

def length_difference(s1, s2):
    """Absolute difference in string length."""
    return abs(len(safe_str(s1)) - len(safe_str(s2)))


def length_ratio(s1, s2):
    """Ratio of shorter to longer string length."""
    l1, l2 = len(safe_str(s1)), len(safe_str(s2))
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)


# ============================================================
# SPECIALTY FEATURES
# ============================================================

def exact_match(s1, s2):
    """Exact string match (case-insensitive)."""
    s1, s2 = safe_str(s1).lower().strip(), safe_str(s2).lower().strip()
    if not s1 and not s2:
        return 1
    if not s1 or not s2:
        return 0
    return 1 if s1 == s2 else 0


def prefix_match(s1, s2, n=5):
    """Whether first n characters match."""
    s1, s2 = safe_str(s1).lower().strip(), safe_str(s2).lower().strip()
    if not s1 or not s2:
        return 0
    return 1 if s1[:n] == s2[:n] else 0


def first_token_match(s1, s2):
    """Whether first meaningful token matches."""
    t1 = safe_str(s1).lower().split()
    t2 = safe_str(s2).lower().split()
    if not t1 or not t2:
        return 0
    return 1 if t1[0] == t2[0] else 0


def country_match(c1, c2):
    """Whether countries match."""
    c1, c2 = safe_str(c1).lower().strip(), safe_str(c2).lower().strip()
    if not c1 or not c2:
        return 0
    return 1 if c1 == c2 else 0


def acronym_similarity(name1, name2):
    """Compare acronyms of business names."""
    def make_acronym(name):
        tokens = safe_str(name).split()
        return ''.join(t[0] for t in tokens if t and t[0].isalpha()).lower()
    
    a1 = make_acronym(name1)
    a2 = make_acronym(name2)
    if not a1 and not a2:
        return 1.0
    if not a1 or not a2:
        return 0.0
    return levenshtein_similarity(a1, a2)


# ============================================================
# FEATURE VECTOR COMPUTATION
# ============================================================

def compute_pair_features(name1, name2, addr1, addr2, country1, country2,
                          norm_name1, norm_name2, norm_addr1, norm_addr2):
    """
    Compute all comparison features for a candidate pair.
    Returns a dict of feature_name: value.
    """
    features = {}
    
    # Country
    features['country_match'] = country_match(country1, country2)
    
    # Name features (on normalized names)
    features['name_exact'] = exact_match(norm_name1, norm_name2)
    features['name_levenshtein'] = levenshtein_similarity(norm_name1, norm_name2)
    features['name_jaro_winkler'] = jaro_winkler_similarity(norm_name1, norm_name2)
    features['name_token_sort'] = token_sort_similarity(norm_name1, norm_name2)
    features['name_token_set'] = token_set_similarity(norm_name1, norm_name2)
    features['name_partial'] = partial_ratio(norm_name1, norm_name2)
    features['name_token_jaccard'] = token_jaccard(norm_name1, norm_name2)
    features['name_token_overlap'] = token_overlap_count(norm_name1, norm_name2)
    features['name_token_overlap_ratio'] = token_overlap_ratio(norm_name1, norm_name2)
    features['name_char_ngram'] = char_ngram_similarity(norm_name1, norm_name2, n=3)
    features['name_length_diff'] = length_difference(norm_name1, norm_name2)
    features['name_length_ratio'] = length_ratio(norm_name1, norm_name2)
    features['name_prefix5'] = prefix_match(norm_name1, norm_name2, n=5)
    features['name_first_token'] = first_token_match(norm_name1, norm_name2)
    features['name_acronym'] = acronym_similarity(norm_name1, norm_name2)
    
    # Address features (on normalized addresses)
    features['addr_exact'] = exact_match(norm_addr1, norm_addr2)
    features['addr_levenshtein'] = levenshtein_similarity(norm_addr1, norm_addr2)
    features['addr_jaro_winkler'] = jaro_winkler_similarity(norm_addr1, norm_addr2)
    features['addr_token_sort'] = token_sort_similarity(norm_addr1, norm_addr2)
    features['addr_token_set'] = token_set_similarity(norm_addr1, norm_addr2)
    features['addr_token_jaccard'] = token_jaccard(norm_addr1, norm_addr2)
    features['addr_token_overlap'] = token_overlap_count(norm_addr1, norm_addr2)
    features['addr_token_overlap_ratio'] = token_overlap_ratio(norm_addr1, norm_addr2)
    features['addr_char_ngram'] = char_ngram_similarity(norm_addr1, norm_addr2, n=3)
    features['addr_length_ratio'] = length_ratio(norm_addr1, norm_addr2)
    features['addr_numeric_jaccard'] = numeric_token_jaccard(norm_addr1, norm_addr2)
    features['addr_numeric_overlap'] = common_numeric_tokens(norm_addr1, norm_addr2)
    
    # Cross features
    features['name_addr_combined'] = (features['name_levenshtein'] + features['addr_levenshtein']) / 2
    features['name_high_addr_low'] = 1 if features['name_levenshtein'] > 0.7 and features['addr_levenshtein'] < 0.3 else 0
    features['both_high'] = 1 if features['name_levenshtein'] > 0.7 and features['addr_levenshtein'] > 0.5 else 0
    
    # Missing value indicators
    features['addr1_missing'] = 1 if not safe_str(norm_addr1).strip() else 0
    features['addr2_missing'] = 1 if not safe_str(norm_addr2).strip() else 0
    features['any_addr_missing'] = 1 if features['addr1_missing'] or features['addr2_missing'] else 0
    
    return features


# Feature column names in order
FEATURE_COLUMNS = [
    'country_match',
    'name_exact', 'name_levenshtein', 'name_jaro_winkler',
    'name_token_sort', 'name_token_set', 'name_partial',
    'name_token_jaccard', 'name_token_overlap', 'name_token_overlap_ratio',
    'name_char_ngram', 'name_length_diff', 'name_length_ratio',
    'name_prefix5', 'name_first_token', 'name_acronym',
    'addr_exact', 'addr_levenshtein', 'addr_jaro_winkler',
    'addr_token_sort', 'addr_token_set', 'addr_token_jaccard',
    'addr_token_overlap', 'addr_token_overlap_ratio', 'addr_char_ngram',
    'addr_length_ratio', 'addr_numeric_jaccard', 'addr_numeric_overlap',
    'name_addr_combined', 'name_high_addr_low', 'both_high',
    'addr1_missing', 'addr2_missing', 'any_addr_missing',
]
