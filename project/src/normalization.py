#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Normalization module for business entity resolution.
Handles name and address normalization for matching across sources.
"""

import re
import unicodedata


# ============================================================
# LEGAL SUFFIXES & ABBREVIATION MAPS
# ============================================================

LEGAL_SUFFIX_MAP = {
    # English
    'incorporated': 'inc', 'inc.': 'inc', 'incorporation': 'inc',
    'corporation': 'corp', 'corp.': 'corp',
    'company': 'co', 'co.': 'co',
    'limited': 'ltd', 'ltd.': 'ltd', 'lmtd': 'ltd',
    'llc': 'llc', 'l.l.c.': 'llc', 'l.l.c': 'llc',
    'llp': 'llp', 'l.l.p.': 'llp', 'l.l.p': 'llp',
    'pvt': 'pvt', 'private': 'pvt', 'pvt.': 'pvt',
    'plc': 'plc', 'p.l.c.': 'plc',
    'assoc': 'assoc', 'associates': 'assoc', 'association': 'assoc',
    'intl': 'intl', 'international': 'intl',
    'grp': 'grp', 'group': 'grp',
    'svcs': 'svcs', 'services': 'svcs', 'service': 'svc',
    'enterprises': 'ent', 'enterprise': 'ent',
    'industries': 'ind', 'industry': 'ind',
    'solutions': 'solutions', 'soln': 'solutions',
    'technologies': 'tech', 'technology': 'tech',
    'consultants': 'consult', 'consulting': 'consult', 'consultancy': 'consult',
    'partners': 'partners', 'partnership': 'partners',
    'foundation': 'fdn', 'fdn': 'fdn',
    'holdings': 'holdings', 'holding': 'holdings',
    # French
    'sarl': 'sarl', 's.a.r.l.': 'sarl', 's.a.r.l': 'sarl',
    'sas': 'sas', 's.a.s.': 'sas', 's.a.s': 'sas',
    'sa': 'sa', 's.a.': 'sa',
    'sci': 'sci', 's.c.i.': 'sci',
    'eurl': 'eurl', 'e.u.r.l.': 'eurl',
    'sasu': 'sasu',
    'snc': 'snc',
    'gie': 'gie',
    # Indian
    'nidhi': 'nidhi',
    'opc': 'opc',
}

ADDRESS_ABBREV_MAP = {
    'street': 'st', 'st.': 'st',
    'road': 'rd', 'rd.': 'rd',
    'avenue': 'ave', 'ave.': 'ave', 'av': 'ave', 'av.': 'ave',
    'boulevard': 'blvd', 'blvd.': 'blvd',
    'drive': 'dr', 'dr.': 'dr',
    'lane': 'ln', 'ln.': 'ln',
    'court': 'ct', 'ct.': 'ct',
    'place': 'pl', 'pl.': 'pl',
    'circle': 'cir', 'cir.': 'cir',
    'highway': 'hwy', 'hwy.': 'hwy',
    'parkway': 'pkwy', 'pkwy.': 'pkwy',
    'terrace': 'ter', 'ter.': 'ter',
    'trail': 'trl', 'trl.': 'trl',
    'way': 'way',
    'north': 'n', 'n.': 'n',
    'south': 's', 's.': 's',
    'east': 'e', 'e.': 'e',
    'west': 'w', 'w.': 'w',
    'northeast': 'ne', 'ne.': 'ne',
    'northwest': 'nw', 'nw.': 'nw',
    'southeast': 'se', 'se.': 'se',
    'southwest': 'sw', 'sw.': 'sw',
    'apartment': 'apt', 'apt.': 'apt',
    'suite': 'ste', 'ste.': 'ste',
    'building': 'bldg', 'bldg.': 'bldg',
    'floor': 'fl', 'fl.': 'fl',
    'number': 'no', 'no.': 'no', 'num': 'no',
    'mount': 'mt', 'mt.': 'mt',
    'saint': 'st', 'fort': 'ft', 'ft.': 'ft',
    # French address terms
    'rue': 'rue', 'r.': 'rue',
    'avenue': 'ave',
    'chemin': 'chemin', 'ch.': 'chemin',
    'allée': 'allee', 'allee': 'allee',
    'impasse': 'impasse', 'imp.': 'impasse',
    'passage': 'passage',
    'place': 'pl',
    'route': 'rte', 'rte.': 'rte',
}

# Indian state abbreviations
INDIA_STATE_MAP = {
    'andhra pradesh': 'ap', 'ap': 'ap',
    'arunachal pradesh': 'ar', 'ar': 'ar',
    'assam': 'as', 'as': 'as',
    'bihar': 'br', 'br': 'br',
    'chhattisgarh': 'cg', 'cg': 'cg', 'ct': 'cg',
    'goa': 'ga', 'ga': 'ga',
    'gujarat': 'gj', 'gj': 'gj',
    'haryana': 'hr', 'hr': 'hr',
    'himachal pradesh': 'hp', 'hp': 'hp',
    'jharkhand': 'jh', 'jh': 'jh',
    'karnataka': 'ka', 'ka': 'ka',
    'kerala': 'kl', 'kl': 'kl',
    'madhya pradesh': 'mp', 'mp': 'mp',
    'maharashtra': 'mh', 'mh': 'mh',
    'manipur': 'mn', 'mn': 'mn',
    'meghalaya': 'ml', 'ml': 'ml',
    'mizoram': 'mz', 'mz': 'mz',
    'nagaland': 'nl', 'nl': 'nl',
    'odisha': 'od', 'od': 'od', 'or': 'od', 'orissa': 'od',
    'punjab': 'pb', 'pb': 'pb',
    'rajasthan': 'rj', 'rj': 'rj',
    'sikkim': 'sk', 'sk': 'sk',
    'tamil nadu': 'tn', 'tn': 'tn',
    'telangana': 'ts', 'ts': 'ts', 'tg': 'ts',
    'tripura': 'tr', 'tr': 'tr',
    'uttar pradesh': 'up', 'up': 'up',
    'uttarakhand': 'uk', 'uk': 'uk', 'ua': 'uk',
    'west bengal': 'wb', 'wb': 'wb',
    'delhi': 'dl', 'dl': 'dl', 'new delhi': 'dl',
    'chandigarh': 'ch', 'ch': 'ch',
    'puducherry': 'py', 'py': 'py', 'pondicherry': 'py',
    'jammu and kashmir': 'jk', 'jk': 'jk', 'jammu & kashmir': 'jk',
    'ladakh': 'la', 'la': 'la',
    'andaman and nicobar': 'an', 'an': 'an',
    'dadra and nagar haveli': 'dn', 'dn': 'dn',
    'daman and diu': 'dd', 'dd': 'dd',
    'lakshadweep': 'ld', 'ld': 'ld',
}

US_STATE_MAP = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar',
    'california': 'ca', 'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de',
    'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
    'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks',
    'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
    'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny',
    'north carolina': 'nc', 'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok',
    'oregon': 'or', 'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut',
    'vermont': 'vt', 'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv',
    'wisconsin': 'wi', 'wyoming': 'wy', 'district of columbia': 'dc',
}


def unicode_normalize(text):
    """Apply Unicode NFC normalization."""
    if not isinstance(text, str):
        return text
    return unicodedata.normalize('NFC', text)


def strip_accents(text):
    """Remove accents but keep base characters."""
    if not isinstance(text, str):
        return text
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def basic_clean(text):
    """Basic text cleaning: lowercase, strip, normalize whitespace."""
    if not isinstance(text, str):
        return ''
    text = unicode_normalize(text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def normalize_punctuation(text):
    """Normalize common punctuation variations."""
    if not isinstance(text, str):
        return ''
    # Normalize quotes
    text = re.sub(r'[''`]', "'", text)
    text = re.sub(r'[""«»]', '"', text)
    # Normalize dashes
    text = re.sub(r'[–—−]', '-', text)
    # Normalize ampersand
    text = re.sub(r'\s*&\s*', ' and ', text)
    # Remove special leading characters
    text = re.sub(r'^[<>«»\-–—\*#@!]+\s*', '', text)
    text = re.sub(r'\s*[<>«»\-–—\*#@!]+$', '', text)
    return text


def normalize_business_name(name, strip_legal_suffix=False):
    """
    Normalize a business name for matching.
    Keeps the core name tokens comparable across sources.
    """
    if not isinstance(name, str) or not name.strip():
        return ''
    
    text = basic_clean(name)
    text = normalize_punctuation(text)
    text = strip_accents(text)
    
    # Remove periods in abbreviations (but not in numbers like 1.5)
    text = re.sub(r'(?<![0-9])\.(?![0-9])', '', text)
    
    # Normalize ampersand
    text = text.replace(' & ', ' and ')
    
    # Normalize common business prefixes
    text = re.sub(r'^(the|shri|shree|sri|sree|m/s|ms|messrs)\s+', '', text)
    
    # Normalize legal suffixes
    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        clean_token = token.strip('.,;:')
        if clean_token in LEGAL_SUFFIX_MAP:
            if not strip_legal_suffix:
                normalized_tokens.append(LEGAL_SUFFIX_MAP[clean_token])
        else:
            normalized_tokens.append(token)
    
    text = ' '.join(normalized_tokens)
    
    # Final whitespace cleanup
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def normalize_address(address, country=None):
    """
    Normalize a business address for matching.
    """
    if not isinstance(address, str) or not address.strip():
        return ''
    
    text = basic_clean(address)
    text = normalize_punctuation(text)
    text = strip_accents(text)
    
    # Remove periods
    text = re.sub(r'(?<![0-9])\.(?![0-9])', '', text)
    
    # Normalize address abbreviations
    tokens = text.split()
    normalized_tokens = []
    for token in tokens:
        clean_token = token.strip('.,;:')
        if clean_token in ADDRESS_ABBREV_MAP:
            normalized_tokens.append(ADDRESS_ABBREV_MAP[clean_token])
        else:
            normalized_tokens.append(token)
    
    text = ' '.join(normalized_tokens)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_name_tokens(name):
    """Extract meaningful tokens from a normalized name."""
    if not isinstance(name, str) or not name.strip():
        return []
    # Remove punctuation for tokenization
    clean = re.sub(r'[^\w\s]', ' ', name)
    tokens = clean.lower().split()
    # Remove very short tokens (likely noise)
    return [t for t in tokens if len(t) > 1]


def extract_address_tokens(address):
    """Extract meaningful tokens from a normalized address."""
    if not isinstance(address, str) or not address.strip():
        return []
    clean = re.sub(r'[^\w\s]', ' ', address)
    tokens = clean.lower().split()
    return [t for t in tokens if len(t) > 1]


def extract_numeric_tokens(text):
    """Extract numeric tokens from text (street numbers, postal codes, etc.)."""
    if not isinstance(text, str):
        return []
    return re.findall(r'\b\d+\b', text)


def extract_postal_code(address, country=None):
    """Extract postal/PIN code from address."""
    if not isinstance(address, str):
        return ''
    
    # Indian PIN codes (6 digits)
    pin_match = re.search(r'\b(\d{6})\b', address)
    if pin_match:
        return pin_match.group(1)
    
    # US ZIP codes (5 digits, optionally with +4)
    zip_match = re.search(r'\b(\d{5})(?:-\d{4})?\b', address)
    if zip_match:
        return zip_match.group(1)
    
    # French postal codes (5 digits, starts with 0-9)
    fr_match = re.search(r'\b(\d{5})\b', address)
    if fr_match:
        return fr_match.group(1)
    
    return ''


def extract_city_from_address(address, country=None):
    """Attempt to extract city from address. This is heuristic."""
    if not isinstance(address, str) or not address.strip():
        return ''
    
    # Split by comma and take likely city component
    parts = [p.strip() for p in address.split(',')]
    if len(parts) >= 2:
        # For US: typically "Street, City, State"
        # For India: varies widely
        # Return second-to-last part as city candidate
        city_candidate = parts[-2] if len(parts) >= 3 else parts[-1]
        # Clean state abbreviations
        city_candidate = re.sub(r'\b[A-Z]{2}\b', '', city_candidate).strip()
        city_candidate = re.sub(r'\d+', '', city_candidate).strip()
        return city_candidate.lower().strip()
    
    return ''


def get_name_prefix(name, n=3):
    """Get first n characters of normalized name for blocking."""
    if not isinstance(name, str) or not name.strip():
        return ''
    clean = re.sub(r'[^\w]', '', name.lower())
    return clean[:n] if len(clean) >= n else clean


def get_first_significant_token(name):
    """Get first meaningful token from business name."""
    tokens = extract_name_tokens(name)
    # Skip common prefixes
    skip = {'the', 'shri', 'shree', 'sri', 'sree', 'ms', 'mr', 'dr', 'llc', 'inc', 'pvt', 'ltd', 'co'}
    for t in tokens:
        if t not in skip and len(t) > 1:
            return t
    return tokens[0] if tokens else ''


if __name__ == '__main__':
    # Quick tests
    tests = [
        ("Orelee's Barbershop", "orelees barbershop"),
        ("B+ Retail Inc", "b+ retail inc"),
        ("राम मार्केटिंग प्राइवेट लिमिटेड", ""),
        ("SCI Ptit Àmicale", "sci ptit amicale"),
        ("-- Holloway Peak Inc Seafood", "holloway peak inc seafood"),
    ]
    for raw, expected in tests:
        result = normalize_business_name(raw)
        print(f"  '{raw}' -> '{result}'")
    
    addr_tests = [
        "1795 Westchester Drive, High Point, NC",
        "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine",
    ]
    for addr in addr_tests:
        result = normalize_address(addr)
        print(f"  '{addr}' -> '{result}'")
