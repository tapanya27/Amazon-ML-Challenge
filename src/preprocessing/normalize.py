"""
Business name, address, and country normalization for entity resolution.

This module provides reusable normalizer classes that create normalized versions
of text fields while preserving the original values. Designed to handle:
- US business names and addresses
- Indian business names and addresses (including Hindi/Devanagari)
- French business names and addresses
- Open-set countries (no hard-coded country restrictions)
"""

import re
import unicodedata
import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# LEGAL SUFFIX MAPPINGS
# ============================================================
# Maps verbose legal suffixes to short canonical forms.
# Order matters: longer forms are checked first to avoid partial matches.
LEGAL_SUFFIX_MAP = {
    # English
    'private limited': 'pvt ltd',
    'pvt limited': 'pvt ltd',
    'private ltd': 'pvt ltd',
    'pvt ltd': 'pvt ltd',
    'limited liability partnership': 'llp',
    'limited liability company': 'llc',
    'limited': 'ltd',
    'incorporated': 'inc',
    'corporation': 'corp',
    'company': 'co',
    'enterprises': 'ent',
    'enterprise': 'ent',
    'international': 'intl',
    'technologies': 'tech',
    'technology': 'tech',
    'solutions': 'soln',
    'industries': 'ind',
    'manufacturing': 'mfg',
    'services': 'svc',
    'consultants': 'consult',
    'consulting': 'consult',
    'associates': 'assoc',
    'foundation': 'fdn',
    'laboratories': 'labs',
    'laboratory': 'labs',
    'management': 'mgmt',
    'development': 'dev',
    'construction': 'const',
    'engineering': 'engg',
    'association': 'assn',
    'partners': 'ptnrs',
    'partnership': 'ptnrshp',
    'holdings': 'hldgs',
    'investments': 'inv',
    'properties': 'prop',
    'ventures': 'vent',
    'communications': 'comm',
    'pharmaceuticals': 'pharma',
    'logistics': 'log',
    'marketing': 'mktg',
    'education': 'edu',
    'financial': 'fin',
    'distributors': 'dist',
    # French
    'societe anonyme': 'sa',
    'société anonyme': 'sa',
    'societe a responsabilite limitee': 'sarl',
    'société à responsabilité limitée': 'sarl',
    'societe par actions simplifiee': 'sas',
    'société par actions simplifiée': 'sas',
    'entreprise individuelle': 'ei',
    'groupement dinteret economique': 'gie',
    # Hindi transliterations
    'praiveta limiteda': 'pvt ltd',
    'praiveta': 'pvt',
    'limiteda': 'ltd',
}

# Short-form legal abbreviations (exact token match)
LEGAL_ABBREV_MAP = {
    'ltd': 'ltd',
    'inc': 'inc',
    'corp': 'corp',
    'co': 'co',
    'pvt': 'pvt',
    'llc': 'llc',
    'llp': 'llp',
    'plc': 'plc',
    'sa': 'sa',
    'sarl': 'sarl',
    'sas': 'sas',
    'srl': 'srl',
    'gmbh': 'gmbh',
    'ag': 'ag',
    'bv': 'bv',
    'nv': 'nv',
}

# ============================================================
# ADDRESS ABBREVIATION MAPPINGS
# ============================================================
ADDRESS_ABBREV_MAP = {
    # Road types
    'street': 'st',
    'road': 'rd',
    'avenue': 'ave',
    'boulevard': 'blvd',
    'drive': 'dr',
    'court': 'ct',
    'place': 'pl',
    'lane': 'ln',
    'terrace': 'ter',
    'circle': 'cir',
    'highway': 'hwy',
    'parkway': 'pkwy',
    'freeway': 'fwy',
    'expressway': 'expy',
    'trail': 'trl',
    'way': 'way',
    'path': 'path',
    'alley': 'aly',
    'pike': 'pike',
    'route': 'rte',
    'crescent': 'cres',
    'close': 'cl',
    # Unit types
    'apartment': 'apt',
    'suite': 'ste',
    'unit': 'unit',
    'floor': 'fl',
    'building': 'bldg',
    'room': 'rm',
    'department': 'dept',
    'number': 'no',
    'house': 'house',
    # Directions
    'north': 'n',
    'south': 's',
    'east': 'e',
    'west': 'w',
    'northeast': 'ne',
    'northwest': 'nw',
    'southeast': 'se',
    'southwest': 'sw',
    # Place types
    'mount': 'mt',
    'saint': 'st',
    'fort': 'ft',
    'junction': 'jct',
    'heights': 'hts',
    'village': 'vlg',
    'crossing': 'xing',
    'extension': 'ext',
    'square': 'sq',
    'center': 'ctr',
    'centre': 'ctr',
    'point': 'pt',
    'creek': 'crk',
    'ridge': 'rdg',
    'valley': 'vly',
    'meadow': 'mdw',
    'garden': 'gdn',
    'gardens': 'gdns',
    'springs': 'spgs',
    'station': 'sta',
    'harbor': 'hbr',
    'harbour': 'hbr',
    # Indian
    'nagar': 'ngr',
    'colony': 'col',
    'sector': 'sec',
    'phase': 'ph',
    'block': 'blk',
    'market': 'mkt',
    'industrial': 'indl',
    'district': 'dist',
    'township': 'twp',
    'county': 'cty',
    'panchayat': 'pnchyt',
    'taluk': 'tlk',
    'mandal': 'mndl',
    'tehsil': 'thsl',
    'mohalla': 'moh',
    'gali': 'gali',
    'marg': 'marg',
    'chowk': 'chk',
    'bazaar': 'bzr',
    'bazar': 'bzr',
    # French
    'rue': 'rue',
    'avenue': 'ave',
    'boulevard': 'blvd',
    'place': 'pl',
    'chemin': 'ch',
    'impasse': 'imp',
    'passage': 'psg',
    'allee': 'all',
    'allée': 'all',
    'cours': 'crs',
    'quai': 'quai',
    'cedex': 'cedex',
}

# Indian state abbreviations
INDIAN_STATE_ABBREV = {
    'andhra pradesh': 'ap',
    'arunachal pradesh': 'ar',
    'assam': 'as',
    'bihar': 'br',
    'chhattisgarh': 'cg',
    'goa': 'ga',
    'gujarat': 'gj',
    'haryana': 'hr',
    'himachal pradesh': 'hp',
    'jharkhand': 'jh',
    'karnataka': 'ka',
    'kerala': 'kl',
    'madhya pradesh': 'mp',
    'maharashtra': 'mh',
    'manipur': 'mn',
    'meghalaya': 'ml',
    'mizoram': 'mz',
    'nagaland': 'nl',
    'odisha': 'od',
    'orissa': 'od',
    'punjab': 'pb',
    'rajasthan': 'rj',
    'sikkim': 'sk',
    'tamil nadu': 'tn',
    'telangana': 'tg',
    'tripura': 'tr',
    'uttar pradesh': 'up',
    'uttarakhand': 'uk',
    'west bengal': 'wb',
    'delhi': 'dl',
    'new delhi': 'dl',
    'chandigarh': 'ch',
    'puducherry': 'py',
    'pondicherry': 'py',
    'jammu and kashmir': 'jk',
    'ladakh': 'la',
    'andaman and nicobar islands': 'an',
    'dadra and nagar haveli': 'dn',
    'daman and diu': 'dd',
    'lakshadweep': 'ld',
}

# US state abbreviations (2-letter already standard, map full names)
US_STATE_ABBREV = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar',
    'california': 'ca', 'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de',
    'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi', 'idaho': 'id',
    'illinois': 'il', 'indiana': 'in', 'iowa': 'ia', 'kansas': 'ks',
    'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn',
    'mississippi': 'ms', 'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne',
    'nevada': 'nv', 'new hampshire': 'nh', 'new jersey': 'nj',
    'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
    'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or',
    'pennsylvania': 'pa', 'rhode island': 'ri', 'south carolina': 'sc',
    'south dakota': 'sd', 'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut',
    'vermont': 'vt', 'virginia': 'va', 'washington': 'wa',
    'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy',
    'district of columbia': 'dc',
}

# French region abbreviations
FRENCH_REGION_ABBREV = {
    'ile-de-france': 'idf',
    'île-de-france': 'idf',
    'provence-alpes-cote-dazur': 'paca',
    'provence-alpes-côte-dazur': 'paca',
    'nouvelle-aquitaine': 'na',
    'occitanie': 'occ',
    'auvergne-rhone-alpes': 'ara',
    'auvergne-rhône-alpes': 'ara',
    'grand-est': 'ge',
    'hauts-de-france': 'hdf',
    'bretagne': 'bre',
    'normandie': 'nor',
    'pays-de-la-loire': 'pdl',
    'bourgogne-franche-comte': 'bfc',
    'centre-val-de-loire': 'cvl',
    'corse': 'cor',
}


def _strip_latin_accents(text: str) -> str:
    """Strip accents/diacritics from Latin characters only.
    
    Preserves non-Latin scripts (Devanagari, Kannada, Tamil, etc.) intact.
    Only removes combining marks that follow Latin base characters.
    """
    result = []
    prev_was_latin = False
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith('M'):  # Combining mark
            if prev_was_latin:
                # Skip combining mark after Latin char (strip accent)
                continue
            else:
                # Keep combining mark for non-Latin scripts
                result.append(char)
        else:
            # Check if this is a Latin letter
            script_name = unicodedata.name(char, '')
            prev_was_latin = 'LATIN' in script_name
            result.append(char)
    return ''.join(result)


class NameNormalizer:
    """Normalizes business names while preserving original values.
    
    Handles:
    - Unicode normalization with script-aware accent stripping
    - Lowercase
    - Punctuation handling (& → and, apostrophes, etc.)
    - Legal suffix canonicalization
    - Whitespace normalization
    - Common abbreviation expansion/normalization
    - URL extraction (keeps domain name as token)
    """
    
    # URL with explicit separator (pipe, preceding space after content, or standalone www/http)
    _URL_WITH_SEPARATOR = re.compile(
        r'\|\s*(?:https?://)?(?:www\.)?[\w.-]+\.(?:com|org|net|co|io|in|edu|gov|fr|biz|info|us)(?:/[\w.-]*)*',
        re.IGNORECASE
    )
    # Standalone URL with protocol or www prefix (safe to extract domain from)
    _URL_WITH_PROTOCOL = re.compile(
        r'(?:https?://|www\.)[\w.-]+\.(?:com|org|net|co|io|in|edu|gov|fr|biz|info|us)(?:/[\w.-]*)*',
        re.IGNORECASE
    )
    _MULTI_SPACE = re.compile(r'\s+')
    _APOSTROPHE_PATTERN = re.compile(r"[''`´\u2019\u2018]")
    _DBA_PATTERN = re.compile(r'\b(?:d/?b/?a|doing business as|trading as|t/?a)\b', re.IGNORECASE)
    # Matches something.com/org/etc as a standalone name (whole string is a domain)
    _DOMAIN_NAME_PATTERN = re.compile(
        r'^[\w.-]+\.(?:com|org|net|co|io|in|edu|gov|fr|biz|info|us)$',
        re.IGNORECASE
    )
    
    def __init__(self):
        # Build legal suffix regex - longer forms first for greedy matching
        sorted_suffixes = sorted(LEGAL_SUFFIX_MAP.keys(), key=len, reverse=True)
        escaped = [re.escape(s) for s in sorted_suffixes]
        self._legal_suffix_re = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE
        )
        
        # Build abbreviation token set
        self._abbrev_set = set(LEGAL_ABBREV_MAP.keys())
    
    @staticmethod
    def _extract_domain_name(url_str: str) -> str:
        """Extract meaningful tokens from a domain name.
        E.g., 'www.shivshakti.com' → 'shivshakti'
        """
        # Remove protocol, www, and TLD
        name = re.sub(r'(?:https?://)?(?:www\.)?', '', url_str, flags=re.IGNORECASE)
        name = re.sub(r'\.(?:com|org|net|co|io|in|edu|gov|fr|biz|info|us).*$', '', name, flags=re.IGNORECASE)
        return name
    
    def normalize(self, name: Optional[str]) -> str:
        """Normalize a business name string.
        
        Args:
            name: Raw business name (may be None/NaN)
            
        Returns:
            Normalized name string (empty string for missing values)
        """
        if pd.isna(name) or not isinstance(name, str) or not name.strip():
            return ''
        
        text = str(name)
        
        # 1. Unicode NFKD decomposition + Latin-only accent stripping
        text = unicodedata.normalize('NFKD', text)
        text = _strip_latin_accents(text)
        
        # 2. Lowercase
        text = text.lower()
        
        # 3. Handle URLs:
        #    - If the entire name is a domain, extract meaningful part
        #    - If URL has a pipe separator, remove it (it's metadata)
        #    - If URL has www/http prefix, extract domain tokens
        stripped = text.strip()
        if self._DOMAIN_NAME_PATTERN.match(stripped):
            # Entire name is a domain - extract the name part
            text = self._extract_domain_name(stripped)
        else:
            # Remove pipe-separated URLs (metadata links)
            text = self._URL_WITH_SEPARATOR.sub(' ', text)
            # Replace protocol URLs with their domain name tokens
            text = self._URL_WITH_PROTOCOL.sub(
                lambda m: ' ' + self._extract_domain_name(m.group(0)) + ' ',
                text
            )
        
        # 4. Handle DBA / "doing business as" - keep both parts
        text = self._DBA_PATTERN.sub(' ', text)
        
        # 5. Handle "&" → "and"
        text = text.replace('&', ' and ')
        
        # 6. Handle apostrophes - remove them to merge possessives
        text = self._APOSTROPHE_PATTERN.sub('', text)
        
        # 7. Normalize legal suffixes (multi-word first, then single tokens)
        text = self._legal_suffix_re.sub(
            lambda m: LEGAL_SUFFIX_MAP.get(m.group(0).lower(), m.group(0).lower()),
            text
        )
        
        # 8. Remove remaining punctuation (keep alphanumeric, whitespace)
        res = []
        for c in text:
            if c == '&':
                res.append(c)
            elif unicodedata.category(c).startswith(('P', 'S')):
                res.append(' ')
            else:
                res.append(c)
        text = ''.join(res)
        
        # 9. Normalize whitespace
        text = self._MULTI_SPACE.sub(' ', text).strip()
        
        # 10. Final token-level abbreviation normalization
        tokens = text.split()
        normalized_tokens = []
        for token in tokens:
            if token in self._abbrev_set:
                normalized_tokens.append(LEGAL_ABBREV_MAP[token])
            else:
                normalized_tokens.append(token)
        
        return ' '.join(normalized_tokens)
    
    def normalize_series(self, series: pd.Series) -> pd.Series:
        """Vectorized normalization for a pandas Series.
        
        Args:
            series: Series of business names
            
        Returns:
            Series of normalized names
        """
        return series.fillna('').astype(str).map(self.normalize)


class AddressNormalizer:
    """Normalizes business addresses while preserving original values.
    
    Handles:
    - Unicode normalization
    - Lowercase
    - Punctuation normalization
    - Road/street abbreviation standardization
    - Unit/apartment/suite abbreviation standardization
    - Numeric information preservation
    - Indian, US, and French address patterns
    - State/region name abbreviation
    """
    
    _MULTI_SPACE = re.compile(r'\s+')
    _SEPARATOR_PATTERN = re.compile(r'[,;|\-/]+')
    _PO_BOX_PATTERN = re.compile(r'\bp\.?\s*o\.?\s*box\b', re.IGNORECASE)
    _HASH_NUM_PATTERN = re.compile(r'#\s*(\d+)')
    _NEAR_PATTERN = re.compile(r'\b(?:near|opp(?:osite)?|behind|beside|next\s+to|adjacent\s+to|in\s+front\s+of)\b', re.IGNORECASE)
    
    def __init__(self):
        # Build address abbreviation regex - longer forms first
        sorted_abbrevs = sorted(ADDRESS_ABBREV_MAP.keys(), key=len, reverse=True)
        escaped = [re.escape(s) for s in sorted_abbrevs]
        self._addr_abbrev_re = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE
        )
        
        # Build state/region regex for normalization
        all_states = {}
        all_states.update(US_STATE_ABBREV)
        all_states.update(INDIAN_STATE_ABBREV)
        all_states.update(FRENCH_REGION_ABBREV)
        self._state_map = all_states
        
        sorted_states = sorted(all_states.keys(), key=len, reverse=True)
        escaped_states = [re.escape(s) for s in sorted_states]
        self._state_re = re.compile(
            r'\b(' + '|'.join(escaped_states) + r')\b',
            re.IGNORECASE
        )
    
    def normalize(self, address: Optional[str]) -> str:
        """Normalize a business address string.
        
        Args:
            address: Raw business address (may be None/NaN)
            
        Returns:
            Normalized address string (empty string for missing values)
        """
        if pd.isna(address) or not isinstance(address, str) or not address.strip():
            return ''
        
        text = str(address)
        
        # 1. Unicode normalization + Latin-only accent stripping
        text = unicodedata.normalize('NFKD', text)
        text = _strip_latin_accents(text)
        
        # 2. Lowercase
        text = text.lower()
        
        # 3. Normalize "P.O. Box" variations
        text = self._PO_BOX_PATTERN.sub('po box', text)
        
        # 4. Normalize "#123" → "no 123"
        text = self._HASH_NUM_PATTERN.sub(r'no \1', text)
        
        # 5. Normalize separators (commas, semicolons, pipes → comma + space)
        text = self._SEPARATOR_PATTERN.sub(', ', text)
        
        # 6. Remove non-essential punctuation but keep digits, commas, slashes, hyphens
        res = []
        for c in text:
            if c in ',/-#':
                res.append(c)
            elif unicodedata.category(c).startswith(('P', 'S')):
                res.append(' ')
            else:
                res.append(c)
        text = ''.join(res)
        
        # 7. Normalize "near/opposite" landmarks - keep them as tokens
        text = self._NEAR_PATTERN.sub('near', text)
        
        # 8. Normalize address abbreviations
        text = self._addr_abbrev_re.sub(
            lambda m: ADDRESS_ABBREV_MAP.get(m.group(0).lower(), m.group(0).lower()),
            text
        )
        
        # 9. Normalize state/region names
        text = self._state_re.sub(
            lambda m: self._state_map.get(m.group(0).lower(), m.group(0).lower()),
            text
        )
        
        # 10. Normalize whitespace
        text = self._MULTI_SPACE.sub(' ', text).strip()
        
        return text
    
    def normalize_series(self, series: pd.Series) -> pd.Series:
        """Vectorized normalization for a pandas Series."""
        return series.fillna('').astype(str).map(self.normalize)


class CountryNormalizer:
    """Normalizes country names without restricting to a fixed list.
    
    Handles:
    - Case normalization
    - Whitespace normalization
    - Common variant standardization (e.g., 'USA' → 'us', 'Inde' → 'india')
    - Open-set: unknown countries pass through lowercase
    """
    
    # Known country variants mapped to canonical lowercase
    COUNTRY_VARIANTS = {
        'us': 'us',
        'usa': 'us',
        'u.s.': 'us',
        'u.s.a.': 'us',
        'united states': 'us',
        'united states of america': 'us',
        'india': 'india',
        'in': 'india',
        'ind': 'india',
        'bharat': 'india',
        'france': 'france',
        'fr': 'france',
        'fra': 'france',
        'république française': 'france',
        'republique francaise': 'france',
        'uk': 'uk',
        'united kingdom': 'uk',
        'great britain': 'uk',
        'gb': 'uk',
        'germany': 'germany',
        'de': 'germany',
        'deutschland': 'germany',
        'canada': 'canada',
        'ca': 'canada',
        'australia': 'australia',
        'au': 'australia',
        'japan': 'japan',
        'jp': 'japan',
        'china': 'china',
        'cn': 'china',
        'brazil': 'brazil',
        'br': 'brazil',
    }
    
    def normalize(self, country: Optional[str]) -> str:
        """Normalize a country string.
        
        Args:
            country: Raw country name (may be None/NaN)
            
        Returns:
            Normalized country string (empty string for missing values)
        """
        if pd.isna(country) or not isinstance(country, str) or not country.strip():
            return ''
        
        text = str(country).strip().lower()
        text = re.sub(r'\s+', ' ', text)
        
        # Check known variants
        if text in self.COUNTRY_VARIANTS:
            return self.COUNTRY_VARIANTS[text]
        
        # Open-set: return lowercase version
        return text
    
    def normalize_series(self, series: pd.Series) -> pd.Series:
        """Vectorized normalization for a pandas Series."""
        return series.fillna('').astype(str).map(self.normalize)


def preprocess_dataframe(df: pd.DataFrame, source_label: str = '') -> pd.DataFrame:
    """Apply all normalization steps to a source dataframe.
    
    Creates new columns with '_normalized' suffix while preserving originals.
    
    Args:
        df: DataFrame with columns: entity_id, business_name, business_address, country
        source_label: Label for logging (e.g., 'S1-train')
        
    Returns:
        DataFrame with additional normalized columns
    """
    logger.info(f"Preprocessing {source_label}: {len(df)} rows")
    
    # Initialize normalizers
    name_norm = NameNormalizer()
    addr_norm = AddressNormalizer()
    country_norm = CountryNormalizer()
    
    # Create normalized columns (preserving originals)
    df = df.copy()
    
    logger.info(f"  Normalizing business names...")
    df['name_normalized'] = name_norm.normalize_series(df['business_name'])
    
    logger.info(f"  Normalizing business addresses...")
    df['address_normalized'] = addr_norm.normalize_series(df['business_address'])
    
    logger.info(f"  Normalizing countries...")
    df['country_normalized'] = country_norm.normalize_series(df['country'])
    
    # Log statistics
    name_empty = (df['name_normalized'] == '').sum()
    addr_empty = (df['address_normalized'] == '').sum()
    country_empty = (df['country_normalized'] == '').sum()
    
    logger.info(f"  Normalization complete for {source_label}:")
    logger.info(f"    Empty name_normalized: {name_empty}")
    logger.info(f"    Empty address_normalized: {addr_empty}")
    logger.info(f"    Empty country_normalized: {country_empty}")
    logger.info(f"    Country distribution: {df['country_normalized'].value_counts().to_dict()}")
    
    return df
