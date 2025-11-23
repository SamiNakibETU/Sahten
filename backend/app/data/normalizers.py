"""
Text normalization utilities for Sahtein 3.0
Handles French text normalization for better matching
"""

import re
from unidecode import unidecode


def normalize_text(text: str) -> str:
    """
    Normalize French text for better matching and search

    - Lowercase
    - Remove accents (é → e, à → a, etc.)
    - Remove HTML entities
    - Normalize whitespace
    - Remove special characters except hyphens and apostrophes
    """
    if not text:
        return ""

    # Decode HTML entities
    text = text.replace("&#039;", "'")
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", "&")
    text = re.sub(r'&[a-z]+;', '', text)

    # Lowercase
    text = text.lower()

    # Remove accents
    text = unidecode(text)

    # Keep only alphanumeric, spaces, hyphens, and apostrophes
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_slug_from_url(url: str) -> str:
    """Extract the slug from an OLJ article URL"""
    # Example: https://www.lorientlejour.com/cuisine-liban-a-table/1227694/le-vrai-taboule-de-kamal-mouzawak.html
    # Returns: le-vrai-taboule-de-kamal-mouzawak

    parts = url.rstrip('/').split('/')
    if parts:
        last_part = parts[-1]
        # Remove .html extension
        slug = last_part.replace('.html', '')
        # Remove numeric ID prefix if present
        slug = re.sub(r'^\d+-', '', slug)
        return slug
    return ""


def normalize_recipe_name(name: str) -> str:
    """
    Normalize recipe name for matching
    Handles special Lebanese dish variations
    """
    normalized = normalize_text(name)

    # Common variations
    variations = {
        "houmous": "hummus",
        "hommos": "hummus",
        "taboule": "tabbouleh",
        "tabboule": "tabbouleh",
        "kebbe": "kibbeh",
        "kibbe": "kibbeh",
        "kafta": "kofta",
        "kefta": "kofta",
        "labne": "labneh",
        "labné": "labneh",
        "moutabbal": "mutabbal",
        "mtabbal": "mutabbal",
    }

    for variant, canonical in variations.items():
        normalized = normalized.replace(variant, canonical)

    return normalized


def extract_keywords(text: str) -> list[str]:
    """
    Extract meaningful keywords from text
    Removes stopwords and keeps substantive terms
    """
    # French stopwords
    stopwords = {
        "le", "la", "les", "un", "une", "des", "de", "du", "au", "aux",
        "et", "ou", "mais", "donc", "or", "ni", "car", "pour", "par",
        "avec", "sans", "sur", "sous", "dans", "en", "a", "à",
        "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
        "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
        "qui", "que", "quoi", "dont", "ou", "où", "quand", "comment",
        "d", "l", "c", "s", "m", "t", "n", "j",
    }

    normalized = normalize_text(text)
    words = normalized.split()
    keywords = [w for w in words if w not in stopwords and len(w) > 2]

    return keywords


def create_searchable_text(fields: list[str]) -> str:
    """
    Combine multiple fields into searchable text
    Used for creating search indexes
    """
    combined = " ".join(str(f) for f in fields if f)
    return normalize_text(combined)
