"""
CSV to Canonical JSON Converter
===============================

Converts the OLJ recipe export CSV to the canonical JSON format used by Sahten.

CSV Columns:
  - Content ID: Unique identifier
  - Image: Image URL
  - Title: Recipe title
  - Category: Category string (may contain multiple categories)
  - Publish date: Publication date
  - Author: Author name
  - Descriptive Summary: Short description
  - Contents: Instructions in HTML
  - Summary: Ingredients in HTML
  - Keyword: Keywords/tags (comma-separated)

Output: data/olj_canonical.json

Usage:
    python scripts/convert_csv_to_canonical.py path/to/export.csv
"""

import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Optional


# ============================================================================
# CATEGORY MAPPING
# ============================================================================

CATEGORY_MAP = {
    # Direct mappings
    "dessert": "dessert",
    "entrée": "entree",
    "mezzé": "mezze_froid",
    "mezze": "mezze_froid",
    "salade": "salade",
    "soupe": "soupe",
    "plat principal": "plat_principal",
    "boisson": "boisson",
    "sauce": "sauces",
    # OLJ specific categories
    "variations autour d'un thème": "plat_principal",
    "un produit, une recette 100% libanaise": "plat_principal",
    "le noël des chefs libanais": "plat_principal",
    "recettes de noël": "plat_principal",
    "recettes de pâques": "plat_principal",
    "recettes de ramadan": "plat_principal",
}


def normalize_category(raw_category: str) -> str:
    """Normalize a raw category string to canonical format."""
    if not raw_category:
        return "autre"
    
    # Clean and lowercase
    cat_lower = raw_category.lower().strip()
    
    # Check for direct matches first
    for key, value in CATEGORY_MAP.items():
        if key in cat_lower:
            return value
    
    # Check keywords in category for hints
    if "dessert" in cat_lower:
        return "dessert"
    if "entrée" in cat_lower or "entree" in cat_lower:
        return "entree"
    if "mezzé" in cat_lower or "mezze" in cat_lower:
        return "mezze_froid"
    if "salade" in cat_lower:
        return "salade"
    if "soupe" in cat_lower:
        return "soupe"
    if "boisson" in cat_lower:
        return "boisson"
    
    return "plat_principal"  # Default for recipes


def detect_category_from_keywords(keywords: List[str]) -> Optional[str]:
    """Try to detect category from keywords."""
    keywords_lower = [k.lower() for k in keywords]
    
    if any("dessert" in k for k in keywords_lower):
        return "dessert"
    if any("entrée" in k or "entree" in k for k in keywords_lower):
        return "entree"
    if any("mezzé" in k or "mezze" in k for k in keywords_lower):
        return "mezze_froid"
    if any("salade" in k for k in keywords_lower):
        return "salade"
    if any("soupe" in k for k in keywords_lower):
        return "soupe"
    if any("plat principal" in k for k in keywords_lower):
        return "plat_principal"
    
    return None


# ============================================================================
# HTML CLEANING
# ============================================================================

def strip_html_tags(html: str) -> str:
    """Remove HTML tags and clean up text."""
    if not html:
        return ""
    
    # Remove image placeholders like [[image id=286087]]
    text = re.sub(r'\[\[image[^\]]*\]\]', '', html)
    
    # Replace common HTML entities
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&nbsp;', ' ')
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def extract_ingredients_from_html(html: str) -> List[str]:
    """Extract ingredient items from HTML list."""
    if not html:
        return []
    
    ingredients = []
    
    # Find all <li> items
    li_pattern = re.compile(r'<li>(.*?)</li>', re.IGNORECASE | re.DOTALL)
    matches = li_pattern.findall(html)
    
    for match in matches:
        # Clean the ingredient text
        text = strip_html_tags(match).strip()
        
        # Skip section headers (like "Pour la pâte sucrée")
        if text.startswith("Pour ") or text.startswith("Ingrédients"):
            continue
        
        # Skip empty or very short items
        if len(text) < 3:
            continue
        
        # Extract main ingredient (first word or two, excluding quantities)
        # E.g., "250 g de farine" -> "farine"
        ingredient_match = re.search(r'(?:de |d\'|du |des )(\w+)', text.lower())
        if ingredient_match:
            ingredients.append(ingredient_match.group(1))
        else:
            # Try to get the last significant word
            words = text.split()
            if words:
                # Skip quantity words
                for word in reversed(words):
                    word_clean = re.sub(r'[^\w]', '', word.lower())
                    if word_clean and not word_clean.isdigit() and len(word_clean) > 2:
                        if word_clean not in ['cuillère', 'soupe', 'café', 'tasse', 'verre', 'pincée']:
                            ingredients.append(word_clean)
                            break
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for ing in ingredients:
        if ing not in seen:
            seen.add(ing)
            unique.append(ing)
    
    return unique[:10]  # Limit to 10 main ingredients


# ============================================================================
# CONVERSION
# ============================================================================

def convert_row_to_canonical(row: dict) -> dict:
    """Convert a CSV row to canonical recipe format."""
    content_id = row.get("Content ID", "").strip()
    title = row.get("Title", "").strip()
    author = row.get("Author", "").strip()
    image_url = row.get("Image", "").strip()
    raw_category = row.get("Category", "").strip()
    publish_date = row.get("Publish date", "").strip()
    description = row.get("Descriptive Summary", "").strip()
    contents_html = row.get("Contents", "").strip()
    summary_html = row.get("Summary", "").strip()
    keywords_raw = row.get("Keyword", "").strip()
    
    # Parse keywords/tags
    tags = [k.strip() for k in keywords_raw.split(",") if k.strip()]
    
    # Normalize category
    category = normalize_category(raw_category)
    
    # Override with keyword-based detection if more specific
    keyword_category = detect_category_from_keywords(tags)
    if keyword_category and category == "plat_principal":
        category = keyword_category
    
    # Build URL from content ID
    url = f"https://www.lorientlejour.com/article/{content_id}"
    
    # Clean HTML content
    instructions_text = strip_html_tags(contents_html)
    ingredients_text = strip_html_tags(summary_html)
    
    # Extract main ingredients
    main_ingredients = extract_ingredients_from_html(summary_html)
    
    # Build search_text (composite of all searchable content)
    search_parts = [
        title,
        author,
        description,
        instructions_text,
        ingredients_text,
        " ".join(tags),
    ]
    search_text = " ".join(filter(None, search_parts))
    
    # Detect if Lebanese
    is_lebanese = any(
        kw in keywords_raw.lower()
        for kw in ["libanaise", "libanais", "liban", "lebanese"]
    ) or "liban" in title.lower()
    
    # Build canonical document
    canonical = {
        "url": url,
        "title": title,
        "chef_name": author if author else None,
        "cuisine_type": "libanaise" if is_lebanese else "méditerranéenne",
        "is_lebanese": is_lebanese,
        "is_recipe": True,
        "category_canonical": category,
        "difficulty_canonical": "non_specifie",
        "tags": tags,
        "main_ingredients": main_ingredients,
        "aliases": [],
        "search_text": search_text,
        "source": "olj",
        "raw_category": raw_category,
        "raw_difficulty": None,
        "raw_enrichment_present": True,
        # Extra fields for reference (not used by retriever but useful)
        "_image_url": image_url,
        "_publish_date": publish_date,
        "_content_id": content_id,
    }
    
    return canonical


def convert_csv_to_canonical(csv_path: Path, output_path: Path) -> int:
    """Convert CSV file to canonical JSON."""
    recipes = []
    
    print(f"Reading CSV from: {csv_path}")
    
    # Use utf-8-sig to handle BOM (Byte Order Mark) in CSV files
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                canonical = convert_row_to_canonical(row)
                recipes.append(canonical)
            except Exception as e:
                print(f"Error processing row: {e}")
                print(f"Row data: {row.get('Title', 'Unknown')}")
                continue
    
    print(f"Converted {len(recipes)} recipes")
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(recipes, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to: {output_path}")
    
    # Print category distribution
    categories = {}
    for r in recipes:
        cat = r["category_canonical"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory distribution:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")
    
    return len(recipes)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_csv_to_canonical.py <csv_path>")
        print("\nExample:")
        print("  python convert_csv_to_canonical.py ../../../export_content.csv")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)
    
    # Output path relative to script location
    script_dir = Path(__file__).parent
    output_path = script_dir.parent.parent / "data" / "olj_canonical.json"
    
    count = convert_csv_to_canonical(csv_path, output_path)
    
    print(f"\nDone! {count} recipes converted.")
