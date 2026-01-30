"""
Sahten HTML Response Composer
============================

Formats structured LLM responses into clean, semantic HTML for the frontend.
Follows "Editorial Chef" design principles: centered text, elegant typography.
"""

import html
import re
from typing import List, Optional
from ..schemas.responses import SahtenResponse, RecipeCard


def _escape_html(text: str) -> str:
    """Escape HTML special characters to prevent XSS."""
    if not text:
        return ""
    return html.escape(text, quote=True)

def compose_html_response(response: SahtenResponse) -> str:
    """Convert SahtenResponse to HTML string."""
    parts = []
    
    # 1. Narrative (The "Voice" of the Chef)
    # The CSS class .sahten-narrative handles the centered alignment and typography.
    if response.narrative:
        # Check if it's a string (legacy/fallback) or RecipeNarrative object
        if isinstance(response.narrative, str):
            raw_text = response.narrative
        else:
            # Reconstruct the narrative flow from the structured object
            parts_list = [
                response.narrative.hook,
                response.narrative.cultural_context
            ]
            if response.narrative.teaser:
                parts_list.append(response.narrative.teaser)
            parts_list.append(response.narrative.cta)
            parts_list.append(response.narrative.closing)
            raw_text = "\n\n".join(parts_list)

        formatted_text = _format_text(raw_text)
        parts.append(f'<div class="sahten-narrative">{formatted_text}</div>')

    # 2. Recipe Cards (The "Menu")
    if response.recipes:
        parts.append('<div class="sahten-recipe-grid">')
        for recipe in response.recipes:
            parts.append(_format_recipe_card(recipe))
        parts.append('</div>')
        
    # 3. Fallback / No Results
    if not response.recipes and response.intent_detected == "recipe_specific" and not response.narrative:
        parts.append('<div class="sahten-narrative"><em>Je n\'ai pas trouvé cette recette exacte dans mes archives, mais je serais ravi de vous proposer une alternative.</em></div>')

    return "".join(parts)


def _format_text(text: str) -> str:
    """
    Text formatter with proper Markdown-to-HTML conversion.
    
    Supported syntax:
    - **bold** → <strong>bold</strong>
    - *italic* → <em>italic</em>  
    - __underline__ → <u>underline</u>
    - Double newlines → paragraph breaks
    """
    if not text:
        return ""
    
    # Normalize newlines
    text = text.replace("\r\n", "\n")
    
    # Apply Markdown transformations (order matters: longest patterns first)
    # Bold: **text** → <strong>text</strong>
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Underline: __text__ → <u>text</u>
    text = re.sub(r'__(.+?)__', r'<u>\1</u>', text)
    
    # Italic: *text* → <em>text</em> (after bold to avoid conflicts)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Split into paragraphs and wrap each in <p>
    paragraphs = text.split("\n\n")
    html_paragraphs = []
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # Handle single newlines within paragraphs as <br>
        p = p.replace("\n", "<br>")
        html_paragraphs.append(f"<p>{p}</p>")
    
    return "".join(html_paragraphs)


def _format_recipe_card(card: RecipeCard) -> str:
    """
    Format a single recipe card with optional grounding/citation.
    All user-provided content is escaped to prevent XSS.
    """
    # Escape all user-provided content
    title = _escape_html(card.title) if card.title else "Recette sans titre"
    url = _escape_html(card.url) if card.url else "#"
    
    image_class = "recipe-image"
    image_style = ""
    if card.image_url:
        escaped_image_url = _escape_html(card.image_url)
        image_style = f'style="background-image: url(\'{escaped_image_url}\')"'
    else:
        image_class = "recipe-image no-image"

    category_html = ""
    if card.category:
        cat_display = _escape_html(card.category.replace("_", " ").upper())
        category_html = f'<span class="recipe-category">{cat_display}</span>'

    chef_html = ""
    if card.chef:
        escaped_chef = _escape_html(card.chef)
        chef_html = f'<span class="recipe-chef">Par {escaped_chef}</span>'

    # Grounding: show cited passage if available
    citation_html = ""
    if card.cited_passage:
        # Escape and truncate citation for display
        citation_text = card.cited_passage[:200]
        if len(card.cited_passage) > 200:
            citation_text += "..."
        escaped_citation = _escape_html(citation_text)
        citation_html = f'<blockquote class="recipe-citation">&ldquo;{escaped_citation}&rdquo;</blockquote>'

    card_html = f"""
    <article class="recipe-card">
        <a href="{url}" target="_blank" class="recipe-card-link-wrapper">
            <div class="{image_class}" {image_style}></div>
            <div class="recipe-content">
                {category_html}
                <h3 class="recipe-title">{title}</h3>
                {citation_html}
                <div class="recipe-meta">
                    {chef_html}
                </div>
            </div>
        </a>
    </article>
    """
    return card_html
