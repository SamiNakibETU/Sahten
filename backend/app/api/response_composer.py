"""
Sahten HTML Response Composer
============================

Formats structured LLM responses into clean, semantic HTML for the frontend.
Follows "Editorial Chef" design principles: centered text, elegant typography.
"""

from typing import List, Optional
from ..schemas.responses import SahtenResponse, RecipeCard

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
    Minimal text formatter. 
    - Converts newlines to paragraph breaks.
    - Handles basic markdown for emphasis: **bold**, *italic*, __underline__.
    """
    # Normalize newlines
    text = text.replace("\r\n", "\n")
    paragraphs = text.split("\n\n")
    
    html_paragraphs = []
    for p in paragraphs:
        if not p.strip():
            continue
            
        # Basic Markdown-to-HTML shim
        # Bold: **text** -> <strong>text</strong>
        p = p.replace("**", "<strong>")
        
        # Underline: __text__ -> <u>text</u> (Custom requirement)
        p = p.replace("__", "<u>")
        
        # Italic: *text* -> <em>text</em>
        # Note: simplistic replacement, works for well-formed simple markdown
        p = p.replace("*", "<em>") 
        
        # Close tags blindly? No, replacement pairs assume valid markdown input.
        # Given we trust the LLM's structured output capability, this suffices for a MVP.
        
        html_paragraphs.append(f"<p>{p.strip()}</p>")
        
    return "".join(html_paragraphs)


def _format_recipe_card(card: RecipeCard) -> str:
    """
    Format a single recipe card.
    """
    title = card.title or "Recette sans titre"
    url = card.url or "#"
    
    image_style = ""
    if card.image_url:
        image_style = f'style="background-image: url(\'{card.image_url}\')"'
    else:
        image_style = 'class="recipe-image no-image"'

    category_html = ""
    if card.category:
        cat_display = card.category.replace("_", " ").upper()
        # Use a safe color class or inline style if needed, but CSS handles .recipe-category
        category_html = f'<span class="recipe-category">{cat_display}</span>'

    chef_html = ""
    if card.chef:
        chef_html = f'<span class="recipe-chef">Par {card.chef}</span>'

    html = f"""
    <article class="recipe-card">
        <a href="{url}" target="_blank" class="recipe-card-link-wrapper">
            <div class="recipe-image" {image_style}></div>
            <div class="recipe-content">
                {category_html}
                <h3 class="recipe-title">{title}</h3>
                <div class="recipe-meta">
                    {chef_html}
                </div>
            </div>
        </a>
    </article>
    """
    return html
