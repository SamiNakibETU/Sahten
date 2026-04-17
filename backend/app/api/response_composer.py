"""
Sahten HTML Response Composer
============================

Formats structured LLM responses into clean, semantic HTML for the frontend.
Follows "Editorial Chef" design principles: centered text, elegant typography.
"""

import html
import re
import urllib.parse
from typing import List, Optional
from ..schemas.responses import SahtenResponse, RecipeCard, ConversationBlock


def _escape_html(text: str) -> str:
    """Escape HTML special characters to prevent XSS."""
    if not text:
        return ""
    return html.escape(text, quote=True)

def compose_html_response(response: SahtenResponse) -> str:
    """Convert SahtenResponse to HTML string."""
    parts = []
    
    # 1. Conversation blocks (new format)
    if response.conversation_blocks:
        parts.append(_format_conversation_blocks(response.conversation_blocks))
    # 1b. Legacy narrative fallback
    elif response.narrative:
            if isinstance(response.narrative, str):
                raw_text = response.narrative
                formatted_text = _format_text(raw_text)
                parts.append(f'<div class="sahten-narrative">{formatted_text}</div>')
            else:
                n = response.narrative
                narrative_html = []
                if n.hook:
                    narrative_html.append(f'<p class="sn-hook">{_escape_html(n.hook)}</p>')
                if n.cultural_context:
                    # Support line breaks in detail
                    for sent in n.cultural_context.split("\n\n"):
                        sent = sent.strip()
                        if sent:
                            narrative_html.append(f'<p>{_escape_html(sent)}</p>')
                if n.cta:
                    narrative_html.append(f'<p class="sn-cta">{_escape_html(n.cta)}</p>')
                if n.teaser:
                    narrative_html.append(f'<p class="sn-followup">{_escape_html(n.teaser)}</p>')
                if n.closing:
                    narrative_html.append(f'<p class="sn-closing">{_escape_html(n.closing)}</p>')
                parts.append(f'<div class="sahten-narrative">{"".join(narrative_html)}</div>')

    # 2. Recipe Cards (The "Menu")
    if response.recipes:
        parts.append('<div class="sahten-recipe-grid">')
        for recipe in response.recipes:
            parts.append(_format_recipe_card(recipe))
        parts.append('</div>')

    # 2b. OLJ CTA (clarification, recipe_not_found, recipe_base2)
    if response.olj_recommendation:
        rec = response.olj_recommendation
        url_esc = _escape_html(rec.url) if rec.url else "#"
        title_esc = _escape_html(rec.title) if rec.title else "L'Orient-Le Jour"
        reason_esc = _escape_html(rec.reason) if rec.reason else "Découvre sur L'Orient-Le Jour"
        parts.append(
            f'<div class="sahten-olj-cta">'
            f'<a href="{url_esc}" target="_blank" rel="noopener" class="olj-cta-link">'
            f'{title_esc}</a>'
            f'<p class="olj-cta-reason">{reason_esc}</p>'
            f'</div>'
        )
        
    # 3. Fallback / No Results
    if not response.recipes and response.intent_detected == "recipe_specific" and not response.narrative:
        parts.append('<div class="sahten-narrative"><em>Je n\'ai pas trouvé cette recette exacte dans mes archives, mais je serais ravi de vous proposer une alternative.</em></div>')

    return "".join(parts)


def _format_conversation_blocks(blocks: List[ConversationBlock]) -> str:
    """Render typed conversation blocks to HTML."""
    html_parts: List[str] = []
    for block in blocks:
        css_class = f"sahten-block sahten-block-{block.block_type}"
        text = _format_text(block.text)
        html_parts.append(f'<div class="{css_class}">{text}</div>')
    return "".join(html_parts)


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
    # Pour les recettes Base2 sans URL, générer un lien de recherche OLJ
    if card.url:
        url = _escape_html(card.url)
    else:
        url = "https://www.lorientlejour.com/cuisine-liban-a-table?q=" + urllib.parse.quote(card.title or "", safe="")
    
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

    is_base2 = getattr(card, "source", None) == "base2"

    # For Base2 recipes: show ingredient list directly (no article URL)
    # For OLJ recipes: show cited passage if available
    extra_html = ""
    if is_base2 and card.ingredients:
        # Show up to 8 ingredients as a compact pill list
        ings = [_escape_html(str(i)) for i in card.ingredients[:8] if i]
        if ings:
            pills = "".join(f'<span class="recipe-ing">{i}</span>' for i in ings)
            extra_html = f'<div class="recipe-ingredients">{pills}</div>'
    elif card.cited_passage:
        citation_text = card.cited_passage[:200]
        if len(card.cited_passage) > 200:
            citation_text += "..."
        escaped_citation = _escape_html(citation_text)
        extra_html = f'<blockquote class="recipe-citation">&ldquo;{escaped_citation}&rdquo;</blockquote>'

    # Base2 badge so user knows this is an archive recipe
    source_badge = ""
    if is_base2:
        source_badge = '<span class="recipe-source-badge">Archive</span>'

    card_html = f"""
    <article class="recipe-card{' recipe-card--base2' if is_base2 else ''}">
        <a href="{url}" target="_blank" class="recipe-card-link-wrapper">
            <div class="{image_class}" {image_style}></div>
            <div class="recipe-content">
                {category_html}
                <h3 class="recipe-title">{title}</h3>
                {extra_html}
                <div class="recipe-meta">
                    {chef_html}
                    {source_badge}
                </div>
            </div>
        </a>
    </article>
    """
    return card_html
