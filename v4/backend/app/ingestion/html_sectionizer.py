"""Découpe le HTML d'un article WhiteBeard en sections sémantiques.

Reconnaît, dans l'ordre :
  - heading      : <h1>..<h4>
  - bio          : encadré "Bio" / "À propos du chef" / "Lire moins"
  - ingredients_list : section "Ingrédients" (liste suivante)
  - recipe_steps : section "Préparation" / "Recette" (liste/paragraphes)
  - quote        : <blockquote>
  - sidebar      : <aside>, divs avec classe "encadre" / "sidebar"
  - list         : <ul>/<ol> hors ingrédients
  - paragraph    : tout le reste (chaque <p> fait sa section)

Chaque section conserve son HTML brut + son texte propre + des metadata
(level, is_premium_marker, headings_above, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag


_BIO_HINTS = re.compile(
    r"\b(bio|biographie|à propos|lire moins|lire plus|sur le chef)\b",
    re.IGNORECASE,
)
_INGREDIENTS_HINTS = re.compile(
    r"\b(ingrédients?|liste des ingrédients|pour la recette)\b", re.IGNORECASE
)
_STEPS_HINTS = re.compile(
    r"\b(préparation|recette|étapes?|méthode|instructions?|réalisation)\b",
    re.IGNORECASE,
)


@dataclass
class Section:
    position: int
    kind: str
    heading: str | None
    html: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _clean_text(node: Tag | NavigableString) -> str:
    txt = node.get_text(separator=" ", strip=True) if isinstance(node, Tag) else str(node)
    return re.sub(r"\s+", " ", txt).strip()


def _classify_heading(heading_text: str | None) -> str | None:
    if not heading_text:
        return None
    if _BIO_HINTS.search(heading_text):
        return "bio"
    if _INGREDIENTS_HINTS.search(heading_text):
        return "ingredients_list"
    if _STEPS_HINTS.search(heading_text):
        return "recipe_steps"
    return None


_KNOWN_KINDS = {
    "recipe_meta",
    "recipe_summary",
    "recipe_history",
    "intro",
    "ingredients_list",
    "recipe_steps",
    "chef_astuce",
    "chef_bio",
}


def sectionize(html: str) -> list[Section]:
    """Convertit du HTML CMS en liste de sections normalisées.

    Si le HTML est balisé en ``<section data-kind="…">`` (cas du mapper
    WhiteBeard moderne), on respecte directement ces marqueurs sémantiques
    plutôt que de les ré-inférer. Sinon on retombe sur l'heuristique
    historique (heading + classes ``sidebar``/``encadre``).
    """
    if not html or not html.strip():
        return []

    soup = BeautifulSoup(html, "lxml")
    for sel in ("script", "style", "noscript"):
        for n in soup.select(sel):
            n.decompose()

    explicit = soup.find_all("section", attrs={"data-kind": True}, recursive=True)
    if explicit:
        sections: list[Section] = []
        pos = 0
        for sec_tag in explicit:
            kind = (sec_tag.get("data-kind") or "").strip() or "paragraph"
            heading_tag = sec_tag.find(["h2", "h3", "h4"])
            heading = _clean_text(heading_tag) if heading_tag else None
            inner_html = "".join(str(c) for c in sec_tag.children).strip()
            inner_text = sec_tag.get_text(separator="\n", strip=True)
            if not inner_text:
                continue
            sections.append(Section(
                position=pos,
                kind=kind if kind in _KNOWN_KINDS else "paragraph",
                heading=heading,
                html=inner_html,
                text=inner_text,
                metadata={"explicit": True, "raw_kind": kind},
            ))
            pos += 1
        return sections

    sections = []
    pos = 0
    current_heading: str | None = None
    current_kind_hint: str | None = None

    body = soup.body or soup
    for child in body.find_all(recursive=False) if body.find_all(recursive=False) else body.children:
        if isinstance(child, NavigableString):
            txt = str(child).strip()
            if not txt:
                continue
            sections.append(Section(
                position=pos, kind="paragraph", heading=current_heading,
                html=str(child), text=txt,
            ))
            pos += 1
            continue
        if not isinstance(child, Tag):
            continue
        sections.extend(_walk(child, pos, current_heading, current_kind_hint))
        if child.name in {"h1", "h2", "h3", "h4"}:
            current_heading = _clean_text(child)
            current_kind_hint = _classify_heading(current_heading)
        pos = sections[-1].position + 1 if sections else pos

    return _post_process(sections)


def _walk(
    node: Tag,
    start_pos: int,
    heading_ctx: str | None,
    kind_hint: str | None,
) -> list[Section]:
    pos = start_pos
    out: list[Section] = []
    name = node.name or ""

    if name in {"h1", "h2", "h3", "h4"}:
        heading_text = _clean_text(node)
        out.append(Section(
            position=pos, kind="heading", heading=heading_text,
            html=str(node), text=heading_text,
            metadata={"level": int(name[1])},
        ))
        return out

    if name in {"ul", "ol"}:
        text_lines = [_clean_text(li) for li in node.find_all("li", recursive=False)]
        text = "\n".join(f"- {t}" for t in text_lines if t)
        kind = kind_hint or "list"
        out.append(Section(
            position=pos, kind=kind, heading=heading_ctx,
            html=str(node), text=text,
            metadata={"items_count": len(text_lines), "ordered": name == "ol"},
        ))
        return out

    if name == "blockquote":
        out.append(Section(
            position=pos, kind="quote", heading=heading_ctx,
            html=str(node), text=_clean_text(node),
        ))
        return out

    if name in {"aside"} or (name == "div" and _is_sidebar(node)):
        text = _clean_text(node)
        sub_kind = "bio" if _BIO_HINTS.search(text or "") else "sidebar"
        out.append(Section(
            position=pos, kind=sub_kind, heading=heading_ctx,
            html=str(node), text=text,
        ))
        return out

    if name == "p":
        text = _clean_text(node)
        if not text:
            return out
        out.append(Section(
            position=pos, kind="paragraph", heading=heading_ctx,
            html=str(node), text=text,
        ))
        return out

    # Conteneur générique : descendre récursivement
    for ch in node.find_all(recursive=False):
        sub = _walk(ch, pos, heading_ctx, kind_hint)
        if sub:
            out.extend(sub)
            pos = out[-1].position + 1
    return out


def _is_sidebar(div: Tag) -> bool:
    classes = " ".join(div.get("class") or []).lower()
    if any(token in classes for token in ("sidebar", "encadre", "bio", "callout", "notice")):
        return True
    return False


def _post_process(sections: list[Section]) -> list[Section]:
    """Renumérote, fusionne les paragraphes vides, attache les listes
    aux titres "Ingrédients"/"Préparation"."""
    cleaned: list[Section] = []
    for sec in sections:
        if not (sec.text or sec.html.strip()):
            continue
        cleaned.append(sec)

    # Si une section "list" suit immédiatement un heading "ingredients_list" /
    # "recipe_steps", elle hérite du kind correspondant (sécurité).
    for i in range(1, len(cleaned)):
        prev = cleaned[i - 1]
        cur = cleaned[i]
        if cur.kind == "list" and prev.kind == "heading" and prev.heading:
            hint = _classify_heading(prev.heading)
            if hint:
                cur.kind = hint

    for i, sec in enumerate(cleaned):
        sec.position = i
    return cleaned
