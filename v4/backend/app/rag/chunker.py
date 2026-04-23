"""Chunking sémantique aligné sur les sections HTML.

Stratégie hybride
-----------------
1. **Chunk d'ancrage** en tête : titre + résumé + chef + metadata pratiques
   en un seul bloc. Indispensable pour les requêtes "qui a fait quoi ?" ou
   "donne-moi une recette de…" sans ambiguïté.
2. **Sections "liste" sémantiques** (``ingredients_list``, ``recipe_steps``,
   ``recipe_history``) : **un chunk par item** (``<li>`` ou ``<p>``
   numéroté). C'est la granularité qui maximise la précision des réponses
   conversationnelles ("quelle est l'étape 3 ?", "donne-moi le 6ᵉ
   commandement", etc.) sans diluer l'embedding.
3. **Sections de prose** (``chef_bio``, ``recipe_summary``, ``intro``,
   ``chef_astuce``…) : si <= ``MAX_TOKENS`` → 1 chunk ; sinon découpe
   par phrases (cf. ``_split_long``) avec recadrage propre sur les espaces.
4. Chaque chunk est préfixé d'un header ``[titre | kind | heading]`` pour
   aider l'embedding à rester ancré sur le bon article/section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken
from bs4 import BeautifulSoup

from ..ingestion.html_sectionizer import Section


MAX_TOKENS = 320
OVERLAP_TOKENS = 40
ENCODING = "cl100k_base"  # compatible text-embedding-3-large

# Sections où l'on veut **un chunk par item** (ingrédient, étape,
# commandement…). Pour ces sections, le `text` plat est insuffisant : on
# repart du HTML pour récupérer la structure ``<li>``/``<p>``.
_ITEMIZED_KINDS = {"ingredients_list", "recipe_steps", "recipe_history", "list"}

_enc = tiktoken.get_encoding(ENCODING)


@dataclass
class ChunkRecord:
    position: int
    section_position: int | None
    kind: str
    text: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text or "", disallowed_special=()))


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _split_long(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Découpe sans couper au milieu d’un mot : phrases d’abord, puis fenêtres
    avec borne droite sur un espace (plus de fragments du type « bour… » + « , versez »).
    """
    t = (text or "").strip()
    if not t:
        return []
    if _count_tokens(t) <= max_tokens:
        return [t]

    # 1) Blocs par phrases courtes
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(t) if s.strip()]
    if len(sentences) <= 1:
        sentences = [t]

    parts: list[str] = []
    buf: list[str] = []
    tok = 0
    gap = 1  # espace entre phrases

    def flush() -> None:
        nonlocal buf, tok
        if buf:
            parts.append(" ".join(buf))
            buf = []
            tok = 0

    for sent in sentences:
        stoks = _count_tokens(sent)
        if stoks > max_tokens:
            flush()
            parts.extend(_split_long_overflow_sentence(sent, max_tokens, overlap))
            continue
        if not buf:
            buf.append(sent)
            tok = stoks
            continue
        if tok + gap + stoks <= max_tokens:
            buf.append(sent)
            tok += gap + stoks
        else:
            flush()
            buf = [sent]
            tok = stoks
    flush()

    out: list[str] = []
    for p in parts:
        if _count_tokens(p) <= max_tokens:
            out.append(p)
        else:
            out.extend(_split_long_overflow_sentence(p, max_tokens, overlap))
    return [x for x in out if x.strip()]


def _split_long_overflow_sentence(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Découpe un long paragraphe : fenêtre max tokens puis reculer à la dernière espace.

    Pas de chevauchement token-par-token (causait des morceils « , suite… » au début du 2ᵉ chunk).
    Le paramètre ``overlap`` est conservé pour compat ; le découpage reste propre.
    """
    _ = overlap  # conservé pour compat d’API ; overlap optionnel pour une V2
    n = len(text)
    chunks: list[str] = []
    start = 0
    while start < n:
        lo, hi = start + 1, n
        cut = start
        while lo <= hi:
            mid = (lo + hi) // 2
            frag = text[start:mid]
            if frag.strip() and _count_tokens(frag) <= max_tokens:
                cut = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if cut <= start:
            cut = min(start + 1, n)
        piece = text[start:cut]
        if cut < n:
            sp = piece.rfind(" ")
            if sp > max(20, len(piece) // 4):
                piece = piece[:sp]
                cut = start + sp
        piece = piece.strip()
        if piece:
            chunks.append(piece)
        if cut >= n:
            break
        start = cut
    return [c for c in chunks if c.strip()]


def _items_from_section(sec: Section) -> list[str]:
    """Extrait les items d'une section "liste" depuis son HTML.

    On regarde d'abord les ``<li>`` (cas standard), puis les ``<p>``
    (cas du payload "9 commandements" rangé dans `introduction` en
    paragraphes numérotés ``1- ...``).
    """
    if not sec.html:
        return []
    soup = BeautifulSoup(sec.html, "lxml")
    lis = soup.find_all("li")
    if lis:
        return [li.get_text(" ", strip=True) for li in lis if li.get_text(strip=True)]
    ps = soup.find_all("p")
    items: list[str] = []
    for p in ps:
        t = p.get_text(" ", strip=True)
        if t:
            items.append(t)
    return items


def _build_anchor_chunk(
    *,
    article_external_id: int,
    article_title: str,
    sections: list[Section],
    pos: int,
) -> ChunkRecord | None:
    """Chunk d'ancrage : titre + résumé éditorial + chef + infos pratiques.

    Permet aux requêtes type "qu'est-ce que X ?" ou "qui a fait Y ?" de
    retomber sur un fragment auto-suffisant qui cite la bonne recette.
    """
    pieces: list[str] = [f"Titre : {article_title}"]
    by_kind: dict[str, Section] = {}
    for sec in sections:
        if sec.kind not in by_kind:
            by_kind[sec.kind] = sec
    if "recipe_summary" in by_kind:
        txt = by_kind["recipe_summary"].text.strip()
        if txt:
            pieces.append(f"Résumé : {txt}")
    if "recipe_meta" in by_kind:
        txt = by_kind["recipe_meta"].text.strip()
        if txt:
            pieces.append(f"Infos pratiques : {txt}")
    if "chef_bio" in by_kind:
        head = by_kind["chef_bio"].heading or "Chef"
        pieces.append(head)
    if len(pieces) <= 1:
        return None
    body = "\n".join(pieces)
    header = f"[{article_title} | anchor]"
    text = f"{header}\n\n{body}"
    return ChunkRecord(
        position=pos,
        section_position=None,
        kind="anchor",
        text=text,
        token_count=_count_tokens(text),
        metadata={
            "article_external_id": article_external_id,
            "anchor": True,
        },
    )


def chunk_article(
    *,
    article_external_id: int,
    article_title: str,
    sections: list[Section],
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[ChunkRecord]:
    """Convertit les sections d'un article en chunks vectorisables.

    Stratégie hybride :
    - 1 chunk d'ancrage (titre + résumé + chef + infos),
    - 1 chunk **par item** pour les sections "liste" sémantiques,
    - 1 ou plusieurs chunks "phrase" pour la prose.
    """
    chunks: list[ChunkRecord] = []
    pos = 0

    anchor = _build_anchor_chunk(
        article_external_id=article_external_id,
        article_title=article_title,
        sections=sections,
        pos=pos,
    )
    if anchor is not None:
        chunks.append(anchor)
        pos += 1

    for sec in sections:
        if not (sec.text and sec.text.strip()) and not sec.html:
            continue

        if sec.kind in _ITEMIZED_KINDS:
            items = _items_from_section(sec)
            # Garde-fou : si on n'a pas pu extraire d'items, on retombe
            # sur le découpage prose (ne perd rien).
            if items:
                total = len(items)
                for i, item in enumerate(items):
                    item_label = _item_label(sec.kind, i, total)
                    header = f"[{article_title} | {sec.kind}"
                    if sec.heading:
                        header += f" | {sec.heading}"
                    header += f" | {item_label}]"
                    text = f"{header}\n\n{item.strip()}"
                    chunks.append(
                        ChunkRecord(
                            position=pos,
                            section_position=sec.position,
                            kind=sec.kind,
                            text=text,
                            token_count=_count_tokens(text),
                            metadata={
                                "article_external_id": article_external_id,
                                "section_heading": sec.heading,
                                "item_index": i,
                                "item_total": total,
                                "item_label": item_label,
                            },
                        )
                    )
                    pos += 1
                continue

        pieces = _split_long(sec.text, max_tokens=max_tokens, overlap=overlap)
        for i, piece in enumerate(pieces):
            header = f"[{article_title} | {sec.kind}"
            if sec.heading:
                header += f" | {sec.heading}"
            header += "]"
            text = f"{header}\n\n{piece.strip()}"
            chunks.append(
                ChunkRecord(
                    position=pos,
                    section_position=sec.position,
                    kind=sec.kind,
                    text=text,
                    token_count=_count_tokens(text),
                    metadata={
                        "article_external_id": article_external_id,
                        "section_heading": sec.heading,
                        "split_index": i,
                        "split_total": len(pieces),
                    },
                )
            )
            pos += 1
    return chunks


def _item_label(kind: str, idx: int, total: int) -> str:
    """Étiquette lisible pour un item (sert dans le header du chunk)."""
    if kind == "ingredients_list":
        return f"ingrédient {idx + 1}/{total}"
    if kind == "recipe_steps":
        return f"étape {idx + 1}/{total}"
    if kind == "recipe_history":
        return f"point {idx + 1}/{total}"
    return f"item {idx + 1}/{total}"
