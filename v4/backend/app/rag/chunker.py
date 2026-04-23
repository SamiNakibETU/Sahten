"""Chunking sémantique aligné sur les sections HTML.

Stratégie
---------
1. Une `ArticleSection` courte (<= MAX_TOKENS) devient **1 chunk** ; on
   préserve son `kind` (bio, ingredients_list, recipe_steps, quote, ...).
2. Une section longue est découpée en sous-chunks de `MAX_TOKENS` tokens
   avec `OVERLAP_TOKENS` de chevauchement pour ne rien perdre au bord.
3. On préfixe chaque chunk avec un *header contextualisant* :
   ``[<title> | <kind> | <heading>]\\n\\n<text>`` — ce préfixe aide
   énormément l'embedding à rester ancré sur le bon article/section.

Sortie : liste de `ChunkRecord` indépendants, prêts à être embeddés et
insérés dans la table `chunks`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tiktoken

from ..ingestion.html_sectionizer import Section


MAX_TOKENS = 320
OVERLAP_TOKENS = 40
ENCODING = "cl100k_base"  # compatible text-embedding-3-large

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


def chunk_article(
    *,
    article_external_id: int,
    article_title: str,
    sections: list[Section],
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[ChunkRecord]:
    """Convertit les sections d'un article en chunks vectorisables."""
    chunks: list[ChunkRecord] = []
    pos = 0
    for sec in sections:
        if not sec.text or not sec.text.strip():
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
