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


def _split_long(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Split greedy par tokens, avec chevauchement pour préserver le contexte."""
    ids = _enc.encode(text, disallowed_special=())
    if len(ids) <= max_tokens:
        return [text]
    out: list[str] = []
    step = max(1, max_tokens - overlap)
    for start in range(0, len(ids), step):
        window = ids[start : start + max_tokens]
        if not window:
            break
        out.append(_enc.decode(window))
        if start + max_tokens >= len(ids):
            break
    return out


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
