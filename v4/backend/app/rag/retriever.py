"""Retriever hybride 100 % SQL : pgvector cosine + tsvector BM25 + RRF.

Pourquoi en SQL pur :
  - Latence Postgres ~ 5-30 ms même à 50k chunks
  - Pas de second network hop (Python -> Redis -> Python)
  - RRF est trivial à exprimer en CTE et reste maintenable
  - Permet d'appliquer les filtres (chef, ingrédient, catégorie) avant le
    fan-out (économise le coût des index)

API publique : `HybridRetriever.search(query, ...)` -> liste de `Hit`s.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..settings import get_settings
from .embeddings import EmbeddingProvider


@dataclass
class Hit:
    chunk_id: int
    article_id: int
    article_external_id: int
    article_title: str
    article_url: str
    section_kind: str
    chunk_text: str
    score_lex: float | None
    score_vec: float | None
    score_rrf: float
    metadata: dict[str, Any]


HYBRID_SQL = text(
    """
WITH
params AS (
    SELECT
        CAST(:rrf_k AS int)        AS rrf_k,
        CAST(:top_k_lex AS int)    AS top_k_lex,
        CAST(:top_k_vec AS int)    AS top_k_vec
),
filters AS (
    SELECT
        CAST(:chef_slugs AS text[])      AS chef_slugs,
        CAST(:ingredient_slugs AS text[]) AS ingredient_slugs,
        CAST(:category_slugs AS text[])   AS category_slugs,
        CAST(:keyword_slugs AS text[])    AS keyword_slugs
),
candidate_articles AS (
    -- Sélectionne les articles correspondant aux filtres (ou tous si pas de filtre)
    SELECT a.id
    FROM articles a
    WHERE
        (
            (SELECT cardinality(chef_slugs) FROM filters) = 0 OR EXISTS (
                SELECT 1 FROM article_authors aa
                JOIN persons p ON p.id = aa.person_id
                WHERE aa.article_id = a.id
                  AND aa.role = 'featured_chef'
                  AND p.slug = ANY ((SELECT chef_slugs FROM filters))
            )
        )
        AND (
            (SELECT cardinality(ingredient_slugs) FROM filters) = 0 OR EXISTS (
                SELECT 1 FROM article_ingredients ai
                JOIN ingredients i ON i.id = ai.ingredient_id
                WHERE ai.article_id = a.id
                  AND i.slug = ANY ((SELECT ingredient_slugs FROM filters))
            )
        )
        AND (
            (SELECT cardinality(category_slugs) FROM filters) = 0 OR EXISTS (
                SELECT 1 FROM article_categories ac
                JOIN categories c ON c.id = ac.category_id
                WHERE ac.article_id = a.id
                  AND c.slug = ANY ((SELECT category_slugs FROM filters))
            )
        )
        AND (
            (SELECT cardinality(keyword_slugs) FROM filters) = 0 OR EXISTS (
                SELECT 1 FROM article_keywords ak
                JOIN keywords k ON k.id = ak.keyword_id
                WHERE ak.article_id = a.id
                  AND k.slug = ANY ((SELECT keyword_slugs FROM filters))
            )
        )
),
lex AS (
    SELECT
        c.id AS chunk_id,
        ts_rank_cd(c.search_tsv, websearch_to_tsquery('french', :query)) AS score,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(c.search_tsv, websearch_to_tsquery('french', :query)) DESC
        ) AS rnk
    FROM chunks c
    WHERE c.article_id IN (SELECT id FROM candidate_articles)
      AND c.search_tsv @@ websearch_to_tsquery('french', :query)
    ORDER BY score DESC
    LIMIT (SELECT top_k_lex FROM params)
),
vec AS (
    SELECT
        c.id AS chunk_id,
        1 - (c.embedding <=> CAST(:qvec AS vector)) AS score,
        ROW_NUMBER() OVER (
            ORDER BY c.embedding <=> CAST(:qvec AS vector) ASC
        ) AS rnk
    FROM chunks c
    WHERE c.article_id IN (SELECT id FROM candidate_articles)
      AND c.embedding IS NOT NULL
    ORDER BY c.embedding <=> CAST(:qvec AS vector) ASC
    LIMIT (SELECT top_k_vec FROM params)
),
fused AS (
    SELECT
        chunk_id,
        SUM(rrf_score) AS score_rrf,
        MAX(score_lex) AS score_lex,
        MAX(score_vec) AS score_vec
    FROM (
        SELECT chunk_id,
               1.0 / ((SELECT rrf_k FROM params) + rnk) AS rrf_score,
               score AS score_lex,
               NULL::float AS score_vec
        FROM lex
        UNION ALL
        SELECT chunk_id,
               1.0 / ((SELECT rrf_k FROM params) + rnk) AS rrf_score,
               NULL::float,
               score
        FROM vec
    ) u
    GROUP BY chunk_id
)
SELECT
    c.id          AS chunk_id,
    c.article_id  AS article_id,
    a.external_id AS article_external_id,
    a.title       AS article_title,
    a.url         AS article_url,
    c.kind        AS section_kind,
    c.text        AS chunk_text,
    f.score_lex,
    f.score_vec,
    f.score_rrf,
    c.metadata_json AS metadata
FROM fused f
JOIN chunks c   ON c.id = f.chunk_id
JOIN articles a ON a.id = c.article_id
ORDER BY f.score_rrf DESC
LIMIT :final_limit
""",
).bindparams(
    bindparam("chef_slugs", type_=None),
    bindparam("ingredient_slugs", type_=None),
    bindparam("category_slugs", type_=None),
    bindparam("keyword_slugs", type_=None),
)


class HybridRetriever:
    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder
        self.settings = get_settings()

    async def search(
        self,
        session: AsyncSession,
        query: str,
        *,
        chef_slugs: list[str] | None = None,
        ingredient_slugs: list[str] | None = None,
        category_slugs: list[str] | None = None,
        keyword_slugs: list[str] | None = None,
        final_limit: int = 30,
    ) -> list[Hit]:
        if not query or not query.strip():
            return []
        embeddings = await self.embedder.embed([query])
        qvec = embeddings[0]

        params: dict[str, Any] = {
            "query": query.strip(),
            "qvec": qvec,
            "rrf_k": self.settings.rag_rrf_k,
            "top_k_lex": self.settings.rag_hybrid_top_k_lexical,
            "top_k_vec": self.settings.rag_hybrid_top_k_vector,
            "chef_slugs": chef_slugs or [],
            "ingredient_slugs": ingredient_slugs or [],
            "category_slugs": category_slugs or [],
            "keyword_slugs": keyword_slugs or [],
            "final_limit": final_limit,
        }
        result = await session.execute(HYBRID_SQL, params)
        rows = result.mappings().all()
        return [
            Hit(
                chunk_id=r["chunk_id"],
                article_id=r["article_id"],
                article_external_id=r["article_external_id"],
                article_title=r["article_title"],
                article_url=r["article_url"],
                section_kind=r["section_kind"],
                chunk_text=r["chunk_text"],
                score_lex=r["score_lex"],
                score_vec=r["score_vec"],
                score_rrf=float(r["score_rrf"]),
                metadata=r["metadata"] or {},
            )
            for r in rows
        ]
