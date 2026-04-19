"""Initial schema: extensions, tables, indexes pgvector + tsvector.

Revision ID: 0001_initial
Revises:
Create Date: 2026-04-19
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

from backend.app.settings import get_settings

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBED_DIM = get_settings().embedding_dim


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    op.execute("CREATE EXTENSION IF NOT EXISTS unaccent")

    # --- referentiels ----------------------------------------------------
    op.create_table(
        "persons",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("external_id", sa.BigInteger, unique=True, index=True),
        sa.Column("name", sa.String(255), nullable=False, index=True),
        sa.Column("slug", sa.String(255), nullable=False, unique=True),
        sa.Column("role", sa.String(64)),
        sa.Column("department", sa.String(128)),
        sa.Column("biography_html", sa.Text),
        sa.Column("biography_text", sa.Text),
        sa.Column("description", sa.Text),
        sa.Column("image_url", sa.Text),
        sa.Column("raw_payload", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "categories",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("external_id", sa.BigInteger, unique=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("slug", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "keywords",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("external_id", sa.BigInteger, unique=True),
        sa.Column("name", sa.String(255), nullable=False, unique=True),
        sa.Column("slug", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "ingredients",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("slug", sa.String(128), nullable=False, unique=True),
        sa.Column("aliases", JSONB),
        sa.Column("category", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )

    # --- articles --------------------------------------------------------
    op.create_table(
        "articles",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("external_id", sa.BigInteger, nullable=False, unique=True),
        sa.Column("url", sa.Text, nullable=False, unique=True),
        sa.Column("slug", sa.String(255), nullable=False, unique=True),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("subtitle", sa.Text),
        sa.Column("summary", sa.Text),
        sa.Column("introduction", sa.Text),
        sa.Column("signature", sa.Text),
        sa.Column("body_html", sa.Text),
        sa.Column("body_text", sa.Text),
        sa.Column("content_length", sa.Integer),
        sa.Column("time_to_read", sa.Integer),
        sa.Column("is_premium", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("cover_image_url", sa.Text),
        sa.Column("cover_image_caption", sa.Text),
        sa.Column("first_published_at", sa.DateTime(timezone=True)),
        sa.Column("last_updated_at", sa.DateTime(timezone=True)),
        sa.Column("seo", JSONB),
        sa.Column("raw_payload", JSONB),
        sa.Column("ingestion_source", sa.String(32), nullable=False, server_default="whitebeard"),
        sa.Column("ingestion_status", sa.String(32), nullable=False, server_default="ok"),
        sa.Column("ingestion_notes", sa.Text),
        sa.Column(
            "search_tsv",
            TSVECTOR,
            sa.Computed(
                "setweight(to_tsvector('french', coalesce(title,'')), 'A') || "
                "setweight(to_tsvector('french', coalesce(summary,'')), 'B') || "
                "setweight(to_tsvector('french', coalesce(body_text,'')), 'C')",
                persisted=True,
            ),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_articles_search_tsv", "articles", ["search_tsv"], postgresql_using="gin")
    op.create_index("ix_articles_first_published_at", "articles", ["first_published_at"])

    op.create_table(
        "article_sections",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("article_id", sa.Integer,
                  sa.ForeignKey("articles.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("heading", sa.Text),
        sa.Column("html", sa.Text, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("metadata_json", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_article_sections_kind", "article_sections", ["article_id", "kind"])

    # --- liens N-N -------------------------------------------------------
    op.create_table(
        "article_authors",
        sa.Column("article_id", sa.Integer,
                  sa.ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("person_id", sa.Integer,
                  sa.ForeignKey("persons.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", sa.String(32), nullable=False, server_default="contributor"),
        sa.Column("position", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "article_keywords",
        sa.Column("article_id", sa.Integer,
                  sa.ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("keyword_id", sa.Integer,
                  sa.ForeignKey("keywords.id", ondelete="CASCADE"), primary_key=True),
    )
    op.create_table(
        "article_categories",
        sa.Column("article_id", sa.Integer,
                  sa.ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("category_id", sa.Integer,
                  sa.ForeignKey("categories.id", ondelete="CASCADE"), primary_key=True),
    )
    op.create_table(
        "article_ingredients",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("article_id", sa.Integer,
                  sa.ForeignKey("articles.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("ingredient_id", sa.Integer,
                  sa.ForeignKey("ingredients.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("quantity", sa.String(64)),
        sa.Column("unit", sa.String(32)),
        sa.Column("raw_text", sa.Text),
        sa.Column("is_main", sa.Boolean, server_default=sa.text("false")),
        sa.UniqueConstraint("article_id", "ingredient_id", name="uq_article_ingredient"),
    )

    # --- chunks (vector) -------------------------------------------------
    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("article_id", sa.Integer,
                  sa.ForeignKey("articles.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("section_id", sa.Integer,
                  sa.ForeignKey("article_sections.id", ondelete="SET NULL"),
                  index=True),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer),
        sa.Column("metadata_json", JSONB),
        sa.Column("embedding_model", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("article_id", "position", name="uq_chunk_article_position"),
    )
    op.execute(f"ALTER TABLE chunks ADD COLUMN embedding vector({EMBED_DIM})")
    op.execute(
        "ALTER TABLE chunks ADD COLUMN search_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('french', coalesce(text,''))) STORED"
    )
    op.create_index("ix_chunks_search_tsv", "chunks", ["search_tsv"], postgresql_using="gin")
    # Index HNSW pgvector (cosine, m=16 ef_construction=64 = SOTA defaults)
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )

    # --- audit + eval ----------------------------------------------------
    op.create_table(
        "ingestion_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("article_external_id", sa.BigInteger, index=True),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("duration_ms", sa.Integer),
        sa.Column("error_message", sa.Text),
        sa.Column("payload_size", sa.Integer),
        sa.Column("started_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "eval_queries",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("query", sa.Text, nullable=False),
        sa.Column("expected_article_ids", JSONB),
        sa.Column("expected_answer_must_contain", JSONB),
        sa.Column("tags", JSONB),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
    )
    op.create_table(
        "eval_runs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("started_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("git_sha", sa.String(64)),
        sa.Column("config_json", JSONB),
        sa.Column("faithfulness", sa.Float),
        sa.Column("answer_relevancy", sa.Float),
        sa.Column("context_precision", sa.Float),
        sa.Column("context_recall", sa.Float),
        sa.Column("notes", sa.Text),
    )


def downgrade() -> None:
    for table in (
        "eval_runs", "eval_queries", "ingestion_logs",
        "chunks",
        "article_ingredients", "article_categories", "article_keywords", "article_authors",
        "article_sections", "articles",
        "ingredients", "keywords", "categories", "persons",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    # On laisse les extensions (vector, pg_trgm, unaccent) en place.
