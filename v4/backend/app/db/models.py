"""Schéma normalisé Sahteïn v4.

Choix structurants
------------------
1. *Une ligne par article* (`Article`), enrichie de tous les champs WhiteBeard.
2. *Sections HTML découpées* (`ArticleSection`) : c'est l'unité de chunking
   *avant* embeddings. Permet de cibler "la section technique du chef" ou
   "les commandements du taboulé".
3. *Auteurs/contributeurs normalisés* (`Person`, `ArticleAuthor`) : on stocke
   biographies + descriptions + rôle (chef invité, journaliste, traducteur).
4. *Mots-clés et catégories* (`Keyword`, `Category`) référencés par lien
   pour permettre des filtres natifs en SQL.
5. *Ingrédients structurés* (`Ingredient`, `RecipeIngredient`) — source de
   vérité pour la recherche par ingrédient (plus de regex magiques).
6. *Chunks vectorisés* (`Chunk`) : grain de retrieval avec embedding pgvector
   et `tsvector` natif Postgres pour BM25.

Toutes les colonnes timestampées en UTC.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..settings import get_settings
from .base import Base

if TYPE_CHECKING:
    pass


_EMBED_DIM = get_settings().embedding_dim


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


# =====================================================================
# Personnes (chefs, journalistes, contributeurs)
# =====================================================================
class Person(Base, TimestampMixin):
    __tablename__ = "persons"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[int | None] = mapped_column(
        BigInteger, unique=True, index=True,
        comment="ID WhiteBeard authors[].id",
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    role: Mapped[str | None] = mapped_column(String(64))  # 'chef', 'journalist', 'translator'
    department: Mapped[str | None] = mapped_column(String(128))
    biography_html: Mapped[str | None] = mapped_column(Text)
    biography_text: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    image_url: Mapped[str | None] = mapped_column(Text)
    raw_payload: Mapped[dict | None] = mapped_column(JSONB)

    article_links: Mapped[list["ArticleAuthor"]] = relationship(
        back_populates="person", cascade="all, delete-orphan"
    )


# =====================================================================
# Catégories & mots-clés (référentiels normalisés)
# =====================================================================
class Category(Base, TimestampMixin):
    __tablename__ = "categories"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text)


class Keyword(Base, TimestampMixin):
    __tablename__ = "keywords"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[int | None] = mapped_column(BigInteger, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    description: Mapped[str | None] = mapped_column(Text)


class Ingredient(Base, TimestampMixin):
    """Référentiel d'ingrédients (lemmatisé, FR canonique)."""
    __tablename__ = "ingredients"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    slug: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    aliases: Mapped[list[str] | None] = mapped_column(JSONB)
    category: Mapped[str | None] = mapped_column(String(64))  # legume, viande, epice...


# =====================================================================
# Article (1 ligne par contenu CMS)
# =====================================================================
class Article(Base, TimestampMixin):
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    external_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, index=True, nullable=False,
        comment="ID WhiteBeard (data[0].id).",
    )
    url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    subtitle: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    introduction: Mapped[str | None] = mapped_column(Text)
    signature: Mapped[str | None] = mapped_column(Text)
    body_html: Mapped[str | None] = mapped_column(Text)
    body_text: Mapped[str | None] = mapped_column(Text)
    content_length: Mapped[int | None] = mapped_column(Integer)
    time_to_read: Mapped[int | None] = mapped_column(Integer)
    is_premium: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    cover_image_url: Mapped[str | None] = mapped_column(Text)
    cover_image_caption: Mapped[str | None] = mapped_column(Text)

    first_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    seo: Mapped[dict | None] = mapped_column(JSONB)
    raw_payload: Mapped[dict | None] = mapped_column(
        JSONB, comment="Payload WhiteBeard brut (audit + retraitements)."
    )
    ingestion_source: Mapped[str] = mapped_column(
        String(32), default="whitebeard", nullable=False,
        comment="whitebeard | webhook | playwright | manual",
    )
    ingestion_status: Mapped[str] = mapped_column(
        String(32), default="ok", nullable=False,
        comment="ok | partial | failed | needs_playwright",
    )
    ingestion_notes: Mapped[str | None] = mapped_column(Text)

    search_tsv: Mapped[str | None] = mapped_column(  # noqa: F811
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('french', coalesce(title,'')), 'A') || "
            "setweight(to_tsvector('french', coalesce(summary,'')), 'B') || "
            "setweight(to_tsvector('french', coalesce(body_text,'')), 'C')",
            persisted=True,
        ),
    )

    sections: Mapped[list["ArticleSection"]] = relationship(
        back_populates="article", cascade="all, delete-orphan",
        order_by="ArticleSection.position",
    )
    authors: Mapped[list["ArticleAuthor"]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )
    keyword_links: Mapped[list["ArticleKeyword"]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )
    category_links: Mapped[list["ArticleCategory"]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )
    ingredient_links: Mapped[list["ArticleIngredient"]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )


Index("ix_articles_search_tsv", Article.search_tsv, postgresql_using="gin")


# =====================================================================
# Sections HTML d'un article (h2/h3, listes, citations, encadrés)
# =====================================================================
class ArticleSection(Base, TimestampMixin):
    __tablename__ = "article_sections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(
        String(32), nullable=False,
        comment="paragraph | heading | list | quote | sidebar | recipe_steps | ingredients_list | bio",
    )
    heading: Mapped[str | None] = mapped_column(Text)
    html: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB)

    article: Mapped[Article] = relationship(back_populates="sections")


Index(
    "ix_article_sections_kind",
    ArticleSection.article_id,
    ArticleSection.kind,
)


# =====================================================================
# Liens N–N
# =====================================================================
class ArticleAuthor(Base, TimestampMixin):
    __tablename__ = "article_authors"
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True
    )
    person_id: Mapped[int] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), primary_key=True
    )
    role: Mapped[str] = mapped_column(
        String(32), nullable=False, default="contributor",
        comment="featured_chef | journalist | photographer | contributor",
    )
    position: Mapped[int] = mapped_column(Integer, default=0)

    article: Mapped[Article] = relationship(back_populates="authors")
    person: Mapped[Person] = relationship(back_populates="article_links")


class ArticleKeyword(Base):
    __tablename__ = "article_keywords"
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True
    )
    keyword_id: Mapped[int] = mapped_column(
        ForeignKey("keywords.id", ondelete="CASCADE"), primary_key=True
    )
    article: Mapped[Article] = relationship(back_populates="keyword_links")
    keyword: Mapped[Keyword] = relationship()


class ArticleCategory(Base):
    __tablename__ = "article_categories"
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), primary_key=True
    )
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"), primary_key=True
    )
    article: Mapped[Article] = relationship(back_populates="category_links")
    category: Mapped[Category] = relationship()


class ArticleIngredient(Base):
    __tablename__ = "article_ingredients"
    __table_args__ = (
        UniqueConstraint("article_id", "ingredient_id", name="uq_article_ingredient"),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), index=True, nullable=False
    )
    ingredient_id: Mapped[int] = mapped_column(
        ForeignKey("ingredients.id", ondelete="CASCADE"), index=True, nullable=False
    )
    quantity: Mapped[str | None] = mapped_column(String(64))
    unit: Mapped[str | None] = mapped_column(String(32))
    raw_text: Mapped[str | None] = mapped_column(Text)
    is_main: Mapped[bool] = mapped_column(Boolean, default=False)

    article: Mapped[Article] = relationship(back_populates="ingredient_links")
    ingredient: Mapped[Ingredient] = relationship()


# =====================================================================
# Chunks (grain de retrieval RAG, embedding pgvector + tsvector)
# =====================================================================
class Chunk(Base, TimestampMixin):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("article_id", "position", name="uq_chunk_article_position"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[int] = mapped_column(
        ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    section_id: Mapped[int | None] = mapped_column(
        ForeignKey("article_sections.id", ondelete="SET NULL"), index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer)
    metadata_json: Mapped[dict | None] = mapped_column(JSONB)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(_EMBED_DIM))
    embedding_model: Mapped[str | None] = mapped_column(String(64))

    search_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed(
            "to_tsvector('french', coalesce(text,''))",
            persisted=True,
        ),
    )

    article: Mapped[Article] = relationship(back_populates="chunks")


Index("ix_chunks_search_tsv", Chunk.search_tsv, postgresql_using="gin")
# Index HNSW pgvector créé via migration Alembic (DDL custom).


# =====================================================================
# Logs d'ingestion (audit)
# =====================================================================
class IngestionLog(Base):
    __tablename__ = "ingestion_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_external_id: Mapped[int | None] = mapped_column(BigInteger, index=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer)
    error_message: Mapped[str | None] = mapped_column(Text)
    payload_size: Mapped[int | None] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# =====================================================================
# Évaluations RAG (golden set + scores RAGAS)
# =====================================================================
class EvalQuery(Base, TimestampMixin):
    __tablename__ = "eval_queries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    expected_article_ids: Mapped[list[int] | None] = mapped_column(JSONB)
    expected_answer_must_contain: Mapped[list[str] | None] = mapped_column(JSONB)
    tags: Mapped[list[str] | None] = mapped_column(JSONB)


class EvalRun(Base):
    __tablename__ = "eval_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    git_sha: Mapped[str | None] = mapped_column(String(64))
    config_json: Mapped[dict | None] = mapped_column(JSONB)
    faithfulness: Mapped[float | None] = mapped_column(Float)
    answer_relevancy: Mapped[float | None] = mapped_column(Float)
    context_precision: Mapped[float | None] = mapped_column(Float)
    context_recall: Mapped[float | None] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text)
