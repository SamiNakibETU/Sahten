"""
Link Index for Precise Article Resolution
Article-level index for ultra-precise URL resolution
Only operates on OLJ articles (Base 1)
"""

import logging
from datetime import datetime, timezone
from typing import Literal
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schemas import LinkDocument, RecipeArticle
from app.data.normalizers import normalize_text, create_searchable_text

logger = logging.getLogger(__name__)


class LinkIndex:
    """
    Article-level index for link resolution
    Ensures all URLs come from real OLJ articles
    """

    def __init__(self):
        self.documents: list[LinkDocument] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_vectors: np.ndarray | None = None
        self._is_built = False

        # Quick lookup maps
        self._article_by_id: dict[str, RecipeArticle] = {}
        self._article_by_normalized_title: dict[str, list[RecipeArticle]] = {}

    def add_articles(self, articles: list[RecipeArticle]):
        """Add articles to link index"""
        logger.info(f"Adding {len(articles)} articles to link index")

        for article in articles:
            # Create searchable text for link resolution
            searchable_parts = [
                article.normalized_title,
                " ".join(article.tags),
                article.chef or "",
                article.slug,
            ]

            searchable_text = create_searchable_text(searchable_parts)

            doc = LinkDocument(
                article=article,
                searchable_text=searchable_text,
            )

            self.documents.append(doc)

            # Build lookup maps
            self._article_by_id[article.article_id] = article

            if article.normalized_title not in self._article_by_normalized_title:
                self._article_by_normalized_title[article.normalized_title] = []
            self._article_by_normalized_title[article.normalized_title].append(article)

    def build(self):
        """Build the link index"""
        if not self.documents:
            logger.warning("No articles in link index")
            return

        logger.info(f"Building link index with {len(self.documents)} articles")

        # Extract searchable texts
        texts = [doc.searchable_text for doc in self.documents]

        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            sublinear_tf=True,
        )

        self.doc_vectors = self.vectorizer.fit_transform(texts)
        self._is_built = True

        logger.info("Link index built successfully")

    def find_exact_match(self, query: str) -> RecipeArticle | None:
        """
        Find exact match by normalized title
        This is the highest confidence match
        """
        normalized_query = normalize_text(query)

        # Try direct lookup
        if normalized_query in self._article_by_normalized_title:
            articles = self._article_by_normalized_title[normalized_query]
            # Return most recent if multiple matches
            return max(articles, key=lambda a: a.publish_date or a.modified_date or "")

        return None

    def find_best_match(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.1,
    ) -> list[tuple[RecipeArticle, float, str]]:
        """
        Find best matching articles using similarity search

        Returns list of (article, score, strategy) tuples
        strategy: "exact", "high_similarity", "moderate_similarity"
        """
        if not self._is_built:
            logger.error("Link index not built")
            return []

        # First try exact match
        exact_match = self.find_exact_match(query)
        if exact_match:
            return [(exact_match, 1.0, "exact")]

        # Otherwise use similarity search
        normalized_query = normalize_text(query)
        query_vector = self.vectorizer.transform([normalized_query])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(similarities[idx])

            if score < min_score:
                continue

            article = self.documents[idx].article

            # Determine strategy based on score
            if score > 0.7:
                strategy = "high_similarity"
            elif score > 0.4:
                strategy = "moderate_similarity"
            else:
                strategy = "low_similarity"

            results.append((article, score, strategy))

        return results

    def get_fallback_articles(
        self,
        strategy: Literal["recent", "popular", "editor_pick"] = "recent",
        count: int = 3,
    ) -> list[RecipeArticle]:
        """
        Get fallback articles when no good match is found
        Used for off-topic queries or when no relevant recipe exists
        """
        articles = [doc.article for doc in self.documents]

        def _safe_date(article: RecipeArticle) -> datetime:
            """Return a timezone-aware datetime for sorting, never raising."""
            candidate = article.modified_date or article.publish_date
            if not candidate:
                return datetime.min.replace(tzinfo=timezone.utc)
            if candidate.tzinfo is None:
                try:
                    return candidate.replace(tzinfo=timezone.utc)
                except Exception:
                    return datetime.min.replace(tzinfo=timezone.utc)
            return candidate

        if strategy == "recent":
            # Sort by most recent publication/modification
            sorted_articles = sorted(
                articles,
                key=_safe_date,
                reverse=True,
            )

        elif strategy == "popular":
            # Sort by popularity score
            sorted_articles = sorted(
                articles,
                key=lambda a: a.popularity_score,
                reverse=True,
            )

        elif strategy == "editor_pick":
            # Filter editor picks first, then by recency
            editor_picks = [a for a in articles if a.is_editor_pick]
            if editor_picks:
                sorted_articles = sorted(
                    editor_picks,
                    key=_safe_date,
                    reverse=True,
                )
            else:
                # Fallback to recent
                sorted_articles = sorted(
                    articles,
                    key=_safe_date,
                    reverse=True,
                )

        else:
            sorted_articles = articles

        return sorted_articles[:count]

    def get_article_by_id(self, article_id: str) -> RecipeArticle | None:
        """Get article by ID"""
        return self._article_by_id.get(article_id)

    def get_articles_by_tag(self, tag: str, limit: int = 5) -> list[RecipeArticle]:
        """Get articles with a specific tag"""
        normalized_tag = normalize_text(tag)
        results = []

        for doc in self.documents:
            article_tags = [normalize_text(t) for t in doc.article.tags]
            if normalized_tag in article_tags or any(normalized_tag in t for t in article_tags):
                results.append(doc.article)

            if len(results) >= limit:
                break

        return results

    def get_articles_by_chef(self, chef: str, limit: int = 5) -> list[RecipeArticle]:
        """Get articles by a specific chef"""
        normalized_chef = normalize_text(chef)
        results = []

        for doc in self.documents:
            if doc.article.chef and normalized_chef in normalize_text(doc.article.chef):
                results.append(doc.article)

            if len(results) >= limit:
                break

        return results

    @property
    def is_built(self) -> bool:
        """Check if index is built"""
        return self._is_built

    def __len__(self) -> int:
        """Number of articles in index"""
        return len(self.documents)
