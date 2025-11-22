"""
Content Index for RAG Retrieval
Implements BM25/TF-IDF based lexical search over recipe content
"""

import logging
from typing import Literal
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.models.schemas import ContentDocument, RecipeArticle, StructuredRecipe
from app.data.normalizers import normalize_text, create_searchable_text

logger = logging.getLogger(__name__)


class ContentIndex:
    """
    Content index for RAG retrieval
    Uses TF-IDF for lexical similarity
    """

    def __init__(self):
        self.documents: list[ContentDocument] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_vectors: np.ndarray | None = None
        self._is_built = False

    def add_olj_articles(self, articles: list[RecipeArticle]):
        """Add OLJ articles to the index"""
        logger.info(f"Adding {len(articles)} OLJ articles to content index")

        for article in articles:
            # Create searchable content from article fields
            content_parts = [
                article.title,
                article.description,
                article.anecdote,
                " ".join(article.tags),
                article.chef or "",
            ]

            content = create_searchable_text(content_parts)

            doc = ContentDocument(
                doc_id=f"olj_{article.article_id}",
                source="olj",
                content=content,
                metadata={
                    "article_id": article.article_id,
                    "title": article.title,
                    "url": article.url,
                    "chef": article.chef,
                    "tags": article.tags,
                },
            )

            self.documents.append(doc)

    def add_structured_recipes(self, recipes: list[StructuredRecipe]):
        """Add structured recipes to the index"""
        logger.info(f"Adding {len(recipes)} structured recipes to content index")

        for recipe in recipes:
            # Create searchable content
            ingredients_text = " ".join(ing.nom for ing in recipe.ingredients)
            steps_text = " ".join(recipe.steps)

            content_parts = [
                recipe.name,
                recipe.category,
                ingredients_text,
                steps_text,
                " ".join(recipe.tags),
            ]

            content = create_searchable_text(content_parts)

            doc = ContentDocument(
                doc_id=f"base2_{recipe.recipe_id}",
                source="base2",
                content=content,
                metadata={
                    "recipe_id": recipe.recipe_id,
                    "name": recipe.name,
                    "category": recipe.category,
                    "ingredients": [ing.nom for ing in recipe.ingredients],
                    "difficulty": recipe.difficulty,
                },
            )

            self.documents.append(doc)

    def build(self):
        """Build the TF-IDF index"""
        if not self.documents:
            logger.warning("No documents to index")
            return

        logger.info(f"Building content index with {len(self.documents)} documents")

        # Extract all content texts
        contents = [doc.content for doc in self.documents]

        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1,
            max_df=0.8,
            sublinear_tf=True,
        )

        self.doc_vectors = self.vectorizer.fit_transform(contents)
        self._is_built = True

        logger.info("Content index built successfully")

    def search(
        self,
        query: str,
        top_k: int = 10,
        source_filter: Literal["olj", "base2", "all"] = "all",
    ) -> list[tuple[ContentDocument, float]]:
        """
        Search the content index

        Returns list of (document, score) tuples, sorted by relevance
        """
        if not self._is_built:
            logger.error("Index not built. Call build() first.")
            return []

        # Normalize query
        normalized_query = normalize_text(query)

        # Vectorize query
        query_vector = self.vectorizer.transform([normalized_query])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1]

        # Filter by source if requested
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = similarities[idx]

            # Apply source filter
            if source_filter != "all" and doc.source != source_filter:
                continue

            results.append((doc, float(score)))

            if len(results) >= top_k:
                break

        return results

    def search_by_ingredients(
        self,
        ingredients: list[str],
        top_k: int = 10,
    ) -> list[tuple[ContentDocument, float]]:
        """
        Search for recipes by ingredients
        Specialized search that prioritizes ingredient matches with equivalence support
        """
        if not self._is_built:
            return []

        # Import ingredient normalizer
        from app.data.ingredient_normalizer import ingredient_normalizer

        # Create query from ingredients (with equivalents for broader search)
        expanded_ingredients = ingredient_normalizer.normalize_ingredient_list(ingredients)
        query = " ".join(expanded_ingredients[:10])  # Limit to avoid too long query
        normalized_query = normalize_text(query)

        # Search
        results = self.search(normalized_query, top_k=top_k * 2, source_filter="base2")

        # Re-score based on ingredient overlap with equivalence matching
        rescored_results = []
        for doc, base_score in results:
            doc_ingredients = doc.metadata.get("ingredients", [])

            # Use ingredient normalizer for matching with equivalences
            matches, match_ratio = ingredient_normalizer.match_ingredients(
                query_ingredients=ingredients,
                doc_ingredients=doc_ingredients,
            )

            # Boost score based on match ratio
            # Higher weight on ingredient matches for ingredient-based queries
            final_score = base_score * 0.3 + match_ratio * 0.7

            rescored_results.append((doc, final_score))

        # Re-sort by final score
        rescored_results.sort(key=lambda x: x[1], reverse=True)

        return rescored_results[:top_k]

    def get_document_by_id(self, doc_id: str) -> ContentDocument | None:
        """Get a document by ID"""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None

    @property
    def is_built(self) -> bool:
        """Check if index is built"""
        return self._is_built

    def __len__(self) -> int:
        """Number of documents in index"""
        return len(self.documents)
