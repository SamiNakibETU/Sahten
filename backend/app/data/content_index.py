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
from app.models.config import settings
from app.services.embedding_client import EmbeddingClient, get_embedding_client

logger = logging.getLogger(__name__)


class ContentIndex:
    """
    Content index for RAG retrieval
    Uses TF-IDF for lexical similarity
    """

    def __init__(self, embedding_client: EmbeddingClient | None = None):
        self.documents: list[ContentDocument] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.doc_vectors: np.ndarray | None = None
        self._is_built = False
        self.doc_lookup: dict[str, ContentDocument] = {}
        self.embedding_client = embedding_client
        self.enable_embeddings = settings.enable_embeddings
        self.olj_embedding_matrix: np.ndarray | None = None
        self.olj_embedding_ids: list[str] = []
        self.base2_embedding_matrix: np.ndarray | None = None
        self.base2_embedding_ids: list[str] = []

    def add_olj_articles(self, articles: list[RecipeArticle]):
        """Add OLJ articles to the index"""
        logger.info(f"Adding {len(articles)} OLJ articles to content index")

        for article in articles:
            # Create searchable content from article fields
            content_parts = [
                article.title,
                article.description,
                article.anecdote,
                article.doc_text,
                " ".join(article.tags),
                article.chef or "",
                " ".join(article.main_ingredients),
                article.course or "",
                article.diet or "",
            ]

            content = create_searchable_text(content_parts)

            embedding_fields = [
                article.title,
                article.short_summary,
                article.doc_text,
                " ".join(article.main_ingredients),
                article.dish_name or "",
            ]
            embedding_text = create_searchable_text(embedding_fields)

            doc_text = article.doc_text or " ".join(
                part for part in [article.description, article.short_summary, article.anecdote] if part
            )
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
                    "course": article.course,
                    "diet": article.diet,
                    "main_ingredients": article.main_ingredients,
                    "editorial_score": article.editorial_score,
                    "recency_score": article.recency_score,
                    "is_recipe": article.is_recipe,
                    "embedding_text": embedding_text,
                    "doc_text": doc_text,
                },
            )

            self.documents.append(doc)
            self.doc_lookup[doc.doc_id] = doc

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
            embedding_text = create_searchable_text(
                [recipe.name, recipe.category, ingredients_text]
            )

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
                    "embedding_text": embedding_text,
                },
            )

            self.documents.append(doc)
            self.doc_lookup[doc.doc_id] = doc

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

        if self.enable_embeddings:
            self._build_embedding_index()

        logger.info("Content index built successfully")

    def _build_embedding_index(self):
        """Build embedding matrices for semantic retrieval."""
        try:
            if self.embedding_client is None:
                self.embedding_client = get_embedding_client()
        except Exception as exc:
            self.enable_embeddings = False
            logger.warning("Embeddings disabled (client init failed): %s", exc)
            return

        for source in ("olj", "base2"):
            docs = [doc for doc in self.documents if doc.source == source]
            if not docs:
                continue

            texts = [
                doc.metadata.get("embedding_text") or doc.content
                for doc in docs
            ]

            try:
                vectors = np.asarray(self.embedding_client.embed(texts), dtype=np.float32)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to build %s embedding index: %s", source, exc)
                self.enable_embeddings = False
                return

            if vectors.size == 0:
                continue

            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms

            if source == "olj":
                self.olj_embedding_matrix = vectors
                self.olj_embedding_ids = [doc.doc_id for doc in docs]
            else:
                self.base2_embedding_matrix = vectors
                self.base2_embedding_ids = [doc.doc_id for doc in docs]

    def _select_embedding_matrix(
        self, source_filter: Literal["olj", "base2"]
    ) -> tuple[np.ndarray | None, list[str]]:
        if source_filter == "olj":
            return self.olj_embedding_matrix, self.olj_embedding_ids
        if source_filter == "base2":
            return self.base2_embedding_matrix, self.base2_embedding_ids
        return None, []

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        source_filter: Literal["olj", "base2"] = "olj",
    ) -> list[tuple[ContentDocument, float]]:
        """Semantic (embedding) search helper."""
        if not self.enable_embeddings or not self.embedding_client:
            return []

        matrix, doc_ids = self._select_embedding_matrix(source_filter)
        if matrix is None or not doc_ids:
            return []

        query_text = create_searchable_text([query])

        try:
            query_vector = np.asarray(
                self.embedding_client.embed([query_text])[0],
                dtype=np.float32,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to embed query: %s", exc)
            return []

        norm = np.linalg.norm(query_vector)
        if norm:
            query_vector = query_vector / norm

        scores = matrix @ query_vector
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[ContentDocument, float]] = []
        for idx in top_indices:
            doc_id = doc_ids[idx]
            doc = self.doc_lookup.get(doc_id)
            if not doc:
                continue
            results.append((doc, float(scores[idx])))

        return results

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
        normalized_struct = ingredient_normalizer.normalize_ingredient_list(ingredients)
        expanded_ingredients = [item.normalized for item in normalized_struct]
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
        return self.doc_lookup.get(doc_id)

    @property
    def is_built(self) -> bool:
        """Check if index is built"""
        return self._is_built

    def __len__(self) -> int:
        """Number of documents in index"""
        return len(self.documents)
