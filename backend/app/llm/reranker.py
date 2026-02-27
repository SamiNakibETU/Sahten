"""
LLM Reranker
============

Implements the second stage of retrieval:
  retrieve(topK) -> rerank(topK) -> select(topN)

We use OpenAI JSON mode to get a stable output schema.
This reranker runs only on a small topK to control cost/latency.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from ..core.config import get_settings

logger = logging.getLogger(__name__)


class RerankItem(BaseModel):
    url: str
    score: float = Field(ge=0.0, le=1.0)
    cited_passage: Optional[str] = Field(
        default=None,
        description="Passage pertinent extrait du document qui justifie sa pertinence"
    )


class RerankResult(BaseModel):
    items: List[RerankItem] = Field(default_factory=list)


@dataclass(frozen=True)
class RerankCandidate:
    url: str
    title: str
    search_text_excerpt: str
    category: str
    cuisine_type: Optional[str]
    is_lebanese: bool


RERANK_SYSTEM_PROMPT = """Tu es un reranker de documents pour un assistant culinaire de L'Orient-Le Jour.

Objectif: classer les documents selon leur pertinence pour la requête utilisateur et extraire un passage justificatif.

Règles:
- Privilégie les RECETTES et les résultats directement pertinents.
- Si l'utilisateur demande cuisine libanaise ou contexte libanais, privilégie is_lebanese=true.
- Si la requête mentionne un ingrédient, favorise les recettes qui le contiennent (ou en parlent clairement).
- Donne un score entre 0 et 1 (1 = parfait).
- Pour chaque résultat, extrais un passage court (1-2 phrases, max 150 caractères) de l'excerpt qui justifie pourquoi ce document est pertinent pour la requête.
- Le passage doit être une citation directe de l'excerpt, pas une reformulation.
"""


class LLMReranker:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        settings = get_settings()
        self.api_key = api_key or settings.openai_api_key
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = model or settings.rerank_model or settings.openai_model

    async def rerank(
        self,
        query: str,
        candidates: List[RerankCandidate],
        *,
        max_items: int = 10,
    ) -> List[RerankItem]:
        if not candidates:
            return []

        # Offline/durable fallback: no API key => no network call, just return stable fallback order.
        if not self.api_key:
            out: list[RerankItem] = []
            n = len(candidates)
            for i, c in enumerate(candidates[:max_items]):
                out.append(RerankItem(url=c.url, score=(n - i) / max(n, 1)))
            return out

        # Keep prompt small and stable
        packed: list[dict[str, Any]] = []
        for c in candidates:
            packed.append(
                {
                    "url": c.url,
                    "title": c.title,
                    "excerpt": c.search_text_excerpt[:350],
                    "category": c.category,
                    "cuisine_type": c.cuisine_type,
                    "is_lebanese": c.is_lebanese,
                }
            )

        user_prompt = (
            "Requête utilisateur:\n"
            f"{query}\n\n"
            "Candidats (à classer):\n"
            f"{json.dumps(packed, ensure_ascii=False)}\n\n"
            f"Retourne au plus {max_items} items."
        )

        def _safe_json_loads(content: str) -> dict:
            s = (content or "").strip()
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                s = s[start : end + 1]
            return json.loads(s)

        # Use tool/function calling for robust JSON (more reliable than free-form JSON mode).
        tool_schema = {
            "type": "function",
            "function": {
                "name": "rerank_result",
                "description": "Return the reranked items with scores and cited passages.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "url": {"type": "string"},
                                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                    "cited_passage": {
                                        "type": "string",
                                        "description": "Passage court (1-2 phrases) extrait de l'excerpt justifiant la pertinence"
                                    },
                                },
                                "required": ["url", "score"],
                                "additionalProperties": False,
                            },
                        }
                    },
                    "required": ["items"],
                    "additionalProperties": False,
                },
            },
        }

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                tools=[tool_schema],
                tool_choice={"type": "function", "function": {"name": "rerank_result"}},
                messages=[
                    {"role": "system", "content": RERANK_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_completion_tokens=500,
            )

            msg = resp.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                args = tool_calls[0].function.arguments
                data = _safe_json_loads(args)
            else:
                # Fallback: try content
                data = _safe_json_loads(msg.content or "{}")

            parsed = RerankResult(**data)
            allowed = {c.url for c in candidates}
            filtered = [it for it in parsed.items if it.url in allowed]
            filtered.sort(key=lambda x: -x.score)
            return filtered[:max_items]

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning("Rerank parse failure, falling back to lexical order: %s", e)
        except Exception as e:
            logger.warning("Rerank call failed, falling back to lexical order: %s", e)

        # Fallback: return in given order with descending synthetic scores
        out: list[RerankItem] = []
        n = len(candidates)
        for i, c in enumerate(candidates[:max_items]):
            out.append(RerankItem(url=c.url, score=(n - i) / max(n, 1)))
        return out
