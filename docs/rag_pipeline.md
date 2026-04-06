# Pipeline RAG Sahten (référence)

## Flux

1. **SafetyGate** — filtrage déterministe (injection, etc.).
2. **Compréhension** — `route_intent_deterministic` (cas triviaux : `about_bot`, menu entrée/plat/dessert, salutation courte, off-topic fort, clarification « c’est quoi… », et **`pattern_override_plan`** pour quelques motifs ultra stables). Sinon **`QueryAnalyzer.analyze`** : sortie **`QueryPlan`** validé (JSON schema strict ou `json_object` + Pydantic), projeté en **`QueryAnalysis`** via `query_plan_to_analysis` pour compat pipeline. Même **`pattern_override_plan`** est réévalué en tête d’analyse si le routeur n’a pas pris la main (tests / cohérence). Les motifs **`mood_intent_patterns`** ne servent qu’au **fallback offline** quand l’appel LLM échoue (`query_analyzer.py`).
3. **Retrieval** — `HybridRetriever.search_with_rerank` :
   - Allowlist de catégories et préfixe rerank pilotés par **`analysis.plan`** (`task`, `course`, `cuisine_scope`, `retrieval_focus`) quand présent.
   - Construction de `retrieval_query` : focus **`effective_retrieval_focus()`** (plan) sinon champs classiques ; **`maybe_rewrite_for_retrieval`** est court-circuité si le plan fournit déjà un `retrieval_focus` non vide.
   - **Lexical** : TF-IDF mots + caractères.
   - **Sémantique** : embeddings si `ENABLE_EMBEDDINGS=true` (OpenAI ou mock via `embedding_client`), matrice mise en cache disque si `embedding_cache_enabled` et provider OpenAI.
   - **Fusion** : RRF sur listes lexical + sémantique.
   - **Rerank** : cross-encoder optionnel (`ENABLE_CROSS_ENCODER_RERANK`) puis rerank LLM (ou CE seul si `RERANK_LLM_AFTER_CROSS_ENCODER=false`).
4. **Garde** — pour `recipe_specific`, vérification titre vs plat demandé.
5. **Alternatives** — si vide : ingrédients partagés / catégorie (`retrieval_constants`, `get_alternative_by_shared_ingredient`).
6. **Génération** — `ResponseGenerator` ; payload JSON inclut **`conversation_recent`** si session avec historique ; par recette OLJ : **`cited_passage`** (élargi), **`recipe_lead`**, **`story_snippet`** (extraits déterministes depuis `search_text`, voir `editorial_snippets.py`).

## Mémoire de session (onglet)

- Le client envoie **`session_id`** (ex. `sessionStorage`) ; `SessionManager` garde tours, URLs déjà proposées, titres de fiches (`recipe_titles` sur chaque tour).
- **`get_context_for_continuation()`** expose `recent_turns`, **`recent_recipe_titles`**, `recipes_proposed_count`, etc. Paramètres : `SESSION_MAX_TURNS`, `SESSION_TURN_MAX_CHARS`, `SESSION_RETRIEVAL_CONTEXT_MAX_CHARS` (voir `Settings`).
- **`format_session_hints_for_analyzer`** enrichit le prompt du `QueryAnalyzer` (coréférences) ; le rerank LLM reçoit un préfixe `[Session]` avec les titres récents (`LLMReranker.session_prefix`).
- Limite : stockage **mémoire process** (LRU/TTL) — sans Redis partagé, pas d’historique entre workers non sticky.

## Clarification pilotée par le plan

- Si **`ENABLE_PLAN_CLARIFICATION=true`** et `QueryPlan.needs_clarification` avec `clarification_question` non vide : réponse type **`clarification`** **sans retrieval** (voir `bot.py`).

## Fichiers clés

| Fichier | Rôle |
|---------|------|
| `backend/app/schemas/query_plan.py` | Modèle `QueryPlan` + schéma OpenAI strict |
| `backend/app/schemas/query_plan_mapper.py` | `QueryPlan` → `QueryAnalysis` |
| `backend/app/core/query_plan_patterns.py` | Motifs déterministes → `QueryPlan` (ex. N mezze, salade de pâtes) |
| `backend/app/llm/query_analyzer.py` | Appel LLM `QueryPlan`, température 0 |
| `backend/app/bot.py` | Orchestration, session, rewrite, timings |
| `backend/app/rag/retriever.py` | Index, search, rerank, sélection cartes |
| `backend/app/rag/retrieval_constants.py` | Tokens génériques, alternatives |
| `backend/app/rag/citation_quality.py` | Filtre titres/passages type interview |
| `backend/app/rag/editorial_snippets.py` | Chapô + anecdote depuis `search_text` pour le générateur |
| `backend/app/rag/session_manager.py` | Sessions, `format_session_hints_for_analyzer` |
| `backend/app/services/embedding_cache.py` | Cache `.npz` + métadonnées |
| `backend/app/llm/query_rewriter.py` | Réécriture retrieval (JSON) |
| `backend/app/llm/cross_encoder_rerank.py` | Rerank local optionnel |
| `data/rag_golden_queries.yaml` | Jeu d’évaluation |
| `backend/scripts/run_rag_golden.py` | Batch diagnostic JSONL |

## Variables d’environnement (extrait)

- `ENABLE_EMBEDDINGS` — active la recherche dense + cache (OpenAI).
- `EMBEDDING_CACHE_ENABLED` — cache disque sous `data_dir/.cache/` (défaut config).
- `ENABLE_CROSS_ENCODER_RERANK` — nécessite `sentence-transformers` (+ `torch`).
- `ENABLE_RETRIEVAL_QUERY_REWRITE` — réécriture avant retrieval (défaut true).
- `SESSION_MAX_TURNS`, `SESSION_TURN_MAX_CHARS`, `SESSION_RETRIEVAL_CONTEXT_MAX_CHARS` — fenêtre mémoire session.
- `ENABLE_PLAN_CLARIFICATION` — question courte avant retrieval si le `QueryPlan` le demande (défaut false).

## Évaluation

```bash
cd backend
python scripts/run_rag_golden.py --out ../.rag_report.jsonl
```

Comparer `checks` (intent, optionnellement `expected_task` / `expected_course` / `expected_cuisine_scope`, URL attendue) dans la sortie JSONL.
