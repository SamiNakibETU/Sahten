# Sahteïn v4 — RAG SOTA (avril 2026)

Refonte complète du backend Sahteïn pour atteindre l'état de l'art RAG :
**Postgres + pgvector + tsvector + RRF SQL + cross-encoder rerank +
grounding phrase-par-phrase**, avec ingestion exhaustive de l'API
WhiteBeard de L'Orient-Le Jour (toutes les biographies de chefs, tous
les encadrés éditoriaux, toutes les listes de commandements/ingrédients
sont désormais capturés et exploités).

> Cette branche `sota/v4` coexiste avec la prod actuelle (`backend/` au
> niveau supérieur). Quand v4 est validée, elle remplacera l'ancienne.

## TL;DR architecture

```
Article OLJ
   │ (API WhiteBeard /content/{id}, exhaustif)
   ▼
backend/app/ingestion/whitebeard_client.py
   │ payload brut JSON
   ▼
backend/app/ingestion/mapper.py + html_sectionizer.py
   │ MappedArticle  ─►  sections {bio, ingredients_list, recipe_steps, quote, ...}
   ▼                     persons (avec biography_text)
   │                     keywords / categories
backend/app/ingestion/repository.py
   │ upsert idempotent Postgres normalisé
   ▼
backend/app/rag/chunker.py + embeddings.py + indexer.py
   │ chunks vectorisés (text-embedding-3-large)
   ▼
backend/app/rag/retriever.py  (1 seule requête SQL hybride RRF)
   │ + reranker.py (Cohere rerank-multilingual-v3.0)
   ▼
backend/app/llm/query_understanding.py  (filtres + rewrite, JSON schema)
backend/app/llm/response_generator.py   (réponse + grounding par phrase)
   │
   ▼
backend/app/api/chat.py  →  POST /api/chat
```

## Démarrer en local

```bash
# 1. Stack infra (Postgres pgvector + Redis)
docker compose -f v4/infra/docker-compose.yml up -d

# 2. Venv + deps
cd v4
python -m venv .venv && .venv\Scripts\activate
pip install -e .[dev]

# 3. .env (copier puis remplir OPENAI_API_KEY, OLJ_API_KEY, etc.)
cp .env.example .env

# 4. Migrations
cd infra/alembic && alembic -c alembic.ini upgrade head && cd ../..

# 5. Tests rapides
pytest tests/test_html_sectionizer.py tests/test_mapper.py -v

# 6. Ingestion d'un article spécifique (le taboulé de Mouzawak)
python scripts/ingest_cli.py one 1227694

# 7. Évaluation RAG (golden set)
#    Windows: set PYTHONPATH=backend
#    Linux/macOS: export PYTHONPATH=backend
python scripts/run_rag_eval.py --golden data/golden_eval_fr.json
#    Voir docs/eval-golden-set.md ; RAGAS optionnel via pip install -e .[eval]

# 8. Serveur dev
uvicorn backend.main:app --reload
```

## Phases livrées

| Phase | Statut | Livrables |
|-------|--------|-----------|
| 0 | Probe API rejouable | `scripts/fetch_whitebeard.py`, `scripts/probe_auth.py`, fixture mock 1227694 |
| 1 | Fondation | `pyproject.toml`, `settings.py`, `docker-compose.yml`, `.env.example` |
| 2 | Schéma DB | `db/models.py`, migration Alembic `0001_initial_schema.py` |
| 3 | Ingestion | `whitebeard_client`, `html_sectionizer`, `mapper`, `repository`, `service` (10/10 tests verts) |
| 4 | RAG hybride | `chunker`, `embeddings`, `indexer`, `retriever` (RRF SQL pur), `reranker` |
| 5 | LLM | `query_understanding`, `response_generator` (grounding), `pipeline`, API `/api/chat` |
| 6 | Worker | `worker/tasks.py` (arq), `scripts/ingest_cli.py` |
| 7 | Évaluation | `data/golden_eval_fr.json`, `scripts/run_rag_eval.py`, `docs/eval-golden-set.md` |
| 8 | Déploiement | `Dockerfile.{web,worker,migrations}`, `railway.toml`, `.gitignore` |
| 9 | Frontend | (à faire — voir section *Reste à faire*) |

## Preuves de captation

10 tests unitaires (`pytest tests/test_html_sectionizer.py tests/test_mapper.py`) prouvent
que la chaîne d'ingestion v4 capte bien, sur le mock 1227694 :

- la **bio complète** de Kamal Mouzawak (« Souk el-Tayeb », « Tawlet », « Beit »)
- la liste des **9 commandements** (`recipe_steps`/`list` ordered)
- la **liste d'ingrédients** structurée (persil, bourghol, menthe, citron…)
- les **catégories** (Cuisine), les **mots-clés** (Taboulé, Souk el-Tayeb…)
- la **citation éditoriale** (« S'il existe un point commun entre tous les Libanais… »)

C'est exactement ce que la v3 perdait à l'ingestion.

## Reste à faire (post-merge v4)

- Phase 0 réelle : rejouer `scripts/fetch_whitebeard.py 1227694` avec la
  clé Railway valide pour confirmer la structure exacte du payload (et
  ajuster `mapper.py` au besoin).
- Backfill complet : `python scripts/ingest_cli.py backfill --max-pages 200`
  sur Railway, après migrations.
- Frontend widget : remettre les hash de cache-bust (`?v=hash` injecté
  par un script de build), URL embed canonique documentée.
- Aplatir le dépôt Git : voir `docs/repo-flatten.md` (à créer) — opération
  destructive, à valider explicitement avec l'utilisateur.

## Suppression de l'ancien code (Phase 6 deslop, à exécuter)

Une fois la v4 validée en staging, supprimer du dépôt courant :

- `backend/app/intent_router.py`
- `backend/app/mood_intent_patterns.py`
- `backend/app/query_plan_patterns.py`
- `backend/app/rag/editorial_snippets.py`
- `backend/scripts/convert_csv_to_canonical.py`
- `backend/data/Data_base_2.json` et toute trace de `_search_base2`
- `data/olj_canonical.json` (remplacé par Postgres)
