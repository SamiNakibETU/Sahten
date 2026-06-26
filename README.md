# Sahteïn

Assistant culinaire libanais de **L'Orient-Le Jour** — un RAG sur les recettes de
chefs publiées dans « Liban à Table ».

> ⚠️ **L'application vivante est dans [`v4/`](v4/).** L'ancienne app v2.1
> (`backend/`, `frontend/`, JSON canonique) a été supprimée le 2026-06-25 :
> elle est remplacée par la stack Postgres + pgvector de `v4/`.
> Récupérable dans l'historique git si besoin (`git log -- backend`).

## Architecture (v4)

FastAPI + **Postgres/pgvector + tsvector** + **Redis** (cache + queue `arq`),
Docker, Alembic. Pipeline RAG hybride (dense + lexical → RRF → rerank Cohere →
rerank article → génération groundée phrase-par-phrase).

```
v4/
├── backend/app/        # API + RAG + LLM + ingestion + db
├── web_static/         # frontend (widget, demo, admin, dashboard)
├── scripts/            # ingest_cli, run_rag_eval, audits
├── data/               # golden_eval_fr.json (marqueurs), olj_seed_ids.json
├── docs/               # runbooks, go-live, audit RAG/Epicure
└── Dockerfile.web      # image déployée par railway.toml
```

## Démarrage local

```bash
cd v4
pip install -e .[dev]
cp .env.example .env          # OPENAI_API_KEY, DATABASE_URL, REDIS_URL, COHERE_API_KEY...
docker compose -f infra/docker-compose.yml up -d   # Postgres + Redis
cd infra/alembic && alembic upgrade head
python -m scripts.ingest_cli reindex-all --publication 17 --content-type 4 --seed-file data/olj_seed_ids.json
python -m uvicorn backend.main:app --reload
```

## Tests & marqueurs d'acceptation

```bash
cd v4 && set PYTHONPATH=.
pytest -q                                              # 60 tests unitaires
python scripts/run_rag_eval.py --golden data/golden_eval_fr.json   # golden set (DB peuplée)
```

Voir [`Specifications.md`](Specifications.md) pour le périmètre, l'arbitrage
modèle et la définition de « FINI ».

## Déploiement

Railway : service **web** (`v4/Dockerfile.web`) + **Postgres** (pgvector) +
**Redis**. Health check `GET /healthz`. Détails : [`v4/docs/railway-runbook.md`](v4/docs/railway-runbook.md).

## Licence

Usage interne L'Orient-Le Jour.
