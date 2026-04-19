# Déploiement Railway — 4 services

## Vue d'ensemble

| Service | Image | Rôle |
|---------|-------|------|
| `sahten-postgres` | `pgvector/pgvector:pg16` (Railway plugin "PostgreSQL" custom image) | Base de données + extension vector + FTS |
| `sahten-redis` | `redis:7-alpine` (Railway plugin "Redis") | Queue arq + cache |
| `sahten-web` | Build `v4/Dockerfile.web` | API FastAPI + webhook |
| `sahten-worker` | Build `v4/Dockerfile.worker` | Worker arq (ingestion / reindex) |
| `sahten-migrations` | Build `v4/Dockerfile.migrations` (one-off / job) | `alembic upgrade head` |

## Variables d'environnement (par service)

### Communes
```
APP_ENV=production
LOG_LEVEL=INFO
DATABASE_URL=${{ sahten-postgres.DATABASE_URL }}        # remplacer asyncpg via init Python
REDIS_URL=${{ sahten-redis.REDIS_URL }}
OPENAI_API_KEY=...
OLJ_API_BASE=https://api.lorientlejour.com/cms
OLJ_API_KEY=...               # clé valide (la clé locale 21-char retourne 401)
WEBHOOK_SECRET=...            # généré : python -c "import uuid; print(uuid.uuid4())"
COHERE_API_KEY=...            # rerank-multilingual-v3.0
LLM_MODEL=gpt-4.1
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
```

> Astuce : `DATABASE_URL` exposée par Railway commence par `postgresql://`.
> Notre code ajoute automatiquement `+asyncpg` via `Settings`. Si Railway
> exige le DSN brut, remplacer dans `settings.py` ou utiliser un alias
> `DATABASE_URL_ASYNC`.

## Procédure de bootstrap initial

1. Créer le projet Railway, ajouter Postgres et Redis.
2. Sur le **Postgres Railway**, exécuter une fois (via psql) :
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS pg_trgm;
   CREATE EXTENSION IF NOT EXISTS unaccent;
   ```
   (la migration le fait aussi, mais Postgres Railway peut bloquer
   `CREATE EXTENSION` si l'utilisateur n'a pas les bons droits ;
   demander à Railway support si besoin).
3. Déployer `sahten-migrations` une seule fois.
4. Déployer `sahten-web` (healthcheck `/healthz`).
5. Déployer `sahten-worker` (pas de healthcheck HTTP).
6. Lancer le backfill manuel :
   ```bash
   railway run --service sahten-worker python -m scripts.ingest_cli backfill --max-pages 200
   ```
7. Vérifier dans `eval_runs` que les scores RAGAS sont > 0.7
   (`scripts/eval_rag.py`).

## Webhook OLJ -> Sahteïn

URL à configurer côté CMS WhiteBeard :
```
POST https://<sahten-web>.up.railway.app/api/webhook/recipe
Header X-Signature-256: sha256=<hmac_sha256(WEBHOOK_SECRET, body)>
```

Body attendu :
```json
{ "event": "article.published", "article_id": 1227694 }
```

## Rollback

- `railway rollback <service>` revient à la version précédente.
- Les chunks/embeddings sont recalculables : il suffit de
  `python -m scripts.ingest_cli reindex <id>` ou un nouveau backfill.
