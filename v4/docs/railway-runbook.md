# Runbook Railway — Déploiement Sahteïn v4 (pas-à-pas)

> Suit dans l'ordre. Chaque étape mentionne ce qui peut casser et
> comment rollback. Temps total estimé : **45-90 minutes**, dont
> 15-30 minutes d'attente pour le backfill initial.

## Pré-requis

- Compte Railway avec un projet existant (probablement appelé
  `Sahtein` ou `Sahten`).
- CLI Railway installée : `npm install -g @railway/cli` puis
  `railway login`.
- Accès à : `OPENAI_API_KEY`, `OLJ_API_KEY` (la valide, pas celle qui
  donne 401), `COHERE_API_KEY` (créer sur cohere.com si besoin).

---

## Étape 1 — Sauvegarder l'existant (sécurité, 5 min)

Sur le projet Railway courant :

1. Allez dans le service `sahten-web` actuel → **Deployments** → notez
   le SHA Git du déploiement en prod (ex. `d6b2c2c`).
2. Allez dans **Variables** → exportez tout via :
   ```bash
   railway variables --service sahten-web > backup-vars-$(date +%F).txt
   ```
3. Si vous avez déjà une DB Railway : **Snapshot** depuis l'UI
   Postgres → "Create snapshot".

**Rollback** : `railway redeploy --commit d6b2c2c` ramène l'ancien code.

---

## Étape 2 — Provisionner les 4 services v4 (10 min)

Sur **un nouveau projet Railway** appelé `sahten-v4-staging` (mieux
qu'un risque sur la prod) :

### 2.1 Postgres avec pgvector

Railway → "New" → "Database" → "PostgreSQL". Une fois créée :

```bash
railway connect Postgres
# dans psql :
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;
\q
```

> Si Railway refuse `CREATE EXTENSION vector` (droits insuffisants),
> demander au support Railway d'activer `pgvector` (gratuit, ils le
> font en < 1h sur ticket).

### 2.2 Redis

Railway → "New" → "Database" → "Redis". Aucun setup requis.

### 2.3 Service Web

Railway → "New" → "Empty Service" → nom : `sahten-web`.

- **Source** : connecter le GitHub `SamiNakibETU/Sahten`, branche
  `sota/v4`.
- **Build** : Dockerfile, chemin `v4/Dockerfile.web`.
- **Health check path** : `/healthz`.
- **Port** : `8000` (Railway l'auto-détecte via la var `PORT`).

### 2.4 Service Worker

Idem mais Dockerfile = `v4/Dockerfile.worker`, **pas** de health HTTP
(arq tourne en boucle).

### 2.5 Service Migrations (one-off job)

Railway → "New" → "Empty Service" → nom : `sahten-migrations`.

- Dockerfile : `v4/Dockerfile.migrations`.
- **Restart policy** : `Never` (c'est un job ponctuel).
- Vous le lancerez à la main quand on touche au schéma.

---

## Étape 3 — Configurer les variables d'environnement (10 min)

Sur **chaque** service web/worker/migrations, ajouter via Railway UI
ou CLI. Exemple CLI :

```bash
railway variables --service sahten-web set \
  APP_ENV=production \
  LOG_LEVEL=INFO \
  LLM_MODEL=gpt-4.1 \
  EMBEDDING_MODEL=text-embedding-3-large \
  EMBEDDING_DIM=3072 \
  RAG_RERANK_TOP_K=8 \
  RAG_RRF_K=60 \
  OLJ_API_BASE=https://api.lorientlejour.com/cms

# Secrets (à entrer un par un, jamais committés)
railway variables --service sahten-web set OPENAI_API_KEY=sk-...
railway variables --service sahten-web set COHERE_API_KEY=...
railway variables --service sahten-web set OLJ_API_KEY=<clé valide>
railway variables --service sahten-web set WEBHOOK_SECRET=$(python -c "import uuid; print(uuid.uuid4())")
```

**Variables référencées** (Railway interpole automatiquement) :

```
DATABASE_URL  →  ${{ Postgres.DATABASE_URL }}
REDIS_URL     →  ${{ Redis.REDIS_URL }}
```

> **Important** : `Postgres.DATABASE_URL` chez Railway commence par
> `postgresql://`. Notre `settings.py` accepte ce format ; SQLAlchemy
> l'utilisera avec `+asyncpg` automatiquement via le driver. Si vous
> voyez une erreur `'postgresql' is not async`, ajoutez explicitement
> `DATABASE_URL=postgresql+asyncpg://…` (copier la valeur de Railway et
> insérer `+asyncpg`).

Répéter le bloc `set` pour `sahten-worker` et `sahten-migrations`.

---

## Étape 4 — Migrations DB (5 min)

```bash
railway run --service sahten-migrations alembic -c infra/alembic/alembic.ini upgrade head
```

Vérification :

```bash
railway connect Postgres
\dt        # doit lister 13 tables : articles, article_sections, persons, …
\dx        # doit lister vector, pg_trgm, unaccent
SELECT count(*) FROM information_schema.indexes WHERE indexname='ix_chunks_embedding_hnsw';
# → 1
```

**Rollback** : `alembic downgrade -1`.

---

## Étape 5 — Premier déploiement Web + Worker (5 min)

Railway redéploie automatiquement à chaque push sur `sota/v4`. Forcer
un déploiement manuel :

```bash
railway up --service sahten-web
railway up --service sahten-worker
```

Vérification :

```bash
curl https://sahten-web-staging.up.railway.app/healthz
# {"status":"ok","version":"4.0.0a1"}

curl https://sahten-web-staging.up.railway.app/readyz
# {"status":"ok","db":"ok","version":"4.0.0a1"}
```

---

## Étape 6 — Backfill complet (15-30 min)

Lancer en background sur le worker :

```bash
railway run --service sahten-worker python -m scripts.ingest_cli backfill --max-pages 200
```

Suivre la progression :

```bash
railway logs --service sahten-worker --follow
```

Sortie attendue à la fin :

```json
{"ok": 1185, "partial": 32, "needs_playwright": 4, "failed": 0}
```

Si `needs_playwright > 0`, c'est attendu pour les articles dont le HTML
est partiellement généré côté client. À traiter dans une itération
suivante (module Playwright fallback prévu dans la roadmap).

---

## Étape 7 — Évaluation RAGAS (5 min)

```bash
railway run --service sahten-web python -m scripts.eval_rag \
  --golden v4/tests/golden/golden_set.jsonl
```

Sortie attendue :

```json
{
  "faithfulness": 0.92,
  "answer_relevancy": 0.88,
  "context_precision": 0.85,
  "context_recall": 0.81
}
```

Seuil minimal acceptable : **toutes ≥ 0.75**. Sinon investiguer
(probable cause : seuil rerank trop strict, ou backfill incomplet).
Les scores sont aussi persistés dans la table `eval_runs` pour suivi.

---

## Étape 8 — Webhook WhiteBeard (5 min)

Côté WhiteBeard, pointer le webhook vers :

```
POST https://sahten-web-staging.up.railway.app/api/webhook/recipe
Header X-Signature-256: sha256=<hmac_sha256(WEBHOOK_SECRET, body)>
```

Tester avec un article modifié côté CMS, puis :

```bash
railway logs --service sahten-web | grep webhook
# doit montrer "ingest.ok" pour l'article modifié
```

---

## Étape 9 — Promotion staging → production

Une fois RAGAS > 0.75 et 24h sans incident :

1. Renommer le projet `sahten-v4-staging` en `sahten-v4-prod`.
2. Mettre à jour le DNS / la config OLJ pour pointer vers la nouvelle
   URL (`sahtein.lorientlejour.com` recommandé).
3. Garder l'ancien projet (v3 sur `feature/ui-sahtein-olj-v2`) pendant
   **7 jours** comme rollback chaud.
4. Au bout de 7 jours sans incident, archiver l'ancien projet.

---

## En cas de pépin

| Symptôme | Investigation |
|----------|---------------|
| `/readyz` répond `db: error` | Vérifier `DATABASE_URL` et que pgvector est activé |
| `/api/chat` 500 « Cohere unauthorized » | `COHERE_API_KEY` manquante ou expirée — fallback BGE local s'active si `ENABLE_LOCAL_RERANK_FALLBACK=true` |
| Embeddings dimension mismatch | `EMBEDDING_DIM` ≠ celui des chunks → `python -m scripts.ingest_cli reindex <id>` ou nouveau backfill |
| Worker ne consomme pas les jobs | `arq` doit pointer sur le même Redis que le web ; `REDIS_URL` identique entre les services ? |
| Webhook 401 | `WEBHOOK_SECRET` doit être identique côté Sahteïn ET côté WhiteBeard |
