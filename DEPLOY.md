# Déploiement Sahten (Railway et GitHub)

## Branche déployée

- **Railway** peut être configuré pour déployer une branche précise (ex. `feature/ui-sahtein-olj-v2` ou `main`).
- **Source de vérité recommandée** : une seule branche de production, par exemple `main`.
- Si l’équipe utilise une branche fonctionnelle pour Railway, synchroniser explicitement après merge :

```bash
git checkout main
git pull
git push origin main:feature/ui-sahtein-olj-v2
```

(À adapter selon le nom de branche déployée.)

Vérifier dans le dashboard Railway : **Settings → Source** (branche et dépôt).

## Variables d’environnement

- `OPENAI_API_KEY` (obligatoire pour le chat)
- `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN` (sessions Redis, traces)
- Autres variables selon `backend/app/core/config.py` / `.env.example` si présent

## Santé

- `railway.toml` définit `healthcheckPath = "/api/health"` (ou équivalent selon votre `main.py`).

## CI GitHub

- Workflow `.github/workflows/ci.yml` : `pytest -m "not llm"` (pas de clé OpenAI requise).
- Les tests marqués `llm` (`tests/test_eval.py`) nécessitent `OPENAI_API_KEY` : les lancer en local ou ajouter un job avec le secret `OPENAI_API_KEY` dans les réglages du dépôt.

## Données canoniques

- Fichier `data/olj_canonical.json` : qualité des `main_ingredients` et `search_text` critique pour le RAG.
- Script d’audit : `python backend/scripts/audit_canonical_data.py` (depuis la racine du clone, avec `cd backend` si besoin selon la doc du script).
