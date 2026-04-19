# Suppression des anciennes versions — plan en 3 phases

> **Aucune suppression destructive ne sera faite sans validation
> explicite de votre part.** Ce document liste l'ordre, les commandes,
> et les rollbacks.

## Inventaire à supprimer

### A — Fichiers/dossiers du **workspace local** (hors repo)

À la racine `d:\…\V3\` :

| Élément | Type | Action |
|---------|------|--------|
| `key.txt` | Secret en clair | **À supprimer immédiatement et rotater la clé** |
| `mdp.txt` | Secret en clair | **À supprimer immédiatement et rotater le mdp** |
| `dev_conv` | Conv dev (peut contenir secrets) | À examiner puis supprimer |
| `sahten/`, `sahten-dev/`, `sahten-mvp/`, `sahten-scraper/` | Variantes obsolètes du projet | À déplacer dans un dossier `_archive_/2026-04/` puis supprimer après 30 jours |
| `archive/`, `api/`, `scripts/`, `assets/`, `DESIGN/`, `Logo Animation Design/` | Brouillons divers | Idem |
| `inspect_data.py`, `show_block.py`, `patch_welcome_emoji.py`, `test_chef_extraction.py` | Scripts dev temporaires | À déplacer dans `_archive_/2026-04/old-scripts/` |
| 11 logos legacy (`logo_*.svg`, `v5_logo.svg`, `v6_*.svg`, `v7_*.svg`, `sahten_logo_v*.svg`) | Anciens jets | À déplacer dans `_archive_/2026-04/old-logos/` |
| `.pytest_cache/` | Cache pytest | Suppression directe (généré) |
| `vercel.json` à la racine | Config orpheline | À supprimer (pas de Vercel dans la stack) |
| `requirements.txt` à la racine | Doublonne `sahten_github/backend/requirements.txt` | À supprimer |
| `VERSION_AND_STRUCTURE.md`, `EMAIL_TO_DEV_TEAM.txt`, `dev_conv` | Docs racine | Examiner, déplacer dans `_archive_/2026-04/old-docs/` |

**Script de cleanup** (dry-run par défaut) :
[`v4/scripts/cleanup_workspace.py`](../scripts/cleanup_workspace.py).

### B — Fichiers du **repo Git** (anciens code v3 à supprimer côté repo)

À supprimer dans `sahten_github/` une fois la v4 validée en staging
(seuil RAGAS > 0.75 stable 48h) :

| Fichier | Raison |
|---------|--------|
| `backend/app/intent_router.py` | Heuristiques regex remplacées par `query_understanding.py` |
| `backend/app/mood_intent_patterns.py` | Idem |
| `backend/app/query_plan_patterns.py` | Idem |
| `backend/app/rag/editorial_snippets.py` | Remplacé par sectionizer HTML structuré |
| `backend/app/rag/retriever.py` (l'ancien) | Remplacé par `v4/backend/app/rag/retriever.py` |
| `backend/scripts/convert_csv_to_canonical.py` | CSV obsolète, source = API WhiteBeard |
| `backend/data/Data_base_2.json` | Base secondaire ad-hoc, supprimée |
| `data/olj_canonical.json` | Remplacé par Postgres |

Procédure :

```bash
git checkout sota/v4
git rm backend/app/intent_router.py \
       backend/app/mood_intent_patterns.py \
       backend/app/query_plan_patterns.py \
       backend/app/rag/editorial_snippets.py \
       backend/app/rag/retriever.py \
       backend/scripts/convert_csv_to_canonical.py \
       backend/data/Data_base_2.json \
       data/olj_canonical.json
git commit -m "chore(v4): remove v3 heuristics + canonical JSON (replaced by Postgres+pgvector)"
```

> **Ne pas faire avant** que la v4 réponde à toutes les questions du
> golden set de manière satisfaisante en prod.

### C — Aplatissement du **repo distant** (action destructive)

Voir [`v4/docs/repo-flatten.md`](repo-flatten.md). Cette opération
réécrit l'historique Git, **change tous les SHAs**, et **invalide
tous les forks/clones existants**.

Pré-requis stricts :

1. v4 en prod et stable depuis 7 jours.
2. Backup mirror du repo (`git clone --mirror`) sauvegardé hors-ligne.
3. **Rotation de tous les secrets** qui ont déjà été dans
   l'historique : `OLJ_API_KEY`, `OPENAI_API_KEY`, `WEBHOOK_SECRET`,
   `COHERE_API_KEY`, mots de passe BD éventuels. Tant que
   l'historique git contient les fichiers `key.txt`, `mdp.txt`,
   `.env`, ces secrets sont à considérer **compromis**.
4. Annonce à l'équipe OLJ + WhiteBeard : "demain à <heure>, le repo
   `Sahten` sera réécrit ; merci de re-cloner après l'opération".

---

## Calendrier recommandé

| Jour | Action |
|------|--------|
| J0 (aujourd'hui) | Push `sota/v4` (FAIT), provisionner Railway staging, lancer migrations + backfill |
| J1 | Évaluation RAGAS, fix éventuels, tests utilisateurs internes |
| J3 | Promotion staging → prod si RAGAS stable |
| J7 | Confirmation prod stable → suppression code v3 (étape B) |
| J10 | Rotation des secrets compromis |
| J14 | `git filter-repo` (étape C), prévenir équipes |
| J15 | Re-clones par toutes les équipes |
| J30 | Suppression définitive de `_archive_/2026-04/` du workspace local |
