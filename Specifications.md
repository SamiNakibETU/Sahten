# Sahteïn — Spécifications & Marqueurs d'acceptation

> Assistant culinaire libanais de **L'Orient-Le Jour**, adossé à un RAG sur les
> recettes de chefs publiées dans « Liban à Table ». Ce document est la **source
> de vérité** du projet : périmètre, architecture, comportement attendu, et les
> **marqueurs** (golden set) qui prouvent que « tout marche ».
>
> Version vivante (branche `sota/v4`). Dernière mise à jour : 2026-06-25.

---

## 1. Vision & périmètre (la décision qui définit « fini »)

Sahteïn **n'est pas** une encyclopédie de cuisine libanaise. C'est un guide qui
ouvre vers les **recettes de chefs publiées par OLJ**. Cette distinction est la
décision produit centrale, car elle transforme un « bug » en comportement voulu :

- ✅ **Dans le périmètre** : recettes de chefs OLJ (taboulé de Kamal Mouzawak,
  cailles de Karim Haïdar, plats d'Aline Kamakian, **manouche/manaïiche** — cf.
  art. 1474718 « Les manaïichs du Chouf de Salim Azzam »…), bios de chefs,
  ingrédients, astuces — **tout ce qui est sourcé dans le corpus**.
- ❌ **Hors périmètre** : cuisines non libanaises/levantines (sushi, pizza,
  burger…) — *absence à confirmer au cas par cas, ne pas présumer*.

**Règle d'or :** hors périmètre ⇒ **abstention honnête**, jamais une description
inventée ni une carte hors-sujet. **Dans** le périmètre ⇒ il faut **trouver**
l'article, quelle que soit la graphie.

> ⚠️ **Diagnostic corrigé (juin 2026)** — vérifié sur l'API live : le manouche
> N'EST PAS hors corpus. L'article 1474718 existe et est indexé, mais seule la
> graphie quasi-exacte (`manaiche`) le retrouve ; `manouche`, `manouché`,
> `man'ouché`, `manakish` échouent (réponse « pas trouvé » + carte hors-sujet
> kishk/lahm). **C'est un bug de translittération au retrieval**, pas un trou de
> corpus ni un problème de modèle. Correctif = canonicalisation des
> translittérations (query + alias d'article) = P1 « vocabulaire canonique »
> (cf. `v4/docs/rag-epicure-audit.md`). Marqueurs : `anti-regression-manouche-*`.

---

## 2. Architecture (état réel)

Le dépôt contient **deux applications** ; une seule est vivante.

| Dossier | Statut | Description |
|---|---|---|
| **`v4/`** | ✅ **PRODUCTION** | FastAPI + **Postgres/pgvector + tsvector**, **Redis** (cache + queue `arq`), Docker, Alembic. C'est ce que `railway.toml` déploie (`v4/Dockerfile.web`). |
| `backend/` | 🗑️ **LEGACY v2.1** | Ancienne app JSON + scikit-learn. Le `Procfile` pointe encore dessus (à corriger). **À supprimer** après confirmation v4. |

Infra Railway : service **web** (FastAPI) + **Postgres** (pgvector) + **Redis**.

### Pipeline RAG (`v4/backend/app/`)

```
Requête utilisateur
 1. llm/query_understanding.py   → QueryPlan JSON strict (intent, ingrédients, chef, reformulation, anaphores)
 2. rag/retriever.py             → HYBRIDE : pgvector (dense, top 50) + tsvector (lexical, top 50) → fusion RRF (k=60)
 3. rag/reranker.py              → Cohere rerank-multilingual-v3.0 (fallback local BAAI/bge-reranker-v2-m3)
 4. rag/pipeline.py              → widening, interleaving par article, rerank article-level (2e passe), priorité source, fallback Base2
 5. llm/response_generator.py    → réponse GROUNDÉE phrase-par-phrase (chaque phrase cite ≥1 chunk_id ; validateur post-LLM)
 6. rag/html_renderer.py         → cartes recette / chef + liens vers l'article OLJ
```

Embeddings : `text-embedding-3-small` (1536d, HNSW pgvector). **Ne pas** passer à
`large`/3072 (casse la compat HNSW sans gain prouvé — cf. `v4/docs/rag-epicure-audit.md`).

---

## 3. Modèles LLM — arbitrage (coût min × qualité FR max)

État actuel : OpenAI uniquement, `gpt-4.1-mini` partout (`v4/backend/app/llm/models_config.py`), temp 0.

**Cible recommandée — router à 2 étages** (à implémenter en Phase 4, après les correctifs) :

| Tâche | Modèle cible | Pourquoi |
|---|---|---|
| Query understanding + rerank article (JSON) | OpenAI `gpt-4.1-nano` | structuré, fiabilité `json_schema` stricte, quasi gratuit |
| **Génération (prose FR visible)** | **Mistral Large 3** (~0,50/1,50 $/M) | français natif, culturellement juste, résidence EU (gouvernance OLJ), API compatible OpenAI |
| Génération « qualité maximale » (option) | Claude Sonnet 4.6 (3/15 $/M) | meilleure plume nuancée si le ton prime |
| Embeddings | `text-embedding-3-small` (inchangé) | compat HNSW, suffisant |

À **éviter** pour la prose lecteur : modèles chinois (DeepSeek/Qwen) — gouvernance
données + nuance FR. OK pour l'étage structuré bon marché si la réduction de coût
devient prioritaire.

> ⚠️ Le changement de modèle est le **petit levier**. Le bug manouche vient du
> gating + prompt + couverture, pas du modèle. Faire la Phase 1 d'abord.

---

## 4. Données & corpus

- Source de vérité runtime : **Postgres** (ingestion API OLJ/WhiteBeard).
- Ingestion : `python -m scripts.ingest_cli reindex-all --publication 17 --content-type 4 --seed-file data/olj_seed_ids.json`.
- `v4/data/olj_seed_ids.json` : ~150 IDs CMS de recettes (fusionnés avec la
  pagination API — pagination à confirmer avec Joseph/WhiteBeard, cf. go-live).
- Fichiers JSON racine (`Data_base_2.json`, `data_base_OLJ_enriched.json`) :
  **legacy** ; `Data_base_2.json` sert encore de *last-resort fallback* (à retirer
  une fois la couverture Postgres complète).
- Schéma DB (`v4/backend/app/db/models.py`) : `Article`, `ArticleSection`,
  `Person`, `Category`, `Keyword`, `Ingredient`, `ArticleIngredient`, `Chunk`.

---

## 5. Contrat d'API (extrait)

- `POST /api/chat` → `{ answer_sentences[], recipe_card?, recipe_card_secondary?, chef_card?, follow_up, confidence, sources[] }`
- `GET /healthz` (Railway health check) · `GET /api/admin/stats` · `/admin` · `/dashboard` · `/embed` (widget)
- Webhook CMS : `POST /api/webhook` (secret `WEBHOOK_SECRET`).
- CORS : domaines `lorientlejour.com` + URL Railway (jamais `*` en prod).

---

## 6. Règles de comportement (contrat produit)

1. **Faithfulness** : chaque phrase cite ≥1 `chunk_id` du contexte (validateur post-LLM).
2. **Answerability / abstention** (à durcir en Phase 1, *en code* pas en prompt) :
   si le meilleur score rerank < seuil **et** que titre/ingrédients ne matchent
   pas la demande ⇒ abstention franche, **aucune carte**, `confidence ≤ 0.35`.
3. **Discipline des cartes** : `recipe_card` seulement si l'article a une vraie
   section recette **et** matche le plat/ingrédient demandé. Une seule carte
   cliquable. L'ingrédient demandé doit figurer dans la recette proposée.
4. **Mention ≠ recette** : un chunk qui *mentionne* un plat n'autorise pas à le
   présenter comme une recette.
5. **Ton** : guide chaleureux, FR soutenu accessible, vouvoiement, sans emoji ni
   jargon technique (« contexte », « extraits », « corpus », « chunk »… interdits
   côté utilisateur).
6. **Trafic article** : ne pas recopier la recette complète ; donner envie d'ouvrir
   la fiche OLJ (cartes avec `ingredients: []`, `steps: []`).

> Dette à résorber : le `SYSTEM_PROMPT` de `response_generator.py` empile **24
> règles** (souvent redondantes/contradictoires). Cible Phase 1 : ~6 règles, le
> reste passe en logique de code testée.

---

## 7. Marqueurs d'acceptation — le golden set

**Fichier :** [`v4/data/golden_eval_fr.json`](v4/data/golden_eval_fr.json) — 30
marqueurs, 8 catégories. C'est la définition opérationnelle de « tout marche ».

| Catégorie | Cas | Doit passer |
|---|---|---|
| `recipe_present` | 6 | maintenant |
| `chef_bio` | 3 | maintenant |
| `ingredient` | 4 | maintenant |
| `alias` | 3 | maintenant |
| `constraint` | 2 | Phase 1 |
| `coherence` | 4 | Phase 1 |
| `out_of_corpus` | 3 | **Phase 1 (critique)** |
| `anti_regression` | 5 | **Phase 1 (critique)** |

Le marqueur **n°1** est `anti-regression-manouche-recipe` : « recette manouche »
doit **retrouver l'article 1474718** (« Les manaïichs du Chouf de Salim Azzam »),
qui existe pourtant déjà dans l'index. Il **échoue sur le code actuel**
(translittération) et passera après le correctif d'alias. Variantes :
`anti-regression-manouche-accent`, `anti-regression-manakish`. Champs d'assertion
supportés par le runner : `expected_article_external_ids`,
`answer_must_contain`, `answer_any_contains`, `answer_must_not_contain`,
`answer_prefix`, `follow_up_must_contain`, `max_recipe_cards`, `require_recipe_card`,
`require_chef_card`.

### Exécuter les marqueurs

```bash
# Mode base locale (DB peuplée + clés API) — depuis v4/
cd v4 && set PYTHONPATH=backend
python scripts/run_rag_eval.py --golden data/golden_eval_fr.json --top-k 12

# Mode live (contre staging/prod, pas besoin de DB locale)
python scripts/run_rag_eval.py --golden data/golden_eval_fr.json \
  --base-url https://web-sahtein-19-04-staging.up.railway.app

# Validation structure seule (CI, sans LLM)
pytest v4/tests/test_golden_json.py
```

Sortie : JSON avec `n_passed/n_items` et `all_passed`. Code 0 si tout passe.

> **À faire** : renseigner les `expected_article_external_ids` réels (depuis
> `/admin`) pour les cas marqués `note` afin de durcir la couverture retrieval.

---

## 8. Définition de « FINI »

Le projet est **présentable / livrable** quand :

1. **Repo propre** : `sota/v4` = `main` ; branches `feature/ui-*` et `mvp`
   supprimées (mvp archivée en tag) ; app legacy `backend/` supprimée ; `Procfile`
   corrigé. → un seul code, une seule version.
2. **Comportement correct** : **100 %** des marqueurs `anti_regression` +
   `out_of_corpus` passent, et **≥ 90 %** du reste (plus de manouche→cailles).
3. **Non-régression garantie** : le golden set tourne en CI (au moins
   `test_golden_json.py` ; idéalement `run_rag_eval` live nightly).
4. **Périmètre assumé** (§1) : l'abstention est une qualité, pas une gêne.

Le reste (router Mistral, cooccurrence ingrédients/Epicure, navigation culinaire)
est **amélioration**, pas « finir ».

---

## 9. Plan d'exécution (phases)

- **P0 — Hygiène repo** (½ j) : promotion `sota/v4`→`main`, suppression branches
  mortes + legacy `backend/`, fix `Procfile`. *(non destructif sans validation)*
- **P1 — Correctif visible** (2-3 j) : answerability gating + card gating **en
  code**, prompt 24→~6 règles, séparation fait-sourcé / culture-générale. ⇒ fait
  passer les marqueurs critiques.
- **P2 — Éval** (2 j) : golden set (✅ fait, ce document), extension du runner
  (MRR, taux d'abstention, diversité article), CI.
- **P3 — Couverture & vocabulaire** : backfill complet WhiteBeard, canonicalisation
  ingrédients (alias en données auditées), décision éditoriale §1.
- **P4 — Modèle & navigation** : router Mistral/nano, cooccurrence ingrédients
  (NPMI), contrôles « même ingrédient autre registre » / substitutions.

Référence détaillée : [`v4/docs/rag-epicure-audit.md`](v4/docs/rag-epicure-audit.md),
[`v4/docs/go-live-sota-v4.md`](v4/docs/go-live-sota-v4.md).

---

## 10. Sécurité (état vérifié)

- ✅ **Aucun secret réel commité** dans l'historique GitHub (les `key.txt`/`mdp.txt`
  sont dans le workspace local `V3\`, hors repo) ⇒ la réécriture d'historique /
  rotation décrite dans `suppression-anciennes-versions.md` **n'est pas nécessaire**
  pour le repo.
- En prod : `APP_ENV=production`, `SAHTEN_ADMIN_API_TOKEN` (≥16 car.),
  `SAHTEN_CORS_ORIGINS` explicite (jamais `*`), secrets `OPENAI_API_KEY`,
  `OLJ_API_KEY`, `WEBHOOK_SECRET`, `COHERE_API_KEY`.
