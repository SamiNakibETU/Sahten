# Prompt — Agent de recherche exhaustive « alias & translittérations » (Sahteïn)

> À copier-coller à un agent **Claude Opus 4.8** (autonome, avec accès shell +
> web + à l'API live). Objectif : rendre **tout** le corpus retrouvable quelle
> que soit la graphie, de façon **déterministe et auditable**, puis poser les
> bases d'un agent qui améliore les alias en continu.

---

## CONTEXTE

Sahteïn est un RAG culinaire de L'Orient-Le Jour (recettes de chefs « Liban à
Table »). Stack : Postgres + pgvector + tsvector → RRF → rerank Cohere →
génération groundée. Le repo est sur la branche `sota/v4`. L'app vivante est dans
`v4/`.

**Bug confirmé (la raison de cette mission).** La recette de manouche EXISTE et
est indexée : article **1474718 « Les manaïichs du Chouf de Salim Azzam »**.
Pourtant, seule la graphie quasi-identique à l'index la retrouve :

| Graphie tapée par l'utilisateur | Retrouve l'article 1474718 ? |
|---|---|
| `manaiche` | ✅ oui |
| `manouche` | ❌ « pas trouvé » |
| `manouché`, `man'ouché` | ❌ mauvais article (pâtes au kishk 1487133) |
| `manakish` | ❌ carte hors-sujet (Lahm bi aajine) |

C'est un **bug de translittération au retrieval**, pas un trou de corpus ni un
problème de modèle. Il touche potentiellement TOUS les plats et ingrédients
translittérés de l'arabe (houmous/hommos, taboulé/tabbouleh, mouloukhié, kebbé,
fattouche, kibbeh, sfiha, etc.).

**Découverte clé sur la méthode (à respecter absolument).** Injecter *beaucoup*
de variantes dilue l'embedding et fait ÉCHOUER la recherche :
- `recette manouche manaiche` → ✅ trouvé (rang 2, avec carte)
- `recette manouche manaiche manaiches` → ✅ trouvé (rang 1)
- `manouche manaiche manakich galette zaatar` → ❌ (dilué par des mots hors-sujet)
- `manakish manaiche` → ❌ (manakish tire vers d'autres articles)

→ La bonne stratégie est de **canoniser vers la (ou les) forme(s) réellement
indexée(s)** d'un plat, en ajoutant un **minimum** de variantes ciblées, PAS un
dictionnaire entier. Chaque ajout doit être **validé empiriquement** contre l'API.

## INFRASTRUCTURE EXISTANTE À ÉTENDRE (ne pas réinventer)

- `v4/backend/app/rag/pipeline.py`
  - `_ALIAS_GROUPS` : tuples de variantes de **plats**. `_expand_query_with_aliases(q)`
    ajoute à la requête toutes les variantes d'un groupe si l'une est présente.
    *(Actuellement : houmous, moghrabieh seulement.)*
  - `_INGREDIENT_ARABIC_ALIASES` : alias arabes par ingrédient.
  - `_BASE2_RECIPE_ALIASES` + `_base2_canonicalize_aliases` : REMPLACE les alias
    par la forme canonique (meilleur pour l'embedding — modèle à suivre).
- `v4/backend/app/rag/ingredient_match.py`
  - `INGREDIENT_SLUG_ALIASES`, `_INGREDIENT_SLUG_CANONICAL`, `_KNOWN_ING_WORDS`.
- `v4/backend/app/llm/query_understanding.py` : le LLM produit `rewritten_query` +
  `ingredient_slugs` ; on peut lui demander d'émettre les variantes.
- Tests : `v4/tests/test_query_aliases.py`, `v4/tests/test_ingredient_match.py`.

## ACCÈS / OUTILS

- **API live (vérité terrain)** : `POST https://web-sahtein-19-04-staging.up.railway.app/api/chat`
  body `{"query": "...", "session_id": "alias-research-<n>"}` → réponse avec
  `answer_sentences`, `recipe_card`, `sources[].article_external_id` & `article_title`.
  C'est l'oracle : une variante « marche » si l'article cible apparaît dans
  `sources` (idéalement rang 1-3) et/ou produit la bonne `recipe_card`.
- **Liste du corpus** : `/admin` (liste recettes) et `GET /api/admin/stats` —
  peuvent exiger l'en-tête `X-Sahten-Admin-Token` (demander le token, sinon
  reconstituer le corpus en sondant `/api/chat` par chef/plat).
- Données locales d'appoint (historique git) : `git show HEAD:data_base_OLJ_enriched.json`
  contient des titres/chefs scrappés (utile pour énumérer les plats, ex. le titre
  exact « Les manaïichs du Chouf de Salim Azzam »).
- Shell Python + `urllib`/`requests` pour automatiser les sondes API.

## MISSION (déterministe + grounded + validée)

### Étape 1 — Énumérer le corpus réel
Lister **tous** les plats, chefs et ingrédients réellement indexés (titres
d'articles, noms de chefs, `ingredients_list`). Pour chaque entité, noter la/les
**forme(s) de surface exacte(s) telle(s) qu'indexée(s)** (ex. plat = « manaïichs »,
chef = « Salim Azzam »).

### Étape 2 — Construire l'espace des graphies utilisateur
Pour chaque plat/ingrédient, énumérer **toutes** les graphies qu'un utilisateur
francophone/anglophone taperait : translittération arabe→FR et arabe→EN, avec/sans
accents, apostrophes (`'` `'` `` ` ``), variantes ch/sh/š, ou/u/oo, k/q/c,
e/é/è, singulier/pluriel, fautes fréquentes, et variantes régionales
(libanaise/syrienne/arménienne). T'appuyer sur des sources : Wikipédia FR/EN/AR,
guides de cuisine levantine, et ta connaissance linguistique.

### Étape 3 — Mapper et VALIDER contre l'API
Pour chaque entité, déterminer le **jeu MINIMAL de variantes à injecter** qui fait
remonter l'article cible au rang 1-3 **sans diluer** (cf. découverte clé).
Procédure par entité :
1. tester chaque graphie utilisateur seule → noter found/rank ;
2. pour les graphies qui échouent, trouver la plus petite expansion
   `<graphie> + <forme(s) indexée(s)>` qui réussit ;
3. enregistrer la forme canonique = la forme indexée qui maximise le rang.
Tout est **mesuré**, pas supposé.

### Étape 4 — Produire les livrables
1. **`v4/data/aliases_dishes.json`** et **`v4/data/aliases_ingredients.json`**
   (alias comme **donnée auditable**, pas en dur — recommandation Epicure P1) :
   ```json
   {
     "canonical": "manouche",
     "indexed_forms": ["manaïichs", "manaiche"],
     "user_variants": ["manouche","manouché","man'ouché","manakish","manaeesh"],
     "inject": ["manaiche","manaiches"],
     "validation": [{"query":"manouche","found":false},{"query":"manouche manaiche","found":true,"rank":2}],
     "source_article_ids": [1474718]
   }
   ```
2. Un **patch** étendant `_ALIAS_GROUPS` / `INGREDIENT_SLUG_ALIASES` à partir de
   ces JSON (ou, mieux, un loader qui lit les JSON au démarrage → aliases =
   donnée, plus de hardcode).
3. **Tests** dans `test_query_aliases.py` / `test_ingredient_match.py` couvrant
   manouche + au moins 20 plats et 20 ingrédients translittérés.
4. Un **rapport** `v4/docs/alias-validation-report.md` : tableau graphie → found/rank
   avant/après, et la liste des plats encore non couverts (gaps réels de corpus).

### Contraintes
- Ne JAMAIS dégrader un cas qui marche déjà (houmous, taboulé, Salim Azzam) :
  re-tester après chaque ajout.
- Préférer la **canonicalisation** (remplacer par la forme indexée) à l'ajout
  massif, pour ne pas diluer l'embedding.
- Respecter le style du code existant (cf. fonctions ci-dessus).
- Idempotent, déterministe, ré-exécutable.

## PARTIE 2 — Agent auto-améliorant (concevoir, pour plus tard)

Concevoir (pas forcément déployer) une boucle qui **apprend les alias manquants
en continu** :
1. **Signal** : logguer les requêtes qui s'abstiennent (« pas trouvé ») ou
   reçoivent un « Cette réponse vous a-t-elle aidé ? **Non** » (voir
   `v4/backend/app/analytics_store.py`).
2. **Hypothèse** : un LLM propose, pour chaque terme orphelin, la forme canonique
   probable + l'article cible candidat (via recherche sémantique).
3. **Validation** : tester la proposition contre l'API (même oracle qu'en Étape 3).
   N'accepter que si l'article remonte au rang 1-3 sans régression.
4. **Écriture** : ajouter l'alias validé au JSON (avec provenance + date), ou le
   mettre en file de revue humaine si confiance < seuil.
5. **Garde-fou** : exécuter le golden set (`v4/data/golden_eval_fr.json`) après
   chaque ajout ; rollback si une régression.
Livrable : un doc d'architecture `v4/docs/alias-self-improving-agent.md` +
un script squelette `v4/scripts/mine_missing_aliases.py`.

## CRITÈRE DE SUCCÈS
- « manouche », « manouché », « man'ouché », « manakish » retrouvent tous
  l'article 1474718 (rang ≤ 3) via l'API.
- ≥ 95 % d'un panel de 40+ plats et 40+ ingrédients translittérés retrouvent le
  bon article, **mesuré** dans le rapport de validation.
- Aucune régression sur le golden set ni sur les cas déjà fonctionnels.
- Les alias vivent comme **donnée auditable**, pas en dur.
