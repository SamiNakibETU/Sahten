# Chantier #2 — Métadonnées & graphe culinaire (design SOTA)

> Objectif : mieux exploiter les **métadonnées** (chef, catégorie, mots-clés,
> ingrédients) et un **graphe** ingrédient-recette, au service du retrieval, de la
> navigation et de la **rédaction**. Suit la prudence de l'audit Epicure :
> mesurer/fonder avant d'intégrer.

## État (ce qui est fait)
- **Fondation graphe** : `scripts/build_cooccurrence.py` → `data/ingredient_cooccurrence.json` ;
  helper `rag/cooccurrence.py` (`related_ingredients`) + tests. Voir
  `cooccurrence-report.md`. (Non branché au runtime.)
- **Métadonnées présentes en base** : `Article`, `Person` (chef), `Category`,
  `Keyword`, `Ingredient`, `ArticleIngredient`, `Chunk` (`db/models.py`). Le
  retrieval filtre déjà par chef/ingrédient/catégorie/keyword (`retriever.py`).

## Axe A — Comprendre les métadonnées (retrieval)
1. **Régénérer le graphe depuis `ArticleIngredient`** (slugs propres) — qualité
   prod (le legacy sert de prototype).
2. **Signaux séparés et mesurables** (Epicure §2) : pour chaque requête, logguer
   d'où vient le bon article (lexical / vectoriel / filtre ingrédient / chef /
   widening / rerank). L'endpoint `/api/admin/diagnose-retrieval` (déjà enrichi
   avec les ids) est la base ; ajouter la contribution par étape.
3. **Filtres métadonnées plus fins** : exploiter `Category`/`Keyword` pour les
   contraintes de registre (salé/sucré, entrée/plat/dessert) au lieu de le porter
   uniquement par le prompt.

## Axe B — Navigation culinaire (graphe)
Brancher `cooccurrence.related_ingredients` **après évaluation**, pour des intents
explicites seulement (pas d'expansion aveugle) :
- « avec quoi cuisiner X » → voisins NPMI de X ;
- « même ingrédient, autre registre » → exclure articles vus + diversifier via
  catégorie/keyword ;
- « substitution » → voisins de même rôle (à valider : nécessite une géométrie
  fiable, cf. Epicure P3).
Garde-fou : une suggestion **calculée** (graphe) doit être étiquetée comme telle,
distincte d'un **fait sourcé** OLJ (le grounding phrase-par-phrase reste pour les faits).

## Axe C — Rédaction (adapter au modèle)
1. **Contexte enrichi** : ajouter au `_format_context` de `response_generator` les
   métadonnées structurées de l'article (chef, catégorie, mots-clés) — le modèle
   écrit mieux et plus juste quand il sait « plat de chef X, registre Y ».
2. **Router modèle** (cf. benchmark) : prose FR → Mistral Large 3 ; structuré
   (query understanding / rerank) → nano. La rédaction gagne en finesse sans coût.
3. **Prompt allégé** : déplacer les décisions binaires (carte ou pas, abstention)
   du prompt vers le code (déjà commencé : gating carte/plat) → le prompt ne garde
   que le ton éditorial.

## Séquence recommandée (sans casser l'existant)
1. Régénérer le graphe via `ArticleIngredient` + évaluer (Axe A.1, B).
2. Logs de contribution par étape (Axe A.2) — observabilité d'abord.
3. Enrichir le contexte de génération avec métadonnées (Axe C.1) — gain rédaction
   immédiat, faible risque.
4. Brancher la navigation graphe (Axe B) derrière des intents explicites.
5. Router modèle (Axe C.2) en dernier, une fois le comportement stable.

## Anti-objectif
Pas d'intégration graphe « magique » dans le retrieval général : l'audit montre
que ça ajoute du risque sans gain mesuré tant que vocabulaire + géométrie ne sont
pas stabilisés. Mesurer, puis intégrer par intent.
