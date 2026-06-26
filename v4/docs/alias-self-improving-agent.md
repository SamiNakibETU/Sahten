# Agent auto-améliorant des alias (conception)

> Boucle qui **apprend en continu** les graphies (translittérations) manquantes,
> les valide contre l'API, et enrichit `data/aliases_dishes.json` /
> `data/aliases_ingredients.json` — sans régression. Conçu, pas encore déployé.

## Pourquoi
Le bug manouche (graphie `manouche` ≠ index `manaïichs`) ne se voit pas dans le
code : il se voit dans les **requêtes réelles qui échouent**. La source de vérité
de « ce qu'il faut aliaser » est donc la **production**, pas la spéculation.

## Signal (déjà disponible en prod)
`backend/app/analytics_store.py` :
- `record_feedback_rating(...)` → empile les avis négatifs dans la liste Redis
  `sahten:feedback:negative_reasons` (les **requêtes/contextes** que l'utilisateur
  a jugés « Cette réponse vous a-t-elle aidé ? **Non** »).
- `record_chat_trace(...)` → traces ; compteur `sahten:metrics:recipe_not_found_count`.
- `get_traces(limit)` / `get_feedback_stats()` exposent ces données.

→ Les **candidats alias** = requêtes ayant déclenché « pas trouvé » OU un « Non ».

## Boucle (5 étapes)
1. **Collecte** : lire les requêtes en échec (traces `recipe_not_found` +
   `negative_reasons`). Normaliser, dédupliquer, garder les termes « plat-like »
   (≠ humeur/ingrédient déjà couvert).
2. **Hypothèse (LLM)** : pour chaque terme orphelin, demander à un petit modèle
   (gpt-4.1-nano / mistral-small — voir benchmark) la **forme canonique probable**
   + l'**article cible candidat** (via une recherche sémantique : on requête la
   forme normalisée et on regarde si un article cohérent existe déjà dans l'index).
3. **Validation (oracle = API)** : réutiliser `scripts/validate_aliases.py` —
   le terme orphelin + l'injection proposée doivent faire remonter l'article
   cible au **rang ≤ 3**, sans diluer. N'accepter que si mesuré OK.
4. **Écriture** : ajouter l'alias validé au JSON avec **provenance** (`source:
   "auto-mined"`, `date`, `evidence: [{query, rank}]`). Si confiance < seuil
   (cible ambiguë, plusieurs articles plausibles) → file de **revue humaine**
   (`data/aliases_review_queue.json`) au lieu d'auto-merge.
5. **Garde-fou** : exécuter le golden set (`scripts/run_rag_eval.py`) +
   `pytest tests/test_query_aliases.py` après chaque lot. **Rollback** si une
   seule régression. Commit séparé, jamais auto-poussé sur prod sans CI verte.

## Cadence & sécurité
- Job hebdomadaire (cron/arq) lisant les N dernières requêtes en échec.
- Lot borné (ex. 20 candidats/run) pour limiter les appels API.
- Idempotent : ne re-propose pas un alias déjà présent ou déjà rejeté.
- Audit : chaque entrée JSON trace d'où elle vient (manuel vs auto-miné).

## Anti-objectif
Ne PAS laisser le LLM écrire directement dans le code/retrieval. Le LLM
**propose** ; l'**API valide** ; un **humain arbitre** les cas ambigus. C'est ce
qui empêche la dérive (le piège des heuristiques accumulées dans `pipeline.py`).

## Fichiers
- `scripts/mine_missing_aliases.py` — squelette du mineur (ci-joint).
- `data/aliases_dishes.json` / `aliases_ingredients.json` — cible d'écriture.
- `data/aliases_review_queue.json` — file de revue humaine (créée au besoin).
