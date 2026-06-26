# Rapport — graphe de cooccurrence ingrédients (Epicure P2)

Généré par `scripts/build_cooccurrence.py` → `data/ingredient_cooccurrence.json`.
Lu par `backend/app/rag/cooccurrence.py` (`related_ingredients`). **Diagnostic /
fondation — non branché au retrieval** (cf. `metadata-graph-sota.md`).

## Données
- 102 recettes du corpus, **76 ingrédients** retenus (≥ 4 recettes).
- Score = **NPMI** (cooccurrence normalisée) ; voisins triés décroissant.

## Le signal culinaire est présent
Exemples de voisinages produits (NPMI) :

| Ingrédient | Voisins top |
|---|---|
| aubergine | poivron, courgette, tomate, menthe, yaourt |
| basilic | pignon, courgette, pain *(pesto)* |
| bouillon | parmesan, champignon, échalote, vin *(risotto)* |
| boulghour | mélange, menthe, oignon *(taboulé/kebbé)* |
| bicarbonate | gingembre, vanille, levure *(pâtisserie)* |

→ Ces relations sont **culinairement justes** : la géométrie de cooccurrence
capture des familles réelles (méditerranéen, pâtisserie, mijoté).

## Limites (honnêtes)
- **Bruit d'extraction** : la source ici est le corpus *legacy* (lignes de recette
  brutes) → extraction heuristique (quantités/unités retirées). On voit des
  artefacts (`cuilleree`, `baton`, `noir`, `d'oignon`, `ufs`). Le signal domine
  mais n'est pas propre.
- **Petit corpus** (102 recettes) → NPMI bruité sur les paires rares.

## Recommandation
1. **Production** : régénérer depuis Postgres `ArticleIngredient` (slugs canoniques
   propres) au lieu du legacy → graphe nettement plus net. Le builder accepte déjà
   un JSON `{title, ingredient_slugs:[...]}` (mode privilégié).
2. **Évaluer** la qualité voisin-par-voisin sur 20 ingrédients libanais clés
   (concombre, menthe, persil, boulgour, pois chiche, aubergine, citron, yaourt…)
   **avant** toute intégration runtime.
3. Puis intégrer comme **signal de navigation** (voir `metadata-graph-sota.md`),
   jamais comme expansion aveugle de requête (risque de dilution déjà constaté).
