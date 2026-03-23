# Grille de scoring qualité conversationnelle — Voix Sahten

Objectif : mesurer la qualité des réponses alternatives (plats non présents dans OLJ) selon la voix cible : **court, oral, pas magazine**.

---

## Critères et pondération

| Critère | Poids | Description | Pass |
|---------|-------|-------------|------|
| **Brevity** | 1 | Longueur du bloc narrative (hook + context + teaser) < 450 caractères | ✓ si ≤ 450 |
| **Mentions query** | 1 | Le plat demandé est explicitement cité (ex. "fajitas", "boeuf bourguignon") | ✓ si présent |
| **Mentions recipe** | 1 | Le titre de la recette proposée est cité | ✓ si présent |
| **Mentions ingredient** | 1 | Au moins un ingrédient commun prouvé est cité (quand preuve existe) | ✓ si présent |
| **No banned phrases** | 2 | Aucune phrase vague ou magazine détectée | ✓ si absent |

**Score max** : 6 points. **Seuil de succès** : ≥ 5 pour un cas.

---

## Phrases bannies (style magazine / vague)

- « Une recette libanaise qui partage au moins un ingrédient avec ce que tu cherchais »
- « expérience gustative similaire »
- « savoureux et convivial » (sans détail concret)
- « incarne cette simplicité » / « incarne cette richesse »
- « raviver » (souvenirs)

---

## 10 cas cibles

| ID | Requête | Type attendu |
|----|---------|--------------|
| fajitas | recette fajitas | not_found_with_alternative |
| pizza | recette pizza | not_found_with_alternative |
| ramen | recette ramen | not_found_with_alternative |
| boeuf_bourguignon | recette boeuf bourguignon | not_found_with_alternative |
| tacos | recette tacos | not_found_with_alternative |
| curry | recette curry | not_found_with_alternative |
| pad_thai | recette pad thai | not_found_with_alternative |
| burger | recette burger | not_found_with_alternative |
| sushi | recette sushi | not_found_with_alternative |
| croque_monsieur | recette croque-monsieur | not_found_with_alternative |

---

## Exemples de bon / mauvais ton

### ❌ Mauvais (magazine, long)

> Bienvenue à la table de L'Orient-Le Jour. Je suis désolé, mais je n'ai pas cette recette dans mes carnets. Mais pour me faire pardonner je peux te proposer une recette libanaise qui incarne cette simplicité et cette richesse de la cuisine libanaise, où l'on remplit des légumes de saveurs. Cette recette offre une expérience gustative similaire aux fajitas, en utilisant des légumes frais et des épices.

### ✓ Bon (court, oral)

> Je suis désolé, mais je n'ai pas cette recette dans mes carnets. Mais pour me faire pardonner je peux te proposer les légumes farcis au quinoa de Joanna Kassem : on garde le poulet et les légumes des fajitas, mais façon Liban. Découvre sur L'Orient-Le Jour !

---

## Exécution des tests

```bash
cd backend
pytest tests/test_voix_sahten.py -v
```

Avec clé API (pour LLM) :
```bash
OPENAI_API_KEY=sk-... pytest tests/test_voix_sahten.py -v
```

Sans clé : les tests utilisent le fallback déterministe (validations structurelles uniquement).
