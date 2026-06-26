# Rapport de validation — alias / translittérations

Mission « rendre tout le corpus retrouvable quelle que soit la graphie ».
Date : 2026-06-26. Branche `sota/v4`.

## Résultat

- **Bug racine confirmé** : l'article 1474718 « Les manaïichs du Chouf de Salim
  Azzam » existe et est indexé, mais seules les graphies proches de l'index le
  retrouvent. Idem pour d'autres plats translittérés.
- **Correctif livré** : canonicalisation des graphies utilisateur → forme
  **indexée**, en données auditables (`data/aliases_dishes.json`) + code
  (`pipeline._canonicalize_dish_aliases`, branché dans `_expand_query_with_aliases`).
- **80 règles** de canonicalisation chargées ; **66 tests** verts (dont 6 nouveaux).

## Méthode : canonicalize_replace (pas append)

L'empirie (sondes API live) a tranché : **ajouter** la forme indexée ne suffit pas,
il faut **remplacer** la graphie utilisateur.

| Requête | append (`variante + indexée`) | remplacement (`→ indexée`) |
|---|---|---|
| `tabbouleh` | ❌ (`tabbouleh taboulé` échoue) | ✅ (`taboulé` rang 1) |
| `kofta` | ❌ (`kofta kafta` échoue) | ✅ (`kafta` rang 1) |
| `molokheya` | ❌ | ✅ (`mouloukhiyé`) |
| `manouche` | ✅ (`manouche manaiche` rang 2) | ✅ (`manaiche`) |

Raison : la graphie « étrangère » (tabbouleh, kofta…) tire l'embedding/lexical
vers d'autres articles ; tant qu'elle reste dans la requête, elle pollue. La
remplacer élimine le bruit. C'est le modèle déjà présent dans
`_base2_canonicalize_aliases`, généralisé au retrieval principal.

## Données empiriques (API live, partielles — API instable sous sondage rapide)

| Plat | cible | graphies OK seules | graphies réparées par canonicalisation |
|---|---|---|---|
| manouche | 1474718 | manaiche | manouche, manouché, man'ouché, manakish |
| taboulé | 1469385 | taboule, taboulah | tabboule, tabbouleh, tabbouli |
| mouloukhiyé | 1469013 | molokhia, mloukhiyeh | mouloukhia, moloukhieh, mloukhieh, molokheya |
| kafta | 1350035 | kafta, kafta batata | kefta, kofta, köfte |

> ⚠️ L'API `/api/chat` s'est révélée **instable sous sondage rapide** (réponses
> parfois vides, rang non-déterministe). La validation par boîte noire n'est donc
> pas fiable à grande échelle.

## Oracle propre pour la suite : `/api/admin/diagnose-retrieval`

L'endpoint admin **existe déjà** et a été **enrichi** pour renvoyer les
`sample_article_ids` par étape de retrieval. Comme il passe par
`_retrieval_fallback_queries → _expand_query_with_aliases`, il reflète
**automatiquement** la canonicalisation. C'est l'oracle déterministe (sans
génération LLM) à utiliser après déploiement :

```
GET /api/admin/diagnose-retrieval?q=manouche   (en-tête X-Sahten-Admin-Token)
```

## Couverture

- **Validé empiriquement** (live) : manouche, taboulé, mouloukhiyé, kafta.
- **Déterministe depuis titres corpus** (rang live à confirmer via l'endpoint
  ci-dessus) : maamoul, kebbé, fatteh, su-beureg, ourfa-kebab, ouayamat,
  samboussek, fattouche, houmous.
- **Ingrédients** : `data/aliases_ingredients.json` (zaatar, sumac, tahini,
  labneh, freekeh, boulgour, kishk, mélasse de grenade, fleur d'oranger…) —
  curé, à brancher dans `_INGREDIENT_ARABIC_ALIASES` (loader ingrédient = TODO).

## Gaps / prochaines étapes

1. **Déployer** `sota/v4` sur staging, puis valider via `diagnose-retrieval`
   (rang ≤ 3 pour chaque graphie du panel). Mettre à jour ce tableau.
2. ✅ **Loader ingrédients** : FAIT — `aliases_ingredients.json` chargé par le même
   canoniseur (`_build_canon_rules`), cible = `inject[0]` (forme dans les chunks).
   136 règles au total (plats + ingrédients). `pois-chiche` exclu (graphies
   hommos/houmous gérées comme PLAT). za'atar→zaatar, labneh→labné, etc.
3. **Élargir** le panel à 40+ plats / 40+ ingrédients via l'agent auto-améliorant
   (`docs/alias-self-improving-agent.md` + `scripts/mine_missing_aliases.py`).
4. **Aliases côté article** (indexation) : à terme, stocker les translittérations
   comme champ searchable des articles (plus robuste que la réécriture de requête).
