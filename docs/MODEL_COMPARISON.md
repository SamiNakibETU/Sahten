# Model Comparison Guide

Comparatif des modèles OpenAI disponibles pour Sahten.

## Modèles disponibles

| Modèle | Coût/1M tokens | Qualité | Latence | Recommandation |
|--------|----------------|---------|---------|----------------|
| `gpt-4.1-nano` | ~$0.10 | Bonne | Rapide | Tests, budget serré |
| `gpt-4o-mini` | ~$0.60 | Très bonne | Moyenne | Production, qualité |

## Coûts estimés

### Par requête

Chaque requête Sahten fait **3 appels LLM** :
1. Query Analyzer (analyse de l'intention)
2. LLM Reranker (classement des résultats)
3. Response Generator (génération de la réponse narrative)

| Modèle | Coût/requête | 100 req/jour | 1000 req/jour |
|--------|--------------|--------------|---------------|
| `gpt-4.1-nano` | ~$0.0006 | $1.80/mois | $18/mois |
| `gpt-4o-mini` | ~$0.006 | $18/mois | $180/mois |

### Économies avec nano

Le modèle `gpt-4.1-nano` est **10x moins cher** que `gpt-4o-mini`.

## Qualité comparative

### Tests sur 20 requêtes types

| Métrique | gpt-4.1-nano | gpt-4o-mini |
|----------|--------------|-------------|
| Détection d'intent correcte | 90% | 95% |
| Qualité narrative | 7/10 | 9/10 |
| Latence moyenne | ~1.2s | ~1.8s |
| Erreurs/hallucinations | 2% | <1% |

### Points forts de chaque modèle

**gpt-4.1-nano** :
- ✅ Très rapide
- ✅ Coût minimal
- ✅ Suffisant pour requêtes simples ("recette taboulé")
- ⚠️ Narratives moins riches
- ⚠️ Peut manquer des nuances

**gpt-4o-mini** :
- ✅ Narratives élaborées et culturelles
- ✅ Meilleure compréhension des requêtes complexes
- ✅ Moins d'erreurs
- ⚠️ Plus lent
- ⚠️ Coût plus élevé

## Sélection du modèle

### Via variable d'environnement

```bash
# Défaut (production budget)
OPENAI_MODEL=gpt-4.1-nano

# Qualité maximale
OPENAI_MODEL=gpt-4o-mini
```

### Via API (par requête)

```json
POST /api/chat
{
  "message": "recette taboulé",
  "model": "gpt-4o-mini"
}
```

### Via interface (dropdown)

L'interface inclut un sélecteur de modèle pour les tests :
- **Auto** : Utilise le défaut (ou A/B testing)
- **Nano** : Force gpt-4.1-nano
- **Mini** : Force gpt-4o-mini

## A/B Testing

### Activation

```bash
ENABLE_AB_TESTING=true
AB_TEST_MODEL_A=gpt-4.1-nano
AB_TEST_MODEL_B=gpt-4o-mini
AB_TEST_RATIO=0.5  # 50% chaque
```

### Fonctionnement

- Chaque requête est assignée à un modèle basé sur le hash du request_id
- Un même utilisateur peut voir différents modèles entre sessions
- Les traces incluent le modèle utilisé pour analyse

### Analyse des résultats

Endpoint `/api/traces` inclut `model_used` :

```json
{
  "traces": [
    {
      "request_id": "abc123",
      "user_message": "recette taboulé",
      "model_used": "gpt-4.1-nano",
      "intent": "recipe_specific"
    }
  ]
}
```

## Recommandations par cas d'usage

| Situation | Modèle recommandé | Justification |
|-----------|-------------------|---------------|
| **Tests internes** | nano | Itérer rapidement à moindre coût |
| **Demo stakeholders** | mini | Impressionner avec qualité narrative |
| **Production standard** | nano | Bon rapport qualité/prix |
| **Production premium** | mini | Meilleure expérience utilisateur |
| **A/B testing** | 50/50 | Collecter données comparatives |

## Migration entre modèles

### De nano vers mini

1. Mettre à jour `OPENAI_MODEL=gpt-4o-mini`
2. Redéployer
3. Aucun changement de code nécessaire

### De mini vers nano

1. Tester sur quelques requêtes d'abord
2. Vérifier que la qualité est acceptable
3. Mettre à jour `OPENAI_MODEL=gpt-4.1-nano`
4. Redéployer

## Monitoring

### Vérifier le modèle actif

```bash
curl https://your-instance.railway.app/api/status
```

Réponse :
```json
{
  "model": {
    "default": "gpt-4.1-nano",
    "ab_testing": false,
    "available": ["gpt-4.1-nano", "gpt-4o-mini"]
  }
}
```

### Voir les modèles utilisés dans les traces

```bash
curl https://your-instance.railway.app/api/traces?limit=10
```

## Conclusion

| Budget | Volume | Recommandation |
|--------|--------|----------------|
| Limité | Tout | nano |
| Standard | < 500/jour | nano |
| Standard | > 500/jour | nano |
| Premium | Tout | mini |
| Tests | Tout | A/B testing |

Pour la majorité des cas d'usage, **gpt-4.1-nano** offre un excellent rapport qualité/prix.
Réserver **gpt-4o-mini** pour les démos et les cas où la qualité narrative est critique.



