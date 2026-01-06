# Embeddings Guide

Guide pour activer et configurer les embeddings sémantiques dans Sahten.

## État actuel : OFF par défaut

Les embeddings sont **désactivés** par défaut car :
- TF-IDF + LLM Reranker suffit pour 145 recettes
- Le LLM Reranker compense déjà le manque de compréhension sémantique
- Évite les coûts supplémentaires d'API embeddings

## Quand activer les embeddings ?

| Situation | Embeddings | Justification |
|-----------|------------|---------------|
| 145 recettes (actuel) | ❌ OFF | TF-IDF + Reranker suffit |
| 500 recettes | ⚠️ Évaluer | Tester la performance |
| 1000+ recettes | ✅ ON | Améliore le rappel |
| Requêtes sémantiques fréquentes | ✅ ON | "quelque chose de frais" |

### Requêtes bénéficiant des embeddings

```
✓ "quelque chose de réconfortant pour l'hiver"
✓ "un plat qui rappelle la montagne"
✓ "une recette festive"
✓ "quelque chose de léger pour l'été"
```

Ces requêtes fonctionnent *déjà* grâce au LLM Reranker, mais les embeddings améliorent le rappel initial.

## Comment activer

### 1. Variable d'environnement

```bash
ENABLE_EMBEDDINGS=true
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

### 2. Redéployer

Le retriever construira automatiquement l'index d'embeddings au démarrage.

## Coûts

| Opération | Coût |
|-----------|------|
| Indexation initiale (145 recettes) | ~$0.01 |
| Par nouvelle recette | ~$0.0001 |
| Par requête | ~$0.0001 |

### Impact sur le coût total

```
Sans embeddings:
  3 appels LLM/requête = ~$0.0006 (nano)

Avec embeddings:
  3 appels LLM + 1 embedding = ~$0.0007 (nano)
  
Différence: +16% par requête
```

## Architecture avec embeddings

```
┌─────────────────────────────────────────────────────────────┐
│                     QUERY                                   │
│                       │                                     │
│           ┌───────────┴───────────┐                         │
│           ▼                       ▼                         │
│    ┌─────────────┐         ┌─────────────┐                  │
│    │   TF-IDF    │         │  Embeddings │  ← OPTIONNEL     │
│    │  (lexical)  │         │ (semantic)  │                  │
│    └─────────────┘         └─────────────┘                  │
│           │                       │                         │
│           └───────────┬───────────┘                         │
│                       ▼                                     │
│              ┌─────────────┐                                │
│              │ RRF Fusion  │                                │
│              └─────────────┘                                │
│                       │                                     │
│                       ▼                                     │
│              ┌─────────────┐                                │
│              │ LLM Reranker│                                │
│              └─────────────┘                                │
│                       │                                     │
│                       ▼                                     │
│                   RESULTS                                   │
└─────────────────────────────────────────────────────────────┘
```

## Configuration avancée

```python
# config.py
class Settings:
    # Activer/désactiver
    enable_embeddings: bool = False
    
    # Provider: "openai" ou "local" (future)
    embedding_provider: str = "openai"
    
    # Modèle OpenAI
    embedding_model: str = "text-embedding-3-small"
    
    # Dimension (pour text-embedding-3-small)
    embedding_dimension: int = 1536
```

## Monitoring

### Vérifier l'état

```bash
curl https://your-instance.railway.app/api/status
```

Réponse :
```json
{
  "embeddings": {
    "enabled": true,
    "provider": "openai"
  }
}
```

### Logs au démarrage

Quand les embeddings sont activés :
```
INFO - Building embeddings for 145 documents...
INFO - Embeddings built: shape (145, 1536)
```

## Embeddings locaux (future)

Pour éviter les coûts API, une future version pourrait supporter les embeddings locaux :

```bash
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

**Avantages** :
- Gratuit après setup
- Pas de dépendance réseau

**Inconvénients** :
- Nécessite plus de mémoire
- Plus lent sur CPU

## Recommandation finale

| Phase | Recommandation |
|-------|----------------|
| MVP (maintenant) | ❌ OFF - TF-IDF + Reranker suffit |
| 500+ recettes | Tester avec embeddings |
| 1000+ recettes | ✅ ON recommandé |
| Requêtes sémantiques fréquentes | ✅ ON |

## Résumé

```
ENABLE_EMBEDDINGS=false  # MVP (145 recettes)
ENABLE_EMBEDDINGS=true   # Quand nécessaire (1000+ ou sémantique)
```

Le flag `ENABLE_EMBEDDINGS` permet d'activer les embeddings sans changer le code.
Le système fonctionne très bien sans embeddings grâce au LLM Reranker.



