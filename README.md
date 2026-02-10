# Sahten MVP

Chatbot culinaire libanais pour L'Orient-Le Jour avec sélection de modèle flexible et intégration CMS.

## Fonctionnalités MVP

- ✅ **Sélection de modèle** : nano (économique) ou mini (qualité)
- ✅ **A/B Testing** : Comparer les modèles automatiquement
- ✅ **Webhook CMS** : Intégration des nouvelles recettes
- ✅ **Enrichissement automatique** : Métadonnées ajoutées via LLM
- ✅ **Embeddings optionnels** : OFF par défaut, activable au besoin
- ✅ **Logging Upstash** : Traces persistantes pour analyse

## Démarrage rapide

### 1. Installation

```bash
cd Sahten_MVP/backend
python -m venv venv
.\venv\Scripts\Activate  # Windows
# ou: source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Configuration

Créer `.env` dans `backend/` :

```env
OPENAI_API_KEY=sk-...

# Modèle par défaut
OPENAI_MODEL=gpt-4.1-nano

# A/B Testing (optionnel)
ENABLE_AB_TESTING=false

# Embeddings (optionnel, OFF par défaut)
ENABLE_EMBEDDINGS=false

# Webhook CMS (optionnel)
WEBHOOK_SECRET=your-secret

# Upstash Redis (optionnel)
UPSTASH_REDIS_REST_URL=https://...
UPSTASH_REDIS_REST_TOKEN=...
```

### 3. Lancement

```bash
cd Sahten_MVP/backend
python main.py
```

Frontend : http://localhost:8000

## Endpoints API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/chat` | POST | Chat avec sélection de modèle |
| `/api/models` | GET | Liste des modèles disponibles |
| `/api/status` | GET | État du système |
| `/api/traces` | GET | Traces des conversations |
| `/api/webhook/recipe` | POST | Webhook pour CMS |
| `/api/health` | GET | Health check |

### Exemple : Chat avec modèle spécifique

```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "recette taboulé", "model": "gpt-4o-mini"}'
```

## Sélection de modèle

### Priorité

1. Paramètre `model` dans la requête API
2. A/B testing (si activé)
3. Variable d'environnement `OPENAI_MODEL`

### Modèles disponibles

| Modèle | Coût | Qualité | Usage |
|--------|------|---------|-------|
| `gpt-4.1-nano` | $0.10/1M tokens | Bonne | Tests, production budget |
| `gpt-4o-mini` | $0.60/1M tokens | Excellente | Démos, production premium |

### A/B Testing

```env
ENABLE_AB_TESTING=true
AB_TEST_MODEL_A=gpt-4.1-nano
AB_TEST_MODEL_B=gpt-4o-mini
AB_TEST_RATIO=0.5
```

## Intégration CMS

Voir [docs/CMS_INTEGRATION.md](docs/CMS_INTEGRATION.md) pour le guide complet.

```bash
curl -X POST "http://localhost:8000/api/webhook/recipe" \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: your-secret" \
  -d '{
    "id": "article_123",
    "title": "Fattouch",
    "url": "https://lorientlejour.com/article/123",
    "content": "La fattouch est une salade..."
  }'
```

## Enrichissement des recettes

### Script de re-enrichissement

```bash
cd Sahten_MVP/backend
python scripts/enrich_recipes.py --dry-run  # Prévisualisation
python scripts/enrich_recipes.py            # Exécution
```

### Champs enrichis

- `prep_time_minutes` : Temps de préparation
- `main_ingredients` : Ingrédients principaux
- `occasion` : quotidien, fête, ramadan, etc.
- `mood` : frais, réconfortant, léger, etc.
- `dietary` : végétarien, vegan, sans-gluten

## Structure du projet

```
Sahten_MVP/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes.py         # Endpoints principaux
│   │   │   ├── webhook.py        # Webhook CMS
│   │   │   └── response_composer.py
│   │   ├── core/
│   │   │   ├── config.py         # Configuration
│   │   │   └── model_selector.py # A/B testing
│   │   ├── enrichment/
│   │   │   └── enricher.py       # Enrichissement LLM
│   │   ├── llm/
│   │   ├── rag/
│   │   │   └── retriever.py      # TF-IDF + embeddings optionnels
│   │   └── bot.py
│   ├── scripts/
│   │   └── enrich_recipes.py
│   └── main.py
├── frontend/
│   ├── index.html                # Avec dropdown modèle
│   ├── css/sahten.css
│   └── js/sahten.js
├── data/
│   └── olj_canonical.json
└── docs/
    ├── CMS_INTEGRATION.md
    ├── MODEL_COMPARISON.md
    └── EMBEDDINGS_GUIDE.md
```

## Déploiement

### Railway (Backend)

1. Push sur GitHub
2. Connecter Railway au repo
3. Ajouter les variables d'environnement
4. Déployer

### Vercel (Frontend optionnel)

Le frontend peut être servi par Railway ou déployé séparément sur Vercel.

## Documentation

- [Guide d'intégration CMS](docs/CMS_INTEGRATION.md)
- [Comparaison des modèles](docs/MODEL_COMPARISON.md)
- [Guide des embeddings](docs/EMBEDDINGS_GUIDE.md)

## Variables d'environnement

| Variable | Requis | Défaut | Description |
|----------|--------|--------|-------------|
| `OPENAI_API_KEY` | ✅ | - | Clé API OpenAI |
| `OPENAI_MODEL` | ❌ | `gpt-4.1-nano` | Modèle par défaut |
| `ENABLE_AB_TESTING` | ❌ | `false` | Activer A/B testing |
| `ENABLE_EMBEDDINGS` | ❌ | `false` | Activer embeddings |
| `WEBHOOK_SECRET` | ❌ | - | Secret pour webhook CMS |
| `UPSTASH_REDIS_REST_URL` | ❌ | - | URL Upstash |
| `UPSTASH_REDIS_REST_TOKEN` | ❌ | - | Token Upstash |

## Licence

Projet interne L'Orient-Le Jour.
