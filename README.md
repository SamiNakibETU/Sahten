# Sahten

Dépôt **Python (FastAPI)** + **frontend statique** : assistant recettes pour *L’Orient-Le Jour*, appuyé sur un index de fiches « Liban à Table ».  
Le backend enchaîne analyse de requête, recherche dans le corpus, éventuellement rerank / LLM, et renvoie du HTML (cartes recette avec liens OLJ). Un webhook permet d’ingérer les publications CMS.

## Arborescence

| Élément | Rôle |
|--------|------|
| `backend/` | API (`main.py`, préfixe `/api`), bot, RAG, webhook, schémas Pydantic, tests. |
| `frontend/` | `index.html` (page par défaut `/`), `widget.html` (`/embed`), `css/`, `js/`, `assets/`. |
| `data/` | Données canoniques servies au retriever (ex. `olj_canonical.json`). |
| `railway.toml`, `Procfile` | Déploiement type Railway. |

## Exécution locale

```bash
cd backend
pip install -r requirements.txt
# Copier .env.example vers .env et renseigner au minimum OPENAI_API_KEY
python main.py
```

URL : http://localhost:8000/ (démo). API : `/api/chat`, `/api/health`, etc.

## Configuration

Voir `backend/.env.example` (clé OpenAI, CORS, Redis optionnel pour traces/analytics, secret webhook).

## Licence

Usage interne L’Orient-Le Jour.
