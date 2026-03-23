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

## Déploiement (ex. Railway) & intégration OLJ

- **Le déploiement est OK** si `https://…/api/health` répond et le widget charge. Le texte « mode test local » sur la page d’accueil était uniquement une **phrase statique** dans `frontend/index.html` (remplacée par un libellé neutre).
- **Pages utiles** (même origine que l’app déployée) :
  - `/` — démo complète
  - `/embed` — page widget (référence intégration)
  - `/dashboard` — **analytics** (métriques + traces si Redis configuré)
  - `/admin` — liste recettes canoniques
- **Analytics** : `GET /api/analytics` + dashboard nécessitent **Upstash Redis en REST** (`UPSTASH_REDIS_REST_URL` = URL `https://…upstash.io`, `UPSTASH_REDIS_REST_TOKEN`). La variable Railway **`REDIS_URL` en `redis://` ne suffit pas** pour ce code : créer une base gratuite sur [Upstash](https://console.upstash.com) et copier REST URL + token. Sans ça, le chat marche, traces/stats non (plus d’erreur 500 sur `/api/traces`).
- **Production** : `DEBUG=false`, `OPENAI_API_KEY` (ou fournisseur configuré), `WEBHOOK_SECRET` si le CMS appelle le webhook.
- **CORS** : les domaines OLJ sont déjà dans `app/core/config.py` ; ajoute l’URL exacte de ton service Railway si besoin (ex. `https://sahten.up.railway.app`).
- **Intégration site OLJ** : charger le script / iframe depuis ton domaine d’hébergement Sahten ; le widget utilise en général `window.location.origin + "/api"` pour l’API (même host). Si l’API est sur un autre domaine, configurer l’URL API côté init du widget et **autoriser le domaine du site OLJ** dans `cors_origins`.

## Licence

Usage interne L’Orient-Le Jour.
