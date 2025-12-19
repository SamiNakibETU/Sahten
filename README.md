# Sahten - Production Deployment (Railway)

> Lebanese Culinary Assistant by L'Orient-Le Jour

This is the **production-ready** version of Sahten, configured for Railway deployment with Upstash Redis logging.

---

## 📁 Structure

```
Sahten/
├── backend/
│   ├── app/              # FastAPI application
│   ├── main.py           # Entry point
│   └── requirements.txt  # Backend dependencies
├── frontend/
│   ├── css/sahten.css
│   ├── js/sahten.js
│   └── index.html
├── data/
│   └── olj_canonical.json
├── data_base_OLJ_enriched.json
├── Data_base_2.json
├── requirements.txt      # Root dependencies (for Railway)
├── Procfile              # Railway start command
└── railway.toml          # Railway configuration
```

---

## 🚀 Déploiement sur Railway

### Étape 1 : Créer un compte Upstash (pour les traces)

1. Aller sur [upstash.com](https://upstash.com)
2. Créer un compte gratuit
3. Créer une base **Redis**
4. Copier les credentials :
   - `UPSTASH_REDIS_REST_URL`
   - `UPSTASH_REDIS_REST_TOKEN`

### Étape 2 : Déployer sur Railway

1. Aller sur **[railway.app](https://railway.app)**
2. Cliquer **"Start a New Project"**
3. Choisir **"Deploy from GitHub repo"**
4. Sélectionner **SamiNakibETU/Sahten**
5. Railway détecte automatiquement Python et le Procfile

### Étape 3 : Configurer les variables d'environnement

Dans Railway → **Variables** :

| Variable                   | Valeur                  | Description                          |
| -------------------------- | ----------------------- | ------------------------------------ |
| `OPENAI_API_KEY`           | `sk-...`                | Clé API OpenAI (requise)             |
| `UPSTASH_REDIS_REST_URL`   | `https://...upstash.io` | URL Redis Upstash                    |
| `UPSTASH_REDIS_REST_TOKEN` | `AX...`                 | Token Redis Upstash                  |
| `PORT`                     | (auto)                  | Railway le configure automatiquement |

### Étape 4 : Générer un domaine

1. Aller dans **Settings** → **Networking**
2. Cliquer **"Generate Domain"**
3. Tu obtiens une URL comme : `sahten-production.up.railway.app`

---

## 🌐 Accéder à l'application

Après déploiement :

- **API Health** : `https://ton-app.up.railway.app/api/health`
- **API Status** : `https://ton-app.up.railway.app/api/status`
- **Chat API** : `https://ton-app.up.railway.app/api/chat`
- **Traces** : `https://ton-app.up.railway.app/api/traces`

### Frontend

Le frontend (`frontend/index.html`) peut être :

1. Ouvert localement (il appellera l'API Railway)
2. Hébergé sur GitHub Pages / Netlify / Vercel (statique)

Pour configurer l'URL de l'API dans le frontend, modifier `frontend/js/sahten.js` :

```javascript
const chat = new SahtenChat({
  apiBase: "https://ton-app.up.railway.app/api",
});
```

---

## 📊 Voir les conversations

### Via l'API

```
https://ton-app.up.railway.app/api/traces?limit=100
```

### Via les logs Railway

Dashboard Railway → **Deployments** → **View Logs**

---

## 🧪 Test local

```bash
cd Sahten/backend
pip install -r requirements.txt

# Configurer les variables
$env:OPENAI_API_KEY="sk-..."

# Lancer le serveur
python -m uvicorn main:app --reload --port 8000

# Ouvrir http://localhost:8000
```

---

## 📊 Endpoints API

| Méthode | Endpoint      | Description                  |
| ------- | ------------- | ---------------------------- |
| `GET`   | `/`           | Interface chat (frontend)    |
| `POST`  | `/api/chat`   | Envoyer un message           |
| `GET`   | `/api/health` | Health check                 |
| `GET`   | `/api/status` | Statut détaillé              |
| `GET`   | `/api/traces` | Historique des conversations |

---

## 💰 Coûts Railway

- **Free tier** : $5 de crédit gratuit/mois
- **Usage estimé Sahten** : ~$0-3/mois (selon trafic)
- Pas de limite de taille comme Vercel !

---

_Sahten ! 🇱🇧_
