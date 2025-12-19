# Sahten - Production Deployment

> Lebanese Culinary Assistant by L'Orient-Le Jour

This is the **production-ready** version of Sahten, configured for Vercel deployment with Upstash Redis logging.

---

## 📁 Structure

```
Sahten/
├── api/
│   └── index.py          # Vercel serverless entry point
├── backend/
│   └── app/              # FastAPI application
├── frontend/
│   ├── css/sahten.css
│   ├── js/sahten.js
│   └── index.html
├── data/
│   └── olj_canonical.json
├── data_base_OLJ_enriched.json
├── Data_base_2.json
├── requirements.txt
└── vercel.json
```

---

## 🚀 Déploiement sur Vercel

### Étape 1 : Créer un compte Upstash (pour les traces)

1. Aller sur [upstash.com](https://upstash.com)
2. Créer un compte gratuit
3. Créer une base **Redis**
4. Copier les credentials :
   - `UPSTASH_REDIS_REST_URL`
   - `UPSTASH_REDIS_REST_TOKEN`

### Étape 2 : Push sur GitHub

```bash
# Dans le dossier Sahten/
git init
git add .
git commit -m "Initial Sahten deployment"
git branch -M main
git remote add origin https://github.com/VOTRE_USERNAME/sahten.git
git push -u origin main
```

### Étape 3 : Configurer Vercel

1. Aller sur [vercel.com](https://vercel.com)
2. "New Project" → Importer depuis GitHub
3. Sélectionner le repo `sahten`
4. **Framework Preset** : "Other"
5. **Root Directory** : `.` (laisser vide, c'est la racine)
6. **Environment Variables** (Settings → Environment Variables) :

| Variable | Valeur | Description |
|----------|--------|-------------|
| `OPENAI_API_KEY` | `sk-...` | Clé API OpenAI (requise) |
| `UPSTASH_REDIS_REST_URL` | `https://...upstash.io` | URL Redis Upstash |
| `UPSTASH_REDIS_REST_TOKEN` | `AX...` | Token Redis Upstash |

7. Cliquer **Deploy**

---

## 🔍 Voir les traces (conversations)

### Via l'API

Après déploiement, accéder à :

```
https://votre-app.vercel.app/api/traces?limit=50
```

Retourne les 50 dernières conversations avec :
- Question utilisateur
- Type de réponse (recette, menu, etc.)
- Intent détecté
- Nombre de recettes retournées

### Via Vercel Logs

Même sans Upstash, les traces sont toujours visibles dans :
**Vercel Dashboard → Project → Logs**

Format : `[TRACE] {"ts":"...","id":"abc","q":"recette taboulé","intent":"recipe_specific","recipes":1}`

---

## 🧪 Test local

```bash
cd Sahten/backend
pip install -r requirements.txt

# Avec clé API OpenAI
$env:OPENAI_API_KEY="sk-..."
python -m uvicorn main:app --reload

# Ouvrir http://localhost:8000
```

---

## 📊 Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Interface chat |
| `POST /api/chat` | Envoyer un message |
| `GET /api/health` | Health check |
| `GET /api/status` | Statut détaillé |
| `GET /api/traces` | Historique conversations (si Upstash) |

---

## 🔒 Notes de sécurité

- Ne jamais committer les clés API dans le code
- Utiliser les Variables d'Environnement Vercel
- Le fichier `.env` est pour le dev local uniquement

---

*Sahten ! 🇱🇧*
