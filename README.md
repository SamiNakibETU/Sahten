# Sahten

Assistant culinaire pour **L’Orient-Le Jour** : recherche dans les recettes « Liban à Table », réponses en français, liens vers les fiches OLJ.

## Intégration site (OLJ)

1. Charger le widget (iframe ou script selon votre gabarit).
2. Définir l’URL du backend :  
   `window.SAHTEN_API_BASE = 'https://<votre-service>.up.railway.app/api';`  
   (à adapter au domaine Railway réel du service.)
3. CORS : le backend doit autoriser l’origine du site OLJ.

Guides détaillés (webhook CMS, architecture, modèles) : dossier **`archive/`** — usage interne équipe technique.

## Prérequis

- Python 3.11+
- Clé OpenAI (`OPENAI_API_KEY`)

## Démarrage local

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# Copier backend/.env.example vers backend/.env et renseigner OPENAI_API_KEY
python main.py
```

Interface : http://localhost:8000  

## API utiles

| Méthode | Chemin | Rôle |
|--------|--------|------|
| POST | `/api/chat` | Message utilisateur → HTML recette(s) |
| GET | `/api/health` | Santé du service |
| POST | `/api/webhook/recipe` | Réception publication CMS (configuré avec WhiteBeard) |

Variables principales : voir `backend/.env.example`.

## Déploiement Railway

1. Connecter le dépôt GitHub au projet Railway.
2. Renseigner les variables d’environnement (OpenAI, secrets webhook, Redis optionnel).
3. Vérifier que l’URL publique correspond à celle configurée côté CMS.

## Observabilité

Sans **Upstash Redis**, traces et événements widget restent dans les **logs Railway**. Avec Redis configuré, persistance listée côté API (`/api/traces`, analytics dans `routes.py`). Voir `archive/README.md`.

## Licence

Usage interne L’Orient-Le Jour.
