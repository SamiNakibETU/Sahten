# Archive — documentation technique détaillée

Ce répertoire regroupe la documentation longue et les guides d’exploitation du projet **Sahten** (assistant recettes L’Orient-Le Jour). Le dépôt principal expose un `README.md` court ; l’archive sert de référence pour l’installation, l’intégration du widget, le CMS et les options d’indexation.

## Contenu du dossier

| Fichier / dossier | Description |
|-------------------|-------------|
| `INTEGRATION_GUIDE.md` | Intégration du widget sur un site (configuration, CORS, API de base). |
| `ARCHITECTURE.md` | Vue d’ensemble des composants backend, flux de requête et déploiement. |
| `docs/CMS_INTEGRATION.md` | Endpoint webhook, format de payload, authentification et synchronisation des contenus. |
| `docs/MODEL_COMPARISON.md` | Comparaison des modèles de langage (coût, usage). |
| `docs/EMBEDDINGS_GUIDE.md` | Activation et rôle des embeddings dans la recherche. |
| `docs/LIBAN_A_TABLE_ANALYSIS.md` | Notes d’analyse sur le corpus recettes. |
| `docs/SCRAPING_PROMPT_CODEX.md` | Prompt et procédure d’extraction de données (usage ponctuel). |

## Télémétrie, traces et indicateurs métier

Le backend expose un **périmètre analytics dédié au widget Sahten** : requêtes utilisateur, recettes affichées, clics vers les articles OLJ, retours utilisateur (pouces), répartition des intentions et des types de réponse.

### Prérequis : persistance Redis (Upstash)

Les compteurs, listes d’événements et agrégations sont stockés dans **Redis** lorsque les variables d’environnement `UPSTASH_REDIS_REST_URL` et `UPSTASH_REDIS_REST_TOKEN` sont correctement définies sur l’environnement d’exécution (ex. Railway). Sans Redis, les événements restent visibles dans les **logs applicatifs** (`[EVENT:…]`, `[TRACE]`) mais **ne sont pas agrégés** pour les API ci-dessous.

### Endpoints principaux

| Méthode | Chemin | Fonction |
|---------|--------|----------|
| POST | `/api/events` | Enregistrement d’événements widget : `impression` (carte recette vue), `click` (lien recette), `feedback` (satisfaction). |
| GET | `/api/analytics` | Synthèse : volumes d’événements, **CTR** (clics / impressions), feedback positif/négatif, top recettes par clics, répartition intents / modèles / types de réponse, indicateurs qualité (match exact, alternative prouvée, etc.). |
| GET | `/api/traces` | Dernières traces de conversation (requête, intention, recettes proposées, modèle). |
| GET | `/api/feedback/stats` | Statistiques de feedback et motifs récents des avis négatifs. |

### Données côté Redis (schéma logique)

- Listes `sahten:events:{type}` : flux récents par type d’événement.  
- Compteurs globaux `sahten:events:{type}:count`.  
- Tables de comptage par URL de recette : `sahten:recipe:impressions`, `sahten:recipe:clicks`.  
- Métriques produit : `sahten:metrics:*` (requêtes, intents, modèles, types de réponse, etc.).  
- Liste `sahten:traces:recent` : journal des requêtes pour audit et amélioration continue.

### Interface de suivi

Le fichier **`frontend/dashboard.html`** consomme `GET /api/analytics` (configurer la base URL de l’API pour pointer vers le même hôte que le backend déployé). Il permet un **tableau de bord opérationnel** sans stack externe, tant que Redis est actif.

### Périmètre : widget vs site entier

- **Mesuré par Sahten** : tout ce qui se passe **dans le widget** (question posée, recette(s) recommandée(s) affichée(s), clic vers la fiche OLJ, satisfaction). C’est la base pour analyser la **performance des recommandations** et le **trafic qualifié** envoyé vers les URLs recettes.  
- **Non couvert par défaut** : le trafic global du site (autres pages, SEO, campagnes) relève en général des **outils d’analyse du site hôte** (ex. balise GA4, Matomo, stack éditoriale). Ces outils sont **complémentaires** : ils ne remplacent pas les événements Sahten, et inversement.  
- **Responsabilités** : l’intégration CMS / publication des articles est distincte de la **collecte d’événements Sahten**. Une équipe tierce peut gérer le CMS ; l’équipe produit Sahten peut toutefois **piloter l’analytics widget** via Redis + API + dashboard, sans dépendre de cette équipe pour les métriques listées ci-dessus.

### Évolution possible (hors périmètre actuel du code)

Pour des besoins avancés (requêtes SQL, rétention longue durée, rapports éditoriaux croisés avec d’autres sources), une évolution naturelle consiste à **répliquer** les événements ou traces vers un entrepôt (base relationnelle, data lake) ou vers un outil BI, en s’appuyant sur les mêmes payloads que `/api/events`.
