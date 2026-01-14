# CMS Integration Guide

Guide complet pour l'équipe de développement du journal pour intégrer Sahten avec le CMS.

## Vue d'ensemble

Sahten utilise une intégration **hybride PUSH + PULL** :
1. Le CMS envoie une notification (webhook) avec l'`article_id`
2. Sahten récupère les données complètes via l'API OLJ
3. Sahten enrichit et indexe automatiquement la recette

```
┌──────────────┐  POST /webhook/recipe   ┌──────────────┐  GET /cms/content/{id}  ┌──────────────┐
│              │  { article_id: 123 }    │              │  ─────────────────────► │              │
│   Votre CMS  │  ────────────────────►  │   Sahten     │  ◄───────────────────── │   OLJ API    │
│              │  ◄────────────────────  │              │   { title, content... } │              │
└──────────────┘  {"status": "indexed"}  └──────────────┘                         └──────────────┘
       │                                        │
       │ Vous développez              Nous développons │
       ▼                                        ▼
  Appeler le webhook               Récupérer + Enrichir
  avec l'article_id                + Indexer + Répondre
```

## Endpoint Webhook

```
POST /api/webhook/recipe
```

### Authentification

Header requis :
```http
X-Webhook-Secret: votre-secret-partage
```

Le secret est configuré côté Sahten via `WEBHOOK_SECRET`.

### Format du Payload (Simplifié)

```json
{
  "article_id": 1227694,
  "action": "publish"
}
```

| Champ        | Type   | Requis | Description                                        |
| ------------ | ------ | ------ | -------------------------------------------------- |
| `article_id` | int    | ✅     | Content ID de l'article (ex: 1227694)              |
| `action`     | string | ❌     | `"publish"` (défaut), `"update"`, ou `"delete"`    |

### Actions supportées

| Action    | Comportement                                      |
| --------- | ------------------------------------------------- |
| `publish` | Ajoute la recette (ignore si déjà existante)      |
| `update`  | Met à jour la recette existante                   |
| `delete`  | Supprime la recette de l'index                    |

## Réponses

### Succès - Recette indexée (200)

```json
{
  "status": "indexed",
  "article_id": 1227694,
  "message": "Recipe 'Le taboulé de Kamal Mouzawak' added and indexed",
  "enriched": true
}
```

### Recette déjà existante (200)

```json
{
  "status": "skipped",
  "article_id": 1227694,
  "message": "Recipe already exists",
  "enriched": false
}
```

### Recette supprimée (200)

```json
{
  "status": "deleted",
  "article_id": 1227694,
  "message": "Recipe 1227694 deleted",
  "enriched": false
}
```

### Erreur d'authentification (401)

```json
{
  "detail": "Invalid webhook secret"
}
```

### Recette non trouvée dans l'API (404)

```json
{
  "detail": "Recipe 1227694 not found in OLJ API"
}
```

### Erreur API OLJ (502)

```json
{
  "detail": "Failed to fetch recipe from OLJ API: ..."
}
```

## Exemples curl

### Publier une nouvelle recette

```bash
curl -X POST "https://sahten-mvp-production.up.railway.app/api/webhook/recipe" \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: votre-secret" \
  -d '{
    "article_id": 1227694,
    "action": "publish"
  }'
```

### Mettre à jour une recette

```bash
curl -X POST "https://sahten-mvp-production.up.railway.app/api/webhook/recipe" \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: votre-secret" \
  -d '{
    "article_id": 1227694,
    "action": "update"
  }'
```

### Supprimer une recette

```bash
curl -X POST "https://sahten-mvp-production.up.railway.app/api/webhook/recipe" \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: votre-secret" \
  -d '{
    "article_id": 1227694,
    "action": "delete"
  }'
```

## Vérification de l'état

```bash
curl "https://sahten-mvp-production.up.railway.app/api/webhook/health"
```

Réponse :

```json
{
  "status": "ready",
  "auto_enrich": true,
  "webhook_configured": true,
  "olj_api_configured": true
}
```

## Configuration côté Sahten (Railway)

Variables d'environnement requises :

```
OPENAI_API_KEY=sk-...
WEBHOOK_SECRET=votre-secret-partage-securise
OLJ_API_KEY=d3037095bcd8ad824767518cf83b9440bf7dc14a17a150f1398c9d8f63a7e623
OLJ_API_BASE=https://api.lorientlejour.com/cms
AUTO_ENRICH_ON_WEBHOOK=true
```

## Enrichissement automatique

Quand Sahten reçoit une notification :
1. Récupère les données via `GET /cms/content/{article_id}`
2. Extrait : titre, contenu, auteur, catégorie, keywords, image
3. Enrichit via LLM (catégorie, ingrédients, tags)
4. Indexe pour la recherche

| Champ enrichi       | Description            | Exemple                            |
| ------------------- | ---------------------- | ---------------------------------- |
| `category_canonical`| Type de plat           | `"mezze_froid"`, `"dessert"`       |
| `is_lebanese`       | Cuisine libanaise ?    | `true`                             |
| `tags`              | Mots-clés de recherche | `["persil", "boulghour", "mezzé"]` |
| `main_ingredients`  | Ingrédients principaux | `["persil", "tomates", "oignons"]` |

## Responsabilités

| Composant                           | Responsable   |
| ----------------------------------- | ------------- |
| Endpoint `/api/webhook/recipe`      | Équipe Sahten |
| Appel API `/cms/content/{id}`       | Équipe Sahten |
| Enrichissement automatique          | Équipe Sahten |
| Appel du webhook à la publication   | Équipe CMS    |
| Génération et partage du secret     | Coordination  |

## Données initiales

Les 151 recettes du CSV export ont été importées. Le webhook est pour les **nouvelles** publications uniquement.

## FAQ

### Quand appeler le webhook ?

- ✅ À la **publication** d'une nouvelle recette
- ✅ À la **mise à jour** d'une recette existante
- ✅ À la **suppression** d'une recette
- ❌ Ne pas appeler pour les brouillons

### Que se passe-t-il si l'API OLJ est indisponible ?

Sahten retourne une erreur 502. Implémentez un système de retry côté CMS.

### Comment tester en développement ?

```bash
# Local (sans secret requis en mode dev)
POST http://localhost:8000/api/webhook/recipe
```

### Comment vérifier qu'une recette est indexée ?

Utilisez le chat pour rechercher la recette par son titre.

## Contact

- Endpoint `/api/health` - état du système
- Endpoint `/api/status` - statistiques détaillées
- Endpoint `/api/webhook/health` - état du webhook
