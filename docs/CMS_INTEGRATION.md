# CMS Integration Guide

Guide complet pour l'équipe de développement du journal pour intégrer Sahten avec le CMS.

## Vue d'ensemble

Sahten supporte l'intégration **PUSH** via webhook : le CMS envoie les nouvelles recettes à Sahten qui les enrichit et les indexe automatiquement.

```
┌──────────────┐     POST /api/webhook/recipe     ┌──────────────┐
│              │  ──────────────────────────────► │              │
│   Votre CMS  │     {title, content, url...}     │   Sahten     │
│              │  ◄────────────────────────────── │              │
└──────────────┘     {"status": "indexed"}        └──────────────┘
       │                                                 │
       │ Vous développez                   Nous développons │
       ▼                                                 ▼
  Appeler le webhook                         Recevoir + Enrichir
  lors de la publication                     + Indexer + Répondre
```

## Endpoint

```
POST /api/webhook/recipe
```

## Authentification

Utiliser le header `X-Webhook-Secret` :

```http
X-Webhook-Secret: votre-secret-partage
```

Le secret est configuré côté Sahten via la variable d'environnement `WEBHOOK_SECRET`.

## Format du Payload

```json
{
  "id": "article_12345",
  "title": "Fattouch aux herbes fraîches",
  "url": "https://lorientlejour.com/article/12345",
  "author": "Maya Sfeir",
  "published_date": "2025-01-05",
  "content": "La fattouch est une salade libanaise traditionnelle...",
  "image_url": "https://cdn.lorientlejour.com/images/12345.jpg"
}
```

### Champs

| Champ            | Type   | Requis | Description                            |
| ---------------- | ------ | ------ | -------------------------------------- |
| `id`             | string | ✅     | Identifiant unique de l'article        |
| `title`          | string | ✅     | Titre de la recette                    |
| `url`            | string | ✅     | URL de l'article sur lorientlejour.com |
| `content`        | string | ✅     | Contenu complet de la recette          |
| `author`         | string | ❌     | Nom du chef/auteur                     |
| `published_date` | string | ❌     | Date de publication (ISO 8601)         |
| `image_url`      | string | ❌     | URL de l'image principale              |

## Réponses

### Succès (200)

```json
{
  "status": "indexed",
  "id": "article_12345",
  "message": "Recipe 'Fattouch aux herbes fraîches' added and indexed",
  "enriched": true
}
```

### Recette déjà existante (200)

```json
{
  "status": "skipped",
  "id": "article_12345",
  "message": "Recipe already exists",
  "enriched": false
}
```

### Erreur d'authentification (401)

```json
{
  "detail": "Invalid webhook secret"
}
```

### Erreur serveur (500)

```json
{
  "detail": "Failed to save recipe: ..."
}
```

## Exemples curl

### Test simple

```bash
curl -X POST "https://your-sahten-instance.railway.app/api/webhook/recipe" \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: your-secret" \
  -d '{
    "id": "test_001",
    "title": "Test Taboulé",
    "url": "https://lorientlejour.com/test",
    "content": "Le taboulé est une salade de persil..."
  }'
```

### Avec tous les champs

```bash
curl -X POST "https://your-sahten-instance.railway.app/api/webhook/recipe" \
  -H "Content-Type: application/json" \
  -H "X-Webhook-Secret: your-secret" \
  -d '{
    "id": "article_67890",
    "title": "Kibbeh nayeh traditionnel",
    "url": "https://lorientlejour.com/article/67890",
    "author": "Chef Antoine",
    "published_date": "2025-01-05T10:30:00Z",
    "content": "Le kibbeh nayeh est un plat emblématique de la cuisine libanaise. Cette préparation de viande crue assaisonnée demande une qualité de viande irréprochable et un savoir-faire traditionnel...",
    "image_url": "https://cdn.lorientlejour.com/images/67890.jpg"
  }'
```

## Enrichissement automatique

Quand Sahten reçoit une recette, il l'enrichit automatiquement via LLM :

| Champ enrichi       | Description            | Exemple                            |
| ------------------- | ---------------------- | ---------------------------------- |
| `categories`        | Type de plat           | `["mezze_froid", "entree"]`        |
| `difficulty`        | Niveau de difficulté   | `"facile"`                         |
| `is_lebanese`       | Cuisine libanaise ?    | `true`                             |
| `keywords`          | Mots-clés de recherche | `["persil", "boulghour", "تبولة"]` |
| `prep_time_minutes` | Temps de préparation   | `30`                               |
| `main_ingredients`  | Ingrédients principaux | `["persil", "tomates", "oignons"]` |
| `occasion`          | Occasion de service    | `["quotidien", "ete", "buffet"]`   |
| `mood`              | Qualités émotionnelles | `["frais", "leger", "healthy"]`    |
| `dietary`           | Régimes alimentaires   | `["vegetarien", "vegan"]`          |

Ces champs permettent une recherche plus précise :

- "recette fraîche pour l'été"
- "dessert pour le ramadan"
- "plat végétarien rapide"

## Vérification de l'état

```bash
# Vérifier que le webhook est prêt
curl "https://your-sahten-instance.railway.app/api/webhook/health"
```

Réponse :

```json
{
  "status": "ready",
  "auto_enrich": true,
  "webhook_configured": true
}
```

## Responsabilités

| Composant                         | Responsable   |
| --------------------------------- | ------------- |
| Endpoint `/api/webhook/recipe`    | Équipe Sahten |
| Enrichissement automatique        | Équipe Sahten |
| Appel du webhook à la publication | Équipe CMS    |
| Génération et stockage du secret  | Coordination  |

## Configuration requise

### Côté Sahten (Railway/Vercel)

Variables d'environnement :

```
WEBHOOK_SECRET=votre-secret-partage-securise
OPENAI_API_KEY=sk-...
AUTO_ENRICH_ON_WEBHOOK=true
```

### Côté CMS

- Stocker le secret de manière sécurisée
- Appeler le webhook après publication/mise à jour
- Gérer les retries en cas d'erreur réseau

## FAQ

### Quand appeler le webhook ?

- ✅ À la **publication** d'une nouvelle recette
- ✅ À la **mise à jour** d'une recette existante (sera ignorée si déjà présente)
- ❌ Ne pas appeler pour les brouillons

### Que se passe-t-il si le webhook échoue ?

La recette n'est pas indexée. Implémentez un système de retry côté CMS ou contactez l'équipe Sahten.

### Les anciennes recettes sont-elles incluses ?

Oui, les 145 recettes existantes sont déjà indexées. Le webhook est pour les **nouvelles** publications.

### Puis-je tester en développement ?

Oui, utilisez l'endpoint local :

```
POST http://localhost:8000/api/webhook/recipe
```

Sans `WEBHOOK_SECRET` configuré, le webhook accepte toutes les requêtes (mode dev).

## Contact

Pour toute question technique :

- Endpoint `/api/status` pour l'état du système
- Endpoint `/api/traces` pour voir les conversations récentes

