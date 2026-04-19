# Brief à l'équipe WhiteBeard (CMS de L'Orient-Le Jour)

**Objet** : Sahteïn — diagnostic d'authentification API + demande de
confirmation du contrat de données pour notre ingestion exhaustive.

---

## Contexte

Nous (Sahteïn) sommes le bot culinaire intégré au site OLJ. Nous
consommons votre API `https://api.lorientlejour.com/cms/content/{id}`
via le header `API-Key`. Nous lançons une refonte v4 (RAG SOTA :
Postgres + pgvector + retrieval hybride + grounding LLM) qui exige une
ingestion **exhaustive** de tous les champs disponibles, pas juste le
HTML principal.

---

## Diagnostic d'authentification

Sonde sur l'article taboulé Mouzawak (1227694) avec la clé que nous
avons en local (`OLJ_API_KEY`, longueur 21, préfixe `yo…re`) :

```
GET /cms/content/1227694
Header API-Key: <clé>
→ HTTP 401 "Unauthorized: Invalid API key"
```

Toutes les variantes (`Token`, `Authorization: Bearer`,
`Authorization: API-Key`, `X-API-Key`, query string `?api_key=`) ont été
testées : elles renvoient « Missing API-Key header », ce qui confirme
que le header `API-Key` est bien le bon et que **notre clé locale est
expirée ou invalide**.

> Script de reproduction (open source) :
> [`v4/scripts/probe_auth.py`](../scripts/probe_auth.py).

### Ce dont nous avons besoin

1. Confirmer si la clé `yo…re` (21 caractères) a été révoquée ou
   rotationnée.
2. Nous fournir une **clé de production valide** (à transmettre au
   responsable Sahteïn par canal sécurisé — jamais en clair par mail) à
   stocker dans Railway.
3. Si possible, une **clé de staging séparée** pour notre CI GitHub
   Actions et notre environnement local de développement.

---

## Demandes de clarification du contrat de données

Notre v4 exige tous les champs ci-dessous en réponse de
`GET /content/{id}`. Merci de confirmer leur présence et de nous
indiquer leur structure exacte.

### Niveau article (`data[0]`)

| Champ attendu | Type | Usage Sahteïn |
|---------------|------|---------------|
| `id` | int | Clé primaire externe |
| `url` | string | Lien canonique vers l'article |
| `slug` | string | Slug pour citation |
| `title`, `subtitle`, `summary` | string | Affichage + indexation |
| `introduction` | string | Texte d'intro (souvent perdu chez nous) |
| `signature` | string | Crédit auteur |
| `firstPublished`, `lastUpdate` | ISO datetime | Tri et delta-sync |
| `time_to_read` | int | UI |
| `premium` | bool | Filtrage |
| `content_length` | int | Métrique |
| `contents.html` | string HTML | **Corps principal — confirmer que c'est bien la version finale, intégrant les encadrés Bio chef + listes Ingrédients + listes Préparation + commandements** |
| `image` ou `cover` | object `{url, caption, credits}` | Affichage |
| `seo` | object `{meta_title, meta_description, canonical}` | Indexation |

### Auteurs/contributeurs (`data[0].authors[]`)

| Champ | Critique pour nous |
|-------|--------------------|
| `id` | clé externe Person |
| `name` | obligatoire |
| `department` | distinguer chef invité / journaliste |
| `description` | sous-titre type « Chef invité — fondateur de Souk el-Tayeb » |
| `biography` | **HTML complet de la bio** (notre v3 ne la stockait pas — c'est la cause #1 du problème "le bot ne connaît pas Mouzawak") |
| `image.url` | photo |

### Mots-clés (`data[0].keywords[]`)

```
{ "id": 1001, "name": "Taboulé", "description": "Plat libanais à base de persil…" }
```

→ Confirmer que `description` est présent quand le mot-clé a une fiche
éditoriale dédiée.

### Catégories (`data[0].categories[]`)

→ Idem, avec `description` quand disponible.

### Pièces jointes / encadrés

| Champ | Question |
|-------|----------|
| `attachments[]` | Liste de PJ (pdf, vidéos) — utile à indexer ? |
| `inline_attachments[]` | Encarts in-body (sidebar éditoriaux) — sont-ils déjà fusionnés dans `contents.html` ou séparés ? |

---

## Webhook publication / mise à jour

Nous exposons :

```
POST https://<sahten-prod>.up.railway.app/api/webhook/recipe
Header X-Signature-256: sha256=<hmac_sha256(WEBHOOK_SECRET, body)>
Body { "event": "article.published" | "article.updated" | "article.deleted",
       "article_id": 1227694 }
```

À chaque événement, nous re-fetcherons `/content/{id}` et reconstruirons
les sections + chunks + embeddings. **Idempotent**.

### Demandes côté webhook

1. Pouvez-vous nous fournir un **endpoint de test** qui rejoue les
   événements à la demande ?
2. Ré-émettez-vous l'événement en cas de 5xx côté nous, ou faut-il un
   reconciler de notre côté ?
3. Quel est le délai max entre publication CMS et émission du webhook ?

---

## Ressources

- Probe API reproductible : [`v4/scripts/probe_auth.py`](../scripts/probe_auth.py)
- Sonde exhaustive avec audit champ-par-champ :
  [`v4/scripts/fetch_whitebeard.py`](../scripts/fetch_whitebeard.py)
- Schéma SQL cible : [`v4/backend/app/db/models.py`](../backend/app/db/models.py)
- Mapper qui consomme votre payload : [`v4/backend/app/ingestion/mapper.py`](../backend/app/ingestion/mapper.py)

Merci d'avance — disponibles pour un call rapide si besoin.
