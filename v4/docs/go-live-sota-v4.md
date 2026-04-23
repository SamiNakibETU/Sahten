# Bascule production — branche `sota/v4` (Sahteïn)

Checklist de référence une fois le staging (Railway) validé. À adapter à votre nommage de services.

## 1. Code et déploiement

- [ ] Dernière version **mergée / poussée** sur `sota/v4` (ou merge vers `main` si c’est la politique de prod).
- [ ] Service Railway **web** : branche / Dockerfile `v4/Dockerfile.web`, health check `GET /healthz` OK.
- [ ] Migrations : job ou déploiement exécute `alembic upgrade head` (voir `docs/railway-runbook.md`).
- [ ] Variables d’environnement prod : mêmes clés qu’en staging, valeurs **prod** (`OPENAI_API_KEY`, `OLJ_API_KEY`, `COHERE_API_KEY`, `DATABASE_URL`, `REDIS_URL`, `WEBHOOK_SECRET`, etc.).

## 2. Données

- [ ] Ingestion recettes : `python -m scripts.ingest_cli reindex-all --publication 17 --content-type 4 --seed-file data/olj_seed_ids.json` (conteneur web ou job one-shot), puis contrôle `GET /api/admin/stats` (articles, chunks, `chunks_embedded` alignés).
- [ ] Vérification spot : `/admin` — article 1227694 (taboulé) avec sections + chunks + embeddings.

## 3. Côté OLJ / intégration

- [ ] CORS : domaines `lorientlejour.com` (et variantes) toujours autorisés sur l’URL API production.
- [ ] Widget : URL du `widget.js` / iframe pointant vers l’hôte **prod** ; `window.SAHTEN_API_BASE` = base `/api` du même hôte.
- [ ] Vérification navigateur : widget **Sahteïn**, clôture session, accroche « carnets » sur requête hors corpus (recette inconnue).

## 4. Emails (équipes WhiteBeard & OLJ)

- [ ] Alerter l’exploitation : URL prod du widget, URL API, fenêtre de bascule.
- [ ] Point **WhiteBeard** (Joseph) : pagination `GET /publication/{id}/content` (paramètre de page) à confirmer pour éviter de fusionner avec un seed file ; biographie chef via `GET /content/{chef_id}`. Modèle proposé ci-dessous.

---

## Modèle d’e-mail (Joseph / OLJ)

**Objet :** Sahteïn v4 en production — accès et points API

Bonjour,

La version **Sahteïn v4** (RAG, branche `sota/v4`) est basculée en production. Pour rappel :

- **Point d’intégration** : le widget (chat) et l’API répondent sur l’hôte [INSÉRER URL PROD, ex. `https://…up.railway.app`], avec l’**Admin Base RAG** sur `/admin` (usage interne / relecture).
- **Côté WhiteBeard** : nous comptons toujours sur `GET /content/{id}` pour l’ingestion ; en complément, la fiche **chef** est chargée en second appel quand l’objet `chef` n’expose pas toute la bio. Il reste à confirmer côté API la **pagination fiable** de `GET /publication/17/content` (récupération de l’ensemble des IDs recettes) : aujourd’hui nous **fusionnons** la liste retournée avec notre `data/olj_seed_ids.json` pour couvrir l’intégralité du total annoncé par le champ `total` de l’API.

Merci de dire si vous préférez un autre point d’appel listant tous les contenus d’une publication.

Bien à vous,  
[Signature]

---

*Document généré pour le dépôt Sahteïn — mettre à jour les URLs et la date à chaque go-live réel.*
