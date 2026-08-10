# Sortir de Railway — déploiement sur le serveur AWS de WhiteBeard

Cible : `ec2-user@3.94.84.219`, Amazon Linux 2023, **aarch64 (ARM)**.
Objectif : Sahteïn tourne intégralement sur ce serveur, Railway est éteint.

Prérequis d'accès (demandés par mail le 5 août 2026, hors de notre contrôle) :

- [ ] IP passerelle `145.241.160.99` autorisée en SSH
- [ ] **Port 80 ouvert** dans le security group (aujourd'hui seul le 22 répond →
      c'est la cause directe du `HTTP 522` sur `sahtenbotapi.lorientlejour.com`)
- [ ] Confirmation que `sahtenbotapi.lorientlejour.com` pointe bien sur cette machine

Le reste ci-dessous ne dépend que de nous.

---

## 0. Pourquoi Docker et pas une installation native

Une installation native avait été commencée (`dnf install postgresql16-server-devel
gcc make`), ce qui implique de **compiler pgvector depuis les sources** sur ARM.
On abandonne cette voie :

- `pgvector/pgvector:pg16` est multi-architecture et embarque déjà l'extension.
- Railway construit **déjà** `v4/Dockerfile.web`. Réutiliser la même image supprime
  toute la classe de bugs « ça marchait sur Railway » : c'est un changement
  d'hébergeur, pas un portage.
- Le rollback est un `docker compose down` + redémarrage de l'image précédente.

Vérifié côté compatibilité ARM : `python:3.11-slim` est multi-arch et **aucune
dépendance de production n'a besoin de compilation** (`torch` et
`sentence-transformers` sont dans l'extra optionnel `local-rerank`, jamais installé).

## 1. Docker sur Amazon Linux 2023

```bash
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user     # se déconnecter/reconnecter pour l'appliquer
# Le plugin compose n'est pas dans les dépôts AL2023 : installation manuelle.
sudo mkdir -p /usr/libexec/docker/cli-plugins
sudo curl -fsSL -o /usr/libexec/docker/cli-plugins/docker-compose \
  https://github.com/docker/compose/releases/latest/download/docker-compose-linux-aarch64
sudo chmod +x /usr/libexec/docker/cli-plugins/docker-compose
docker compose version
```

## 2. Code + configuration

```bash
git clone https://github.com/SamiNakibETU/Sahten.git ~/sahten
cd ~/sahten
git checkout main          # sota/v4 n'existe plus (cf. §6)
cp v4/infra/aws/env.example v4/infra/aws/.env
vi v4/infra/aws/.env       # remplir tous les champs vides
chmod 600 v4/infra/aws/.env
```

Générer les secrets qui doivent être forts :

```bash
openssl rand -base64 32    # POSTGRES_PASSWORD
openssl rand -base64 32    # SAHTEN_ADMIN_API_TOKEN
```

**Ne pas réutiliser** les valeurs qui ont circulé en clair (l'ancien
`SAHTEN_ADMIN_API_TOKEN=aaa3f68...` est compromis, ainsi que toute clé collée
dans une conversation ou un mail).

## 3. Démarrage

```bash
cd ~/sahten
docker compose -f v4/infra/aws/docker-compose.yml --env-file v4/infra/aws/.env up -d --build
docker compose -f v4/infra/aws/docker-compose.yml logs -f web
```

L'entrypoint lance `alembic upgrade head` avant uvicorn : le schéma se crée seul
au premier boot. Vérifier :

```bash
curl -fsS localhost/healthz && echo         # l'app répond
curl -fsS localhost/readyz  && echo         # + base ET Redis joignables
```

`/readyz` est le test qui compte : `/healthz` répond même si la base est morte.

## 4. Reprendre les données de Railway (ne pas ré-indexer)

Le corpus indexé (~197 articles, ~6400 chunks) représente des milliers d'appels
d'embeddings **déjà payés**. Le recréer coûterait de l'argent et plusieurs heures.
On transfère la base.

Depuis le poste local (pas depuis le serveur) — récupérer `DATABASE_PUBLIC_URL`
dans le dashboard Railway, l'URL interne `postgres.railway.internal` n'étant
joignable que depuis Railway :

```bash
pg_dump --no-owner --no-privileges -Fc \
  "postgresql://postgres:<mdp>@<host>.proxy.rlwy.net:<port>/railway" \
  -f sahten_railway.dump
scp sahten_railway.dump olj-prod:~/
```

Sur le serveur :

```bash
docker compose -f v4/infra/aws/docker-compose.yml exec -T postgres \
  psql -U sahten -d sahten -c "CREATE EXTENSION IF NOT EXISTS vector;"
docker cp ~/sahten_railway.dump sahten_pg:/tmp/d.dump
docker compose -f v4/infra/aws/docker-compose.yml exec postgres \
  pg_restore -U sahten -d sahten --no-owner --clean --if-exists /tmp/d.dump
```

`CREATE EXTENSION vector` **avant** le restore : sans elle, `pg_restore` échoue sur
le type `vector` des colonnes d'embeddings.

Contrôle :

```bash
docker compose -f v4/infra/aws/docker-compose.yml exec postgres \
  psql -U sahten -d sahten -c "SELECT count(*) FROM articles; SELECT count(*) FROM chunks;"
```

Si le transfert est impossible, repli : `python -m scripts.ingest_cli reindex-all
--skip-ingest --seed-file data/olj_seed_ids.json` (long, et refacture les embeddings).

## 5. Exposition publique

Cloudflare est déjà devant le domaine. Le conteneur `web` est publié sur le port
**80 de l'hôte**, ce que Cloudflare vient chercher — aucun nginx nécessaire pour
la première mise en ligne.

- Security group : ouvrir **80**, idéalement restreint aux
  [plages d'IP Cloudflare](https://www.cloudflare.com/ips/). Ne jamais exposer 8000.
- `SAHTEN_TRUSTED_PROXY_HOPS=1` (Cloudflare seul devant l'app).
- Postgres et Redis n'ont **aucun** `ports:` — ils ne sortent pas du réseau Docker.

Durcissement à faire ensuite, pas bloquant pour la remise en ligne : générer un
*Cloudflare Origin Certificate*, terminer le TLS sur la machine et passer
Cloudflare en mode « Full (strict) ». Tant que ce n'est pas fait, le segment
Cloudflare → origine circule en clair.

## 6. Faire pointer le déploiement sur `main`

Joseph a signalé le 1er juillet 2026 que leur déploiement suivait encore la branche
**`sota/v4`**, supprimée fin juin lors de la consolidation sur `main`. C'est
l'explication la plus probable de l'arrêt du service. Si une automatisation de leur
côté déploie encore ce dépôt, elle doit être basculée sur `main`.

## 7. Contrôle de bon fonctionnement

```bash
curl -fsS -X POST localhost/api/chat -H 'Content-Type: application/json' \
  -d '{"query":"recette de manouche"}' | head -c 400
```

Attendu : une réponse contenant l'article `1474718` (*Les manaïichs du Chouf de
Salim Azzam*). C'est le cas de non-régression historique du projet.

Puis, une fois le DNS confirmé, depuis l'extérieur :

```bash
curl -sS -o /dev/null -w '%{http_code}\n' https://sahtenbotapi.lorientlejour.com/healthz
```

`200` attendu. Un `522` signifie que rien n'écoute ou que le port reste fermé.

## 8. Éteindre Railway

À ne faire qu'après une journée de fonctionnement nominal :

1. vérifier que `/dashboard` se remplit (trafic réel servi par AWS) ;
2. faire une dernière sauvegarde de la base Railway ;
3. supprimer le service Railway.

## Limite connue du catalogue

L'API CMS annonce `total = 215` recettes mais plafonne `limit` à 100
(`HTTP 400 — "Invalid limit specified. Must be 1-100"`), et **ignore** `page`,
`offset`, `skip`, `start`, `from` et `per_page` : tous renvoient le même premier
lot. Environ 115 recettes sont donc inatteignables par l'API. D'où le
`--seed-file` d'identifiants dans `ingest_cli`. Question posée à WhiteBeard le
5 août 2026 ; en attendant, le seed reste le seul moyen d'avoir le corpus complet.
