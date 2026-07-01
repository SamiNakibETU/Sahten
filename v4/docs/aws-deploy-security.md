# Déploiement AWS — checklist sécurité (prod L'Orient-Le Jour)

Cible : widget Sahteïn embarqué en iframe sur `lorientlejour.com`, backend FastAPI
conteneurisé (`Dockerfile.web`), Postgres+pgvector, Redis.

L'app est déjà durcie côté code (SQL paramétré, HMAC webhook, headers OWASP,
CORS verrouillé en prod, jeton admin à temps constant, HTML échappé + DOMPurify,
conteneur non-root). Cette checklist couvre la **configuration d'infra** à ne pas
rater sur AWS.

## 1. Variables d'environnement (obligatoires en prod)

| Variable | Valeur | Pourquoi |
|---|---|---|
| `APP_ENV` | `production` | active HSTS + **bloque le boot** si CORS=* ou jeton admin faible (`validate_security_at_startup`) |
| `SAHTEN_CORS_ORIGINS` | `https://www.lorientlejour.com,https://lorientlejour.com` | pas de `*` en prod (le boot échoue sinon) |
| `SAHTEN_ADMIN_API_TOKEN` | aléatoire ≥ 32 car. (`openssl rand -base64 32`) | protège /dashboard, /api/analytics, /api/traces, /api/metrics |
| `SAHTEN_TRUSTED_PROXY_HOPS` | `1` (ALB seul) ou `2` (CloudFront→ALB) | **rate-limit** : sinon tous les users partagent un quota (voir §3) |
| `OPENAI_API_KEY`, `COHERE_API_KEY`, `OLJ_API_KEY`, `WEBHOOK_SECRET` | via Secrets Manager | secrets applicatifs (prod échoue si manquants) |
| `DATABASE_URL` | DSN RDS asyncpg | `postgresql+asyncpg://…` |
| `REDIS_URL` | DSN ElastiCache | analytics + sessions (voir §5) |
| `SAHTEN_EXPOSE_OPENAPI` | non défini / `false` | `/docs` masqué en prod |

## 2. Secrets → AWS Secrets Manager / SSM (jamais en clair)
- Stocker chaque secret dans **Secrets Manager** (ou SSM Parameter Store `SecureString`).
- ECS : injecter via `secrets` dans la task definition (mappe le secret → variable d'env). NE PAS mettre les valeurs dans `environment`.
- Rotation : activer la rotation Secrets Manager où c'est possible ; sinon rotation manuelle documentée.
- IAM : la task role ne doit lire QUE ses propres secrets (`secretsmanager:GetSecretValue` scoping par ARN).
- **Ne jamais** committer `.env` (déjà gitignoré ; `.env.example` = placeholders vides).

## 3. Réseau & proxy (point critique rate-limit)
- **L'app ne doit être joignable QUE via l'ALB** (security group : ingress 8000 depuis le SG de l'ALB uniquement, jamais 0.0.0.0/0). Sinon `X-Forwarded-For` est spoofable en tapant l'app en direct.
- Régler `SAHTEN_TRUSTED_PROXY_HOPS` selon la topologie :
  - **ALB seul** → `1` (le client est le dernier IP de XFF).
  - **CloudFront → ALB** → `2` (le dernier IP = edge CloudFront partagé ; le client est l'avant-dernier). Avec `1`, tous les visiteurs tombent dans le même quota de rate-limit.
- Vérifier après déploiement : logs `real_ip_key` doivent montrer des IPs clients variés, pas une seule IP d'edge.

## 4. TLS / en-têtes
- TLS terminé à l'ALB/CloudFront (ACM). Rediriger 80→443.
- HSTS est déjà émis par l'app en `APP_ENV in (staging, production)`.
- CSP `frame-ancestors` autorise déjà `*.lorientlejour.com` (embed iframe). Si le site est servi depuis un autre domaine, l'ajouter dans `security_middleware.py`.

## 5. Données & état
- **Postgres = RDS** avec l'extension `pgvector` (`CREATE EXTENSION vector;`). Chiffrement au repos (KMS) + in-transit (SSL, `sslmode=require`).
- **Redis = ElastiCache PERSISTANT** : les compteurs d'analytics et les sessions y vivent. Sans persistance, les stats se réinitialisent à chaque redéploiement. Activer AUTH + chiffrement in-transit.
- Migrations : l'entrypoint lance `alembic upgrade head` au boot (idempotent). S'assurer que la task a les droits DB.

## 6. Conteneur & image
- Image déjà **non-root** (uid 10001). Ne pas remonter en root.
- Activer **ECR image scanning** (scan-on-push) + `pip-audit` en CI pour les CVE de dépendances.
- Healthchecks : `GET /healthz` (liveness), `GET /readyz` (readiness, vérifie DB/Redis).

## 7. Observabilité
- Logs structurés JSON (structlog) → **CloudWatch Logs**. Chaque requête a un `X-Request-ID`.
- Métier : `/dashboard` (KPIs, coût $, répartition par `answer_strategy`), `/history`, `/admin` — **derrière le jeton admin**. Prévoir soit le jeton, soit une auth réseau (VPN/IP allowlist) devant ces routes.
- Alerte conseillée : 5xx rate, latence p95 du RAG, échecs webhook (signature), pics de rate-limit 429.

## 8. Avant go-live — actions manuelles
- [ ] **Régénérer TOUTES les clés/tokens** qui ont pu transiter en clair (OpenAI, Cohere, OLJ, `SAHTEN_ADMIN_API_TOKEN`, tout autre fournisseur). Voir la procédure de rotation.
- [ ] Fixer `SAHTEN_TRUSTED_PROXY_HOPS` selon ALB seul (1) ou CloudFront+ALB (2).
- [ ] Confirmer le SG : app injoignable hors ALB.
- [ ] Redis persistant + chiffré, RDS chiffré + pgvector.
- [ ] `APP_ENV=production` + `SAHTEN_CORS_ORIGINS` = domaine(s) OLJ (le boot valide).
- [ ] Test de fumée : 10 requêtes types + vérifier /dashboard qui se remplit.
