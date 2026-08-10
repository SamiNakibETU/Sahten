#!/bin/bash
# Auto-déploiement gardé : aligne le serveur sur origin/main dès qu'un commit
# arrive, et RESTAURE la version précédente si /readyz casse après redémarrage.
# Répond au besoin « le site évolue tout seul depuis GitHub » sans mettre la
# prod à la merci d'un commit fautif.
#
# Lancé par sahten-deploy.timer (toutes les ~5 min). Idempotent : si rien de
# nouveau sur origin/main, sort en une seconde sans rien toucher.
set -euo pipefail

REPO="/home/ec2-user/sahten"
BRANCH="main"
SERVICE="sahten"
HEALTH="http://localhost/readyz"
LOG="/home/ec2-user/auto_deploy.log"

log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

cd "$REPO"
git fetch --quiet origin "$BRANCH"

local_sha="$(git rev-parse HEAD)"
remote_sha="$(git rev-parse "origin/$BRANCH")"

if [ "$local_sha" = "$remote_sha" ]; then
    exit 0  # rien de neuf
fi

log "nouveau commit $remote_sha (depuis $local_sha) — déploiement"
prev_sha="$local_sha"

# Applique la nouvelle version (working tree propre garanti côté serveur).
git reset --hard "origin/$BRANCH" --quiet

# Réinstalle les dépendances seulement si pyproject a changé (rare, évite ~1 min).
if ! git diff --quiet "$prev_sha" "$remote_sha" -- v4/pyproject.toml; then
    log "pyproject modifié — pip install"
    (cd v4 && .venv/bin/pip install -q -e . 2>>"$LOG") || log "pip install a échoué (on continue)"
fi

sudo systemctl restart "$SERVICE"

# Garde-fou : /readyz doit répondre 200 dans les 40 s, sinon rollback.
ok=""
for _ in $(seq 1 20); do
    sleep 2
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "$HEALTH" 2>/dev/null || echo 000)"
    if [ "$code" = "200" ]; then ok=1; break; fi
done

if [ -n "$ok" ]; then
    log "OK — $SERVICE sain sur $remote_sha"
else
    log "ÉCHEC /readyz après $remote_sha — ROLLBACK vers $prev_sha"
    git reset --hard "$prev_sha" --quiet
    sudo systemctl restart "$SERVICE"
    log "rollback appliqué"
    exit 1
fi
