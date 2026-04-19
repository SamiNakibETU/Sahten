#!/bin/sh
set -e

# Auto-migration : s'exécute à chaque déploiement Railway.
# Si DATABASE_URL est absent (ex. tout premier boot), on log et on continue
# pour ne pas bloquer /healthz.
if [ -n "$DATABASE_URL" ]; then
  echo "[entrypoint] alembic upgrade head ..."
  cd /app
  alembic -c infra/alembic/alembic.ini upgrade head || {
    echo "[entrypoint] alembic a échoué — l'app démarre quand même pour /healthz"
  }
else
  echo "[entrypoint] DATABASE_URL absent — skip alembic"
fi

PORT="${PORT:-8000}"
echo "[entrypoint] uvicorn main:app --host 0.0.0.0 --port ${PORT}"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}" --proxy-headers
