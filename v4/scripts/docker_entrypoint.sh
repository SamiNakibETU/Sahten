#!/bin/sh
set -e

# Auto-migration : s'exécute à chaque déploiement Railway.
# Si DATABASE_URL est absent (ex. tout premier boot), on log et on continue
# pour ne pas bloquer /healthz.
if [ -n "$DATABASE_URL" ]; then
  echo "[entrypoint] alembic upgrade head ..."
  # `script_location = .` dans alembic.ini est résolu vs CWD : on doit donc
  # se placer dans le dossier qui contient `env.py` et `alembic.ini`.
  cd /app/infra/alembic
  alembic upgrade head || {
    echo "[entrypoint] alembic a échoué — l'app démarre quand même pour /healthz"
  }
  cd /app
else
  echo "[entrypoint] DATABASE_URL absent — skip alembic"
fi

PORT="${PORT:-8000}"
echo "[entrypoint] uvicorn main:app --host 0.0.0.0 --port ${PORT}"
exec uvicorn main:app --host 0.0.0.0 --port "${PORT}" --proxy-headers
