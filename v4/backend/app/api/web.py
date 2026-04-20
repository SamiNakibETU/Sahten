"""Routes HTML : sert le frontend statique v3 et les pages admin/history.

Le dossier `v4/web_static/` contient le widget chat (`index.html`, `js/`,
`css/`, `assets/`) plus les pages secondaires (`admin.html`, `history.html`,
`widget.html`, `dashboard.html`, `demo-olj.html`).

On expose :
- `/`              → `index.html` (widget chat OLJ)
- `/admin`         → `admin.html` (navigation DB)
- `/history`       → `history.html` (sessions enregistrées)
- `/widget.html`   → version embeddable iframe
- `/static/*`      → CSS/JS/assets statiques

L'ordre de mount est important dans `main.py` : tous les routers `/api/*`
DOIVENT être enregistrés AVANT le mount `/static`, sinon FastAPI shadow.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response

router = APIRouter(tags=["web"])

# v4/backend/app/api/web.py → v4/web_static
WEB_STATIC_DIR = Path(__file__).resolve().parents[3] / "web_static"


def _serve(path: str) -> FileResponse:
    full = WEB_STATIC_DIR / path
    if not full.is_file():
        raise HTTPException(status_code=404, detail=f"page introuvable: {path}")
    return FileResponse(full)


@router.get("/", include_in_schema=False)
def home() -> Response:
    return _serve("index.html")


@router.get("/admin", include_in_schema=False)
def admin_page() -> Response:
    return _serve("admin.html")


@router.get("/history", include_in_schema=False)
def history_page() -> Response:
    return _serve("history.html")


@router.get("/widget", include_in_schema=False)
def widget_page() -> Response:
    return _serve("widget.html")


@router.get("/dashboard", include_in_schema=False)
def dashboard_page() -> Response:
    return _serve("dashboard.html")


@router.get("/demo", include_in_schema=False)
def demo_page() -> Response:
    return _serve("demo-olj.html")


@router.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    fav = WEB_STATIC_DIR / "assets" / "sahten_logo_v4.svg"
    if fav.is_file():
        return FileResponse(fav, media_type="image/svg+xml")
    return RedirectResponse(url="/static/assets/sahten_logo_v4.svg", status_code=302)


# Compat : index.html charge `./js/sahten.js` et `./css/sahten.css`
# (chemins relatifs). On expose donc aussi /js/* /css/* /assets/* /img/*
@router.get("/js/{path:path}", include_in_schema=False)
def js_passthrough(path: str) -> Response:
    return _serve(f"js/{path}")


@router.get("/css/{path:path}", include_in_schema=False)
def css_passthrough(path: str) -> Response:
    return _serve(f"css/{path}")


@router.get("/assets/{path:path}", include_in_schema=False)
def assets_passthrough(path: str) -> Response:
    return _serve(f"assets/{path}")


@router.get("/img/{path:path}", include_in_schema=False)
def img_passthrough(path: str) -> Response:
    return _serve(f"img/{path}")
