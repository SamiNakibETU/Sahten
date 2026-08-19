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


_WEB_ROOT = WEB_STATIC_DIR.resolve()


def _serve(path: str) -> FileResponse:
    full = (WEB_STATIC_DIR / path).resolve()
    # Protection path traversal : le fichier résolu DOIT rester sous web_static/.
    if not str(full).startswith(str(_WEB_ROOT) + ("/" if str(_WEB_ROOT)[-1] != "/" else "")):
        raise HTTPException(status_code=404, detail="not found")
    if not full.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(full)


def _serve_fresh(name: str) -> Response:
    """Page d'outillage : toujours revalidee.

    Ces pages evoluent a chaque deploiement et sont consultees par une poignee
    de personnes : le cache navigateur n'y apporte rien et masque les
    corrections (constate le 20/08 — un correctif du tableau de bord etait en
    place sur le serveur mais le navigateur affichait encore l'ancienne page).
    """
    resp = _serve(name)
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@router.get("/", include_in_schema=False)
def home() -> Response:
    return _serve("index.html")


@router.get("/admin", include_in_schema=False)
def admin_page() -> Response:
    return _serve_fresh("admin.html")


@router.get("/history", include_in_schema=False)
def history_page() -> Response:
    return _serve_fresh("history.html")


@router.get("/widget", include_in_schema=False)
def widget_page() -> Response:
    # Surface auto-mise à jour : embarquée en iframe sur lorientlejour.com, elle
    # doit refléter le dernier déploiement. Sans revalidation, le navigateur
    # d'un visiteur garde une version périmée et l'auto-déploiement ne lui
    # parvient jamais (constaté : ancienne mise en page servie depuis le cache).
    resp = _serve("widget.html")
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@router.get("/embed.js", include_in_schema=False)
def embed_loader() -> Response:
    # Chargeur embarquable pour lorientlejour.com (une seule balise <script>).
    # C'est LE point d'entrée de l'intégration : il porte le lanceur, la taille
    # de l'iframe et l'URL du widget. Un `max-age=600` laissait Cloudflare le
    # mettre en cache (cf-cache-status: HIT) et servir une version de la veille,
    # donc l'auto-déploiement ne parvenait pas aux lecteurs. Revalidation
    # systématique, comme /widget : le fichier fait quelques kilo-octets, le
    # coût d'un 304 est négligeable devant une intégration figée.
    resp = _serve("embed.js")
    resp.media_type = "application/javascript"
    resp.headers["Content-Type"] = "application/javascript; charset=utf-8"
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    return resp


@router.get("/dashboard", include_in_schema=False)
def dashboard_page() -> Response:
    return _serve_fresh("dashboard.html")


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
