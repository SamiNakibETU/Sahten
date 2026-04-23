"""Audit du payload WhiteBeard pour la rubrique « Liban à Table » (recettes).

Objectif : déterminer **précisément** ce que renvoie l'API WhiteBeard pour
chaque recette, et savoir si le pipeline d'ingestion peut tout récupérer
sans avoir à scraper le site public.

Pour chaque article auditté :
    1. Sauvegarde du payload brut dans ``--out/<id>.json``
    2. Analyse :
        - taille du HTML principal (``contents.html``),
        - présence des champs périphériques (``aside``, ``sidebar``,
          ``recipe_details``, ``authors[].biography``…),
        - mots-clés sémantiques attendus (« commandement », « ingrédient »,
          « préparation », « bio »…) et leur localisation (corps vs annexes).
    3. Génération de ``--report`` (Markdown) avec un verdict global.

Usage local (depuis V3/sahten_github/v4) ::

    .venv/Scripts/Activate.ps1
    python scripts/audit_whitebeard.py --limit 30 --out tests/fixtures/audit \\
        --report tests/fixtures/audit/audit_report.md

Aucune dépendance Postgres : on tape juste l'API HTTP. Le script lit
``OLJ_API_BASE`` et ``OLJ_API_KEY`` depuis l'environnement, ou depuis
``.env`` (à la racine ``V3/`` ou dans ``v4/``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup

PUBLICATION_ID = 17
CONTENT_TYPE_RECIPES = 4

EXPECTED_KEYWORDS = {
    "commandement": ("commandement", "commandements"),
    "ingredients": ("ingrédient", "ingredients", "ingrédients"),
    "preparation": ("préparation", "preparation", "étapes", "etapes"),
    "biography": ("bio ", "biographie", "biography"),
    "chef": ("chef",),
}

SIDEBAR_HINTS = (
    "recipe_details",
    "recipe-details",
    "sidebar",
    "aside",
    "biography",
    "chef",
    "encadre",
    "encadré",
)


def _load_dotenv_files() -> None:
    """Lit ``.env`` à la racine V3 puis dans v4 (sans écraser env existant)."""
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / ".env",
        here.parents[1] / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            for raw in path.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {path}: {exc}", file=sys.stderr)


@dataclass
class ArticleAudit:
    article_id: int
    title: str
    url: str | None
    raw_size_kb: float
    sections_top_level: list[str]
    contents_len: int                # string HTML — étapes de préparation
    summary_len: int                 # string HTML — ingrédients (souvent <ul><li>)
    introduction_len: int
    description_len: int
    astuce_len: int
    recipe_history_title_len: int
    chef_present: bool
    chef_name: str
    chef_contents_len: int           # bio chef HTML
    chef_summary_len: int
    chef_introduction_len: int
    keywords: list[str]
    categories: list[str]
    n_attachments: int
    has_recipe_meta: bool            # preparation_time / cooking_time / persons / difficulty
    full_html_len: int               # concat de tous les champs HTML pour parsing global
    full_text_len: int
    keyword_hits: dict[str, int]
    notes: list[str] = field(default_factory=list)


def _flatten_keys(node: Any, prefix: str = "") -> list[str]:
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            full = f"{prefix}.{k}" if prefix else k
            out.append(full)
            out.extend(_flatten_keys(v, full))
    elif isinstance(node, list):
        for v in node[:3]:
            out.extend(_flatten_keys(v, prefix + "[]"))
    return out


def _extract_first(payload: dict[str, Any]) -> dict[str, Any]:
    """L'API retourne `{"data": [article]}` ; tolère aussi `data: article`."""
    data = payload.get("data")
    if isinstance(data, list) and data:
        return data[0] if isinstance(data[0], dict) else {}
    if isinstance(data, dict):
        return data
    return payload if isinstance(payload, dict) else {}


def _gather_html(article: dict[str, Any]) -> tuple[str, list[str]]:
    """Concatène toutes les chaînes HTML trouvées et liste leurs chemins."""
    html_blobs: list[str] = []
    paths: list[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str):
            stripped = node.strip()
            if "<" in stripped and ">" in stripped and len(stripped) > 80:
                html_blobs.append(stripped)
                paths.append(path)

    walk(article, "")
    return "\n".join(html_blobs), paths


def _str_html(value: Any) -> str:
    """Coerce a JSON value into an HTML/text string. Handles None/list/dict gracefully."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_str_html(v) for v in value)
    if isinstance(value, dict):
        for k in ("html", "content", "value", "text"):
            if k in value:
                return _str_html(value[k])
        return ""
    return str(value)


def _audit_article(article_id: int, payload: dict[str, Any]) -> ArticleAudit:
    article = _extract_first(payload)
    raw_size_kb = round(len(json.dumps(payload, ensure_ascii=False)) / 1024, 1)

    title = (
        article.get("title")
        or article.get("recipe_name")
        or article.get("name")
        or ""
    ).strip()
    url = article.get("url") or article.get("slug")

    contents_html = _str_html(article.get("contents"))
    summary_html = _str_html(article.get("summary"))
    introduction_html = _str_html(article.get("introduction"))
    description_html = _str_html(article.get("description"))
    astuce_html = _str_html(article.get("astuce"))
    history_title = _str_html(article.get("recipe_history_title"))
    history_html = _str_html(article.get("recipe_history"))

    chef = article.get("chef") or {}
    if not isinstance(chef, dict):
        chef = {}
    chef_present = bool(chef.get("id"))
    chef_name = (chef.get("title") or chef.get("name") or "").strip()
    chef_contents = _str_html(chef.get("contents"))
    chef_summary = _str_html(chef.get("summary"))
    chef_introduction = _str_html(chef.get("introduction"))

    full_html = "\n".join(
        s for s in [
            contents_html, summary_html, introduction_html, description_html,
            astuce_html, history_title, history_html,
            chef_contents, chef_summary, chef_introduction,
        ] if s
    )
    soup = BeautifulSoup(full_html, "lxml") if full_html.strip() else None
    full_text = soup.get_text(" ", strip=True) if soup else ""

    haystack = (full_html + " " + full_text).lower()
    keyword_hits = {
        bucket: sum(haystack.count(kw) for kw in kws)
        for bucket, kws in EXPECTED_KEYWORDS.items()
    }

    keywords_raw = article.get("keywords") or []
    keywords: list[str] = []
    if isinstance(keywords_raw, list):
        for k in keywords_raw:
            if isinstance(k, dict):
                d = k.get("data") or k
                name = (d.get("display_name") or d.get("name") or "").strip()
                if name:
                    keywords.append(name)

    categories_raw = article.get("categories") or []
    categories: list[str] = []
    if isinstance(categories_raw, list):
        for c in categories_raw:
            if isinstance(c, dict):
                name = (c.get("name") or "").strip()
                if name:
                    categories.append(name)

    attachments = article.get("attachments") or []
    n_attachments = len(attachments) if isinstance(attachments, list) else 0

    has_recipe_meta = any(
        article.get(k) not in (None, "", 0)
        for k in ("preparation_time", "cooking_time", "persons", "difficulty")
    )

    sections_top_level = sorted({k for k in article.keys()})
    notes: list[str] = []
    if not contents_html:
        notes.append("`contents` (étapes) vide")
    if not summary_html:
        notes.append("`summary` (ingrédients) vide")
    if chef_present and not chef_contents:
        notes.append("chef présent mais `chef.contents` (bio) vide")

    return ArticleAudit(
        article_id=article_id,
        title=title or "(sans titre)",
        url=url,
        raw_size_kb=raw_size_kb,
        sections_top_level=sections_top_level,
        contents_len=len(contents_html),
        summary_len=len(summary_html),
        introduction_len=len(introduction_html),
        description_len=len(description_html),
        astuce_len=len(astuce_html),
        recipe_history_title_len=len(history_title) + len(history_html),
        chef_present=chef_present,
        chef_name=chef_name,
        chef_contents_len=len(chef_contents),
        chef_summary_len=len(chef_summary),
        chef_introduction_len=len(chef_introduction),
        keywords=keywords,
        categories=categories,
        n_attachments=n_attachments,
        has_recipe_meta=has_recipe_meta,
        full_html_len=len(full_html),
        full_text_len=len(full_text),
        keyword_hits=keyword_hits,
        notes=notes,
    )


async def _list_recipe_ids(
    client: httpx.AsyncClient, *, limit: int, offset: int
) -> tuple[list[int], dict[str, Any]]:
    """Liste des IDs de recettes via /publication/{id}/content?content_type=4."""
    r = await client.get(
        f"/publication/{PUBLICATION_ID}/content",
        params={
            "content_type": CONTENT_TYPE_RECIPES,
            "limit": limit,
            "offset": offset,
        },
    )
    if r.status_code >= 400:
        raise RuntimeError(f"LIST failed HTTP {r.status_code}: {r.text[:500]}")
    payload = r.json()
    items = payload.get("data") or payload.get("items") or []
    if isinstance(items, dict):
        items = list(items.values())
    ids: list[int] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        for k in ("id", "article_id", "content_id", "external_id"):
            v = it.get(k)
            if v is None:
                continue
            try:
                ids.append(int(v))
                break
            except (TypeError, ValueError):
                continue
    return ids, payload


async def _fetch_payload(
    client: httpx.AsyncClient, article_id: int
) -> dict[str, Any] | None:
    try:
        r = await client.get(f"/content/{article_id}")
    except Exception as exc:  # noqa: BLE001
        print(f"[err] {article_id}: {exc}", file=sys.stderr)
        return None
    if r.status_code >= 400:
        print(f"[err] {article_id}: HTTP {r.status_code}: {r.text[:300]}", file=sys.stderr)
        return None
    try:
        return r.json()
    except json.JSONDecodeError as exc:
        print(f"[err] {article_id}: JSON decode failed: {exc}", file=sys.stderr)
        return None


async def _fetch_one(
    client: httpx.AsyncClient,
    article_id: int,
    out_dir: Path,
    *,
    fetch_chef_bio: bool = True,
) -> ArticleAudit | None:
    payload = await _fetch_payload(client, article_id)
    if payload is None:
        return None
    out_path = out_dir / f"{article_id}.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 2nd-pass : si la recette référence un chef sans bio, on tente de
    # récupérer la fiche chef complète (`GET /content/{chef_id}`).
    if fetch_chef_bio:
        article = _extract_first(payload)
        chef = article.get("chef") or {}
        chef_id = chef.get("id") if isinstance(chef, dict) else None
        if chef_id:
            try:
                chef_id_int = int(chef_id)
            except (TypeError, ValueError):
                chef_id_int = None
            if chef_id_int and not _str_html(chef.get("contents")):
                chef_payload = await _fetch_payload(client, chef_id_int)
                if chef_payload is not None:
                    chef_full = _extract_first(chef_payload)
                    if chef_full:
                        article["chef"] = chef_full
                        # On sauvegarde aussi la fiche chef pour traçabilité.
                        chef_out = out_dir / f"chef_{chef_id_int}.json"
                        chef_out.write_text(
                            json.dumps(chef_payload, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )

    return _audit_article(article_id, payload)


def _verdict(audits: list[ArticleAudit]) -> dict[str, Any]:
    if not audits:
        return {"verdict": "no_data", "messages": ["Aucun article auditté avec succès."]}
    n = len(audits)
    full_lengths = sorted(a.full_html_len for a in audits)
    median_full = full_lengths[n // 2]
    pct_contents = sum(1 for a in audits if a.contents_len > 0) / n
    pct_summary = sum(1 for a in audits if a.summary_len > 0) / n
    pct_chef = sum(1 for a in audits if a.chef_present) / n
    pct_chef_bio = sum(1 for a in audits if a.chef_contents_len > 0) / n
    pct_recipe_meta = sum(1 for a in audits if a.has_recipe_meta) / n
    pct_astuce = sum(1 for a in audits if a.astuce_len > 0) / n
    pct_history = sum(1 for a in audits if a.recipe_history_title_len > 0) / n
    pct_commandements = sum(1 for a in audits if a.keyword_hits.get("commandement", 0) > 0) / n
    pct_ingredients = sum(1 for a in audits if a.keyword_hits.get("ingredients", 0) > 0) / n
    pct_preparation = sum(1 for a in audits if a.keyword_hits.get("preparation", 0) > 0) / n

    def lvl(p: float) -> str:
        if p >= 0.9:
            return "presque toujours"
        if p >= 0.5:
            return "souvent"
        if p > 0:
            return "parfois"
        return "jamais"

    msgs: list[str] = []
    if pct_contents >= 0.8 and pct_summary >= 0.6 and pct_chef >= 0.6 and pct_chef_bio >= 0.5:
        verdict_status = "ok_api_complete"
        msgs.append("[OK] L'API renvoie tout l'essentiel : etapes (contents), ingredients (summary), metadonnees recette et biographies chef.")
    elif pct_contents >= 0.5 and pct_summary >= 0.5:
        verdict_status = "ok_with_minor_gaps"
        msgs.append("[~OK] L'API renvoie le coeur (etapes + ingredients) mais certains champs annexes manquent partiellement.")
    else:
        verdict_status = "needs_enrichment"
        msgs.append("[KO] L'API a des trous structurels - il faut ecrire a WhiteBeard.")

    if pct_chef < 0.5:
        msgs.append(f"Champ `chef` présent {lvl(pct_chef)} ({pct_chef:.0%}) — à demander si systématique attendu.")
    if pct_chef_bio < 0.5:
        msgs.append(f"`chef.contents` (bio) {lvl(pct_chef_bio)} ({pct_chef_bio:.0%}).")
    if pct_commandements > 0:
        msgs.append(f"Mot « commandement » trouvé {lvl(pct_commandements)} ({pct_commandements:.0%}).")
    return {
        "verdict": verdict_status,
        "n_articles": n,
        "median_full_html_chars": median_full,
        "pct_contents_present": pct_contents,
        "pct_summary_present": pct_summary,
        "pct_chef_present": pct_chef,
        "pct_chef_bio_present": pct_chef_bio,
        "pct_recipe_meta_present": pct_recipe_meta,
        "pct_astuce_present": pct_astuce,
        "pct_history_present": pct_history,
        "pct_keyword_commandement": pct_commandements,
        "pct_keyword_ingredients": pct_ingredients,
        "pct_keyword_preparation": pct_preparation,
        "messages": msgs,
    }


def _render_report(audits: list[ArticleAudit], verdict: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Audit WhiteBeard — payload « Liban à Table »\n")
    lines.append(f"**Articles audités** : {verdict.get('n_articles', 0)}")
    lines.append(f"**Verdict** : `{verdict.get('verdict')}`\n")
    for m in verdict.get("messages", []):
        lines.append(f"- {m}")
    lines.append("\n## Indicateurs globaux\n")
    keys = [
        ("median_full_html_chars", "Taille médiane HTML total (chars)"),
        ("pct_contents_present", "% articles avec `contents` (étapes)"),
        ("pct_summary_present", "% articles avec `summary` (ingrédients)"),
        ("pct_chef_present", "% articles avec champ `chef` peuplé"),
        ("pct_chef_bio_present", "% articles avec `chef.contents` (bio chef)"),
        ("pct_recipe_meta_present", "% articles avec metadata recette (temps/persos/difficulté)"),
        ("pct_astuce_present", "% articles avec `astuce` chef"),
        ("pct_history_present", "% articles avec `recipe_history` / titre"),
        ("pct_keyword_commandement", "% articles contenant « commandement »"),
        ("pct_keyword_ingredients", "% articles contenant « ingrédient »"),
        ("pct_keyword_preparation", "% articles contenant « préparation »"),
    ]
    for k, label in keys:
        v = verdict.get(k)
        if v is None:
            continue
        if isinstance(v, float) and 0 <= v <= 1 and "pct_" in k:
            lines.append(f"- {label} : **{v:.0%}**")
        else:
            lines.append(f"- {label} : **{v}**")

    field_counter: Counter[str] = Counter()
    for a in audits:
        field_counter.update(a.sections_top_level)
    lines.append("\n## Champs top-level les plus fréquents (sur l'objet article)\n")
    lines.append("| champ | présent dans |")
    lines.append("|---|---|")
    for f_, c in field_counter.most_common(30):
        lines.append(f"| `{f_}` | {c} |")

    lines.append("\n## Détail par article\n")
    lines.append("| ID | Titre | full_html | contents | summary | meta | chef | chef.bio | astuce | history | kw_cmd |")
    lines.append("|---|---|---:|---:|---:|:---:|:---:|---:|---:|---:|---:|")
    for a in audits:
        title = a.title.replace("|", "/")[:55]
        lines.append(
            f"| {a.article_id} | {title} | {a.full_html_len} | {a.contents_len} | {a.summary_len} | "
            f"{'✅' if a.has_recipe_meta else '❌'} | "
            f"{'✅' if a.chef_present else '❌'} | "
            f"{a.chef_contents_len} | {a.astuce_len} | {a.recipe_history_title_len} | "
            f"{a.keyword_hits.get('commandement', 0)} |"
        )

    lines.append("\n## Échantillons (10 premiers articles)\n")
    for a in audits[:10]:
        lines.append(f"\n### Article {a.article_id} — {a.title}")
        lines.append(f"- URL : `{a.url}`")
        lines.append(f"- Taille payload brut : {a.raw_size_kb} KB")
        lines.append(f"- Chef : `{a.chef_name}` (bio: {a.chef_contents_len} chars)")
        lines.append(f"- Keywords : {', '.join(a.keywords[:8]) or '—'}")
        lines.append(f"- Categories : {', '.join(a.categories) or '—'}")
        lines.append(f"- Pièces jointes (images) : {a.n_attachments}")
        lines.append(
            f"- Tailles HTML : contents={a.contents_len}, summary={a.summary_len}, "
            f"intro={a.introduction_len}, description={a.description_len}, "
            f"astuce={a.astuce_len}"
        )
        if a.notes:
            lines.append(f"- Notes : {'; '.join(a.notes)}")
    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> None:
    _load_dotenv_files()
    api_base = (os.environ.get("OLJ_API_BASE") or "").rstrip("/")
    api_key = os.environ.get("OLJ_API_KEY") or ""
    if not api_base or not api_key:
        print(
            "OLJ_API_BASE et OLJ_API_KEY doivent être définis dans l'environnement ou .env",
            file=sys.stderr,
        )
        sys.exit(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report) if args.report else out_dir / "audit_report.md"
    raw_list_path = out_dir / "_list_response.json"

    print(f"[audit] api_base={api_base}")
    print(f"[audit] out_dir={out_dir}")
    print(f"[audit] limit={args.limit} offset={args.offset}")

    headers = {"API-Key": api_key, "Accept": "application/json"}
    async with httpx.AsyncClient(
        base_url=api_base, headers=headers, timeout=30.0
    ) as client:
        if args.ids:
            ids = [int(x) for x in args.ids.split(",") if x.strip()]
            print(f"[audit] using explicit IDs: {ids}")
        else:
            ids, list_payload = await _list_recipe_ids(
                client, limit=args.limit, offset=args.offset
            )
            raw_list_path.write_text(
                json.dumps(list_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[audit] listed {len(ids)} IDs (raw saved to {raw_list_path})")
            if not ids:
                print(
                    "[audit] aucune ID trouvée. Réponse brute sauvegardée. Vérifie le format renvoyé.",
                    file=sys.stderr,
                )
                sys.exit(3)

        if args.include_ids:
            extras = [int(x) for x in args.include_ids.split(",") if x.strip()]
            for x in extras:
                if x not in ids:
                    ids.append(x)
            print(f"[audit] +{len(extras)} extra IDs included; total now {len(ids)}")

        audits: list[ArticleAudit] = []
        sem = asyncio.Semaphore(args.concurrency)

        async def worker(aid: int) -> None:
            async with sem:
                a = await _fetch_one(
                    client, aid, out_dir,
                    fetch_chef_bio=not args.no_chef_bio_fetch,
                )
                if a is not None:
                    audits.append(a)
                    print(
                        f"  [{len(audits)}/{len(ids)}] {aid}: full={a.full_html_len}c "
                        f"contents={a.contents_len} summary={a.summary_len} "
                        f"chef={'Y' if a.chef_present else 'N'}({a.chef_contents_len}) "
                        f"meta={'Y' if a.has_recipe_meta else 'N'} kw_cmd={a.keyword_hits.get('commandement', 0)}"
                    )

        await asyncio.gather(*(worker(i) for i in ids))

    audits.sort(key=lambda a: a.article_id)
    verdict = _verdict(audits)
    report = _render_report(audits, verdict)
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[audit] report written: {report_path}")
    print(f"[audit] verdict: {verdict.get('verdict')}")
    for m in verdict.get("messages", []):
        print(f"  - {m}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=30, help="Nb d'articles à lister via /publication/17/content")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--out", default="tests/fixtures/audit", help="Dossier de sortie pour les payloads bruts + rapport")
    p.add_argument("--report", default=None, help="Chemin du rapport Markdown (défaut : <out>/audit_report.md)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--ids", default=None, help="Liste explicite d'IDs séparés par des virgules (court-circuite --limit)")
    p.add_argument("--include-ids", default=None, help="IDs à ajouter en plus du listing (comma-separated)")
    p.add_argument("--no-chef-bio-fetch", action="store_true", help="Ne pas faire le 2e GET pour récupérer la bio chef")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
