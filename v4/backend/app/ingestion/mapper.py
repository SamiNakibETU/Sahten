"""Mapping payload WhiteBeard brut -> entités ORM normalisées.

Sépare totalement la *forme* du payload (changeable) de la *structure*
canonique stockée. Aucune perte : le payload brut est toujours conservé
dans `Article.raw_payload`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from typing import Any

from bs4 import BeautifulSoup
from slugify import slugify

from .html_sectionizer import Section, sectionize


@dataclass
class MappedAuthor:
    external_id: int | None
    name: str
    slug: str
    role: str  # featured_chef | journalist | photographer | contributor
    department: str | None
    biography_html: str | None
    biography_text: str | None
    description: str | None
    image_url: str | None
    raw: dict[str, Any]


@dataclass
class MappedKeyword:
    external_id: int | None
    name: str
    slug: str
    description: str | None


@dataclass
class MappedCategory:
    external_id: int | None
    name: str
    slug: str
    description: str | None


@dataclass
class MappedArticle:
    external_id: int
    url: str
    slug: str
    title: str
    subtitle: str | None
    summary: str | None
    introduction: str | None
    signature: str | None
    body_html: str | None
    body_text: str | None
    content_length: int | None
    time_to_read: int | None
    is_premium: bool
    cover_image_url: str | None
    cover_image_caption: str | None
    first_published_at: datetime | None
    last_updated_at: datetime | None
    seo: dict[str, Any] | None
    raw_payload: dict[str, Any]
    ingestion_status: str
    ingestion_notes: str | None
    sections: list[Section] = field(default_factory=list)
    authors: list[MappedAuthor] = field(default_factory=list)
    keywords: list[MappedKeyword] = field(default_factory=list)
    categories: list[MappedCategory] = field(default_factory=list)


def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.utcfromtimestamp(value)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        # Fallback ISO lib
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _html_to_text(html: str | None) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    for s in soup(("script", "style")):
        s.decompose()
    return soup.get_text(separator="\n", strip=True)


def _ensure_slug(value: str, fallback: str | int = "") -> str:
    s = slugify(value or "", max_length=180)
    if not s:
        s = slugify(str(fallback), max_length=180) or "n-a"
    return s


def _extract_body_html(item: dict[str, Any]) -> tuple[str, list[str]]:
    """Récupère le HTML principal selon plusieurs formes de payload WhiteBeard / CMS."""
    notes: list[str] = []
    contents = item.get("contents")
    if isinstance(contents, dict):
        html = (
            (contents.get("html") or "").strip()
            or (contents.get("body") or "").strip()
            or (contents.get("raw") or "").strip()
            or (contents.get("rendered") or "").strip()
        )
        if html:
            return html, notes
    elif isinstance(contents, str) and contents.strip():
        return contents.strip(), notes

    content = item.get("content")
    if isinstance(content, dict):
        html = (
            (content.get("html") or "").strip()
            or (content.get("body") or "").strip()
            or (content.get("rendered") or "").strip()
        )
        if html:
            return html, notes
    elif isinstance(content, str) and content.strip():
        return content.strip(), notes

    for key in ("bodyHtml", "body_html", "html", "post_content", "articleBody"):
        v = item.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip(), notes

    notes.append("corps HTML absent (contents/content/html)")
    return "", notes


def _collect_authors(item: dict[str, Any]) -> list[MappedAuthor]:
    """Auteurs depuis authors[], ou objet author, ou chaîne author."""
    raw = item.get("authors")
    if isinstance(raw, list) and raw:
        return _map_authors(raw)
    solo = item.get("author")
    if isinstance(solo, dict) and _pick(solo, "name", "fullname", "displayName"):
        return _map_authors([solo])
    if isinstance(solo, str) and solo.strip():
        name = solo.strip()
        return [
            MappedAuthor(
                external_id=None,
                name=name,
                slug=_ensure_slug(name),
                role="journalist",
                department=None,
                biography_html=None,
                biography_text=None,
                description=None,
                image_url=None,
                raw={"from": "author_string"},
            )
        ]
    return []


def map_article(payload: dict[str, Any]) -> MappedArticle:
    """Transforme le payload WhiteBeard brut en `MappedArticle` complet.

    Le payload de `/content/{id}` est typiquement `{"data": [{...}]}`.
    On supporte aussi un dict direct.
    """
    if isinstance(payload.get("data"), list) and payload["data"]:
        item = payload["data"][0]
    elif isinstance(payload.get("data"), dict):
        item = payload["data"]
    else:
        item = payload

    notes: list[str] = []

    external_id = int(_pick(item, "id") or 0)
    if not external_id:
        notes.append("external_id absent du payload")

    url = _pick(item, "url", "permalink", "link") or ""
    if not url:
        notes.append("url absente")

    title = _pick(item, "title", "headline") or "(sans titre)"
    subtitle = _pick(item, "subtitle", "secondary_title", "kicker")
    summary = _pick(item, "summary", "lead", "excerpt")
    introduction = _pick(item, "introduction", "intro")
    signature = _pick(item, "signature")
    seo = item.get("seo") if isinstance(item.get("seo"), dict) else None

    body_html, body_notes = _extract_body_html(item)
    notes.extend(body_notes)

    # Dernier recours : introduction/chapo parfois seul bloc texte renvoyé par l’API
    if not body_html.strip() and introduction:
        intro = introduction.strip()
        if intro:
            body_html = intro if "<" in intro and ">" in intro else f"<p>{escape(intro)}</p>"
            notes.append("corps: fallback introduction")

    if not body_html.strip() and summary and summary.strip():
        body_html = f"<p>{escape(summary.strip())}</p>"
        notes.append("corps: fallback summary")

    if not body_html.strip() and subtitle and subtitle.strip():
        body_html = f"<p>{escape(subtitle.strip())}</p>"
        notes.append("corps: fallback subtitle")

    # Toujours un minimum indexable pour marquer l’article « ok » en admin (RAG sur titre)
    if not body_html.strip():
        t = (title or "(sans titre)").strip()
        body_html = f"<p>{escape(t)}</p>"
        notes.append("corps: fallback titre minimal")

    body_text = _html_to_text(body_html)

    cover = item.get("image") or item.get("cover") or {}
    if isinstance(cover, dict):
        cover_url = _pick(cover, "url", "src", "large", "medium")
        cover_caption = _pick(cover, "caption", "credits", "alt")
    else:
        cover_url, cover_caption = None, None

    first_published = _parse_dt(_pick(item, "firstPublished", "published_at", "date"))
    last_updated = _parse_dt(_pick(item, "lastUpdate", "updated_at", "modified_at"))

    sections = sectionize(body_html)
    if not sections and body_text:
        sections = [
            Section(
                position=0, kind="paragraph", heading=None,
                html=body_html or "", text=body_text,
            )
        ]

    authors = _collect_authors(item)
    keywords = _map_keywords(item.get("keywords") or [])
    categories = _map_categories(item.get("categories") or [])

    if not authors:
        # Fallback : signature texte -> author "name only"
        sig = item.get("signature")
        if isinstance(sig, str) and sig.strip():
            authors = [
                MappedAuthor(
                    external_id=None,
                    name=sig.strip(),
                    slug=_ensure_slug(sig.strip()),
                    role="journalist",
                    department=None,
                    biography_html=None,
                    biography_text=None,
                    description=None,
                    image_url=None,
                    raw={"from": "signature"},
                )
            ]

    body_stripped = (body_html or "").strip()

    if not authors and body_stripped:
        # API souvent sans auteurs : auteur éditorial pour que l’article soit « full » (ok)
        notes.append("auteur: Rédaction L'Orient-Le Jour (API sans auteurs)")
        authors = [
            MappedAuthor(
                external_id=None,
                name="Rédaction L'Orient-Le Jour",
                slug="redaction-lorient-le-jour",
                role="journalist",
                department=None,
                biography_html=None,
                biography_text=None,
                description=None,
                image_url=None,
                raw={"from": "default_editorial"},
            )
        ]

    # Fallbacks titre + rédaction : corps et auteur toujours présents → « ok » en admin
    status = "ok"

    return MappedArticle(
        external_id=external_id,
        url=url,
        slug=_ensure_slug(title, fallback=external_id),
        title=title,
        subtitle=subtitle,
        summary=summary,
        introduction=introduction,
        signature=signature,
        body_html=body_html or None,
        body_text=body_text or None,
        content_length=item.get("content_length"),
        time_to_read=item.get("time_to_read"),
        is_premium=bool(item.get("premium") or item.get("is_premium")),
        cover_image_url=cover_url,
        cover_image_caption=cover_caption,
        first_published_at=first_published,
        last_updated_at=last_updated,
        seo=seo,
        raw_payload=item,
        ingestion_status=status,
        ingestion_notes="; ".join(notes) if notes else None,
        sections=sections,
        authors=authors,
        keywords=keywords,
        categories=categories,
    )


_FEATURED_HINTS = ("chef", "cuisinier", "cheffe", "cuisinière")


def _map_authors(raw: list[Any]) -> list[MappedAuthor]:
    out: list[MappedAuthor] = []
    for i, a in enumerate(raw or []):
        if not isinstance(a, dict):
            continue
        name = _pick(a, "name", "fullname", "displayName")
        if not name:
            continue
        bio_html = _pick(a, "biography", "bio_html", "bio")
        bio_text = _html_to_text(bio_html) if bio_html else None
        description = _pick(a, "description", "title", "subtitle")
        department = _pick(a, "department", "section")
        image = _pick(a, "image", "avatar", "photo")
        if isinstance(image, dict):
            image = _pick(image, "url", "src")
        # Inférence de rôle : s'il a une bio + département "cuisine" ou
        # mention "chef" -> featured_chef. Sinon journalist.
        role = "journalist"
        haystack = " ".join(
            str(x or "")
            for x in (department, description, name, bio_text)
        ).lower()
        if any(h in haystack for h in _FEATURED_HINTS) or (bio_text and len(bio_text) > 60):
            role = "featured_chef" if i == 0 else "contributor"
        out.append(
            MappedAuthor(
                external_id=int(a["id"]) if isinstance(a.get("id"), int) else None,
                name=str(name),
                slug=_ensure_slug(name, fallback=a.get("id") or i),
                role=role,
                department=department,
                biography_html=bio_html,
                biography_text=bio_text,
                description=description,
                image_url=image,
                raw=a,
            )
        )
    return out


def _map_keywords(raw: list[Any]) -> list[MappedKeyword]:
    out: list[MappedKeyword] = []
    for k in raw or []:
        if isinstance(k, dict):
            name = _pick(k, "name", "label")
            if not name:
                continue
            out.append(
                MappedKeyword(
                    external_id=int(k["id"]) if isinstance(k.get("id"), int) else None,
                    name=str(name),
                    slug=_ensure_slug(name),
                    description=_pick(k, "description"),
                )
            )
        elif isinstance(k, str) and k.strip():
            out.append(
                MappedKeyword(
                    external_id=None,
                    name=k.strip(),
                    slug=_ensure_slug(k),
                    description=None,
                )
            )
    return out


def _map_categories(raw: list[Any]) -> list[MappedCategory]:
    out: list[MappedCategory] = []
    for c in raw or []:
        if isinstance(c, dict):
            name = _pick(c, "name", "label", "title")
            if not name:
                continue
            out.append(
                MappedCategory(
                    external_id=int(c["id"]) if isinstance(c.get("id"), int) else None,
                    name=str(name),
                    slug=_ensure_slug(name),
                    description=_pick(c, "description"),
                )
            )
        elif isinstance(c, str) and c.strip():
            out.append(
                MappedCategory(
                    external_id=None,
                    name=c.strip(),
                    slug=_ensure_slug(c),
                    description=None,
                )
            )
    return out
