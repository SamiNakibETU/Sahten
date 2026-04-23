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


def _coerce_html(value: Any) -> str:
    """Normalise une valeur JSON en chaîne HTML utilisable.

    Le payload WhiteBeard mélange parfois ``str`` et ``dict {html: ...}``
    pour le même champ selon le `contentType`. Cette fonction lisse la
    différence pour que tout le reste du mapping ne traite que des chaînes.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(_coerce_html(v) for v in value).strip()
    if isinstance(value, dict):
        for k in ("html", "body", "raw", "rendered", "content", "value", "text"):
            if k in value:
                return _coerce_html(value[k])
        return ""
    return str(value).strip()


def _wrap_section(kind: str, html: str, *, heading: str | None = None) -> str:
    """Encapsule un bloc HTML dans une section sémantique typée.

    Le sectionizer reconnaît ensuite ces ``<section data-kind="...">`` et
    les transforme directement en ``Section(kind=...)`` sans heuristique.
    Cela évite de re-deviner la structure côté chunker/RAG.
    """
    if not html or not html.strip():
        return ""
    head = f"<h3>{escape(heading)}</h3>" if heading else ""
    return f'<section data-kind="{kind}">{head}{html}</section>'


def _format_recipe_meta(item: dict[str, Any]) -> str:
    """Construit un bloc HTML « infos pratiques » à partir des metadata recette."""
    rows: list[tuple[str, str]] = []
    prep = item.get("preparation_time")
    cook = item.get("cooking_time")
    persons = item.get("persons")
    diff = item.get("difficulty")
    if prep:
        rows.append(("Préparation", f"{prep} min"))
    if cook:
        rows.append(("Cuisson", f"{cook} min"))
    if persons:
        rows.append(("Pour", f"{persons} personnes"))
    if diff:
        rows.append(("Difficulté", str(diff)))
    if not rows:
        return ""
    li = "".join(f"<li><strong>{escape(k)} :</strong> {escape(v)}</li>" for k, v in rows)
    return f"<ul>{li}</ul>"


def _format_chef_block(chef: dict[str, Any]) -> tuple[str, str]:
    """Renvoie ``(heading, html)`` à partir d'un objet chef enrichi.

    On combine le ``chef.title`` (souvent « Kamal Mouzawak, fondateur de … »)
    et la bio HTML (``chef.contents``). On ajoute un lien vers le portrait
    OLJ si ``chef.url`` est dispo, pour que le RAG puisse citer la source.
    """
    if not isinstance(chef, dict):
        return "", ""
    title = (chef.get("title") or chef.get("name") or "").strip()
    bio_html = _coerce_html(chef.get("contents"))
    intro = _coerce_html(chef.get("introduction"))
    summary = _coerce_html(chef.get("summary"))
    url = (chef.get("url") or "").strip()

    parts: list[str] = []
    if title:
        if url:
            parts.append(f'<p><strong><a href="{escape(url)}">{escape(title)}</a></strong></p>')
        else:
            parts.append(f"<p><strong>{escape(title)}</strong></p>")
    if intro:
        parts.append(intro)
    if summary:
        parts.append(summary)
    if bio_html:
        parts.append(bio_html)
    return f"Le chef : {title}" if title else "Le chef", "\n".join(parts).strip()


def _assemble_body_html(item: dict[str, Any]) -> tuple[str, list[str]]:
    """Concatène **toutes** les sections HTML utiles du payload WhiteBeard.

    Ordre choisi pour favoriser la lecture humaine (admin) ET le retrieval :
    metadata → résumé éditorial → histoire (commandements signature)
    → ingrédients → étapes → astuce du chef → bio chef.

    Le HTML produit reste valide et chaque bloc est balisé par un
    ``<section data-kind="…">`` exploité ensuite par ``html_sectionizer``.
    """
    notes: list[str] = []
    blocks: list[str] = []

    meta_html = _format_recipe_meta(item)
    if meta_html:
        blocks.append(_wrap_section("recipe_meta", meta_html, heading="Infos pratiques"))

    description = _coerce_html(item.get("description"))
    if description:
        # Dans le payload WhiteBeard, ``description`` est du texte brut court.
        if "<" not in description:
            description = f"<p>{escape(description)}</p>"
        blocks.append(_wrap_section("recipe_summary", description))

    history_title = (item.get("recipe_history_title") or "").strip()
    history_html = _coerce_html(item.get("recipe_history"))
    introduction = _coerce_html(item.get("introduction"))
    # Cas taboulé : "9 commandements" sont rangés dans `introduction`.
    # On les attache à `recipe_history` quand un titre d'histoire existe,
    # sinon on garde une section "intro" séparée.
    if history_title and not history_html and introduction:
        history_html = introduction
        introduction = ""
    if history_title or history_html:
        blocks.append(_wrap_section("recipe_history", history_html or "", heading=history_title or None))

    if introduction:
        blocks.append(_wrap_section("intro", introduction))

    summary_html = _coerce_html(item.get("summary"))
    if summary_html:
        if "<" not in summary_html:
            summary_html = f"<p>{escape(summary_html)}</p>"
        blocks.append(_wrap_section("ingredients_list", summary_html, heading="Ingrédients"))

    contents_html = _coerce_html(item.get("contents"))
    if contents_html:
        if "<" not in contents_html:
            contents_html = f"<p>{escape(contents_html)}</p>"
        blocks.append(_wrap_section("recipe_steps", contents_html, heading="Préparation"))

    astuce = _coerce_html(item.get("astuce"))
    if astuce:
        if "<" not in astuce:
            astuce = f"<blockquote>{escape(astuce)}</blockquote>"
        blocks.append(_wrap_section("chef_astuce", astuce, heading="L'astuce du chef"))

    chef = item.get("chef") or {}
    chef_heading, chef_html = _format_chef_block(chef) if isinstance(chef, dict) else ("", "")
    if chef_html:
        blocks.append(_wrap_section("chef_bio", chef_html, heading=chef_heading))

    # Fallbacks historiques : certains payloads atypiques utilisent encore
    # un `body_html` plat. On le récupère pour ne rien perdre.
    if not blocks:
        for key in ("body_html", "bodyHtml", "html", "post_content", "articleBody"):
            v = item.get(key)
            if isinstance(v, str) and v.strip():
                blocks.append(v.strip())
                notes.append(f"corps : fallback `{key}` (aucune section structurée)")
                break

    if not blocks:
        notes.append("aucun bloc HTML exploitable trouvé")

    body = "\n".join(blocks).strip()
    return body, notes


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

    title = _pick(item, "title", "recipe_name", "headline") or "(sans titre)"
    subtitle = _pick(item, "subtitle", "secondary_title", "kicker")
    # Pour les recettes WhiteBeard, ``summary`` est le HTML des ingrédients
    # (``<ul><li>…</li></ul>``). Comme champ "résumé" en admin, on prend
    # plutôt ``description`` (résumé éditorial). Les ingrédients restent
    # exhaustivement présents dans ``body_html`` (section ``ingredients_list``).
    description = _pick(item, "description", "lead", "excerpt")
    summary = description if description else _pick(item, "summary")
    introduction = _pick(item, "introduction", "intro")
    signature = _pick(item, "signature")
    seo = item.get("seo") if isinstance(item.get("seo"), dict) else None

    body_html, body_notes = _assemble_body_html(item)
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

    cover_url, cover_caption = _extract_cover(item)

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

    # Le créateur réel d'une recette WhiteBeard est dans ``chef`` (objet
    # enrichi par le service en 2ᵉ appel), distinct du / des journaliste(s)
    # qui ont signé l'article. On promeut donc le chef en `featured_chef`
    # ET on conserve les auteurs existants (la journaliste reste citée).
    chef_author = _chef_to_author(item.get("chef"))
    if chef_author is not None:
        existing_slugs = {a.slug for a in authors}
        if chef_author.slug not in existing_slugs:
            authors.insert(0, chef_author)

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
        is_premium=_is_premium(item),
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


def _is_premium(item: dict[str, Any]) -> bool:
    """Vrai si l'article est payant (paywall OLJ)."""
    if item.get("premium") or item.get("is_premium"):
        return True
    options = item.get("options") if isinstance(item.get("options"), dict) else {}
    if str(options.get("premium") or "").strip() in ("1", "true", "yes"):
        return True
    if item.get("freemium") is True:
        return True
    return False


def _extract_cover(item: dict[str, Any]) -> tuple[str | None, str | None]:
    """Image principale : ``image``, ``cover``, ou 1ʳᵉ pièce jointe image.

    WhiteBeard place les images dans ``attachments`` (liste de dicts avec
    ``url_large``/``url_highres``/etc.) ; la première est la photo de tête.
    """
    cover = item.get("image") or item.get("cover")
    if isinstance(cover, dict):
        url = _pick(cover, "url_large", "url", "src", "large", "medium")
        caption = _pick(cover, "caption", "credits", "alt", "description")
        if url:
            return url, caption

    attachments = item.get("attachments") or []
    if isinstance(attachments, list):
        for a in attachments:
            if not isinstance(a, dict):
                continue
            if a.get("type") and a["type"] != "image":
                continue
            url = _pick(a, "url_large", "url_highres", "url_medium", "url_home_slideshow", "url")
            if url:
                return url, _pick(a, "description", "name")
    return None, None


_FEATURED_HINTS = ("chef", "cuisinier", "cheffe", "cuisinière")


def _chef_to_author(chef: Any) -> MappedAuthor | None:
    """Promotion du champ ``chef`` du payload recette en ``MappedAuthor``.

    Le ``chef.title`` typique est ``"Kamal Mouzawak, fondateur de …"`` :
    on découpe sur la 1ʳᵉ virgule pour avoir un ``name`` propre + une
    ``description`` riche (utile en filtre / facette / réponse LLM).
    """
    if not isinstance(chef, dict):
        return None
    raw_title = (chef.get("title") or chef.get("name") or "").strip()
    if not raw_title:
        return None
    if "," in raw_title:
        name, _, description = raw_title.partition(",")
        name = name.strip()
        description = description.strip() or None
    else:
        name, description = raw_title, None
    bio_html = _coerce_html(chef.get("contents")) or None
    bio_text = _html_to_text(bio_html) if bio_html else None
    image = None
    attachments = chef.get("attachments") or []
    if isinstance(attachments, list) and attachments:
        first = attachments[0]
        if isinstance(first, dict):
            image = first.get("url_large") or first.get("url_medium") or first.get("url")
    ext_id: int | None = None
    cid = chef.get("id")
    try:
        if cid is not None:
            ext_id = int(cid)
    except (TypeError, ValueError):
        ext_id = None
    return MappedAuthor(
        external_id=ext_id,
        name=name,
        slug=_ensure_slug(name, fallback=ext_id or 0),
        role="featured_chef",
        department="cuisine",
        biography_html=bio_html,
        biography_text=bio_text,
        description=description,
        image_url=image,
        raw={"from": "chef_field", "chef_id": ext_id, "url": chef.get("url")},
    )


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
    """Mappe les keywords WhiteBeard.

    Format observé :
        [{"type": "Keyword", "data": {"keywordId": "56638", "name": "plat principal", ...}}, ...]
    L'ancien format à plat ``{"id": ..., "name": ...}`` reste supporté.
    """
    out: list[MappedKeyword] = []
    seen_slugs: set[str] = set()
    for k in raw or []:
        if isinstance(k, dict):
            data = k.get("data") if isinstance(k.get("data"), dict) else k
            name = _pick(data, "display_name", "name", "label")
            if not name:
                continue
            ext_id_raw = data.get("keywordId") or data.get("id")
            try:
                ext_id = int(ext_id_raw) if ext_id_raw is not None else None
            except (TypeError, ValueError):
                ext_id = None
            slug = _ensure_slug(data.get("slug") or name, fallback=ext_id or 0)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            out.append(
                MappedKeyword(
                    external_id=ext_id,
                    name=str(name),
                    slug=slug,
                    description=_pick(data, "description"),
                )
            )
        elif isinstance(k, str) and k.strip():
            slug = _ensure_slug(k)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            out.append(
                MappedKeyword(
                    external_id=None,
                    name=k.strip(),
                    slug=slug,
                    description=None,
                )
            )
    return out


def _to_int(v: Any) -> int | None:
    try:
        return int(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _map_categories(raw: list[Any]) -> list[MappedCategory]:
    """Mappe les catégories. WhiteBeard renvoie les IDs en string.

    Lorsqu'une catégorie a un ``parent`` (ex. parent ``Recettes``), on
    pousse aussi le parent dans la liste pour que le retrieval puisse
    filtrer indifféremment sur la sous-catégorie ou la racine.
    """
    out: list[MappedCategory] = []
    seen_slugs: set[str] = set()

    def _push(c: dict[str, Any]) -> None:
        name = _pick(c, "name", "label", "title")
        if not name:
            return
        ext_id = _to_int(c.get("id"))
        slug = _ensure_slug(c.get("slug") or name, fallback=ext_id or 0)
        if slug in seen_slugs:
            return
        seen_slugs.add(slug)
        out.append(
            MappedCategory(
                external_id=ext_id,
                name=str(name),
                slug=slug,
                description=_pick(c, "description"),
            )
        )

    for c in raw or []:
        if isinstance(c, dict):
            _push(c)
            parent = c.get("parent")
            if isinstance(parent, dict) and parent.get("name"):
                _push(parent)
        elif isinstance(c, str) and c.strip():
            slug = _ensure_slug(c)
            if slug not in seen_slugs:
                seen_slugs.add(slug)
                out.append(
                    MappedCategory(
                        external_id=None,
                        name=c.strip(),
                        slug=slug,
                        description=None,
                    )
                )
    return out
