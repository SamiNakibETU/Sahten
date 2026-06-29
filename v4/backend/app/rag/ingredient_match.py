"""Correspondance ingrédient demandé ↔ chunks indexés (ingredients_list)."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Sequence

from ..llm.query_understanding import QueryPlan
from .reranker import RerankedHit
from .retriever import Hit

INGREDIENT_SLUG_ALIASES: dict[str, tuple[str, ...]] = {
    "concombre": ("concombre", "concombres", "khyar", "khiar"),
    "tomate": ("tomate", "tomates", "banadoura", "bandoura"),
    "poulet": ("poulet", "poulets"),
    "citron": ("citron", "citrons"),
    "aubergine": ("aubergine", "aubergines"),
    "courgette": ("courgette", "courgettes"),
    "pois-chiche": ("pois chiche", "pois chiches", "pois-chiche", "pois-chiches"),
}

# Variantes plurielles / synonymes → slug canonique unique en base.
_INGREDIENT_SLUG_CANONICAL: dict[str, str] = {
    "pois-chiches": "pois-chiche",
    "tomates": "tomate",
    "concombres": "concombre",
    "poulets": "poulet",
    "citrons": "citron",
    "aubergines": "aubergine",
    "courgettes": "courgette",
}
for _canonical in INGREDIENT_SLUG_ALIASES:
    _INGREDIENT_SLUG_CANONICAL.setdefault(_canonical, _canonical)

_INGREDIENT_EXTRACT_RE = re.compile(
    r"(?i)\b(?:avec|au|aux|à base de)\s+"
    r"(?:du|de la|de l'|des|d'|de)\s*"
    r"([\wàâäéèêëïîôùûç\-]+)"
)
_INGREDIENT_RECETTE_RE = re.compile(
    r"(?i)\brecette(?:s)?\s+(?:au|aux|à base de|avec)\s+"
    r"(?:du|de la|de l'|des|d'|de)?\s*"
    r"([\wàâäéèêëïîôùûç\-]+)"
)
_KNOWN_ING_WORDS: dict[str, str] = {
    "concombre": "concombre",
    "concombres": "concombre",
    "tomate": "tomate",
    "tomates": "tomate",
    "poulet": "poulet",
    "poulets": "poulet",
    "citron": "citron",
    "citrons": "citron",
    "aubergine": "aubergine",
    "aubergines": "aubergine",
    "courgette": "courgette",
    "courgettes": "courgette",
    "halloumi": "halloumi",
    "boulgour": "boulgour",
    "menthe": "menthe",
    "persil": "persil",
    "yaourt": "yaourt",
    "khyar": "concombre",
    "khiar": "concombre",
    "banadoura": "tomate",
    "bandoura": "tomate",
    "batinjane": "aubergine",
    "batinghane": "aubergine",
    "kousa": "courgette",
    "kusa": "courgette",
    "djej": "poulet",
    "dajaj": "poulet",
}


_FOLLOW_UP_RE = re.compile(
    r"(?i)^(?:oui|non|ok|merci|d'accord|une autre|encore(?:\s+une)?(?:\s+autre)?|"
    r"autre(?:\s+recette)?|autre chose|la premi(?:è|e)re?|celle(?:-là)?)\.?$"
)
_WANTS_ANOTHER_RE = re.compile(
    r"(?i)\b(une autre|encore une|autre recette|autre chose|autre idée)\b"
)


def _ascii_slug(word: str) -> str:
    s = unicodedata.normalize("NFKD", word.strip().lower())
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or word.strip().lower()


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = cur
    return prev[-1]


def canonical_ingredient_slug(slug: str) -> str:
    s = (slug or "").strip().lower()
    return _INGREDIENT_SLUG_CANONICAL.get(s, s)


_KNOWN_INGREDIENT_SLUGS = set(INGREDIENT_SLUG_ALIASES) | set(_KNOWN_ING_WORDS.values())


def is_known_ingredient_slug(slug: str) -> bool:
    """True si le slug correspond à un ingrédient réel connu (pas un terme parasite
    type 'saveurs'/'tomate-jabaliyeh' que le LLM extrait parfois d'une réponse
    précédente). Sert à nettoyer le plan lors de la rotation « une autre »."""
    return canonical_ingredient_slug((slug or "").strip().lower()) in _KNOWN_INGREDIENT_SLUGS


def slug_search_terms(slug: str) -> tuple[str, ...]:
    raw = canonical_ingredient_slug((slug or "").strip().lower())
    if raw in INGREDIENT_SLUG_ALIASES:
        return INGREDIENT_SLUG_ALIASES[raw]
    base = raw.replace("-", " ")
    terms = {raw, base}
    if base.endswith("s") and len(base) > 3:
        terms.add(base[:-1])
    else:
        terms.add(f"{base}s")
    return tuple(sorted(terms, key=len, reverse=True))


def text_contains_ingredient(text: str, terms: tuple[str, ...]) -> bool:
    low = (text or "").lower().replace("’", "'")
    for term in terms:
        t = term.lower().replace("-", " ")
        if re.search(rf"(?<![\wàâäéèêëïîôùûç-]){re.escape(t)}(?![\wàâäéèêëïîôùûç-])", low):
            return True
    return False


def chunk_confirms_ingredient(
    section_kind: str | None,
    chunk_text: str,
    slug: str,
    *,
    strict: bool = True,
) -> bool:
    sk = (section_kind or "").lower()
    allowed = ("ingredients_list",) if strict else ("ingredients_list", "recipe_summary", "anchor", "recipe_steps")
    if sk not in allowed:
        return False
    return text_contains_ingredient(chunk_text, slug_search_terms(slug))


def _word_to_slug(word: str) -> str | None:
    w = word.strip().lower().replace("’", "'")
    if w in _KNOWN_ING_WORDS:
        return _KNOWN_ING_WORDS[w]
    if len(w) >= 4:
        for canonical in INGREDIENT_SLUG_ALIASES:
            for term in slug_search_terms(canonical):
                t = term.lower().replace("-", " ")
                if len(t) >= 4 and _levenshtein(w, t) <= 1:
                    return canonical
    if len(w) < 3:
        return None
    return _ascii_slug(w)


def scan_known_ingredient_slugs(text: str) -> list[str]:
    """Repère tous les ingrédients connus présents dans un bloc de texte."""
    if not text or not text.strip():
        return []
    low = text.lower().replace("’", "'")
    found: list[str] = []
    seen: set[str] = set()
    for canonical, terms in INGREDIENT_SLUG_ALIASES.items():
        for term in terms:
            t = term.lower().replace("-", " ")
            if len(t) < 3:
                continue
            if re.search(
                rf"(?<![\wàâäéèêëïîôùûç-]){re.escape(t)}(?![\wàâäéèêëïîôùûç-])",
                low,
            ):
                if canonical not in seen:
                    seen.add(canonical)
                    found.append(canonical_ingredient_slug(canonical))
                break
    return found


def extract_ingredient_slugs_from_text(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    found: list[str] = []
    seen: set[str] = set()
    for pattern in (_INGREDIENT_EXTRACT_RE, _INGREDIENT_RECETTE_RE):
        for m in pattern.finditer(text):
            slug = _word_to_slug(m.group(1))
            if slug:
                slug = canonical_ingredient_slug(slug)
            if slug and slug not in seen:
                seen.add(slug)
                found.append(slug)
    for m in re.finditer(r"(?i)\b([\wàâäéèêëïîôùûç-]+)\b", text):
        slug = _word_to_slug(m.group(1))
        if slug:
            slug = canonical_ingredient_slug(slug)
        if slug and slug not in seen and slug in INGREDIENT_SLUG_ALIASES:
            seen.add(slug)
            found.append(slug)
    return found


_EXCLUDE_RE = re.compile(
    r"(?i)\b(?:sans|pas\s+de|pas\s+d'|sans\s+aucun[e]?|ni)\s+"
    r"(?:de\s+|du\s+|des\s+|d'\s*|la\s+|le\s+|les\s+)?"
    r"([\wàâäéèêëïîôùûç-]+)"
)


def extract_excluded_ingredient_slugs(text: str) -> list[str]:
    """Ingrédients EXCLUS par une contrainte négative (« sans tomate », « pas de
    poivron », « ni oignon »). Permet de ne pas les filtrer comme requis et de
    retirer les recettes qui les contiennent."""
    if not text or not text.strip():
        return []
    out: list[str] = []
    seen: set[str] = set()
    for m in _EXCLUDE_RE.finditer(text):
        slug = _word_to_slug(m.group(1))
        if slug:
            slug = canonical_ingredient_slug(slug)
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)
    return out


# ── Extraction data-driven d'un ingrédient depuis une LIGNE de liste de recette ──
# « 300 g de semoule fine » -> 'semoule' ; « 1/2 tasse de jus de citron » -> 'citron'.
# Construit le vocabulaire d'ingrédients À PARTIR DES DONNÉES (pas de liste figée).
_ING_UNITS = (
    r"(?:g|gr|kg|mg|ml|cl|dl|l|càs|càc|c\.?\s*à\s*(?:soupe|caf[ée]|s|c)\.?|"
    r"cuill[èe]r\w*|tasses?|verres?|pinc[ée]es?|gousses?|bottes?|branches?|brins?|"
    r"tranches?|sachets?|bo[îi]tes?|pi[èe]ces?|feuilles?|filets?|tiges?|poign[ée]es?|"
    r"louches?|doses?|barquettes?|paquets?|kg)"
)
_ING_LEAD = re.compile(r"^[\s\d.,/½¼¾⅓⅔⅛°x×+-]+", re.I)
_ING_UNIT = re.compile(rf"^{_ING_UNITS}\b\.?\s*", re.I)
_ING_ART = re.compile(
    r"^(?:de\s+la\s+|de\s+l['’]|du\s+|des\s+|de\s+|d['’]\s*|d\s+|à\s+|au\s+|aux\s+|"
    r"le\s+|la\s+|les\s+|un\s+|une\s+|quelques\s+|environ\s+|gros(?:se)?s?\s+|"
    r"petits?\s+|belles?\s+|beaux?\s+)+",
    re.I,
)
_ING_CONTAINER = {
    "jus", "huile", "filet", "filets", "zeste", "zestes", "coeur", "coeurs",
    "tranche", "tranches", "morceau", "morceaux", "gousse", "gousses", "botte",
    "feuille", "feuilles", "brin", "brins", "poignee",
}
_ING_ADJ = {
    "fine", "fin", "fraiche", "frais", "jaune", "rouge", "vert", "verte", "blanc",
    "blanche", "mure", "mur", "mature", "moyen", "moyenne", "gros", "grosse",
    "petit", "petite", "plat", "plate", "sec", "seche", "hache", "hachee", "rape",
    "rapee", "entier", "entiere", "doux", "douce", "libanais", "liban", "libanaise",
    "bio", "nature", "cuit", "cru", "crue", "grille", "grillee", "confit", "sale",
    "sucre", "noir", "noire", "neuf", "nouvelle", "nouveau",
}
_ING_STOP = {
    "de", "d", "la", "le", "les", "du", "des", "a", "au", "aux", "et", "ou", "pour",
    "selon", "gout", "environ", "quelques", "un", "une", "avec", "sans",
}


def _ing_norm(s: str) -> str:
    t = unicodedata.normalize("NFKD", (s or "").lower()).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 '-]", " ", t).replace("'", " ").strip()


def parse_ingredient_slug(line: str) -> str | None:
    """Nom d'ingrédient canonique extrait d'une ligne de liste, sinon None."""
    s = (line or "").lower().replace("’", "'").strip()
    s = re.split(r"[,(:]| - ", s)[0]
    prev = None
    while prev != s:
        prev = s
        s = _ING_LEAD.sub("", s)
        s = _ING_UNIT.sub("", s)
        s = _ING_ART.sub("", s)
        s = s.strip()
    words = [w for w in _ing_norm(s).split() if w and w not in _ING_STOP and len(w) > 1]
    if not words:
        return None
    head = words[0]
    if head in _ING_CONTAINER and len(words) > 1:
        head = next((w for w in words[1:] if w not in _ING_CONTAINER and w not in _ING_ADJ), head)
    elif head in _ING_ADJ and len(words) > 1:
        head = words[1]
    if len(head) > 4 and head.endswith("s"):
        head = head[:-1]
    if len(head) < 3 or head in _ING_ADJ or head in _ING_CONTAINER:
        return None
    return canonical_ingredient_slug(head)


def parse_ingredient_slugs_from_lines(text: str) -> list[str]:
    """Tous les ingrédients d'un bloc de liste (une ligne = un item)."""
    out: list[str] = []
    seen: set[str] = set()
    for line in (text or "").splitlines():
        slug = parse_ingredient_slug(line)
        if slug and slug not in seen:
            seen.add(slug)
            out.append(slug)
    return out


def supplement_ingredient_slugs(
    plan: QueryPlan,
    user_query: str,
    conversation_history: str | None,
) -> QueryPlan:
    """Enrichit les slugs ingrédient depuis la requête et l'historique.

    Le filtrage des requêtes "humeur" (simple, soir…) est géré en amont dans
    pipeline.py en passant conversation_history=None pour ces cas. Ici on scan
    l'historique inconditionnellement s'il est fourni, afin que les suivis courts
    ("une salde", "une autre", "oui") héritent bien du fil ingrédient courant.
    """
    slugs = list(plan.ingredient_slugs or [])
    seen = set(slugs)

    # Extraction depuis la requête courante (toujours faite).
    for slug in extract_ingredient_slugs_from_text(user_query or ""):
        if slug not in seen:
            seen.add(slug)
            slugs.append(slug)

    # Scanner l'historique si le caller l'a fourni (les requêtes humeur arrivent
    # avec conversation_history=None depuis pipeline.py).
    if conversation_history:
        for slug in extract_ingredient_slugs_from_text(conversation_history):
            if slug not in seen:
                seen.add(slug)
                slugs.append(slug)

    if slugs == (plan.ingredient_slugs or []):
        return plan
    return plan.model_copy(update={"ingredient_slugs": slugs})


def _article_ids_matching_slugs(
    items: Sequence[Hit | RerankedHit],
    slugs: list[str],
    *,
    strict: bool,
) -> set[int]:
    by_article: dict[int, list[Hit]] = defaultdict(list)
    for item in items:
        hit = item.hit if isinstance(item, RerankedHit) else item
        by_article[int(hit.article_external_id)].append(hit)

    valid: set[int] = set()
    for aid, article_hits in by_article.items():
        ok = True
        for slug in slugs:
            if not any(
                chunk_confirms_ingredient(h.section_kind, h.chunk_text, slug, strict=strict)
                for h in article_hits
            ):
                ok = False
                break
        if ok:
            valid.add(aid)
    return valid


def filter_hits_by_ingredient_slugs(
    hits: list[Hit],
    slugs: list[str] | None,
    *,
    strict: bool = True,
) -> list[Hit]:
    if not slugs:
        return hits
    valid = _article_ids_matching_slugs(hits, slugs, strict=strict)
    if not valid and strict:
        valid = _article_ids_matching_slugs(hits, slugs, strict=False)
    if not valid:
        return []
    return [h for h in hits if int(h.article_external_id) in valid]


def filter_hits_by_ingredient_text(
    hits: list[Hit], slugs: list[str] | None
) -> list[Hit]:
    """Filtre TRÈS permissif (dernier recours) : garde les articles dont un chunk
    — quelle que soit la section — contient réellement le terme de l'ingrédient.

    Sert quand le pré-filtre structuré (`article_ingredients`) et le filtre par
    section échouent alors que des recettes mentionnent bien l'ingrédient (ex.
    « concombre » présent dans le titre/le corps mais non lié en base). Reste
    précis : un plat absent (ex. « katayef », qu'aucune recette ne mentionne)
    ressort vide → la génération de dernier recours prend le relais."""
    if not slugs:
        return hits
    by_article: dict[int, list[Hit]] = defaultdict(list)
    for h in hits:
        by_article[int(h.article_external_id)].append(h)
    valid: set[int] = set()
    for aid, article_hits in by_article.items():
        if all(
            any(
                text_contains_ingredient(h.chunk_text, slug_search_terms(slug))
                for h in article_hits
            )
            for slug in slugs
        ):
            valid.add(aid)
    return [h for h in hits if int(h.article_external_id) in valid]


def filter_reranked_by_ingredient_slugs(
    reranked: list[RerankedHit],
    slugs: list[str] | None,
    *,
    strict: bool = True,
    retrieval_hits: list[Hit] | None = None,
) -> list[RerankedHit]:
    if not slugs or not reranked:
        return reranked
    by_chunk: dict[int, Hit | RerankedHit] = {}
    for h in retrieval_hits or []:
        by_chunk[h.chunk_id] = h
    for r in reranked:
        by_chunk[r.hit.chunk_id] = r
    evidence = list(by_chunk.values())

    valid = _article_ids_matching_slugs(evidence, slugs, strict=strict)
    if not valid and strict:
        valid = _article_ids_matching_slugs(evidence, slugs, strict=False)
    if not valid and retrieval_hits:
        # Le pré-filtre SQL a déjà validé l'article ; les chunks rerankés
        # peuvent être recipe_summary/steps sans repasser ingredients_list.
        valid = {int(h.article_external_id) for h in retrieval_hits}
    if not valid:
        return []
    return [r for r in reranked if int(r.hit.article_external_id) in valid]


# Marqueurs d'ACCOMPAGNEMENT dans un titre : l'ingrédient n'est pas le sujet du
# plat mais un à-côté (« … à la tomate », « … au citron », « … et concombre »).
_ACCOMPANIMENT_PREFIX_RE = re.compile(r"(?:a la |a l |au |aux |et |avec |sauce )$")


def _ingredient_central_in_title(title: str, terms: tuple[str, ...]) -> bool:
    """True si l'ingrédient apparaît dans le titre comme SUJET (« kebbé de tomate »)
    et pas seulement en accompagnement (« panade d'aubergines à la tomate »)."""
    low = unicodedata.normalize("NFKD", (title or "").lower()).replace("’", "'")
    low = "".join(c for c in low if not unicodedata.combining(c))
    for term in terms:
        t = term.lower().replace("-", " ")
        for m in re.finditer(
            rf"(?<![\w]){re.escape(t)}(?![\w])", low
        ):
            prefix = low[max(0, m.start() - 7): m.start()]
            if not _ACCOMPANIMENT_PREFIX_RE.search(prefix):
                return True
    return False


def rerank_by_ingredient_centrality(
    reranked: list[RerankedHit],
    slugs: list[str] | None,
    *,
    retrieval_hits: list[Hit] | None = None,
) -> list[RerankedHit]:
    """Ré-ordonne pour faire remonter les recettes où l'ingrédient est CENTRAL.

    Niveaux (tri stable, le score de rerank départage à l'intérieur d'un niveau) :
    0. ingrédient SUJET du titre (« La kebbé de tomate ») ;
    1. ingrédient en accompagnement dans le titre (« … à la tomate ») ;
    2. ingrédient confirmé dans la section ingredients_list ;
    3. simplement mentionné ailleurs (« … peut s'accompagner de tomates »).

    Corrige « recette tomate » → panade d'aubergines/blette au lieu de kebbé-de-tomate."""
    if not slugs or len(reranked) < 2:
        return reranked
    by_article: dict[int, list[Hit]] = defaultdict(list)
    for h in retrieval_hits or []:
        by_article[int(h.article_external_id)].append(h)
    for r in reranked:
        by_article[int(r.hit.article_external_id)].append(r.hit)

    def tier(r: RerankedHit) -> int:
        title = r.hit.article_title or ""
        terms_per_slug = [slug_search_terms(s) for s in slugs]
        if any(_ingredient_central_in_title(title, terms) for terms in terms_per_slug):
            return 0
        if any(text_contains_ingredient(title, terms) for terms in terms_per_slug):
            return 1
        arts = by_article.get(int(r.hit.article_external_id), [])
        if all(
            any(
                chunk_confirms_ingredient(h.section_kind, h.chunk_text, s, strict=True)
                for h in arts
            )
            for s in slugs
        ):
            return 2
        return 3

    return sorted(reranked, key=lambda r: (tier(r), -float(r.rerank_score)))


def filter_reranked_excluding_ingredients(
    reranked: list[RerankedHit],
    excluded_slugs: list[str] | None,
    *,
    retrieval_hits: list[Hit] | None = None,
) -> list[RerankedHit]:
    """Écarte les articles qui contiennent un ingrédient EXCLU (« sans tomate »).
    Ne renvoie jamais vide : si tout est exclu, on garde la liste d'origine plutôt
    que d'abstenir (mieux vaut imparfait que rien)."""
    if not excluded_slugs or not reranked:
        return reranked
    by_article: dict[int, list[Hit]] = defaultdict(list)
    for h in retrieval_hits or []:
        by_article[int(h.article_external_id)].append(h)
    for r in reranked:
        by_article[int(r.hit.article_external_id)].append(r.hit)

    def has_excluded(aid: int) -> bool:
        arts = by_article.get(aid, [])
        return any(
            any(text_contains_ingredient(h.chunk_text, slug_search_terms(s)) for h in arts)
            for s in excluded_slugs
        )

    kept = [r for r in reranked if not has_excluded(int(r.hit.article_external_id))]
    return kept if kept else reranked


def article_external_id_for_recipe_card(
    recipe_card,
    hits: list[RerankedHit],
) -> int | None:
    if recipe_card is None:
        return None
    want = set(recipe_card.source_chunk_ids or [])
    for h in hits:
        if h.hit.chunk_id in want:
            return int(h.hit.article_external_id)
    return None


def recipe_card_matches_required_ingredients(
    recipe_card,
    hits: list[RerankedHit],
    slugs: list[str] | None,
    *,
    strict: bool = True,
) -> bool:
    if not slugs or recipe_card is None:
        return True
    aid = article_external_id_for_recipe_card(recipe_card, hits)
    if aid is None:
        return False
    article_hits = [h.hit for h in hits if int(h.hit.article_external_id) == aid]
    for slug in slugs:
        if not any(
            chunk_confirms_ingredient(h.section_kind, h.chunk_text, slug, strict=strict)
            for h in article_hits
        ):
            if strict and any(
                chunk_confirms_ingredient(
                    h.section_kind, h.chunk_text, slug, strict=False
                )
                for h in article_hits
            ):
                continue
            return False
    return True


def is_short_follow_up(user_query: str) -> bool:
    return bool(_FOLLOW_UP_RE.match((user_query or "").strip()))


def wants_another_recipe(user_query: str) -> bool:
    q = (user_query or "").strip()
    low = q.lower().replace("'", "'")
    if low in {"une autre", "encore", "autre", "autre recette", "autre chose"}:
        return True
    return bool(_WANTS_ANOTHER_RE.search(q))


def is_ingredient_browse_query(required_ingredient_slugs: list[str] | None) -> bool:
    return bool(required_ingredient_slugs)


def ingredient_display_name(slug: str) -> str:
    # Affichage = slug canonique dé-hyphené (concombre, riz, pois chiche) ; évite
    # les pluriels auto-générés ("rizs") et les alias arabes ("khyar").
    name = canonical_ingredient_slug((slug or "").strip()).replace("-", " ").strip()
    return name or (slug or "").replace("-", " ")
