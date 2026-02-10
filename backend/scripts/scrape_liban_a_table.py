#!/usr/bin/env python
"""
Liban a Table Scraper (Playwright)
==================================

Scrape exhaustively the "Liban a Table" section of L'Orient-Le Jour,
including subsections and pagination, then export JSONL (1 article per line).

Install dependencies:
    pip install playwright beautifulsoup4 lxml tqdm
    python -m playwright install

Usage:
    python scripts/scrape_liban_a_table.py
    python scripts/scrape_liban_a_table.py --max-pages 50
    python scripts/scrape_liban_a_table.py --max-articles 200
    python scripts/scrape_liban_a_table.py --output data/liban_a_table.jsonl --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
from tqdm import tqdm

BASE_URL = "https://www.lorientlejour.com"
ROOT_SECTION_URL = "https://www.lorientlejour.com/cuisine-liban-a-table"

DEFAULT_MAX_PAGES = 200
REQUEST_DELAY_MIN = 1.0
REQUEST_DELAY_MAX = 2.0
MAX_RETRIES = 3

FALLBACK_SECTION_SLUGS = [
    "recettes",
    "nos-selections-gourmandes",
    "chefs",
    "restaurants",
    "vins-et-plus",
]

SECTION_LINK_RE = re.compile(r"/cuisine-liban-a-table(?:/[^?]*)?$")
ARTICLE_ID_RE = re.compile(r"/cuisine-liban-a-table/(?:\d{6,}|[^/]+/\d{6,})")


@dataclass
class ArticleRecord:
    url: str
    title: str
    content_type: str
    category: Optional[str]
    subcategory: Optional[str]
    author: Optional[str]
    date: Optional[str]
    image_url: Optional[str]
    description: Optional[str]
    full_text: Optional[str]
    tags: List[str]
    chef_name: Optional[str]
    ingredients: List[str]
    instructions: List[str]
    prep_time: Optional[str]
    cook_time: Optional[str]
    servings: Optional[str]
    restaurant_name: Optional[str]
    location: Dict[str, Optional[str]]
    scraped_at: str


class LibanATableScraper:
    def __init__(
        self,
        output_path: Path,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_articles: Optional[int] = None,
        resume: bool = False,
        headless: bool = True,
    ) -> None:
        self.output_path = output_path
        self.max_pages = max_pages
        self.max_articles = max_articles
        self.resume = resume
        self.headless = headless
        self.logger = self._setup_logger()
        self.scraped_urls: Set[str] = set()

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("liban_a_table_scraper")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        log_dir = self.output_path.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "liban_a_table_scraper.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(message)s"))

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    async def run(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.resume:
            self.scraped_urls = self._load_resume_urls(self.output_path)

        self.logger.info("Starting scrape")
        self.logger.info(f"Output: {self.output_path}")
        if self.scraped_urls:
            self.logger.info(f"Resume enabled: {len(self.scraped_urls)} URLs loaded")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.headless)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
                locale="fr-FR",
            )
            page = await context.new_page()

            section_urls = await self._collect_section_urls(page)
            self.logger.info(f"Section URLs: {len(section_urls)}")

            article_urls = await self._collect_article_urls(page, section_urls)
            sitemap_urls = await self._collect_sitemap_article_urls(page)
            if sitemap_urls:
                self.logger.info(f"Sitemap article URLs: {len(sitemap_urls)}")
                article_urls = list(set(article_urls).union(sitemap_urls))
            self.logger.info(f"Article URLs collected: {len(article_urls)}")

            await self._scrape_articles(page, article_urls)

            await context.close()
            await browser.close()

        self.logger.info("Scrape complete")

    async def _collect_section_urls(self, page) -> List[str]:
        section_urls: Set[str] = set()
        to_visit: List[Tuple[str, int]] = [(ROOT_SECTION_URL, 0)]
        max_depth = 2

        while to_visit:
            current_url, depth = to_visit.pop(0)
            if current_url in section_urls:
                continue

            html = await self._fetch_html(page, current_url, scroll=True)
            soup = BeautifulSoup(html, "lxml")
            section_urls.add(current_url)

            if depth >= max_depth:
                continue

            for a in soup.select("a[href]"):
                href = self._coerce_str(a.get("href"))
                if not href or "/cuisine-liban-a-table" not in href:
                    continue
                full_url = self._normalize_url(urljoin(BASE_URL, href))
                if self._is_section_url(full_url) and full_url not in section_urls:
                    to_visit.append((full_url, depth + 1))

        if len(section_urls) <= 1:
            for slug in FALLBACK_SECTION_SLUGS:
                section_urls.add(f"{ROOT_SECTION_URL}/{slug}")

        return sorted(section_urls)

    async def _collect_sitemap_article_urls(self, page) -> List[str]:
        sitemap_url = f"{BASE_URL}/sitemap.xml"
        urls: Set[str] = set()

        try:
            xml = await self._fetch_text_request(page, sitemap_url)
        except Exception as exc:
            self.logger.warning(f"Sitemap fetch failed: {sitemap_url} ({exc})")
            return []

        soup = BeautifulSoup(xml, "xml")
        sitemap_locs = [loc.get_text(strip=True) for loc in soup.select("sitemap > loc")]
        if not sitemap_locs:
            sitemap_locs = [sitemap_url]

        for loc in sitemap_locs:
            try:
                content = await self._fetch_text_request(page, loc)
            except Exception as exc:
                self.logger.warning(f"Sitemap loc failed: {loc} ({exc})")
                continue

            loc_soup = BeautifulSoup(content, "xml")
            for url_tag in loc_soup.select("url > loc"):
                url_text = url_tag.get_text(strip=True)
                if "/cuisine-liban-a-table/" not in url_text:
                    continue
                normalized = self._normalize_url(url_text)
                if self._is_article_url(normalized):
                    urls.add(normalized)

        return sorted(urls)

    async def _collect_article_urls(self, page, section_urls: Iterable[str]) -> List[str]:
        seen: Set[str] = set()
        for section_url in section_urls:
            self.logger.info(f"Listing section: {section_url}")
            last_page_links: Optional[List[str]] = None
            stagnant_pages = 0
            page_num = 1
            list_url = section_url

            while page_num <= self.max_pages:
                try:
                    html = await self._fetch_html(page, list_url, scroll=True)
                except Exception as exc:
                    self.logger.warning(f"Listing failed: {list_url} ({exc})")
                    break

                soup = BeautifulSoup(html, "lxml")
                links = self._extract_article_links(soup)
                if not links:
                    self.logger.info(f"No articles on page {page_num}, stop pagination")
                    break

                links = [self._normalize_url(u) for u in links]
                new_links = [u for u in links if u not in seen]
                seen.update(new_links)
                self.logger.info(f"Page {page_num}: {len(links)} articles ({len(new_links)} new)")

                if last_page_links and set(links) == set(last_page_links):
                    stagnant_pages += 1
                elif not new_links:
                    stagnant_pages += 1
                else:
                    stagnant_pages = 0

                if stagnant_pages >= 2:
                    self.logger.info("No new links across pages, stop pagination")
                    break

                last_page_links = links

                if self.max_articles and len(seen) >= self.max_articles:
                    return list(seen)[: self.max_articles]

                next_url = self._extract_next_page(soup, list_url)
                if not next_url:
                    next_url = self._with_page_param(section_url, page_num + 1)
                    if next_url == list_url:
                        break

                list_url = next_url
                page_num += 1

        return list(seen)

    async def _scrape_articles(self, page, article_urls: List[str]) -> None:
        to_scrape = [u for u in article_urls if u not in self.scraped_urls]
        if not to_scrape:
            self.logger.info("No new URLs to scrape.")
            return

        self.logger.info(f"Scraping {len(to_scrape)} articles")
        with self.output_path.open("a", encoding="utf-8") as f_out:
            for url in tqdm(to_scrape, desc="Articles", unit="article"):
                try:
                    html = await self._fetch_html(page, url)
                    soup = BeautifulSoup(html, "lxml")
                    record = self._parse_article(url, soup)
                    record = self._validate_record(record)
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    self.scraped_urls.add(url)
                except Exception as exc:
                    self.logger.warning(f"Failed article: {url} ({exc})")

                if self.max_articles and len(self.scraped_urls) >= self.max_articles:
                    self.logger.info("Reached max articles limit")
                    break

    async def _fetch_html(
        self,
        page,
        url: str,
        scroll: bool = False,
        wait_until: str = "networkidle",
        timeout_ms: int = 45000,
        expect_html: bool = True,
    ) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
                if expect_html:
                    try:
                        await page.wait_for_selector("article, .article-card, .teaser, main", timeout=8000)
                    except PlaywrightTimeoutError:
                        pass
                    await self._close_cookie_banner(page)
                    if scroll:
                        await self._scroll_to_bottom(page)
                html = await page.content()
                await asyncio.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
                return html
            except PlaywrightTimeoutError as exc:
                last_error = exc
                self.logger.warning(f"Timeout ({attempt}/{MAX_RETRIES}): {url}")
            except Exception as exc:
                last_error = exc
                self.logger.warning(f"Error ({attempt}/{MAX_RETRIES}): {url} ({exc})")
            await asyncio.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

        raise RuntimeError(f"Failed after retries: {url}") from last_error

    async def _fetch_text_request(self, page, url: str, timeout_ms: int = 30000) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await page.request.get(url, timeout=timeout_ms)
                if not response.ok:
                    raise RuntimeError(f"HTTP {response.status}")
                text = await response.text()
                await asyncio.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))
                return text
            except Exception as exc:
                last_error = exc
                self.logger.warning(f"Request error ({attempt}/{MAX_RETRIES}): {url} ({exc})")
            await asyncio.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

        raise RuntimeError(f"Failed after retries: {url}") from last_error

    async def _scroll_to_bottom(self, page) -> None:
        last_height = 0
        for _ in range(4):
            height = await page.evaluate("document.body.scrollHeight")
            if height == last_height:
                # Try clicking "load more" if present before exiting
                if not await self._click_load_more(page):
                    break
            last_height = height
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(1.0)

    async def _click_load_more(self, page) -> bool:
        for text in ["Voir plus", "Plus d'articles", "Charger plus", "Afficher plus"]:
            locator = page.locator("button, a", has_text=text).first
            try:
                if await locator.is_visible():
                    await locator.click(timeout=2000)
                    await asyncio.sleep(1.0)
                    return True
            except Exception:
                continue
        return False

    async def _close_cookie_banner(self, page) -> None:
        texts = [
            "Accepter",
            "Tout accepter",
            "J'accepte",
            "Fermer",
            "OK",
        ]
        for text in texts:
            locator = page.locator("button, a", has_text=text).first
            try:
                if await locator.is_visible():
                    await locator.click(timeout=2000)
                    await asyncio.sleep(0.5)
                    return
            except Exception:
                continue
        try:
            await page.keyboard.press("Escape")
        except Exception:
            pass

    def _extract_next_page(self, soup: BeautifulSoup, current_url: str) -> Optional[str]:
        selectors = [
            "a[rel='next']",
            ".pagination a[rel='next']",
            ".pagination a.next",
            "a.next",
        ]
        for sel in selectors:
            el = soup.select_one(sel)
            if not el:
                continue
            href = self._coerce_str(el.get("href"))
            if href:
                return self._normalize_url(urljoin(BASE_URL, href))

        for a in soup.select("a[href]"):
            text = a.get_text(strip=True).lower()
            if text in {"suivant", "page suivante", "next"}:
                href = self._coerce_str(a.get("href"))
                if href:
                    return self._normalize_url(urljoin(BASE_URL, href))

        return None

    def _extract_article_links(self, soup: BeautifulSoup) -> List[str]:
        links: List[str] = []
        for a in soup.select("a[href]"):
            href = self._coerce_str(a.get("href"))
            if not href or "/cuisine-liban-a-table/" not in href:
                continue
            full_url = self._normalize_url(urljoin(BASE_URL, href))
            if self._is_article_url(full_url) and full_url not in links:
                links.append(full_url)
        return links

    def _parse_article(self, url: str, soup: BeautifulSoup) -> Dict[str, Any]:
        json_ld = self._extract_json_ld(soup)

        title = self._extract_text(soup, ["h1", ".article-title", ".entry-title"])
        if not title:
            title = self._coerce_str(self._json_ld_first(json_ld, ["headline", "name"]))

        description = self._extract_text(soup, [".article-lead", ".lead", ".excerpt", 'meta[name="description"]'])
        if not description:
            description = self._coerce_str(self._json_ld_first(json_ld, ["description"]))

        author = self._extract_text(soup, [".author-name", ".byline", ".article-author", 'meta[name="author"]'])
        if not author:
            author = self._coerce_str(self._json_ld_first(json_ld, ["author", "creator"]))

        date = self._extract_date(soup)
        if not date:
            date = self._coerce_str(self._json_ld_first(json_ld, ["datePublished"]))

        category, subcategory = self._extract_categories(soup, url)
        image_url = self._extract_image(soup, json_ld)
        full_text = self._extract_content(soup)
        tags = self._extract_tags(soup, json_ld)

        recipe_data = self._extract_recipe_data(soup, json_ld)
        content_type = self._detect_content_type(title, full_text or "", json_ld, recipe_data)

        restaurant_name, location = self._extract_restaurant_info(json_ld, full_text or "", title)

        record = ArticleRecord(
            url=url,
            title=title or "Sans titre",
            content_type=content_type,
            category=category,
            subcategory=subcategory,
            author=author,
            date=date,
            image_url=image_url,
            description=description,
            full_text=full_text,
            tags=tags,
            chef_name=recipe_data.get("chef_name"),
            ingredients=recipe_data.get("ingredients", []),
            instructions=recipe_data.get("instructions", []),
            prep_time=recipe_data.get("prep_time"),
            cook_time=recipe_data.get("cook_time"),
            servings=recipe_data.get("servings"),
            restaurant_name=restaurant_name,
            location=location,
            scraped_at=datetime.now(timezone.utc).isoformat(),
        )

        return asdict(record)

    def _detect_content_type(
        self,
        title: Optional[str],
        content: str,
        json_ld: List[Dict[str, Any]],
        recipe_data: Dict[str, Any],
    ) -> str:
        title_lower = (title or "").lower()
        content_lower = (content or "").lower()

        if recipe_data.get("ingredients") or recipe_data.get("instructions") or self._json_ld_has_type(json_ld, "Recipe"):
            return "recipe"

        if self._json_ld_has_type(json_ld, "Restaurant"):
            return "restaurant"

        if "chef" in title_lower or "rencontre" in title_lower or "interview" in title_lower:
            return "portrait"

        if any(word in title_lower or word in content_lower for word in ["restaurant", "adresse", "où manger", "ou manger"]):
            return "restaurant"

        if "nos " in title_lower or "sélection" in title_lower or "selection" in title_lower:
            return "selection"

        if "vin" in title_lower or "œnologie" in title_lower or "oenologie" in title_lower:
            return "wine"

        if "tout savoir sur" in title_lower:
            return "guide"

        if any(word in title_lower for word in ["histoire", "tradition", "origine"]):
            return "story"

        return "story"

    def _extract_recipe_data(self, soup: BeautifulSoup, json_ld: List[Dict[str, Any]]) -> Dict[str, Any]:
        data: Dict[str, Any] = {}

        recipe_ld = self._json_ld_by_type(json_ld, "Recipe")
        if recipe_ld:
            data["ingredients"] = self._ensure_list(recipe_ld.get("recipeIngredient"))
            instructions = recipe_ld.get("recipeInstructions")
            data["instructions"] = self._normalize_instructions(instructions)
            data["prep_time"] = recipe_ld.get("prepTime")
            data["cook_time"] = recipe_ld.get("cookTime")
            data["servings"] = recipe_ld.get("recipeYield")
            data["chef_name"] = self._json_ld_author_name(recipe_ld)

        if not data.get("ingredients"):
            ingredients = []
            for el in soup.select(".ingredients li, [itemprop='recipeIngredient'], .recipe-ingredient"):
                text = el.get_text(strip=True)
                if text:
                    ingredients.append(text)
            if ingredients:
                data["ingredients"] = ingredients

        if not data.get("instructions"):
            instructions = []
            for el in soup.select(".instructions li, [itemprop='recipeInstructions'] li, .recipe-step"):
                text = el.get_text(strip=True)
                if text:
                    instructions.append(text)
            if instructions:
                data["instructions"] = instructions

        if not data.get("chef_name"):
            chef = self._extract_text(soup, [".chef-name", ".author-name", "[itemprop='author']"])
            if chef:
                data["chef_name"] = chef

        if not data.get("prep_time"):
            prep = soup.select_one("[itemprop='prepTime'], .prep-time")
            if prep:
                data["prep_time"] = prep.get_text(strip=True)

        if not data.get("cook_time"):
            cook = soup.select_one("[itemprop='cookTime'], .cook-time")
            if cook:
                data["cook_time"] = cook.get_text(strip=True)

        if not data.get("servings"):
            servings = soup.select_one("[itemprop='recipeYield'], .servings")
            if servings:
                data["servings"] = servings.get_text(strip=True)

        return data

    def _extract_restaurant_info(
        self,
        json_ld: List[Dict[str, Any]],
        content: str,
        title: Optional[str],
    ) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
        location: Dict[str, Optional[str]] = {
            "city": None,
            "country": None,
            "neighborhood": None,
            "address": None,
        }
        restaurant_name = None

        restaurant_ld = self._json_ld_by_type(json_ld, "Restaurant")
        if restaurant_ld:
            restaurant_name = restaurant_ld.get("name")
            address = restaurant_ld.get("address") or {}
            if isinstance(address, dict):
                location["address"] = address.get("streetAddress")
                location["city"] = address.get("addressLocality")
                location["country"] = address.get("addressCountry")

        if not restaurant_name and title and "restaurant" in title.lower():
            restaurant_name = title

        if not location["address"]:
            address_line = self._extract_address_line(content)
            if address_line:
                location["address"] = address_line

        return restaurant_name, location

    def _extract_address_line(self, content: str) -> Optional[str]:
        for line in content.splitlines():
            if "adresse" in line.lower():
                return line.strip()
        return None

    def _extract_text(self, soup: BeautifulSoup, selectors: List[str]) -> Optional[str]:
        for sel in selectors:
            if sel.startswith("meta"):
                el = soup.select_one(sel)
                if el and el.get("content"):
                    content = self._coerce_str(el.get("content"))
                    if content:
                        return content
            else:
                el = soup.select_one(sel)
                if el:
                    return el.get_text(strip=True)
        return None

    def _extract_date(self, soup: BeautifulSoup) -> Optional[str]:
        for meta in soup.select('meta[property="article:published_time"], meta[name="date"]'):
            content = self._coerce_str(meta.get("content"))
            if content:
                return content
        for sel in [".date", ".publish-date", "time", ".article-date"]:
            el = soup.select_one(sel)
            if el:
                return self._coerce_str(el.get("datetime")) or el.get_text(strip=True)
        return None

    def _extract_categories(self, soup: BeautifulSoup, url: str) -> Tuple[Optional[str], Optional[str]]:
        category = "liban-a-table"
        subcategory = None

        breadcrumbs = soup.select(".breadcrumb a, .breadcrumbs a")
        if len(breadcrumbs) > 1:
            category = breadcrumbs[-1].get_text(strip=True)

        path = urlparse(url).path
        parts = [p for p in path.split("/") if p and p != "cuisine-liban-a-table"]
        parts = [p for p in parts if not p.isdigit()]
        if parts:
            category = parts[0]
        if len(parts) > 1:
            subcategory = parts[1]

        return category, subcategory

    def _extract_image(self, soup: BeautifulSoup, json_ld: List[Dict[str, Any]]) -> Optional[str]:
        og = soup.select_one('meta[property="og:image"]')
        if og and og.get("content"):
            return self._coerce_str(og.get("content"))

        image = self._json_ld_first(json_ld, ["image"])
        if isinstance(image, str):
            return image
        if isinstance(image, dict) and image.get("url"):
            return self._coerce_str(image.get("url"))

        for sel in [".article-image img", ".featured-image img", "article img"]:
            img = soup.select_one(sel)
            if img and img.get("src"):
                src = self._coerce_str(img.get("src"))
                if src:
                    return urljoin(BASE_URL, src)
        return None

    def _extract_content(self, soup: BeautifulSoup) -> Optional[str]:
        for sel in [".article-content", ".entry-content", ".article-body", "article", "main"]:
            el = soup.select_one(sel)
            if el:
                for tag in el.select("script, style, nav, footer, aside"):
                    tag.decompose()
                text = el.get_text(separator="\n", strip=True)
                return self._clean_text(text)
        return None

    def _extract_tags(self, soup: BeautifulSoup, json_ld: List[Dict[str, Any]]) -> List[str]:
        tags: List[str] = []
        for sel in [".tags a", ".article-tags a", ".keywords a", 'meta[name="keywords"]']:
            if sel.startswith("meta"):
                meta = soup.select_one(sel)
                if meta and meta.get("content"):
                    content = self._coerce_str(meta.get("content"))
                    if content:
                        tags.extend([t.strip() for t in content.split(",") if t.strip()])
            else:
                for el in soup.select(sel):
                    text = el.get_text(strip=True)
                    if text:
                        tags.append(text)

        keywords = self._json_ld_first(json_ld, ["keywords"])
        if isinstance(keywords, str):
            tags.extend([t.strip() for t in keywords.split(",") if t.strip()])
        elif isinstance(keywords, list):
            tags.extend([str(t).strip() for t in keywords if str(t).strip()])

        return sorted(set(tags))

    def _extract_json_ld(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        for script in soup.select('script[type="application/ld+json"]'):
            if not script.string:
                continue
            try:
                parsed = json.loads(script.string)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, list):
                data.extend([p for p in parsed if isinstance(p, dict)])
            elif isinstance(parsed, dict):
                if "@graph" in parsed and isinstance(parsed["@graph"], list):
                    data.extend([p for p in parsed["@graph"] if isinstance(p, dict)])
                else:
                    data.append(parsed)
        return data

    def _json_ld_by_type(self, json_ld: List[Dict[str, Any]], type_name: str) -> Optional[Dict[str, Any]]:
        for item in json_ld:
            item_type = item.get("@type")
            if isinstance(item_type, list) and type_name in item_type:
                return item
            if isinstance(item_type, str) and item_type.lower() == type_name.lower():
                return item
        return None

    def _json_ld_has_type(self, json_ld: List[Dict[str, Any]], type_name: str) -> bool:
        return self._json_ld_by_type(json_ld, type_name) is not None

    def _json_ld_first(self, json_ld: List[Dict[str, Any]], keys: List[str]) -> Optional[Any]:
        for item in json_ld:
            for key in keys:
                if key in item:
                    return item[key]
        return None

    def _json_ld_author_name(self, item: Dict[str, Any]) -> Optional[str]:
        author = item.get("author")
        if isinstance(author, dict):
            return author.get("name")
        if isinstance(author, list) and author:
            first = author[0]
            if isinstance(first, dict):
                return first.get("name")
            return str(first)
        if isinstance(author, str):
            return author
        return None

    def _normalize_instructions(self, instructions: Any) -> List[str]:
        if not instructions:
            return []
        if isinstance(instructions, str):
            return [instructions]
        if isinstance(instructions, list):
            normalized = []
            for item in instructions:
                if isinstance(item, str):
                    normalized.append(item)
                elif isinstance(item, dict) and item.get("text"):
                    normalized.append(str(item["text"]).strip())
            return [n for n in normalized if n]
        return []

    def _ensure_list(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        return [str(value).strip()]

    def _coerce_str(self, value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, dict):
            if "name" in value:
                return self._coerce_str(value.get("name"))
            return None
        if isinstance(value, list):
            if not value:
                return None
            return str(value[0]).strip()
        text = str(value).strip()
        return text or None

    def _clean_text(self, text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join([line for line in lines if line])

    def _is_article_url(self, url: str) -> bool:
        return bool(ARTICLE_ID_RE.search(url))

    def _is_section_url(self, url: str) -> bool:
        return bool(SECTION_LINK_RE.search(url)) and not self._is_article_url(url)

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        normalized = parsed._replace(query="", fragment="")
        return urlunparse(normalized).rstrip("/")

    def _with_page_param(self, url: str, page: int) -> str:
        if page <= 1:
            return url
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        params["page"] = [str(page)]
        new_query = urlencode(params, doseq=True)
        return urlunparse(parsed._replace(query=new_query))

    def _load_resume_urls(self, path: Path) -> Set[str]:
        urls: Set[str] = set()
        if not path.exists():
            return urls
        with path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    url = data.get("url")
                    if url:
                        urls.add(url)
                except json.JSONDecodeError:
                    continue
        return urls

    def _validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = [
            "url",
            "title",
            "content_type",
            "category",
            "subcategory",
            "author",
            "date",
            "image_url",
            "description",
            "full_text",
            "tags",
            "chef_name",
            "ingredients",
            "instructions",
            "prep_time",
            "cook_time",
            "servings",
            "restaurant_name",
            "location",
            "scraped_at",
        ]
        for key in required_keys:
            record.setdefault(key, None)
        if record["tags"] is None:
            record["tags"] = []
        if record["ingredients"] is None:
            record["ingredients"] = []
        if record["instructions"] is None:
            record["instructions"] = []
        if record["location"] is None:
            record["location"] = {"city": None, "country": None, "neighborhood": None, "address": None}
        return record


async def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Liban a Table (Playwright)")
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES, help="Max pages per section")
    parser.add_argument("--max-articles", type=int, default=None, help="Limit number of articles")
    parser.add_argument("--output", "-o", default="data/liban_a_table.jsonl", help="Output JSONL path")
    parser.add_argument("--resume", action="store_true", help="Skip URLs already in output")
    parser.add_argument("--headful", action="store_true", help="Run browser in visible mode")
    args = parser.parse_args()

    output_path = Path(__file__).resolve().parent.parent.parent / args.output

    scraper = LibanATableScraper(
        output_path=output_path,
        max_pages=args.max_pages,
        max_articles=args.max_articles,
        resume=args.resume,
        headless=not args.headful,
    )
    await scraper.run()


if __name__ == "__main__":
    asyncio.run(main())
