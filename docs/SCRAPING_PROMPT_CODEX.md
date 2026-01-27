# Prompt optimisé pour GPT Codex / o1 / o3 - Scraping "Liban à Table"

## Contexte d'utilisation
Ce prompt est conçu pour être utilisé avec GPT-4 Codex, o1, o3 ou tout modèle de raisonnement avancé capable d'exécuter du code.

---

## PROMPT

```
<system>
Tu es un expert en web scraping Python. Tu écris du code production-ready, robuste et bien documenté.
</system>

<task>
Scrape EXHAUSTIVEMENT la section "Liban à Table" de L'Orient-Le Jour.
URL racine: https://www.lorientlejour.com/cuisine-liban-a-table
</task>

<constraints>
- Le site bloque les User-Agents basiques (retourne 403)
- Utilise Playwright (headless browser) pour contourner la protection
- Respecte un délai de 1-2 secondes entre requêtes (politeness)
- Gère la pagination (paramètre ?page=N)
- Gère les erreurs et retries (max 3 tentatives)
</constraints>

<structure_site>
5 sections principales à scraper:
1. /cuisine-liban-a-table (page d'accueil + pagination)
2. Sous-sections dans le menu:
   - Recettes (+ sous-catégories: "Variations autour d'un thème", "Les hommes aux fourneaux", etc.)
   - Nos sélections gourmandes
   - Chefs
   - Restaurants (+ "Nos coups de coeur", "Les ouvertures au Liban")
   - Vins et plus
</structure_site>

<output_schema>
Pour chaque article, extrais:
```json
{
  "url": "string (URL complète)",
  "title": "string",
  "content_type": "recipe | portrait | restaurant | selection | story | wine | guide",
  "category": "string (Recettes, Chefs, Restaurants, etc.)",
  "subcategory": "string | null",
  "author": "string | null",
  "date": "string ISO | null",
  "image_url": "string | null",
  "description": "string (lead/excerpt)",
  "full_text": "string (contenu complet)",
  "tags": ["array", "of", "tags"],
  
  // Si recipe:
  "chef_name": "string | null",
  "ingredients": ["array"] | null,
  "instructions": ["array"] | null,
  "prep_time": "string | null",
  "cook_time": "string | null",
  "servings": "string | null",
  
  // Si restaurant:
  "restaurant_name": "string | null",
  "location": {
    "city": "string",
    "country": "string",
    "neighborhood": "string | null",
    "address": "string | null"
  } | null,
  
  // Metadata
  "scraped_at": "ISO timestamp"
}
```
</output_schema>

<detection_rules>
Détermine content_type selon ces règles:
- recipe: présence de "ingrédients", "préparation", balises schema.org Recipe, ou "recette" dans le titre
- portrait: "chef", "rencontre", "interview", "parcours" dans titre/contenu
- restaurant: "restaurant", "adresse", "où manger", nom d'établissement
- selection: "nos X", "sélection", liste de recommandations
- wine: "vin", "œnologie", "brasserie", "bar à vin"
- story: "histoire", "tradition", "patrimoine"
- guide: "tout savoir sur", "comment", tutoriel
</detection_rules>

<implementation>
1. Utilise playwright (async) avec:
   - Browser context avec User-Agent Chrome réaliste
   - Viewport desktop (1920x1080)
   - Locale fr-FR
   - Désactive webdriver detection

2. Stratégie de crawl:
   a) Récupère tous les liens d'articles depuis chaque page de listing
   b) Déduplique par URL
   c) Visite chaque article et extrait les données
   d) Sauvegarde en JSON Lines (1 article par ligne) pour robustesse

3. Sélecteurs CSS à essayer (dans l'ordre):
   - Article links: 'a[href*="/cuisine-liban-a-table/"][href*="/1"]' (URLs avec ID numérique)
   - Title: 'h1', '.article-title', '[itemprop="headline"]'
   - Author: '.author-name', '.byline', '[rel="author"]'
   - Date: 'time[datetime]', '.publish-date', 'meta[property="article:published_time"]'
   - Content: '.article-content', '.entry-content', 'article'
   - Ingredients: '.ingredients li', '[itemprop="recipeIngredient"]'
   - Instructions: '.instructions li', '[itemprop="recipeInstructions"]'
   - Image: 'meta[property="og:image"]', '.article-image img'

4. Output:
   - data/liban_a_table_raw.jsonl (données brutes)
   - data/liban_a_table_stats.json (statistiques du scrape)
</implementation>

<code_structure>
Génère un script Python complet avec:
- Classe LibanATableScraper
- Méthode async main()
- Logging vers console et fichier
- Progress bar (tqdm)
- Resume capability (skip URLs déjà scrapées)
- CLI avec argparse (--max-articles, --output, --resume)
</code_structure>

<quality_checks>
Avant de terminer, vérifie:
- [ ] Gestion des cookies/consent banners (ferme-les automatiquement)
- [ ] Gestion des popups
- [ ] Extraction des données structurées schema.org si présentes
- [ ] Normalisation des dates en ISO 8601
- [ ] Nettoyage du texte (strip, remove extra whitespace)
- [ ] Validation du JSON output
</quality_checks>

Génère le code Python complet, prêt à exécuter.
```

---

## Variante courte (si limite de tokens)

```
Écris un scraper Playwright (Python async) pour https://www.lorientlejour.com/cuisine-liban-a-table

Contraintes:
- Site bloque User-Agent basique → utilise browser headless avec anti-detection
- Scrape TOUTES les pages (pagination ?page=N)
- Scrape TOUTES les sections: Recettes, Chefs, Restaurants, Vins, Sélections

Pour chaque article, extrait: url, title, content_type (recipe/portrait/restaurant/selection/wine), author, date, full_text, image_url, et si recette: chef_name, ingredients[], instructions[]

Output: JSONL + stats JSON
Code production-ready avec retry, logging, progress bar, resume capability.
```

---

## Notes d'utilisation

1. **Avec GPT-4 Codex (API)**: Utilise le prompt complet dans `<task>...</task>`
2. **Avec o1/o3**: Le format structuré aide le raisonnement
3. **Avec Claude**: Ajoute `<thinking>` tags si nécessaire
4. **Avec Cursor Agent**: Copie le prompt et demande l'exécution

## Dépendances requises

```bash
pip install playwright tqdm aiofiles
playwright install chromium
```
