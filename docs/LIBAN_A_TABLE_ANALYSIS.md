# Analyse de la section "Liban à Table" - L'Orient-Le Jour

## Date d'analyse
23 janvier 2026

## Structure de la section

La section "Liban à Table" (https://www.lorientlejour.com/cuisine-liban-a-table) comprend **5 catégories principales** :

### 1. Recettes
Sous-catégories identifiées :
- **Variations autour d'un thème** - Déclinaisons créatives de plats classiques
- **Les hommes aux fourneaux** - Recettes de chefs masculins
- **La cuisine à l'heure des réseaux sociaux** - Tendances food sur les réseaux
- **Tout savoir sur** - Guides sur ingrédients/techniques

**Exemples de contenu :**
- "La soupe de potiron à la muscade de Sophie Schoucair"
- "Le cheesecake à la labné, aux betteraves et au sirop de rose"
- "La salade de zoodle et fenouil de Tarek Alameddine"
- "Les gnocchis de patate douce au bleu d'Andréa Boueiz"
- "Les pâtes au kishk et pesto de zaatar"
- "La tarte craquante au labné"
- "Le gâteau au chocolat infusé au zaatar de Barbara Abdeni Massaad"

### 2. Nos sélections gourmandes
Compilations thématiques et recommandations :
- "Street-food libanaise à Londres : nos 6 adresses, du classique au revisité"
- "Galettes des rois : notre sélection gourmande 100% libanaise"
- "Mots d'artistes et recettes : Notre sélection 100% chocolat"
- "Nos recettes de soupes et veloutés"
- "Nos recettes simples et rapides"
- "Nos tartes sucrées et salées"

### 3. Chefs
Portraits et interviews de chefs libanais :
- "À Milan, Maradona Youssef tisse de savoureux liens entre les cuisines italienne et libanaise"
- "Deux Libanais primés aux TikTok Awards MENA"
- "Rita Maria Kordahi, une cuisine libanaise cousue main à Paris"

**Auteurs/Journalistes identifiés :**
- Rayanne TAWIL
- Layal DAGHER
- Joséphine HOBEIKA

### 4. Restaurants
Sous-catégories :
- **Nos coups de coeur** - Recommandations de la rédaction
- **Les ouvertures au Liban** - Nouveaux restaurants

**Exemples :**
- "O'Zeit : le pari libanais de Tarek Mansour et Zineb Makhlouq à Noisy-le-Grand"
- "À Badaro, Bistrot Lobo mise sur le bistrot français version beyrouthine"
- "Kasr Fakhreddine, une des institutions de la cuisine libanaise à Broummana, investit Beyrouth"

**Localisations :**
- Liban (Beyrouth, Badaro, Broummana)
- France (Paris, Noisy-le-Grand)
- Royaume-Uni (Londres)
- Italie (Milan)

### 5. Vins et plus
Œnologie et boissons :
- "Alchimiste ou sorcier, qui était Guy Accad, le Libanais qui a bouleversé la vinification en Bourgogne ?"
- "Elmir 2.0 : la brasserie libanaise artisanale passe à la vitesse supérieure"
- "Levain : un bar à vin libanais au cœur de Monnot"

---

## Types de contenu identifiés

| Type | Description | Exemple |
|------|-------------|---------|
| `recipe` | Recette complète avec ingrédients/étapes | "La soupe de potiron à la muscade" |
| `portrait` | Interview/portrait de chef | "Rita Maria Kordahi" |
| `restaurant` | Critique/présentation de restaurant | "Bistrot Lobo à Badaro" |
| `selection` | Liste/compilation thématique | "Nos 6 adresses street-food à Londres" |
| `story` | Article culturel/historique | "Guy Accad et la vinification en Bourgogne" |
| `guide` | Guide pratique/tutorial | "Tout savoir sur le kishk" |
| `wine` | Article sur vins/boissons | "Bar à vin Levain" |

---

## Chefs mentionnés (échantillon)

- Sophie Schoucair
- Tarek Alameddine
- Andréa Boueiz
- Barbara Abdeni Massaad
- Maradona Youssef
- Rita Maria Kordahi
- Tarek Mansour
- Jaimee Lee Haddad

---

## Colonnes/Séries récurrentes

- **"Les recettes de Jaimee"** - Série de recettes par Jaimee Lee Haddad
- **"Recette de chef"** - Recettes de chefs invités
- **"Saveurs du Liban"** - Focus sur des recettes traditionnelles

---

## Implications pour le RAG multi-source

### Architecture proposée

```
content_type: 
  - recipe         → Pipeline actuel (retrieval + narrative)
  - portrait       → Biographie + lien vers recettes du chef
  - restaurant     → Fiche avec localisation, prix, ambiance
  - selection      → Liste de liens vers recettes/restaurants
  - story          → Article narratif (full-text retrieval)
  - guide          → FAQ/explications techniques
  - wine           → Recommandations de vins
```

### Nouveaux intents possibles

| Intent | Query exemple | Content type |
|--------|---------------|--------------|
| `chef_info` | "Qui est Barbara Abdeni Massaad ?" | portrait |
| `restaurant_search` | "Un bon restaurant libanais à Paris" | restaurant |
| `restaurant_local` | "Où manger à Badaro ?" | restaurant |
| `wine_pairing` | "Quel vin avec le kebbé ?" | wine |
| `story_search` | "L'histoire du taboulé" | story |
| `selection_request` | "Idées de street-food" | selection |

### Enrichissement des données existantes

1. **Ajouter `chef_id`** aux recettes pour lier aux portraits
2. **Ajouter `location`** pour les restaurants (ville, quartier, pays)
3. **Ajouter `series`** pour les colonnes récurrentes
4. **Ajouter `wine_pairing`** aux recettes si disponible

---

## Prochaines étapes recommandées

### Phase 1 - Scraping complet (semi-automatique)
Le site utilise une protection anti-bot (403 sur requêtes HTTP simples).
Options :
- Utiliser le navigateur MCP intégré pour extraction manuelle
- Demander un accès API à l'équipe OLJ
- Utiliser Playwright/Puppeteer avec anti-detection

### Phase 2 - Enrichissement du dataset
- Classifier le contenu existant par `content_type`
- Extraire les métadonnées (chef, lieu, série)
- Créer des liens entre contenus (chef → recettes)

### Phase 3 - Extension du RAG
- Router par `content_type` après analyse de l'intent
- Cards spécialisées par type (RecipeCard, RestaurantCard, ChefCard)
- Prompts narratifs adaptés au type

---

## Conclusion

La section "Liban à Table" est riche et diversifiée, bien au-delà des simples recettes :
- **~60% recettes** (estimation basée sur l'échantillon)
- **~15% restaurants** (avec couverture internationale)
- **~10% portraits de chefs**
- **~10% sélections/guides**
- **~5% vins/boissons**

Cette diversité permet d'enrichir considérablement l'expérience Sahten en proposant non seulement des recettes mais aussi des recommandations de restaurants, des portraits de chefs inspirants, et du contexte culturel sur la gastronomie libanaise.
