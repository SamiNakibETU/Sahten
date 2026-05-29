# Audit RAG OLJ inspire d'Epicure

## Synthese executive

Sahtein v4 est deja une refonte solide du RAG culinaire OLJ: ingestion WhiteBeard structuree, schema Postgres normalise, chunking par sections, retrieval hybride `pgvector` + `tsvector`, fusion RRF, reranking Cohere, reranking article-level et grounding phrase par phrase. C'est une base beaucoup plus saine que l'ancienne stack JSON + TF-IDF, surtout parce que les donnees sources restent auditables dans Postgres et que le pipeline separe ingestion, retrieval, rerank et generation.

Le papier Epicure ne dit pas simplement "utiliser de meilleurs embeddings". Sa lecon utile pour Sahtein est plus structurelle: dans un domaine culinaire, la qualite vient d'un vocabulaire canonique robuste, de signaux de retrieval explicites et controles, d'une geometrie evaluee, puis seulement d'operateurs de navigation plus avances. Applique a OLJ, cela suggere de ne pas remplacer le pipeline actuel, mais de lui ajouter progressivement une couche de diagnostic et de controle: d'abord mieux evaluer, ensuite consolider les ingredients, puis exploiter les cooccurrences recette-ingredient avant de penser a des embeddings ingredients ou a une navigation type SLERP.

La faiblesse principale actuelle est donc moins le choix `text-embedding-3-small` que l'absence de mesure fine de ce que l'espace vectoriel et les alias culinaires savent vraiment faire. Le golden set est trop petit, les alias ingredients restent en partie manuels, la couverture WhiteBeard n'est pas encore prouvee de bout en bout, et les heuristiques produit accumulees dans `pipeline.py` et `ingredient_match.py` risquent de masquer les echecs au lieu de les rendre mesurables.

## Systeme actuel observe

Le systeme cible est `sahten_github/v4`, decrit dans `README.md` comme une architecture Postgres + pgvector + tsvector + RRF SQL + cross-encoder rerank + grounding. Le chemin principal est:

1. `backend/app/ingestion/whitebeard_client.py`, `mapper.py`, `html_sectionizer.py`, `repository.py` ingestent les articles WhiteBeard et preservent sections, chefs, categories, mots-cles et ingredients.
2. `backend/app/rag/chunker.py` construit des chunks adaptes aux recettes: chunk d'ancrage, item-level pour ingredients/etapes, prose par phrases.
3. `backend/app/rag/embeddings.py` genere les embeddings OpenAI; `backend/app/rag/indexer.py` reinsere les chunks vectorises.
4. `backend/app/rag/retriever.py` execute une requete SQL hybride: filtres structurants, lexical `tsvector`, vectoriel `pgvector`, fusion RRF.
5. `backend/app/rag/reranker.py` applique Cohere `rerank-multilingual-v3.0` ou BGE local.
6. `backend/app/rag/pipeline.py` orchestre query understanding, focus session, widening, interleaving, reranking chunk/article, priorite source, generation et validation du grounding.
7. `backend/app/llm/response_generator.py` produit les reponses sourcees et les cartes recette.
8. `scripts/run_rag_eval.py` verifie un golden set JSON avec hit retrieval et fragments attendus dans la reponse.

Forces importantes:

- Le schema `backend/app/db/models.py` contient deja les bons objets de base: `Article`, `ArticleSection`, `Person`, `Category`, `Keyword`, `Ingredient`, `ArticleIngredient`, `Chunk`.
- Le chunking est domaine-aware: les listes d'ingredients et d'etapes ne sont pas diluees dans de gros blocs de prose.
- Le retrieval combine lexical et vectoriel; c'est crucial pour les noms propres, translitterations et plats libanais.
- Le rerank injecte titre et type de section, ce qui reduit les faux positifs de chunks isoles.
- Le pipeline a deja des mecanismes de diversite article-level: interleaving avant rerank et reranking au niveau article.
- Le grounding phrase par phrase est une vraie protection produit, superieure a une simple reponse LLM avec sources decoratives.

Fragilites principales:

- `data/golden_eval_fr.json` ne contient que quelques cas; il ne couvre pas assez les intentions, les suivis conversationnels, les refus, les ingredients rares, les recettes proches, ni les echecs attendus.
- `ingredient_match.py` contient une liste d'alias utile mais petite et manuelle. Epicure montre que le vocabulaire canonique est un actif central, pas un detail.
- La qualite de l'espace vectoriel n'est pas diagnostiquee: pas de voisinage ingredients, pas de mesure de diversite, pas d'isotropie, pas de stabilite de clusters, pas de separabilite par rubrique/ingredient/cuisine.
- Les heuristiques de `pipeline.py` corrigent des cas reels, mais certaines sont tres specifiques (`concombre`, `fattouche`, Base2 last resort). Sans evaluation plus large, elles peuvent devenir une dette comportementale.
- La documentation operationnelle mentionne encore un risque de configuration `text-embedding-3-large` / `3072` dans le runbook, alors que `settings.py` documente `text-embedding-3-small` / `1536` pour rester compatible HNSW pgvector.
- Le backfill complet et la pagination WhiteBeard restent des preconditions de qualite. Un RAG peut sembler mauvais alors qu'il est surtout incomplet.

## Ce qu'Epicure apporte vraiment

Epicure est un papier sur des embeddings d'ingredients, pas un papier RAG. Les idees transferables sont donc methodologiques.

### 1. Canonicaliser avant d'optimiser

Epicure reduit environ 200k chaines ingredients a 1 790 ingredients canoniques. C'est le point le plus important pour Sahtein. Le RAG actuel a bien une table `ingredients`, des aliases JSON et des filtres SQL, mais la couche canonique semble encore surtout issue de l'ingestion disponible et de corrections manuelles.

Implication Sahtein: avant d'ajouter un modele avance, construire un inventaire auditable des ingredients, aliases, translitterations, pluriels, variantes franco-arabes et plats-ingredients. Chaque alias doit pouvoir etre rattache a un slug canonique, avec exemples d'articles et tests.

### 2. Nommer les signaux de retrieval

Epicure separe cooccurrence recette et signal chimique au lieu de les fusionner dans une boite noire. Sahtein a deja plusieurs signaux, mais ils sont surtout fusionnes dans RRF puis rerank:

- texte lexical francais (`tsvector`);
- embedding de chunk;
- filtres structures chef/ingredient/categorie/keyword;
- boost et filtres ingredients;
- rerank chunk;
- rerank article.

Implication Sahtein: rendre ces signaux mesurables separement. Pour chaque requete de test, enregistrer si le bon article vient du lexical, du vectoriel, du filtre ingredient, du widening, ou du rerank. Sans cela, on ne sait pas quel levier ameliorer.

### 3. Evaluer la geometrie, pas seulement les reponses

Epicure mesure directions, isotropie, separabilite et coherence de modes. Sahtein mesure aujourd'hui surtout "l'article attendu est-il dans top-k" et "la reponse contient-elle quelques mots". C'est necessaire, mais insuffisant.

Implication Sahtein: ajouter une evaluation intrinseque avant de toucher au pipeline:

- voisins d'un ingredient ou d'un plat: les top voisins sont-ils culinaires et utiles?
- separabilite par type de section: les chunks `ingredients_list`, `recipe_steps`, `chef_bio` se melangent-ils trop?
- diversite article-level: top 12 contient-il 12 chunks du meme article ou plusieurs recettes?
- sensibilite aux aliases: `concombre`, `khyar`, `khiar` donnent-ils les memes articles?
- robustesse conversationnelle: "une autre", "pas celle-la", "sans X" changent-ils correctement le contexte?

### 4. Passer de recommandation a navigation

Epicure expose des operateurs: voisins, modes, rotation vers un pole. Pour Sahtein, l'equivalent produit n'est pas necessairement SLERP tout de suite. Les usages concrets seraient:

- "trouve une recette proche mais plus simple";
- "meme ingredient, autre registre";
- "meme plat, autre chef";
- "autour du concombre mais pas salade";
- "recette libanaise proche du couscous";
- "autre angle autour de Kamal Mouzawak / Souk el-Tayeb".

Implication Sahtein: ces comportements peuvent commencer par des signaux controles et evaluables, sans entrainer Metapath2Vec.

## Grille critique Epicure -> Sahtein

### Vocabulaire canonique

Etat actuel: la table `Ingredient` et `ArticleIngredient` existent, et `ingredient_match.py` ajoute des aliases determinants. Le systeme sait corriger certains cas critiques comme `concombre` / `khyar`.

Critique: l'approche est encore trop locale. Les aliases vivent dans le code, pas comme donnees auditees. Les ingredients rares ou variantes libanaises non prevues peuvent etre invisibles au filtre structure, meme si le texte les contient.

Audit recommande:

- Exporter tous les ingredients distincts depuis Postgres avec leurs aliases et frequences.
- Comparer les ingredients structures aux occurrences dans chunks `ingredients_list`.
- Identifier les slugs sans aliases, aliases orphelins, pluriels manquants et translitterations probables.
- Creer un jeu de tests d'alias par ingredient critique.

### Cooccurrence culinaire

Etat actuel: le retrieval sait filtrer par ingredient et retourner des chunks, mais il ne semble pas exploiter explicitement un graphe de cooccurrence ingredient-recette ou ingredient-ingredient.

Critique: pour une requete de type "avec quoi cuisiner X" ou "recette proche mais autre ingredient", l'embedding de chunks est oblige de porter seul une relation culinaire qui serait mieux representee par des cooccurrences.

Audit recommande:

- Construire hors pipeline un rapport de cooccurrence depuis `ArticleIngredient`: ingredient -> articles -> ingredients associes.
- Calculer des voisins simples par NPMI ou PMI positif, sans encore les deployer.
- Comparer ces voisins aux resultats du retrieval actuel sur 20 ingredients frequents.

### Embeddings de chunks

Etat actuel: `text-embedding-3-small` / 1536 dimensions est coherent avec pgvector HNSW, et le chunking donne des textes bien ancres.

Critique: un embedding generaliste de chunk n'est pas un embedding culinaire d'ingredient. Il peut bien faire le retrieval documentaire tout en etant mauvais pour la navigation culinaire.

Audit recommande:

- Ne pas passer a `text-embedding-3-large` avant de prouver un gain, car la dimension 3072 implique une contrainte pgvector/halfvec.
- Evaluer separement recall lexical, recall vectoriel et recall apres RRF.
- Tester si les chunks d'ingredients sont plus proches de chunks d'etapes du meme article que de recettes similaires, ce qui indiquerait une geometrie dominee par l'article.

### Reranking

Etat actuel: Cohere rerank est bien place apres RRF, avec documents enrichis par titre et type de section. Le pipeline ajoute aussi une deuxieme passe article-level.

Critique: le reranker corrige des erreurs de retrieval mais peut aussi masquer un manque de signal ingredient. Sans logs de contribution par etape, on ne sait pas si le rerank ameliore la pertinence culinaire ou seulement la correspondance textuelle.

Audit recommande:

- Pour chaque item de golden set, enregistrer top-k avant rerank, apres chunk rerank, apres article rerank.
- Mesurer MRR article et Recall@K a chaque etape.
- Ajouter des cas ou le bon article n'a pas les memes mots que la requete mais partage une relation culinaire.

### Grounding

Etat actuel: le grounding phrase par phrase est un point fort. Il limite l'invention et force les reponses a rester ancrees dans les chunks.

Critique: pour des fonctions de navigation culinaire, le grounding va devenir plus difficile. Une suggestion de substitution ou de voisinage n'est pas toujours citee mot pour mot dans un article.

Audit recommande:

- Distinguer deux modes de reponse: "information sourcee depuis OLJ" et "suggestion culinaire calculee depuis le graphe".
- Ne pas melanger les deux sans signal UX clair.
- Exiger des sources OLJ pour les faits, mais autoriser des justifications differentes pour les suggestions algorithmiques si elles sont etiquetees.

### Evaluation

Etat actuel: le script `scripts/run_rag_eval.py` est simple, CI-friendly et utile. Le golden set actuel est trop petit pour guider les choix.

Critique: on peut casser des cas conversationnels ou ingredients rares sans que l'eval le voie.

Audit recommande:

- Porter le golden set a 40-80 cas avant tout chantier ML.
- Ajouter des categories explicites: recette precise, ingredient, chef, histoire, astuce, suivi conversationnel, objection, hors corpus, alias, plat levantin.
- Mesurer `retrieval_ok`, `answer_ok`, MRR, precision ingredient, nombre d'articles distincts et presence de source canonique OLJ.
- Garder une section "anti-regression cas sensibles" pour les bugs deja corriges: concombre, fattouche, carnets, Base2, objections.

## Risques actuels prioritaires

1. Le systeme peut paraitre moins bon qu'il ne l'est si le backfill WhiteBeard est incomplet.
2. Le systeme peut paraitre meilleur qu'il ne l'est si le golden set reste trop petit.
3. Les aliases ingredients codes en dur peuvent se multiplier jusqu'a rendre le comportement difficile a expliquer.
4. Les corrections conversationnelles dans `pipeline.py` peuvent devenir une accumulation de cas particuliers.
5. Un passage premature a `text-embedding-3-large` peut ajouter de la complexite infra sans gain mesure.
6. Une couche "geometrie culinaire" de type Epicure serait fragile si le vocabulaire canonique n'est pas d'abord stabilise.
7. Les suggestions culinaires avancees risquent de brouiller la promesse editoriale OLJ si elles ne sont pas distinguees des faits sources.

## Feuille de route priorisee

### P0 - Rendre les echecs mesurables

Objectif: savoir ou le pipeline gagne et ou il perd.

Actions:

- Elargir `data/golden_eval_fr.json` avec 40-80 cas classes par intention.
- Modifier ulterieurement `scripts/run_rag_eval.py` pour capturer top-k avant/apres rerank, MRR et diversite article-level.
- Ajouter un mode rapport qui liste les requetes ou lexical seul trouve mieux que vectoriel, et inversement.
- Ajouter des cas de regression conversationnelle: "une autre", "pas celle-la", "sans X", "pas dessert", "recette simple pour le soir".

Critere de sortie:

- Chaque changement RAG peut etre juge sur des metriques stables, pas seulement sur quelques essais manuels.

### P1 - Stabiliser le vocabulaire culinaire

Objectif: faire du referentiel ingredient un actif produit.

Actions:

- Exporter un audit ingredients avec `scripts/audit_ingredients.py`: slug, nom, aliases, frequence article, frequence chunk `ingredients_list`, exemples.
- Detecter automatiquement les variantes simples: pluriels, accents, tirets, translitterations frequentes.
- Deplacer progressivement les aliases depuis le code vers une donnee auditable si le schema le permet deja via `Ingredient.aliases`.
- Ajouter des tests par ingredient critique dans `tests/test_ingredient_match.py`.

Critere de sortie:

- Les requetes ingredient frequentes et leurs variantes retournent les memes familles de recettes et ne dependent pas de regex disperses.

### P2 - Ajouter une analyse de cooccurrence ingredients-recettes

Objectif: obtenir le premier equivalent pratique du signal Cooc d'Epicure sans changer le runtime.

Actions:

- Generer un rapport offline ingredient-ingredient depuis `ArticleIngredient`.
- Calculer voisins par frequence et NPMI.
- Comparer les voisins cooccurrence aux voisins de retrieval actuel.
- Identifier les expansions utiles pour les requetes "avec quoi", "autre idee", "meme registre".

Critere de sortie:

- On sait si un graphe simple apporte une valeur reelle avant de l'integrer au retrieval.

### P3 - Experimenter une couche d'embeddings ingredients

Objectif: tester une geometrie culinaire separee de l'embedding documentaire.

Actions:

- Construire un prototype offline, non branche au pipeline, avec un vecteur par ingredient canonique.
- Tester deux sources simples: descriptions textuelles OLJ agreges par ingredient, et cooccurrence ingredient-recette.
- Evaluer les voisins sur un panel d'ingredients libanais: concombre, menthe, persil, boulgour, pois chiche, aubergine, courgette, poulet, citron, yaourt.
- Mesurer coherence qualitative et stabilite avant toute integration.

Critere de sortie:

- Les voisins ingredients sont suffisamment utiles pour justifier une integration comme signal supplementaire.

### P4 - Concevoir la navigation culinaire

Objectif: exposer des controles utilisateur inspires d'Epicure seulement si les fondations tiennent.

Actions possibles:

- "Rester proche / explorer plus loin" comme controle de diversite article-level.
- "Meme ingredient / autre plat" via exclusion d'articles deja vus et expansion cooccurrence.
- "Autre registre" via categories, keywords, chefs ou modes de cooccurrence.
- "Substitution" uniquement si l'evaluation ingredient prouve une geometrie fiable.

Critere de sortie:

- Les suggestions avancees sont separees des faits sources OLJ et ont leurs propres metriques d'utilite.

## Critique du papier pour eviter la sur-transposition

Epicure est inspire mais non directement deployable dans Sahtein.

- Le corpus Epicure est massif et multilingue; OLJ cuisine est plus petit, editorial et probablement desequilibre.
- Le signal "chimie" n'a pas d'equivalent direct dans les donnees Sahtein actuelles. Le plus proche serait une taxonomie ingredient/categorie ou une base externe, mais ce serait un nouveau projet.
- Les modes FastICA/GMM sont interpretes et nommes; cela peut produire de beaux labels sans garantir une utilite produit.
- SLERP suppose un espace vectoriel ou les directions sont stables et signifiantes. Rien ne prouve que les embeddings de chunks Sahtein aient cette propriete.
- Le papier utilise des LLM pour canonicalisation et labels; cela reste une source de biais meme si le modele final n'est pas lui-meme un LLM.
- Le code et les artefacts ne sont pas publies; il faut reprendre les principes, pas les hyperparametres.

## Decision recommandee

Ne pas lancer tout de suite un chantier "Epicure complet". La meilleure sequence est:

1. verrouiller evaluation et observabilite;
2. stabiliser le vocabulaire ingredient;
3. produire un rapport de cooccurrence offline;
4. tester une geometrie ingredient separee;
5. seulement ensuite concevoir une UX de navigation culinaire.

Cette approche respecte la direction du papier tout en protegeant ce qui fait deja la valeur de Sahtein v4: un RAG documentaire source, auditable et deployable.

