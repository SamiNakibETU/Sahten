# Rapport de test ULTIME — Sahteïn (staging) — 2026-06-28

Test live sur le déploiement actuel (`main`/Railway), ~120 requêtes croisées avec
la base réelle (198 articles). Verdict **honnête**, pas une promesse.

## TL;DR
**Fonctionnel à ~80 % sur des requêtes réalistes**, mais **pas parfait**. Les
plats courants et la plupart des cas marchent ; il reste **2 défauts systémiques** :
1. **Fallback « recette proche » incohérent** : quand un plat n'est pas dans la base,
   il abstient parfois à vide (falafel, baklava…) au lieu de proposer un proche.
2. **Confusions de synonymes/translittérations** : fattouche↔fatteh, sfiha↔sfouf,
   « feuilles de vigne »↔warak enab (raté), batata harra↔kafta bi batata.

---

## 1. Plats libanais (40 testés, croisés base) — ~26/40 solides

### ✅ Trouvent la bonne recette (≈26)
houmous → Hommos au cumin · baba ganoush → Baba ghanouj · taboulé → taboulé Mouzawak ·
kebbé/kibbeh → kebbé aux noix · chawarma → chawarma de poulet · **manouche/manakish → manaïichs (1474718, pin déterministe ✅)** ·
samboussek · warak enab · moghrabieh · mouloukhié · kafta → kafta bi batata · lahm bi ajin ·
soupe de lentilles · sfouf · maamoul · ouayamat · riz a djej · mjadara · foul → fatté de foul ·
kishk · freekeh → friké · atayef · su beureg · sayadieh → siyadieh.

### ⚠️/❌ Faibles ou faux (≈14)
| Plat | Résultat | Problème |
|---|---|---|
| **fattouche** | carte=∅, src « Fatteh de pois chiches » | ❌ confusion **fattouche↔fatteh** |
| **feuilles de vigne farcies** | « Feuilles de **chou** farcies » | ❌ rate **warak enab (1468901)** qui existe — synonyme non lié |
| **sfiha** | « Le **sfouf** libanais » | ❌ confusion **sfiha (salé)↔sfouf (dessert)** |
| **batata harra** | « kafta bi batata » | ⚠️ match sur « batata » |
| moutabbal | aubergine (panade/ma'loubit) | ⚠️ devrait pointer baba ghanouj |
| labneh | carte « salade petits pois » | ⚠️ mal-carté (un article labné existe) |
| falafel, baklava, basboussa, mouhalabieh, knefe, chich taouk | abstention **à vide** | ⚠️ absents de la base → devrait proposer un **proche** (pois chiche / dessert / poulet), pas rien |

---

## 2. Non-libanais (10 FR / 5 IT / 5 ES / 5 connus) — ✅ bon
- **Abstention propre** sur les vrais absents : bœuf bourguignon, quiche, blanquette,
  gratin, soupe à l'oignon, croque-monsieur, crêpes, pizza, carbonara, lasagnes,
  paella, gazpacho, tortilla, churros, patatas bravas, sushi, burger, ramen.
- **Trouve ceux qui SONT dans le corpus** (revisités OLJ) : ratatouille (Bernard Hage),
  tarte tatin, risotto al pesto, tiramisu, **tacos de chawarma** → cohérent.
- ⚠️ Légers : coq au vin (décrit sans carte), pad thaï (propose le bœuf thaï du corpus).
→ **Pas de hallucination de recette inexistante.** Comportement correct.

## 3. Ingrédients (20) — ~15/20 pertinents, rotation OK
✅ aubergine, courgette, pois chiche, poulet, bœuf, citron, persil, boulgour, tomate,
concombre, zaatar, semoule, yaourt, fleur d'oranger, grenade.
⚠️ agneau (panade d'aubergine), menthe, tahini, pignons (poisson tajine) ;
❌ **labneh → tarte aux tomates séchées** (aucun labné).

## 4. Mood (10) — ~9/10 bon
léger→salade petits pois · rapide→capellini · réconfortant→Lahm bi aajine ·
été→salade estivale · festif→bûche chocolat · végétarien→maghmour · sain→salade
artichauts · ramadan→riz aa djej · dessert peu sucré→ossmalié kaki. ⚠️ « hiver »→langue
de bœuf (un peu hors-sujet).

## 5. Relances multi-tours (10) — ~8/10 bon
- **Diversification « une autre » : OK** (poulet, dessert… changent de recette).
- **« et un plat pour accompagner cette entrée » → propose bien un PLAT** (riz aa djej) ✅.
- ⚠️ « houmous » → « pour tremper » → **redonne houmous** (n'offre pas un autre dip).
- ⚠️ « aubergine » → « plutôt grillé » → même plat (ne s'adapte pas à « grillé »).

---

## Diagnostic honnête
- ✅ **Le bug que tu avais signalé (manouche) est réglé et déterministe.** Les plats
  **courants** et **présents dans la base** marchent bien, avec une **rédaction riche**.
- ✅ **Sécurité, abstention non-libanais, diversification, follow-up « accompagner »** : OK.
- ❌ **Pas parfait** : deux faiblesses systémiques demeurent —
  1. **Synonymes/translittérations non liés** (fattouche/fatteh, sfiha/sfouf,
     feuilles-de-vigne/warak enab). Même cause que manouche, **non résolue pour ces plats**.
  2. **Fallback « recette proche » incohérent** : pour un plat absent, tantôt un proche,
     tantôt une abstention vide.

## Correctifs recommandés (prioritaires)
1. **Étendre les alias/pins** (data-driven, comme manouche) : `feuilles de vigne farcies`→warak enab,
   `fattouche`→article fattouche, séparer `sfiha`(salé) de `sfouf`(dessert), `moutabbal`→baba ghanouj,
   `batata harra` (≠ kafta bi batata). → règle les confusions, déterministe.
2. **Fallback proche systématique** : quand un plat libanais n'est pas dans la base,
   toujours proposer la recette la plus proche (même catégorie/ingrédient) au lieu d'une
   abstention vide — exploiter catégorie + ingrédient principal + graphe de cooccurrence.
3. **Cohérence carte = article décrit** (corrige labneh→tarte tomates, etc.).

## Reproductible
`v4/scripts/qa_grid.py` (tiers + LLM-juge), `live_smoke_eval.py` (20 questions),
+ ce test. À relancer à chaque déploiement.
