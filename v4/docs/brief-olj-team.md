# Brief à l'équipe dev L'Orient-Le Jour (intégration widget Sahteïn)

**Objet** : version du widget Sahteïn déployée sur olj.com — diagnostic
"vous utilisez une vieille version" + checklist d'intégration v4.

---

## Pourquoi vous voyez probablement une ancienne version

Causes possibles, par ordre de probabilité, à vérifier dans l'ordre.

### 1. Cache navigateur / CDN sur les assets statiques

Symptômes : nouveau code poussé, mais l'UI ne change pas.

À vérifier :

- Quel domaine sert le widget chez vous ? (cdn.lorientlejour.com,
  iframe externe, npm package ?). **Réponse attendue de votre part.**
- Inspectez les requêtes réseau de la page d'article : vérifiez le
  `?v=…` ou hash dans l'URL des fichiers `.js`/`.css`/`.svg`.
- En l'état, la branche `feature/ui-sahtein-olj-v2` du repo
  `SamiNakibETU/Sahten` injecte déjà `?v=3` sur les logos
  ([commit d6b2c2c](https://github.com/SamiNakibETU/Sahten/commit/d6b2c2c)).
  Vérifiez que c'est bien cette branche qui se déploie chez vous.
- La nouvelle v4 utilisera un **hash de build** automatique (ex.
  `app.f3a8c1.js`) → fini les caches éternels.

### 2. Mauvaise URL d'embed

À nous confirmer : quelle est l'URL `<iframe src="…">` ou
`<script src="…">` que vous utilisez actuellement dans les pages OLJ ?

Notre URL de production canonique sera, après v4 :
```
https://sahten-web-production.up.railway.app/widget.js
```
(ou domaine custom si vous nous fournissez un sous-domaine
`sahtein.lorientlejour.com` — fortement recommandé pour la confiance
utilisateur et le contournement des bloqueurs de tracking).

### 3. Mauvaise branche déployée sur Railway

À l'heure actuelle, plusieurs branches existent :

| Branche | Usage |
|---------|-------|
| `main` | Historique, **ne pas pointer dessus** |
| `mvp` | Vieille démo, à archiver |
| `feature/ui-sahtein-olj-v2` | Branche prod actuelle (v3 + correctifs UI) |
| `sota/v4` | **Nouvelle architecture (cible)** |

Railway doit pointer sur `feature/ui-sahtein-olj-v2` aujourd'hui, et sur
`sota/v4` une fois validée en staging. Confirmer-vous le branchement
actuel ?

### 4. Anciennes copies statiques de l'app dans votre repo OLJ

Vérifiez que vous n'avez pas, dans votre propre dépôt, une **copie
statique** des assets Sahteïn (ex. `public/widgets/sahtein/…`) qui
sera servie quoi qu'on fasse côté Sahteïn.

---

## Action immédiate côté votre équipe (15 min)

1. Ouvrir une page d'article OLJ contenant le widget.
2. Ouvrir DevTools → Network, filtre `sahtein`.
3. Nous transmettre :
   - L'URL exacte du `<script>` ou `<iframe>` qui charge le widget.
   - Les en-têtes `Cache-Control` et `ETag` des assets servis.
   - La réponse `GET /api/version` du backend (renvoie le numéro de
     version Sahteïn, ajouté dans la v4).

---

## Plan d'intégration v4 (cible)

```html
<!-- À placer dans le template d'article OLJ -->
<div id="sahtein-mount"></div>
<script
  src="https://sahtein.lorientlejour.com/widget.js"
  data-article-id="{{ article.id }}"
  data-locale="fr"
  defer
></script>
```

Le widget v4 :

- détecte l'`article-id` et le passe à `/api/chat` pour pré-conditionner
  le contexte ;
- s'auto-met à jour via le hash de build (cache 1 an pour les fichiers
  hashés, `no-cache` pour `widget.js` qui est un loader léger) ;
- exposera des évènements DOM (`sahtein:opened`, `sahtein:answered`)
  pour vos analytics.

---

## Côté CMS / SEO

- Le widget v4 ajoutera un `link rel="canonical"` vers l'article OLJ
  d'origine dans son embed pour éviter toute pénalité SEO.
- Les réponses citent les sources avec `<a href="<url-article-OLJ>">`,
  donc chaque réponse devient un lien interne supplémentaire vers
  vos pages.

---

## Contact technique

- Repo : https://github.com/SamiNakibETU/Sahten
- PR v4 : https://github.com/SamiNakibETU/Sahten/pull/new/sota/v4
- Health endpoint v4 : `GET /healthz`, `GET /readyz`
- Docs déploiement : [`v4/docs/railway-deploy.md`](railway-deploy.md)
