# Widget Sahteïn — mise à jour pour l'équipe web OLJ

Remplaçant direct de `https://www.lorientlejour.com/assets/js/sahtein.js`
(version actuelle : 21 avril 2026). **Un seul fichier à remplacer, aucune
modification de la page** : le nouveau code garde exactement la même structure
HTML (`#blob`, `#chatWindow`, `#searchBar`, `#chatMessages`, `#chatInput`…).

## Ce que ça corrige

1. **Mémoire de conversation.** L'ancien widget n'envoyait pas d'identifiant de
   session : chaque message ouvrait une conversation neuve côté serveur, donc
   « une autre recette » et les relances ne fonctionnaient pas. Le nouveau génère
   un `session_id` stable (stocké dans le `localStorage` du visiteur) et
   l'envoie à chaque requête.
2. **Double appel API supprimé.** L'ancien fichier avait deux gestionnaires de
   clic sur le bouton d'envoi : chaque clic déclenchait **deux** appels au bot
   (deux fois le coût). Corrigé — un seul point d'envoi.
3. **Coquille du message d'accueil** (« mias je viens de ma lancer » →
   « mais je viens de me lancer »).
4. **Sanitisation client conservée** (DOMPurify si présent, repli inerte sinon).
5. **Robustesse** : délai maximal de 45 s sur l'appel (une requête figée n'immobilise
   plus le widget), verrou anti-double-envoi (une requête à la fois), et message
   dédié quand le serveur répond « trop de demandes » (HTTP 429).

## Déploiement

1. Remplacer le contenu de `/assets/js/sahtein.js` par le fichier
   [`sahtein.js`](./sahtein.js) de ce dossier.
2. **Purger le cache CDN** pour ce fichier. Il est servi avec
   `Cache-Control: max-age=63072000` (deux ans) : sans purge, les visiteurs
   garderont l'ancienne version très longtemps. Le plus simple : soit purger
   l'URL dans Cloudflare, soit changer la référence dans la page en
   `sahtein.js?v=20260810` (et purger l'ancienne).

Aucune clé, aucun réglage à changer : le point d'API
(`sahtenbotapi.lorientlejour.com/api/chat`) et le balisage HTML sont inchangés.

## À noter

- Le serveur accepte les deux formats de requête (`{"query": …}` et
  `{"message": …}`) : ce widget envoie `query`, le canonique.
- La réponse est rendue depuis `data.html` si présent, sinon reconstruite depuis
  `answer_sentences` — robuste aux deux formats de réponse du backend.
