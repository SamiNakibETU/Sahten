# AGENTS.md — Mémoire et règles pour les agents IA (Cursor, etc.)

Ce fichier définit les règles permanentes que tout agent IA doit respecter sur ce projet.

---

## Règles Git

- **JAMAIS** inclure "Made-with: Cursor", "Made with Cursor", "Co-authored-by: Cursor" ou toute mention d'outil IA dans un message de commit ou de push.
- Les messages de commit doivent être en français ou en anglais technique, concis, sans signature automatique.
- Format de commit : `type(scope): description courte` (ex: `fix(bot): corriger hook alternatif`)
- Ne jamais amender un commit déjà poussé sur `origin` sans demande explicite.
- La branche de déploiement Railway est `feature/ui-sahtein-olj-v2` — toujours pousser sur `main` ET `main:feature/ui-sahtein-olj-v2`.

## Règles de déploiement

- Le déploiement Railway se fait depuis la branche `feature/ui-sahtein-olj-v2`.
- Après chaque push, attendre 3-4 minutes avant de vérifier `https://sahten.up.railway.app/`.
- Vérifier `https://sahten.up.railway.app/api/health` pour confirmer le déploiement.

## Règles de code

- Langue des réponses du bot : français, vouvoiement strict (jamais "tu", "te", "ton").
- Le nom du bot est **Sahteïn** (avec tréma sur le ï) — jamais "Sahten".
- Jamais d'emojis dans les réponses du bot ni dans l'interface (sauf si demande explicite).
- Jamais de cadratins (—) dans les textes affichés à l'utilisateur.
- Le `EXACT_ALTERNATIVE_HOOK` est la phrase contractuelle côté serveur — ne jamais le modifier sans validation.

## Contexte projet

- **Projet** : Sahteïn, agent conversationnel culinaire pour L'Orient-Le Jour.
- **Repo** : https://github.com/SamiNakibETU/Sahten (branche principale : `feature/ui-sahtein-olj-v2`)
- **Prod** : https://sahten.up.railway.app/
- **Stack** : FastAPI (Python) + frontend statique HTML/CSS/JS + Upstash Redis + OpenAI GPT-4.1 nano
- **Session** : in-memory (`SessionManager`) + `localStorage` côté widget pour persistence du `session_id`
