# Aplatir le dépôt sur GitHub

But : la racine du dépôt `SamiNakibETU/Sahten` doit contenir **uniquement
le contenu de `sahten_github/`**, sans le dossier parent ni les variantes
`sahten/`, `sahten-dev/`, `sahten-mvp/`.

## Constat actuel

Le workspace local contient à la racine :

```
V3/
├─ sahten/            ← obsolète
├─ sahten-dev/        ← obsolète
├─ sahten-mvp/        ← obsolète
├─ sahten_github/     ← canonique (à promouvoir en racine du repo)
├─ key.txt            ← SECRET, à supprimer du dépôt
├─ mdp.txt            ← SECRET, à supprimer du dépôt
├─ .env               ← SECRET, à supprimer du dépôt
└─ venv/              ← ne doit jamais être versionné
```

Le repo distant `Sahten` contient probablement le même bordel
(historique inclus). Il faut nettoyer **historique + arbre courant**.

## Procédure recommandée (à exécuter avec validation explicite)

```bash
# 1) Cloner un mirror frais du repo
git clone --mirror https://github.com/SamiNakibETU/Sahten.git Sahten.git
cd Sahten.git

# 2) Filter-repo pour ne garder QUE l'arbre sahten_github/, le promouvoir
#    en racine, et purger les fichiers sensibles + venv de tout l'historique
pip install git-filter-repo
git filter-repo \
    --subdirectory-filter sahten_github \
    --path-glob '!key.txt' \
    --path-glob '!mdp.txt' \
    --path-glob '!.env' \
    --path-glob '!venv/**' \
    --path-glob '!**/__pycache__/**' \
    --path-glob '!**/.venv/**'

# 3) Forcer la nouvelle history
git push --force --all
git push --force --tags

# 4) Refaire un clone propre côté équipe
```

## Garde-fous

- Avant `git filter-repo`, faire une **copie de sécurité** du mirror
  (`cp -r Sahten.git Sahten.git.backup`).
- Prévenir l'équipe OLJ : tout le monde doit re-cloner le repo après
  l'opération (les SHAs changent).
- Réinstaller les hooks GitHub (Railway redéploie automatiquement à la
  nouvelle HEAD).
- **Rotation immédiate** des clés `OLJ_API_KEY`, `OPENAI_API_KEY`,
  `WEBHOOK_SECRET` après le force-push : tant qu'ils étaient dans
  l'historique, ils sont à considérer compromis.

## Vérification post-opération

```bash
git clone https://github.com/SamiNakibETU/Sahten.git Sahten-fresh
cd Sahten-fresh
ls           # doit montrer backend/ frontend/ data/ scripts/ v4/ uniquement
git log --all --oneline | grep -E '(key\.txt|mdp\.txt|\.env|venv/)' && echo "FUITE !"
```
