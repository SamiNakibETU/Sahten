"""Cleanup non destructif du workspace racine V3/.

Stratégie : on **déplace** les obsolètes vers `_archive_/<date>/`
plutôt que de supprimer. Les secrets en clair sont signalés mais
**JAMAIS supprimés automatiquement** (l'utilisateur doit les rotater
d'abord, puis les supprimer à la main).

Mode par défaut : DRY RUN (n'effectue aucun changement, affiche le
plan). Pour exécuter réellement : `--apply`.

Usage :
    python v4/scripts/cleanup_workspace.py
    python v4/scripts/cleanup_workspace.py --apply
    python v4/scripts/cleanup_workspace.py --apply --root "d:/.../V3"
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

# Tous les chemins sont relatifs à la racine V3/.
SECRETS_DO_NOT_TOUCH = [
    "key.txt",
    "mdp.txt",
    "dev_conv",
    ".env",
]

OBSOLETE_DIRS = [
    "sahten",
    "sahten-dev",
    "sahten-mvp",
    "sahten-scraper",
    "archive",
    "api",
    "scripts",
    "assets",
    "DESIGN",
    "Logo Animation Design",
    ".pytest_cache",
]

OBSOLETE_FILES = [
    "inspect_data.py",
    "show_block.py",
    "patch_welcome_emoji.py",
    "test_chef_extraction.py",
    "vercel.json",
    "requirements.txt",
    "VERSION_AND_STRUCTURE.md",
    "EMAIL_TO_DEV_TEAM.txt",
]

OLD_LOGOS = [
    "logo_1_OLJ.svg",
    "logo_2_OLJ.svg",
    "logo_sahten_v1.svg",
    "logo_v4.png",
    "sahten_logo_v2.svg",
    "sahten_logo_v3.svg",
    "sahten_logo_v4.svg",
    "v5_logo.svg",
    "v6_sahten_logo.svg",
    "v7_logo_sahten.svg",
]


def _move(src: Path, dest: Path, apply: bool) -> str:
    if not src.exists():
        return f"  SKIP   (absent) : {src.name}"
    action = "MOVE  " if apply else "WOULD "
    if apply:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    return f"  {action} {src.name} -> {dest.relative_to(src.parent.parent)}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[3]),
        help="Racine du workspace (defaut: parent de sahten_github/)",
    )
    p.add_argument("--apply", action="store_true",
                   help="Exécute réellement (sinon dry-run)")
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Racine introuvable: {root}", file=sys.stderr)
        return 1

    archive_root = root / "_archive_" / date.today().isoformat()
    print(f"Racine     : {root}")
    print(f"Archive    : {archive_root}")
    print(f"Mode       : {'APPLY (changements réels)' if args.apply else 'DRY-RUN (aucun changement)'}")
    print()

    # 1. Secrets
    print("=== Secrets (NE PAS supprimer automatiquement) ===")
    found_secret = False
    for s in SECRETS_DO_NOT_TOUCH:
        path = root / s
        if path.exists():
            found_secret = True
            print(f"  /!\\  {path.name} présent à la racine. À rotater puis supprimer manuellement.")
    if not found_secret:
        print("  ok, aucun secret en clair détecté.")
    print()

    # 2. Dossiers obsolètes
    print("=== Dossiers obsolètes (déplacés vers _archive_) ===")
    for d in OBSOLETE_DIRS:
        print(_move(root / d, archive_root / "dirs" / d, args.apply))
    print()

    # 3. Fichiers obsolètes
    print("=== Fichiers obsolètes ===")
    for f in OBSOLETE_FILES:
        print(_move(root / f, archive_root / "files" / f, args.apply))
    print()

    # 4. Logos legacy
    print("=== Logos legacy ===")
    for lg in OLD_LOGOS:
        print(_move(root / lg, archive_root / "old-logos" / lg, args.apply))
    print()

    # 5. venv racine éventuel
    venv = root / "venv"
    if venv.exists():
        print("=== venv racine (à supprimer ; jamais versionné) ===")
        if args.apply:
            shutil.rmtree(venv, ignore_errors=True)
            print("  REMOVED venv/")
        else:
            print("  WOULD REMOVE venv/")
        print()

    if not args.apply:
        print("Dry-run terminé. Relancez avec --apply pour exécuter réellement.")
    else:
        print(f"Cleanup terminé. Archive créée dans : {archive_root}")
        print("Vérifiez le contenu avant suppression définitive (recommandé : 30 jours).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
