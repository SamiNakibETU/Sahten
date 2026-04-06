#!/usr/bin/env python
"""
Test manuel : flux recette non trouvée (ex: recette de tiramisu).
Lance le bot avec une requête qui n'est pas dans la base et vérifie la réponse OLJ.
"""
import asyncio
import os
import sys

# Ajouter le backend au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.bot import get_bot
from app.api.response_composer import compose_html_response


async def main():
    bot = get_bot()
    # Requêtes qui ne sont PAS dans la base OLJ (pour déclencher le flux "non trouvé")
    queries = [
        "recette de carbonara italienne",
        "recette de ramen japonais",
        "recette de sushi",
    ]
    for q in queries:
        print(f"\n--- Query: {q} ---")
        resp, _ = await bot.chat(q, debug=False)
        html = compose_html_response(resp)
        print(f"response_type: {resp.response_type}")
        print(f"recipe_count: {resp.recipe_count}")
        if resp.recipes:
            print(f"Recette: {resp.recipes[0].title}")
            print(f"URL: {resp.recipes[0].url}")
        if resp.narrative:
            hook = resp.narrative.hook if hasattr(resp.narrative, "hook") else str(resp.narrative)[:50]
            print(f"Hook: {hook[:80]}...")
        # Vérifier la formulation OLJ (recette non trouvée)
        hook = resp.narrative.hook if hasattr(resp.narrative, "hook") else ""
        assert "désolé" in hook.lower() or "désolé" in str(resp.narrative).lower(), f"Expected OLJ apology in hook: {hook[:100]}"
        # Soit une recette alternative, soit redirect
        assert resp.recipe_count >= 1 or resp.response_type == "redirect"
        print("OK")


if __name__ == "__main__":
    asyncio.run(main())
