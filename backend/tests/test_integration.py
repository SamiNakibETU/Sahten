"""
Integration tests for Sahtein clean architecture.
"""

import asyncio
import httpx
import pytest


@pytest.fixture
def api_url():
    return "http://localhost:8000/api/chat"


@pytest.mark.asyncio
async def test_hummus_returns_base2():
    """hummus should return Base2 fallback (not in OLJ)."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "http://localhost:8000/api/chat",
            json={"message": "hummus"}
        )
        data = response.json()
        html = data.get("html", "")
        
        assert "card-title" in html, "Should return recipe cards"
        assert "base2-card" in html, "Should use Base2 fallback"


@pytest.mark.asyncio
async def test_taboule_returns_olj():
    """taboulé should return OLJ (exists in OLJ)."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "http://localhost:8000/api/chat",
            json={"message": "taboule"}
        )
        data = response.json()
        html = data.get("html", "")
        
        assert "card-title" in html, "Should return recipe cards"
        # OLJ cards don't have base2-card class
        assert data.get("used_base") == "olj" or "lorientlejour" in html.lower()


@pytest.mark.asyncio
async def test_kebbe_returns_multiple():
    """kebbé should return multiple OLJ recipes."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "http://localhost:8000/api/chat",
            json={"message": "kebbe"}
        )
        data = response.json()
        html = data.get("html", "")
        
        card_count = html.count("card-title")
        assert card_count >= 2, f"Should return multiple cards, got {card_count}"


def run_quick_test():
    """Quick manual test runner."""
    import asyncio
    
    async def test(query):
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                "http://localhost:8000/api/chat",
                json={"message": query}
            )
            data = r.json()
            html = data.get("html", "")
            cards = html.count("card-title")
            base2 = "base2-card" in html
            reco = "olj-recommendation" in html
            return cards, base2, reco

    async def main():
        tests = [
            "hummus",
            "falafel",
            "taboule",
            "kebbe",
        ]
        
        print("Query          Cards  Base2  Reco")
        print("=" * 40)
        for q in tests:
            cards, base2, reco = await test(q)
            b2 = "Y" if base2 else "N"
            rc = "Y" if reco else "N"
            print(f"{q:<14} {cards:<6} {b2:<6} {rc}")

    asyncio.run(main())


if __name__ == "__main__":
    run_quick_test()

