"""SSE /chat/stream endpoint."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


def test_chat_stream_contains_done_with_html(client: TestClient):
    with client.stream("POST", "/api/chat/stream", json={"message": "bonjour"}) as r:
        assert r.status_code == 200
        raw = b"".join(r.iter_bytes())
    text = raw.decode("utf-8", errors="replace")
    assert "data:" in text
    assert "done" in text
    # Parse last data line JSON
    last_json = None
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                try:
                    last_json = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
    assert last_json is not None
    assert last_json.get("type") == "done"
    assert "html" in last_json
    assert last_json.get("request_id")
