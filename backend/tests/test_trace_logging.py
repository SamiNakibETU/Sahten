"""
Tests for persistent trace logging.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import app
from app.rag.trace_logger import TraceLogger, set_trace_logger


@pytest.fixture
def api_client():
    with TestClient(app) as client:
        yield client


def test_chat_request_is_logged(tmp_path: Path, api_client: TestClient):
    """Each chat call should append a JSON line trace."""
    log_path = tmp_path / "chat_traces.jsonl"
    set_trace_logger(TraceLogger(log_path=log_path))

    payload = {"message": "recette taboulé", "debug": True}
    response = api_client.post("/api/chat", json=payload)
    assert response.status_code == 200

    assert log_path.exists(), "Expected trace log file to be created"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "Trace file should contain at least one entry"
    entry = json.loads(lines[-1])

    assert entry["user_message"] == payload["message"]
    assert entry["response_html"]
    assert entry["scenario_id"] == response.json()["scenario_id"]
    assert entry["primary_url"] == response.json()["primary_url"]
    assert "classification" in entry["debug_info"]
    assert entry["metadata"].get("source") == "frontend"

    # Reset logger so other tests fall back to default path
    set_trace_logger(None)
