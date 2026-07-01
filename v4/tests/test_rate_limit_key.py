"""Rate-limit : la clé IP doit isoler le vrai client selon le nombre de proxys.

Couvre le bug AWS : derrière CloudFront->ALB (2 proxys), lire le dernier IP =
l'edge CloudFront partagé -> tous les users dans le même quota. On doit lire
l'avant-dernier (le client). Et l'anti-spoof : injecter des IPs à gauche ne
doit jamais changer la clé.
"""

from __future__ import annotations

import types

import backend.app.settings as settings_mod
from backend.app.limiter_support import real_ip_key


class _FakeReq:
    def __init__(self, xff=None, client_host=None):
        self.headers = {"X-Forwarded-For": xff} if xff else {}
        self.client = types.SimpleNamespace(host=client_host) if client_host else None


def _set_hops(monkeypatch, n):
    monkeypatch.setattr(
        settings_mod, "get_settings", lambda: types.SimpleNamespace(trusted_proxy_hops=n)
    )


def test_single_proxy_takes_last_ip(monkeypatch):
    _set_hops(monkeypatch, 1)
    assert real_ip_key(_FakeReq(xff="203.0.113.7")) == "203.0.113.7"


def test_single_proxy_ignores_spoofed_left_entries(monkeypatch):
    _set_hops(monkeypatch, 1)
    # L'attaquant injecte des IPs à gauche ; le proxy ajoute le vrai client à droite.
    assert real_ip_key(_FakeReq(xff="1.1.1.1, 2.2.2.2, 203.0.113.7")) == "203.0.113.7"


def test_two_proxies_take_client_not_edge(monkeypatch):
    _set_hops(monkeypatch, 2)
    # CloudFront->ALB : "client, edge_cloudfront". Le client est l'avant-dernier.
    assert real_ip_key(_FakeReq(xff="203.0.113.7, 70.132.0.10")) == "203.0.113.7"


def test_two_proxies_ignore_spoof(monkeypatch):
    _set_hops(monkeypatch, 2)
    assert (
        real_ip_key(_FakeReq(xff="9.9.9.9, 203.0.113.7, 70.132.0.10")) == "203.0.113.7"
    )


def test_chain_shorter_than_hops_falls_back_leftmost(monkeypatch):
    _set_hops(monkeypatch, 2)
    assert real_ip_key(_FakeReq(xff="203.0.113.7")) == "203.0.113.7"


def test_no_forwarded_uses_client_host(monkeypatch):
    _set_hops(monkeypatch, 1)
    assert real_ip_key(_FakeReq(client_host="198.51.100.4")) == "198.51.100.4"


def test_no_info_returns_unknown(monkeypatch):
    _set_hops(monkeypatch, 1)
    assert real_ip_key(_FakeReq()) == "unknown"
