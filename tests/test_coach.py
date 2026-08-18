"""Coach tests: retrieval, situation building, fallback, endpoint.

No test here ever calls a live LLM: every provider credential is
scrubbed from the environment, so all paths exercise the deterministic
template. The scrub is autouse and covers both backends — a test that
reached the network would bill a real account.
"""

import pytest
from fastapi.testclient import TestClient

from mahjong.game import GameState, advance_turns
from mahjong.agents import GreedyAgent
from mahjong.coach.corpus import CHUNKS
from mahjong.coach.retrieve import retrieve, situation_tags
from mahjong.coach.explain import (
    _provider, build_situation, fallback_text,
)
from mahjong.server.app import app, manager


COACH_ENV_VARS = [
    "COACH_PROVIDER", "COACH_MODEL", "COACH_BASE_URL", "COACH_API_KEY",
    "ANTHROPIC_API_KEY",
    "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION",
]


@pytest.fixture(autouse=True)
def _no_provider_credentials(monkeypatch):
    for name in COACH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _situation(seed=11, turns=30, seat=0):
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=seed)
    game.setup()
    advance_turns(game, turns)
    game._deal_tile_to(seat)  # 14-shape: a real discard decision
    return game, build_situation(game, seat, {"type": "discard"},
                                 "The trained advisor would discard 5 Wan.")


# ── Corpus hygiene ───────────────────────────────────────────────────

def test_corpus_chunks_are_well_formed():
    ids = [c["id"] for c in CHUNKS]
    assert len(ids) == len(set(ids))
    for c in CHUNKS:
        assert c["title"] and c["text"] and c["tags"]
        assert len(c["text"]) < 700  # chunks stay prompt-sized


# ── Retrieval ────────────────────────────────────────────────────────

def test_claim_window_retrieves_claim_principles():
    situation = {"pending_type": "chow", "shanten": 1, "is_concealed": True,
                 "turn_frac": 0.3, "max_opp_threat": 0.1}
    tags = situation_tags(situation)
    assert "claim_window" in tags and "concealed" in tags
    got = {c["id"] for c in retrieve(situation)}
    assert "principle-concealment" in got or "principle-claim-tempo" in got
    assert "rules-ping-hu-family" in got  # the house rule that gates it


def test_hot_late_table_retrieves_defense():
    situation = {"pending_type": "discard", "shanten": 2,
                 "turn_frac": 0.8, "max_opp_threat": 0.5, "opp_melds": 3,
                 "max_deal_in_prob": 0.06}
    got = {c["id"] for c in retrieve(situation)}
    assert got & {"principle-lateness", "principle-exposed-melds",
                  "principle-push-fold"}


def test_retrieval_is_deterministic():
    situation = {"pending_type": "discard", "shanten": 1,
                 "turn_frac": 0.5, "max_opp_threat": 0.2}
    a = [c["id"] for c in retrieve(situation)]
    b = [c["id"] for c in retrieve(situation)]
    assert a == b and len(a) > 0


# ── Situation building ───────────────────────────────────────────────

def test_situation_uses_engine_numbers_only():
    game, situation = _situation()
    assert situation["pending_type"] == "discard"
    # Concealed tiles + 3 per exposed meld = a full 14-shape hand
    melds = len(situation["exposed_melds"])
    assert len(situation["hand"]) + 3 * melds == 14
    assert situation["shanten"] is not None
    assert situation["turn"] == game.turn
    for d in situation["top_discards"]:
        assert "deal_in_prob" in d and "hand_value_pts" in d


# ── Fallback rendering ───────────────────────────────────────────────

def test_fallback_text_cites_numbers_and_principle():
    _, situation = _situation()
    principles = retrieve(situation)
    text = fallback_text(situation, principles)
    assert "The trained advisor would discard" in text
    assert "Fastest line" in text
    assert "Principle:" in text
    assert "%" in text  # quotes a probability


# ── Endpoint (fallback path, no key) ─────────────────────────────────

def test_explain_endpoint_returns_template_without_key():
    client = TestClient(app)
    created = client.post("/games", json={
        "seed": 4, "human_seat": 0, "bots": ["hybrid"] * 4}).json()
    gid = created["game_id"]
    try:
        r = client.post(f"/games/{gid}/explain")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["source"] == "template"
        assert body["text"]
        assert isinstance(body["principles"], list)
        # Second call is served from the per-decision cache
        assert client.post(f"/games/{gid}/explain").json() == body
    finally:
        manager.remove(gid)


# ── Provider selection (no network in any case) ──────────────────────

def test_provider_defaults_to_template_without_credentials():
    assert _provider() == "template"


def test_azure_credentials_win_by_default(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    assert _provider() == "azure"
    # Anthropic present too: Azure still wins as the configured backend
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    assert _provider() == "azure"


def test_anthropic_used_when_only_anthropic_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    assert _provider() == "anthropic"


def test_coach_provider_pins_the_backend(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setenv("COACH_PROVIDER", "template")
    assert _provider() == "template"


def test_endpoint_falls_back_to_template_when_provider_errors(monkeypatch):
    # Azure "configured" but pointed nowhere: the request must fail
    # inside the provider and still return a usable lesson.
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://localhost:1")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "nonexistent")
    client = TestClient(app)
    created = client.post("/games", json={
        "seed": 6, "human_seat": 0, "bots": ["hybrid"] * 4}).json()
    gid = created["game_id"]
    try:
        body = client.post(f"/games/{gid}/explain").json()
        assert body["source"] == "template"
        assert body["text"]
    finally:
        manager.remove(gid)


# ── Setup diagnostic ─────────────────────────────────────────────────

def test_check_reports_template_and_succeeds_without_credentials(capsys):
    from mahjong.coach.check import main
    assert main() == 0
    out = capsys.readouterr().out
    assert "selected provider          template" in out
    assert "AZURE_OPENAI_DEPLOYMENT" in out


def test_check_never_prints_a_whole_key(monkeypatch, capsys):
    from mahjong.coach.check import main
    secret = "super-secret-key-value-9999"
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", secret)
    monkeypatch.setenv("COACH_PROVIDER", "template")  # no network
    main()
    out = capsys.readouterr().out
    assert secret not in out
    assert "9999" in out  # last 4 shown, enough to tell keys apart


def test_base_url_selects_the_generic_openai_provider(monkeypatch):
    monkeypatch.setenv("COACH_BASE_URL", "http://localhost:11434/v1")
    assert _provider() == "openai"


def test_generic_client_tolerates_a_keyless_local_server(monkeypatch):
    # Local servers ignore the key; the SDK still requires a non-empty one.
    from mahjong.coach.explain import openai_compatible_client
    monkeypatch.setenv("COACH_BASE_URL", "http://localhost:11434/v1")
    client = openai_compatible_client()
    assert str(client.base_url).rstrip("/").endswith("/v1")
    assert client.api_key  # never empty
