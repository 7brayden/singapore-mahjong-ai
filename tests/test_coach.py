"""Coach tests: retrieval, situation building, fallback, endpoint.

No test here ever calls the Anthropic API — the key is scrubbed from
the environment so every path exercises the deterministic template.
"""

import pytest
from fastapi.testclient import TestClient

from mahjong.game import GameState, advance_turns
from mahjong.agents import GreedyAgent
from mahjong.coach.corpus import CHUNKS
from mahjong.coach.retrieve import retrieve, situation_tags
from mahjong.coach.explain import build_situation, fallback_text
from mahjong.server.app import app, manager


@pytest.fixture(autouse=True)
def _no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


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
