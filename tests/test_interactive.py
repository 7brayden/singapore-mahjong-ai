"""Tests for the human-in-the-loop controller and decision generators."""

import json

import pytest

from mahjong.game import (
    GameState, DiscardRequest, ClaimRequest, run_decision,
)
from mahjong.interactive import InteractiveGame
from mahjong.agents import GreedyAgent, HybridAgent


def greedy_agents():
    return [GreedyAgent(f"G{i}") for i in range(4)]


# ── Equivalence: the state machine changes nothing ────────────────────

def test_human_seat_driven_by_bot_matches_pure_bot_game():
    # A human seat whose answers come from the same agent must replay
    # the pure-bot game move for move.
    pure = GameState(greedy_agents(), seed=11).play()

    ig = InteractiveGame(greedy_agents(), human_seats={0}, seed=11)
    ig.start()
    while not ig.game_over:
        ig.submit(ig.game.dispatch_to_agent(ig.pending))

    r = ig.result
    assert (r.winner, r.win_type, r.turns, r.tiles_remaining) == \
           (pure.winner, pure.win_type, pure.turns, pure.tiles_remaining)
    assert r.payments == pure.payments


# ── Pending decisions and submission ──────────────────────────────────

def test_first_pending_is_dealers_discard():
    ig = InteractiveGame(greedy_agents(), human_seats={0}, seed=0)
    ig.start()
    assert isinstance(ig.pending, DiscardRequest)
    assert ig.pending.player == 0
    assert ig.pending.drawn is not None  # the tile they just drew


def test_submit_invalid_discard_raises_and_preserves_state():
    ig = InteractiveGame(greedy_agents(), human_seats={0}, seed=0)
    ig.start()
    hand = ig.game.hands[0]
    not_held = next(t for t in range(34) if hand.counts[t] == 0)

    with pytest.raises(ValueError):
        ig.submit(not_held)
    assert ig.pending is not None  # still waiting on the same decision

    ig.submit(hand.tiles[0])  # a legal discard now works


def test_submit_without_pending_raises():
    ig = InteractiveGame(greedy_agents(), human_seats={0}, seed=0)
    with pytest.raises(RuntimeError):
        ig.submit(0)


def test_start_twice_raises():
    ig = InteractiveGame(greedy_agents(), human_seats={0}, seed=0)
    ig.start()
    with pytest.raises(RuntimeError):
        ig.start()


def test_all_bot_game_runs_to_completion_without_pending():
    ig = InteractiveGame(greedy_agents(), human_seats=set(), seed=0)
    assert ig.start() is None
    assert ig.game_over
    assert ig.result.winner is not None


# ── Views ─────────────────────────────────────────────────────────────

def test_view_redacts_opponent_hands():
    ig = InteractiveGame(greedy_agents(), human_seats={0}, seed=0)
    ig.start()
    view = ig.view_for(0)

    assert view["hand"] == sorted(ig.game.hands[0].tiles)
    for player in view["players"]:
        assert "hand" not in player
        assert "tiles" not in player
        assert player["concealed_count"] == \
               len(ig.game.hands[player["seat"]].tiles)
    # Pending decision belongs to seat 0 and is visible to seat 0 only
    assert view["pending"]["type"] == "discard"
    assert ig.view_for(1)["pending"] is None


def test_view_is_json_serializable():
    ig = InteractiveGame(greedy_agents(), human_seats={0}, seed=3)
    ig.start()
    json.dumps(ig.view_for(0))

    while not ig.game_over:
        ig.submit(ig.game.dispatch_to_agent(ig.pending))
    final = ig.view_for(0)
    json.dumps(final)
    assert final["game_over"]
    assert final["result"]["payments"] is not None


# ── Claim requests surface through the generator ──────────────────────

def test_resolve_discard_yields_claim_request():
    game = GameState(greedy_agents(), seed=0)
    game.wall = [0] * 40
    # P1 holds two 5w — a pong claim is possible on P0's discard
    for t in [4, 4, 9, 13, 17, 18, 22, 26, 27, 28, 31, 32, 33]:
        game.hands[1].add_tile(t)
    for t in [4, 0, 2, 6, 10, 14, 19, 23, 25, 29, 30, 5, 8]:
        game.hands[0].add_tile(t)

    game.hands[0].discard(4)
    gen = game._resolve_discard(0, 4)
    request = next(gen)
    assert isinstance(request, ClaimRequest)
    assert request.player == 1
    assert request.claim_type == "pong"

    # Decline: nobody else can claim, so the generator finishes
    with pytest.raises(StopIteration) as stop:
        gen.send(False)
    assert stop.value.value is True  # game continues
    assert game.hands[1].exposed == []
