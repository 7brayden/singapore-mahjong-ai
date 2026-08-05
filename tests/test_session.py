"""Tests for multi-round sessions: dealer rotation, rounds, and scores.

Rotation logic is tested against a stubbed _play_hand (no real games),
plus one real integration session for determinism.
"""

from mahjong.game import GameResult
from mahjong.session import Session
from mahjong.agents import GreedyAgent
from mahjong.tiles import WIND_START

EAST, SOUTH = WIND_START, WIND_START + 1


def fake_result(winner, payments=None):
    return GameResult(
        winner=winner,
        win_type="ron" if winner is not None else None,
        turns=10,
        final_shanten=[0] * 4,
        deal_ins=[0] * 4,
        flowers_collected=[0] * 4,
        tiles_remaining=20,
        payments=payments or [0, 0, 0, 0],
    )


def stub_session(results, **kwargs):
    """Session whose hands come from a scripted list of results."""
    session = Session([GreedyAgent(f"G{i}") for i in range(4)], seed=0, **kwargs)
    script = list(results)
    session._play_hand = lambda dealer, wind: script.pop(0)
    return session


def test_dealer_repeats_on_own_win_and_draw():
    # Dealer 0 wins, then a draw (repeats twice), then non-dealer wins
    # pass the dealership around until the East round ends.
    script = [fake_result(0), fake_result(None), fake_result(2),
              fake_result(0), fake_result(0), fake_result(0)]
    session = stub_session(script, winds=[EAST])
    result = session.play()

    assert result.hands_played == 6
    assert [rec.dealer for rec in result.records] == [0, 0, 0, 1, 2, 3]


def test_prevailing_wind_advances_after_four_passes():
    # Every hand is won by a non-dealer: dealership passes each hand,
    # so each round is exactly 4 hands.
    dealers = [0, 1, 2, 3, 0, 1, 2, 3]
    script = [fake_result((d + 2) % 4) for d in dealers]
    session = stub_session(script, winds=[EAST, SOUTH])
    result = session.play()

    assert result.hands_played == 8
    assert [rec.dealer for rec in result.records] == dealers
    assert [rec.prevailing_wind for rec in result.records] == \
           [EAST] * 4 + [SOUTH] * 4


def test_scores_accumulate_across_hands():
    script = [
        fake_result(1, payments=[-3, 3, 0, 0]),
        fake_result(2, payments=[-1, -1, 3, -1]),
        fake_result(3, payments=[0, 0, -4, 4]),
        fake_result(0, payments=[6, -2, -2, -2]),
    ]
    session = stub_session(script, winds=[EAST])
    result = session.play()

    assert result.final_scores == [2, 0, -3, 1]
    assert sum(result.final_scores) == 0
    # Running scores recorded per hand
    assert result.records[0].scores_after == [-3, 3, 0, 0]
    assert result.records[1].scores_after == [-4, 2, 3, -1]
    # Ranking: best score first
    assert result.ranking[0] == 0


def test_max_hands_stops_endless_dealer_streak():
    session = Session([GreedyAgent(f"G{i}") for i in range(4)],
                      seed=0, winds=[EAST], max_hands=5)
    session._play_hand = lambda dealer, wind: fake_result(dealer)  # dealer always wins
    result = session.play()
    assert result.hands_played == 5


def test_real_session_is_deterministic():
    def run():
        agents = [GreedyAgent(f"G{i}") for i in range(4)]
        return Session(agents, seed=7, winds=[EAST]).play()

    r1, r2 = run(), run()
    assert r1.final_scores == r2.final_scores
    assert r1.hands_played == r2.hands_played
    assert sum(r1.final_scores) == 0

    # Rotation invariant holds over real games too
    dealer = 0
    for rec in r1.records:
        assert rec.dealer == dealer
        winner = rec.result.winner
        if winner is not None and winner != dealer:
            dealer = (dealer + 1) % 4
