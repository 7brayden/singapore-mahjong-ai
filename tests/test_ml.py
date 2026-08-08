"""Tests for the ML pipeline: features, datagen labels, model, agent."""

import pytest

from mahjong.game import GameState, advance_turns
from mahjong.agents import GreedyAgent, HybridAgent, LearnedAgent
from mahjong.opponent_model import estimate_opponent_threats
from mahjong.ml.features import (
    DANGER_FEATURES, OUTCOME_FEATURES, danger_features, outcome_features,
)
from mahjong.ml.datagen import DANGER_COLUMNS, OUTCOME_COLUMNS, generate_game
from mahjong.ml.model import LinearModel


def _midgame(seed=11, turns=30):
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=seed)
    game.setup()
    advance_turns(game, turns)
    return game


# ── Features ─────────────────────────────────────────────────────────

def test_danger_features_shape_and_range():
    game = _midgame()
    visible = game.get_visible_counts(0)
    threat = estimate_opponent_threats(0, game)
    for tile in set(game.hands[0].tiles):
        x = danger_features(tile, 0, game, visible, threat)
        assert len(x) == len(DANGER_FEATURES)
        assert all(0.0 <= v <= 1.0 for v in x), (tile, x)


def test_outcome_features_shape():
    game = _midgame()
    threat = estimate_opponent_threats(0, game)
    x = outcome_features(0, game, 2, 12, threat)
    assert len(x) == len(OUTCOME_FEATURES)
    assert all(0.0 <= v <= 1.0 for v in x)


# ── Data generation ──────────────────────────────────────────────────

def test_generate_game_rows_and_labels():
    danger_rows, outcome_rows = [], []
    for game_id, seed in enumerate([3, 4, 5]):
        generate_game(game_id, seed,
                      lambda: [GreedyAgent(f"G{i}") for i in range(4)],
                      danger_rows, outcome_rows)

    assert danger_rows and outcome_rows
    assert all(len(r) == len(DANGER_COLUMNS) for r in danger_rows)
    assert all(len(r) == len(OUTCOME_COLUMNS) for r in outcome_rows)

    w = DANGER_COLUMNS.index("waited")
    wl = DANGER_COLUMNS.index("waited_legal")
    ch = DANGER_COLUMNS.index("chosen")
    di = DANGER_COLUMNS.index("dealt_in")
    for r in danger_rows:
        assert r[w] in (0, 1) and r[wl] in (0, 1)
        assert r[wl] <= r[w]           # legal ⊆ waited
        assert r[di] <= r[ch]          # only a thrown tile can deal in

    # Exactly one chosen tile per decision → one outcome row each
    assert sum(r[ch] for r in danger_rows) == len(outcome_rows)

    # decision_id joins the two tables: same (game_id, decision_id) keys
    d_key = DANGER_COLUMNS.index("decision_id")
    o_key = OUTCOME_COLUMNS.index("decision_id")
    o_seat = OUTCOME_COLUMNS.index("seat")
    assert {(r[0], r[d_key]) for r in danger_rows} == \
           {(r[0], r[o_key]) for r in outcome_rows}
    # ...and it is unique per outcome row within a game
    assert len({(r[0], r[o_key]) for r in outcome_rows}) == len(outcome_rows)

    # Outcome labels: 0 or 1 winner per game
    for game_id in (0, 1, 2):
        winners = {r[o_seat] for r in outcome_rows
                   if r[0] == game_id and r[-2] == 1}
        assert len(winners) <= 1


def test_generate_game_is_deterministic():
    a, b = ([], []), ([], [])
    generate_game(0, 42, lambda: [GreedyAgent(f"G{i}") for i in range(4)],
                  *a)
    generate_game(0, 42, lambda: [GreedyAgent(f"G{i}") for i in range(4)],
                  *b)
    assert a == b


# ── Model ────────────────────────────────────────────────────────────

def _toy_model():
    n = len(DANGER_FEATURES)
    coef = [0.0] * n
    coef[DANGER_FEATURES.index("sig_opponent_threat")] = 2.0
    return LinearModel(features=list(DANGER_FEATURES),
                       mean=[0.0] * n, scale=[1.0] * n,
                       coef=coef, intercept=-3.0)


def test_linear_model_predict_and_roundtrip(tmp_path):
    model = _toy_model()
    x0 = [0.0] * len(DANGER_FEATURES)
    x1 = list(x0)
    x1[DANGER_FEATURES.index("sig_opponent_threat")] = 1.0
    p0, p1 = model.predict(x0), model.predict(x1)
    assert 0.0 < p0 < p1 < 1.0    # higher threat → higher probability

    path = tmp_path / "m.json"
    model.save(str(path))
    loaded = LinearModel.load(str(path))
    assert loaded.predict(x1) == pytest.approx(p1)

    top = model.explain(x1)[0]
    assert top["feature"] == "sig_opponent_threat"


# ── Agent ────────────────────────────────────────────────────────────

def test_learned_agent_plays_deterministically():
    def play(seed):
        agents = [LearnedAgent(f"L{i}", model=_toy_model()) for i in range(2)]
        agents += [HybridAgent(f"H{i}") for i in range(2)]
        return GameState(agents, seed=seed).play()

    r1, r2 = play(9), play(9)
    assert (r1.winner, r1.win_type, r1.turns) == (r2.winner, r2.win_type, r2.turns)
