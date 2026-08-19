"""Tests for the ML pipeline: features, datagen labels, model, agent."""

import pytest

from mahjong.game import GameState, advance_turns
from mahjong.agents import GreedyAgent, HybridAgent, LearnedAgent
from mahjong.opponent_model import estimate_opponent_threats
from mahjong.ml.features import (
    DANGER_FEATURES, OUTCOME_FEATURES, danger_features, outcome_features,
)
from mahjong.ml.datagen import DANGER_COLUMNS, OUTCOME_COLUMNS, generate_game
from mahjong.ml.model import (
    LinearModel, CompositeValueModel,
    load_danger_model, load_value_model, load_win_model,
)


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


def _post_discard(game, seat=0):
    """Counts after discarding the seat's first tile — a 13-tile shape."""
    counts = game.hands[seat].copy_counts()
    counts[game.hands[seat].tiles[0]] -= 1
    return counts


def test_outcome_features_shape():
    game = _midgame()
    threat = estimate_opponent_threats(0, game)
    x = outcome_features(0, game, _post_discard(game), 2, 12, threat)
    assert len(x) == len(OUTCOME_FEATURES)
    assert all(0.0 <= v <= 1.0 for v in x)


def test_tai_potential_features_see_hand_value():
    # Two hands at the same shanten: a full-flush track vs mixed junk.
    # The value features must separate them — that's their entire job.
    # Fresh game: seat 0 must have no exposed melds/flowers, since the
    # extractor correctly folds those into the synthetic counts.
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=2)
    game.setup()
    game.hands[0].flowers.clear()
    threat = estimate_opponent_threats(0, game)

    def features_for(tiles):
        counts = [0] * 34
        for t in tiles:
            counts[t] += 1
        return outcome_features(0, game, counts, 2, 10, threat)

    flushy = features_for([0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 0, 1])  # all wan
    # Spread across all three suits with few honors — genuinely junk.
    # (A honor-heavy hand would legitimately score higher: half-flush track.)
    junk = features_for([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29, 31, 33])

    names = {n: i for i, n in enumerate(OUTCOME_FEATURES)}
    assert flushy[names["suit_concentration"]] == 1.0
    assert junk[names["suit_concentration"]] < 0.6
    assert flushy[names["run_progress"]] > junk[names["run_progress"]]
    assert junk[names["honor_frac"]] > flushy[names["honor_frac"]]


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

def _toy_value_model():
    """Identity-link value model: rewards low shanten, run shape, and
    suit concentration — a crude but directionally sane V(state)."""
    n = len(OUTCOME_FEATURES)
    coef = [0.0] * n
    coef[OUTCOME_FEATURES.index("shanten")] = -5.0
    coef[OUTCOME_FEATURES.index("run_progress")] = 1.0
    coef[OUTCOME_FEATURES.index("suit_concentration")] = 1.0
    return LinearModel(features=list(OUTCOME_FEATURES),
                       mean=[0.0] * n, scale=[1.0] * n,
                       coef=coef, intercept=0.5, link="identity")


def test_learned_agent_plays_deterministically():
    def play(seed):
        agents = [LearnedAgent(f"L{i}", model=_toy_model(),
                               value_model=_toy_value_model())
                  for i in range(2)]
        agents += [HybridAgent(f"H{i}") for i in range(2)]
        return GameState(agents, seed=seed).play()

    r1, r2 = play(9), play(9)
    assert (r1.winner, r1.win_type, r1.turns) == (r2.winner, r2.win_type, r2.turns)


# ── The shipped model artifacts ──────────────────────────────────────
#
# Everything above uses a hand-built toy model, so nothing asserted on
# the JSON that actually ships. A corrupted or badly-retrained weights
# file would otherwise pass the whole suite (audit finding F5).

def test_packaged_danger_model_is_sane():
    model = load_danger_model()
    assert model is not None, "danger_model.json missing from the package"
    assert model.features == list(DANGER_FEATURES)  # order matters
    assert len(model.coef) == len(model.mean) == len(model.scale)
    assert all(s > 0 for s in model.scale)

    game = _midgame()
    visible = game.get_visible_counts(0)
    threat = estimate_opponent_threats(0, game)
    probs = [model.predict(danger_features(t, 0, game, visible, threat))
             for t in set(game.hands[0].tiles)]
    assert all(0.0 < p < 1.0 for p in probs)
    # Mid-game deal-in risk is small but not vanishing; a model that
    # collapsed to a constant would fail the spread check.
    assert max(probs) < 0.6
    assert max(probs) > min(probs)


def test_packaged_win_model_is_sane():
    model = load_win_model()
    assert model is not None, "win_model.json missing from the package"
    assert model.features == list(OUTCOME_FEATURES)

    game = _midgame()
    threat = estimate_opponent_threats(0, game)
    counts = _post_discard(game)
    # A tenpai hand must be rated likelier to win than a far-off one
    near = model.predict(outcome_features(0, game, counts, 0, 20, threat))
    far = model.predict(outcome_features(0, game, counts, 5, 4, threat))
    assert 0.0 < far < near < 1.0


def test_packaged_value_model_is_sane():
    model = load_value_model()
    assert model is not None, "value_model.json missing from the package"
    assert model.features == list(OUTCOME_FEATURES)
    assert model.link == "identity"

    game = _midgame()
    threat = estimate_opponent_threats(0, game)
    counts = _post_discard(game)
    near = model.predict(outcome_features(0, game, counts, 0, 20, threat))
    far = model.predict(outcome_features(0, game, counts, 5, 4, threat))
    # Values are points: tenpai must be worth more than hopeless, and
    # both must sit inside plausible point stakes for this table.
    assert near > far
    assert -10.0 < far < near < 40.0


# ── Phase C: composite value model ───────────────────────────────────

def _flat(coef_overrides, intercept=0.0, link="logistic"):
    n = len(OUTCOME_FEATURES)
    coef = [0.0] * n
    for name, v in coef_overrides.items():
        coef[OUTCOME_FEATURES.index(name)] = v
    return LinearModel(features=list(OUTCOME_FEATURES),
                       mean=[0.0] * n, scale=[1.0] * n,
                       coef=coef, intercept=intercept, link=link)


def test_composite_value_predicts_decomposed_expectation():
    comp = CompositeValueModel(
        win=_flat({}, intercept=0.0),          # sigmoid(0) = 0.5
        win_size=_flat({}, intercept=6.0, link="identity"),
        pay=_flat({}, intercept=-1.0986),      # sigmoid ≈ 0.25
        pay_size=_flat({}, intercept=8.0, link="identity"))
    x = [0.0] * len(OUTCOME_FEATURES)
    # V = 0.5·6 − 0.25·8 = 1.0
    assert comp.predict(x) == pytest.approx(1.0, abs=1e-3)


def test_composite_clamps_negative_magnitudes():
    comp = CompositeValueModel(
        win=_flat({}, intercept=10.0),          # P(win) ≈ 1
        win_size=_flat({}, intercept=-5.0, link="identity"),
        pay=_flat({}, intercept=-10.0),         # P(pay) ≈ 0
        pay_size=_flat({}, intercept=3.0, link="identity"))
    x = [0.0] * len(OUTCOME_FEATURES)
    # A negative "size of the win" is extrapolation noise → clamped to 0,
    # never a phantom penalty.
    assert comp.predict(x) == pytest.approx(0.0, abs=1e-3)


def test_composite_roundtrip_and_loader_dispatch(tmp_path):
    comp = CompositeValueModel(
        win=_flat({"shanten": -2.0}), win_size=_flat({}, 5.0, "identity"),
        pay=_flat({}, -1.0), pay_size=_flat({}, 4.0, "identity"),
        metadata={"label": "test"})
    path = tmp_path / "value_model.json"
    comp.save(str(path))
    loaded = CompositeValueModel.load(str(path))
    x = [0.1] * len(OUTCOME_FEATURES)
    assert loaded.predict(x) == pytest.approx(comp.predict(x))
    assert loaded.features == list(OUTCOME_FEATURES)
    assert loaded.link == "identity"


def test_packaged_value_model_is_composite():
    model = load_value_model()
    assert isinstance(model, CompositeValueModel), (
        "value_model.json should be the Phase C composite artifact")


# ── Phase B: claims as branch evaluation ─────────────────────────────

def _value_model_pricing_concealment():
    """Toy value model where concealment and chow shape carry real
    points and shanten matters — enough to make claim trade-offs
    concrete and deterministic for tests."""
    n = len(OUTCOME_FEATURES)
    coef = [0.0] * n
    # Features are normalised (shanten/6), so a coefficient of −18 makes
    # one shanten step worth 3 points against concealment's 2.
    coef[OUTCOME_FEATURES.index("shanten")] = -18.0
    coef[OUTCOME_FEATURES.index("is_concealed")] = 2.0
    coef[OUTCOME_FEATURES.index("run_progress")] = 1.0
    return LinearModel(features=list(OUTCOME_FEATURES),
                       mean=[0.0] * n, scale=[1.0] * n,
                       coef=coef, intercept=1.0, link="identity")


def _quiet_danger_model():
    n = len(DANGER_FEATURES)
    return LinearModel(features=list(DANGER_FEATURES),
                       mean=[0.0] * n, scale=[1.0] * n,
                       coef=[0.0] * n, intercept=-4.0)  # ~1.8% flat


def _claim_fixture():
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=3)
    agent = LearnedAgent("L", model=_quiet_danger_model(),
                         value_model=_value_model_pricing_concealment())
    return game, agent


def test_declines_chow_that_only_breaks_concealment():
    # Already tenpai and fully concealed (45w + three runs + 9s pair,
    # waiting 3w/6w). Chowing the 3w leaves the hand exactly as close
    # (tenpai on the 9s pair instead) but destroys is_concealed —
    # under a value model that prices concealment, decline.
    game, agent = _claim_fixture()
    hand = game.hands[1]
    for t in [3, 4, 9, 10, 11, 18, 19, 20, 21, 22, 23, 26, 26]:
        hand.add_tile(t)
    choice = agent.choose_chow(1, 2, [(3, 4)], game)
    assert choice is None


def test_takes_chow_that_completes_real_progress():
    # Five blocks, 1-shanten (two partials, one spare junk tile). The
    # chow converts 45w into a meld WITHOUT spending a draw: discard
    # the lone White Dragon and the hand is tenpai. Shanten 1 → 0 at
    # -6/shanten dwarfs the 2-point concealment loss: must claim.
    game, agent = _claim_fixture()
    hand = game.hands[1]
    for t in [3, 4, 9, 10, 11, 18, 19, 20, 26, 26, 24, 25, 33]:
        hand.add_tile(t)
    choice = agent.choose_chow(1, 2, [(3, 4)], game)
    assert choice == (3, 4)


def test_claim_decisions_are_deterministic():
    def play(seed):
        agents = [LearnedAgent(f"L{i}", model=_quiet_danger_model(),
                               value_model=_value_model_pricing_concealment())
                  for i in range(4)]
        game = GameState(agents, seed=seed)
        result = game.play()
        melds = sum(game.hands[p].num_exposed_melds for p in range(4))
        return (result.winner, result.win_type, result.turns, melds)

    assert play(21) == play(21)
