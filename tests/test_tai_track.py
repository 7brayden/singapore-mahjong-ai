"""Engine-truth tai arithmetic and the claim legality gate.

Pins the fix for a verified bug: the claim advisor recommended ponging
the pair out of a live ping hu — trading a 4-tai track for a hand that
could not legally win. These tests pin BEHAVIOR, not model weights: the
adversarial-model tests must hold under any future retrain.
"""

import pytest

from mahjong.hand import Hand
from mahjong.game import GameState
from mahjong.agents import GreedyAgent, HybridAgent, LearnedAgent
from mahjong.scoring import ScoreConfig
from mahjong.tai_track import (
    structural_tai_ceiling, legal_win_exists, claim_kills_hand,
    claim_consequence,
)
from mahjong.tiles import WIND_START
from mahjong.ml.model import LinearModel, CompositeValueModel
from mahjong.ml.features import OUTCOME_FEATURES

CFG = ScoreConfig()
EAST = WIND_START

# The workflow's reproduction hand: 2w3w4w 5w 9w 4t 7t 2s3s4s 5s5s 7s —
# concealed, no flowers, a clean chow track holding a 5s pair.
REPRO = [1, 2, 3, 4, 8, 12, 15, 19, 20, 21, 22, 22, 24]
# 1-shanten variant: 234w 567w 11t 2t 567s 9s
VARIANT = [1, 2, 3, 4, 5, 6, 9, 9, 10, 22, 23, 24, 26]


def hand_of(tiles, flowers=(), exposed=()):
    h = Hand()
    for t in tiles:
        h.add_tile(t)
    for f in flowers:
        h.add_tile(f)
    for meld in exposed:
        h.exposed.append(meld)
    return h


def game_with(hand, seat=0, seed=0):
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=seed)
    game.wall = [0] * 60
    game.hands[seat] = hand
    return game


# ── structural_tai_ceiling ───────────────────────────────────────────

def test_ceiling_live_pinghu_track():
    # Clean concealed chow track: ping hu 4 + 门清 1 in reach
    h = hand_of(REPRO)
    assert structural_tai_ceiling(h.counts, h.exposed, h.flowers,
                                  0, EAST, CFG) >= 5


def test_ceiling_dead_after_pair_pong():
    # Pong the 5s pair: no honor tracks, chow-shaped (no triplet or
    # flush trajectory), pinghu dead → structurally 0
    h = hand_of(REPRO)
    counts = h.copy_counts()
    counts[22] -= 2
    exposed = [("pong", [22, 22, 22])]
    assert structural_tai_ceiling(counts, exposed, [], 0, EAST, CFG) == 0


def test_ceiling_survives_value_pongs():
    # A seat-wind pair keeps the hand alive through the same pong
    h = hand_of([1, 2, 3, 4, 5, 6, 9, 10, 11, 19, 20, EAST, EAST])
    counts = h.copy_counts()
    counts[EAST] -= 2
    exposed = [("pong", [EAST] * 3)]
    after = structural_tai_ceiling(counts, exposed, [], 0, EAST, CFG)
    assert after >= 1  # seat+prevailing wind track lives on


def test_ceiling_counts_banked_bonus():
    h = hand_of(REPRO, flowers=[42])  # an animal = 1 banked tai
    counts = h.copy_counts()
    counts[22] -= 2
    exposed = [("pong", [22, 22, 22])]
    assert structural_tai_ceiling(counts, exposed, h.flowers,
                                  0, EAST, CFG) >= 1


# ── legal_win_exists (tenpai legality) ───────────────────────────────

def test_four_claimed_chows_bare_pair_is_dead():
    # All four chows claimed, bare pair wait, no flowers: the reference
    # rules say this can NEVER score ping hu — and with no other tai
    # source the wait is legally dead.
    exposed = [("chow", [0, 1, 2]), ("chow", [3, 4, 5]),
               ("chow", [9, 10, 11]), ("chow", [18, 19, 20])]
    counts = [0] * 34
    counts[24] = 1  # lone 7s waiting to pair
    assert not legal_win_exists(counts, exposed, [], 0, EAST, CFG)


def test_concealed_single_wait_is_alive_via_tsumo():
    # Concealed clean ping hu on a single wait: ron-illegal but
    # tsumo-legal (ping hu + 门清) — the hand is NOT dead.
    h = hand_of([0, 1, 3, 4, 5, 9, 10, 11, 21, 22, 23, 25, 25])
    assert legal_win_exists(h.counts, h.exposed, [], 0, EAST, CFG)


# ── claim_kills_hand (the gate) ──────────────────────────────────────

def test_gate_vetoes_pinghu_killing_pair_pong():
    for tiles, pair in ((REPRO, 22), (VARIANT, 9)):
        h = hand_of(tiles)
        assert claim_kills_hand(h, 0, EAST, CFG, pair, "pong")


def test_gate_allows_tai_bearing_pongs():
    # Dragon pair pong out of a chow hand: forfeits ping hu but the
    # dragon triplet is 1 tai — live→live, the model's call, not ours.
    h = hand_of([1, 2, 3, 4, 5, 6, 9, 10, 11, 19, 20, 31, 31])
    assert not claim_kills_hand(h, 0, EAST, CFG, 31, "pong")


def test_gate_ignores_already_dead_hands():
    # An exposed-pong chicken shape is already dead: claiming more is
    # not the gate's business (resurrection stays possible via pongs).
    h = hand_of([1, 2, 3, 9, 10, 15, 15, 20, 24, 26],
                exposed=[("pong", [13, 13, 13])])
    assert not claim_kills_hand(h, 0, EAST, CFG, 15, "pong")


def test_gate_respects_chicken_config():
    h = hand_of(REPRO)
    allow = ScoreConfig(allow_chicken_hand=True)
    assert not claim_kills_hand(h, 0, EAST, allow, 22, "pong")


# ── claim_consequence (the honest surface) ───────────────────────────

def test_consequence_reports_dead_after():
    h = hand_of(REPRO)
    c = claim_consequence(h, 0, EAST, CFG, 22, "pong")
    assert c["dead_after"] is True
    assert c["kills_pinghu"] is True
    assert c["tai_ceiling_before"] >= 5
    assert c["tai_ceiling_after"] == 0
    assert "cannot win" in c["headline"]


def test_consequence_quiet_on_value_pong():
    h = hand_of([1, 2, 3, 4, 5, 6, 9, 10, 11, 19, 20, 31, 31],
                flowers=[42])
    c = claim_consequence(h, 0, EAST, CFG, 31, "pong")
    assert c["dead_after"] is False


# ── Agent behavior pins (weight-independent) ─────────────────────────

def _adversarial_value_model():
    """A composite model built to MAXIMALLY favor claiming: high
    P(win) everywhere, big wins, no downside. If the gate holds under
    this, it holds under any artifact a retrain can produce."""
    n = len(OUTCOME_FEATURES)
    def flat(intercept, link):
        return LinearModel(features=list(OUTCOME_FEATURES),
                           mean=[0.0] * n, scale=[1.0] * n,
                           coef=[0.0] * n, intercept=intercept, link=link)
    return CompositeValueModel(
        win=flat(4.0, "logistic"),          # P(win) ≈ 0.98
        win_size=flat(50.0, "identity"),    # every win worth 50
        pay=flat(-8.0, "logistic"),         # never pays
        pay_size=flat(0.0, "identity"))


@pytest.fixture
def quiet_danger():
    n = 16
    from mahjong.ml.features import DANGER_FEATURES
    return LinearModel(features=list(DANGER_FEATURES),
                       mean=[0.0] * n, scale=[1.0] * n,
                       coef=[0.0] * n, intercept=-6.0)


def test_learned_agent_refuses_dead_pong_under_any_model(quiet_danger):
    agent = LearnedAgent("L", model=quiet_danger,
                         value_model=_adversarial_value_model())
    for tiles, pair in ((REPRO, 22), (VARIANT, 9)):
        game = game_with(hand_of(tiles))
        assert agent.should_claim(0, pair, "pong", game) is False


def test_clamp_never_flips_value_pongs():
    # Non-inferiority at the unit level: on tai-bearing pong windows
    # (live→live, so the gate stays out of it) the engine clamp must
    # agree with the raw model — it may only correct ILLEGAL optimism,
    # never add anti-claim pressure. The 1k duplicate eval covers the
    # statistical version of this claim.
    class Unclamped(LearnedAgent):
        def _clamped_value(self, x, counts, exposed, shanten,
                           player_idx, game_state):
            return self.value_model.predict(x)

    states = [
        ([18, 19, 20, 21, 22, 23, 24, 25, 26, 26, 31, 31, 14], 31),
        ([1, 2, 3, 4, 5, 6, 9, 10, 11, 19, 20, 31, 31], 31),
        ([1, 2, 3, 4, 5, 6, 9, 10, 11, 19, 20, 27, 27], 27),
    ]
    for tiles, tid in states:
        clamped = LearnedAgent("L").should_claim(
            0, tid, "pong", game_with(hand_of(tiles)))
        raw = Unclamped("U").should_claim(
            0, tid, "pong", game_with(hand_of(tiles)))
        assert clamped == raw


def test_learned_agent_declines_dead_fourth_chow(quiet_danger):
    # Three claimed chows + bare-pair-and-run shape: the 4th chow
    # leaves a bare pair wait that can never score — gate refuses even
    # under the adversarial model.
    agent = LearnedAgent("L", model=quiet_danger,
                         value_model=_adversarial_value_model())
    exposed = [("chow", [0, 1, 2]), ("chow", [9, 10, 11]),
               ("chow", [18, 19, 20])]
    h = hand_of([3, 5, 24, 24], exposed=exposed)  # 4w 6w + 7s pair
    game = game_with(h)
    assert agent.choose_chow(0, 4, [(3, 5)], game) is None


def test_hybrid_agent_shares_the_gate():
    agent = HybridAgent("H")
    for tiles, pair in ((REPRO, 22), (VARIANT, 9)):
        game = game_with(hand_of(tiles))
        game.agents[0] = agent
        assert agent.should_claim(0, pair, "pong", game) is False


def test_consequence_neutral_wind_pong_headline():
    """Regression (screenshot bug): a neutral-wind pair pong on a
    flower-holding hand got narrated with irrelevant ping-hu context.
    The headline must state THIS decision's facts: 0-tai wind, chou
    ping hu killed, concealment broken."""
    h = hand_of([0, 2, 4, 4, 5, 6, 9, 10, 15, 16, 20, 28, 28],
                flowers=[40, 45])  # Blue 3 + centipede; East seat
    c = claim_consequence(h, 0, EAST, CFG, 28, "pong")
    assert c["dead_after"] is False
    assert "neither your seat nor the prevailing wind" in c["headline"]
    assert "chou ping hu" in c["headline"]
    assert "门清" in c["headline"]

    # The same pong for the SOUTH-seat player is their seat wind —
    # a 1-tai claim, and the wind complaint must not appear.
    c = claim_consequence(h, 1, EAST, CFG, 28, "pong")
    assert c["headline"] is None or "neither your seat" not in c["headline"]


# ── Tenpai hysteresis (the "advisor broke my ready hand" bug) ────────

def _acceptance_model(points_per_acceptance):
    """Monolithic identity model: V = k · acceptance feature. Lets a
    test dial exactly how much the model prefers wide unready hands."""
    n = len(OUTCOME_FEATURES)
    coef = [0.0] * n
    coef[OUTCOME_FEATURES.index("acceptance")] = points_per_acceptance
    return LinearModel(features=list(OUTCOME_FEATURES), mean=[0.0] * n,
                       scale=[1.0] * n, coef=coef, intercept=0.0,
                       link="identity")


# The screenshot hand: 1111w 23456w 77t 6s8s + drawn 9w. Discarding 9w
# keeps a LEGAL ready hand (wait 7s, cat = 1 tai); discarding 8s goes
# back to 1-shanten with far more acceptance.
TENPAI_HAND = [0, 0, 0, 0, 1, 2, 3, 4, 5, 15, 15, 23, 25, 8]


def test_agent_holds_legal_tenpai_inside_margin(quiet_danger):
    # The model mildly prefers the wide 1-shanten branch (like the real
    # artifact did, by 0.01 pts) — the hysteresis must hold the ready
    # hand and discard the drawn 9w.
    agent = LearnedAgent("L", model=quiet_danger,
                         value_model=_acceptance_model(0.5))
    game = game_with(hand_of(TENPAI_HAND, flowers=[39, 42]))
    assert agent.choose_discard(0, game) == 8  # 9 Wan keeps the wait


def test_agent_may_break_tenpai_past_margin(quiet_danger):
    # When the alternative genuinely dominates (here: several points of
    # EV), breaking the ready hand is allowed — hysteresis, not a ban.
    agent = LearnedAgent("L", model=quiet_danger,
                         value_model=_acceptance_model(8.0))
    game = game_with(hand_of(TENPAI_HAND, flowers=[39, 42]))
    assert agent.choose_discard(0, game) != 8
