"""Engine-truth tai arithmetic for decisions: what can this hand still score?

Born from a verified bug: the claim advisor recommended ponging the pair
out of a live ping hu — a 4-tai track traded for a hand that could not
even legally win (minimum 1 tai). The value model priced the forfeit
2-5x too low, and NOTHING in the decision pipeline knew the legality
rule at all. The current artifact avoiding the bug was a coefficient
accident; any retrain could bring it back.

This module is the single source of engine truth about tai potential.
The agent's claim gate, the /hint payload, the coach's context, and
(Phase C) the legality features all read it — one implementation, so it
can never drift from scoring.py the way pinghu_live did.

Everything here is a CEILING — a generous upper bound on achievable tai.
Overcounting can never wrongly veto a claim (the gate only fires when
the ceiling hits zero) and can never wrongly clamp a valuation (the cap
sits above the true maximum). Undercounting could do both, so every
judgment call rounds up.
"""

from typing import Dict, List, Optional, Tuple

from mahjong.tiles import (
    NUM_STANDARD_UNIQUE, is_honor, suit_of, Suit,
    WIND_START, DRAGON_START, FLOWER_START, ANIMAL_START,
)
from mahjong.hand import Hand, calculate_shanten, get_winning_tiles
from mahjong.scoring import ScoreConfig, score_win, is_legal_win

# How far off a pattern track a hand may be and still have the track
# counted in its ceiling. 1 = pursuing the track costs at most one
# extra effective draw over the hand's natural line — "actually on
# it", not "could theoretically rebuild toward it". Loosening this can
# only weaken the gate (toward today's behavior); it can never veto a
# good claim.
TRACK_SLACK = 1


def bonus_tai(flowers: List[int], seat_index: int) -> int:
    """Tai banked in bonus tiles under this table's rules: seat flowers
    and animals 1 each, all four animals +1, each complete series +1.

    (Moved here from ml/features.py — the features module imports it,
    so training and decisions share one implementation.)
    """
    fset = set(flowers)
    tai = 0
    for f in fset:
        if f >= ANIMAL_START:
            tai += 1
        elif (f - FLOWER_START) % 4 == seat_index:
            tai += 1
    if fset >= set(range(ANIMAL_START, ANIMAL_START + 4)):
        tai += 1
    for start in (FLOWER_START, FLOWER_START + 4):
        if fset >= set(range(start, start + 4)):
            tai += 1
    return tai


def _shanten_triplets_only(counts: List[int], num_melds: int) -> int:
    """Shanten toward an all-triplets decomposition (碰碰胡 track)."""
    needed = 4 - num_melds
    triplets = sum(1 for t in range(NUM_STANDARD_UNIQUE) if counts[t] >= 3)
    pairs = sum(1 for t in range(NUM_STANDARD_UNIQUE) if counts[t] == 2)
    best = 8
    if pairs > 0:
        # one pair is the eyes; the rest are partial triplets
        melds = min(triplets, needed)
        partials = min(pairs - 1, needed - melds)
        best = min(best, 2 * needed - 2 * melds - partials - 1)
    melds = min(triplets, needed)
    partials = min(pairs, needed - melds)
    best = min(best, 2 * needed - 2 * melds - partials)
    return best


def _shanten_flush(counts: List[int], num_melds: int,
                   suit: Suit, keep_honors: bool) -> int:
    """Shanten toward a one-suit hand: off-suit tiles are simply absent
    from the counts, which the standard shanten treats as draws to come."""
    filtered = [0] * NUM_STANDARD_UNIQUE
    for t in range(NUM_STANDARD_UNIQUE):
        if counts[t] == 0:
            continue
        if is_honor(t):
            if keep_honors:
                filtered[t] = counts[t]
        elif suit_of(t) == suit:
            filtered[t] = counts[t]
    return calculate_shanten(filtered, num_melds)


def structural_tai_ceiling(counts: List[int],
                           exposed: List[Tuple[str, List[int]]],
                           flowers: List[int], seat_index: int,
                           prevailing_wind: int,
                           config: ScoreConfig,
                           shanten: Optional[int] = None) -> int:
    """Generous upper bound on the tai this hand can still score.

    Counts only what the hand demonstrably holds or is genuinely on
    track for: banked bonus tai, honor pairs/triplets of value tiles,
    and the best live pattern track (ping hu family / all triplets /
    flush) within TRACK_SLACK of the hand's real shanten. Lottery tai
    (future flowers, 杠上, 海底) are excluded — including them would
    make the bound vacuous.
    """
    tai = bonus_tai(flowers, seat_index)

    num_melds = len(exposed)
    claimed_melds = sum(1 for kind, _ in exposed if kind != "concealed_kong")
    exposed_pongs = sum(1 for kind, _ in exposed if kind != "chow")
    exposed_chows = num_melds - exposed_pongs

    # Full counts: meld tiles still belong to the hand for tai purposes
    full = counts[:]
    off_suit_meld = {Suit.WAN: False, Suit.TONG: False, Suit.SUO: False}
    meld_has_honor = False
    for kind, tiles in exposed:
        for t in tiles:
            full[t] = min(4, full[t] + 1)
            if is_honor(t):
                meld_has_honor = True
            else:
                for s in off_suit_meld:
                    if suit_of(t) != s:
                        off_suit_meld[s] = True

    # ── Honor tracks: a pair of a value tile is pong-reachable ───────
    for d in range(DRAGON_START, DRAGON_START + 3):
        if full[d] >= 2:
            tai += config.tai_for("dragon_triplet")
    seat_wind = WIND_START + seat_index
    if full[seat_wind] >= 2:
        tai += config.tai_for("seat_wind_triplet")
    if full[prevailing_wind] >= 2:
        tai += config.tai_for("prevailing_wind_triplet")

    if shanten is None:
        shanten = calculate_shanten(counts, num_melds)

    # ── Pattern tracks: best one the hand is actually on ─────────────
    pattern = 0
    # Ping hu family: dead the moment a pong/kong is exposed, and a
    # hand with all four melds claimed can never score it (bare-pair
    # wait rule). Chou ping hu with any bonus tile.
    if exposed_pongs == 0 and num_melds < 4:
        pattern = max(pattern, config.tai_for(
            "chou_ping_hu" if flowers else "ping_hu"))
    # All triplets: dead once a chow is claimed.
    if exposed_chows == 0:
        if _shanten_triplets_only(counts, num_melds) <= shanten + TRACK_SLACK:
            pattern = max(pattern, config.tai_for("all_triplets"))
    # Flush tracks: dead if melds span suits.
    for s in (Suit.WAN, Suit.TONG, Suit.SUO):
        if off_suit_meld[s]:
            continue
        if _shanten_flush(counts, num_melds, s, keep_honors=True) \
                <= shanten + TRACK_SLACK:
            tai_half = config.tai_for("half_flush")
            pattern = max(pattern, tai_half)
        if not meld_has_honor and _shanten_flush(
                counts, num_melds, s, keep_honors=False) \
                <= shanten + TRACK_SLACK:
            pattern = max(pattern, config.tai_for("full_flush"))
    tai += pattern

    # 门清: fully concealed hands keep the self-draw +1 in reach
    if claimed_melds == 0:
        tai += config.tai_for("fully_concealed")

    return tai


def legal_win_exists(counts: List[int],
                     exposed: List[Tuple[str, List[int]]],
                     flowers: List[int], seat_index: int,
                     prevailing_wind: int, config: ScoreConfig) -> bool:
    """At tenpai: can ANY wait complete into a LEGAL win (>= min tai)?

    Checks both self-draw and discard scoring — the ping hu wait rules
    are asymmetric (a single wait scores only on tsumo), so a hand can
    be tsumo-legal yet ron-illegal. Either passing counts as alive.
    """
    num_melds = len(exposed)
    waits = get_winning_tiles(counts[:], num_melds)
    if not waits:
        return False
    probe = Hand()
    probe.counts = counts[:]
    probe.tiles = [t for t in range(NUM_STANDARD_UNIQUE)
                   for _ in range(counts[t])]
    probe.exposed = list(exposed)
    probe.flowers = list(flowers)
    for w in waits:
        probe.counts[w] += 1
        probe.tiles.append(w)
        for tsumo in (True, False):
            score = score_win(probe, w, tsumo, seat_index,
                              prevailing_wind, config)
            if score is not None and is_legal_win(score, config):
                return True
        probe.counts[w] -= 1
        probe.tiles.pop()
    return False


def claim_kills_hand(hand: Hand, seat_index: int, prevailing_wind: int,
                     config: ScoreConfig, tile_id: int, claim_type: str,
                     partners: Optional[Tuple[int, int]] = None) -> bool:
    """The claim gate: does taking this claim turn a structurally
    ALIVE hand into one that cannot reach the minimum tai?

    Fires only on live→dead transitions. A hand that is already dead
    may still claim (resurrection through pongs of value tiles etc.),
    and live→live trades stay the model's decision — this is a
    legality constraint, not strategy.
    """
    if config.allow_chicken_hand:
        return False

    counts = hand.copy_counts()
    before = structural_tai_ceiling(counts, hand.exposed, hand.flowers,
                                    seat_index, prevailing_wind, config)
    if before < config.min_tai:
        return False  # already dead — claiming can't make it worse

    if claim_type == "chow" and partners is not None:
        counts[partners[0]] -= 1
        counts[partners[1]] -= 1
        meld = ("chow", sorted([tile_id, partners[0], partners[1]]))
    elif claim_type in ("pong", "kong"):
        needed = 2 if claim_type == "pong" else 3
        if counts[tile_id] < needed:
            return False
        counts[tile_id] -= needed
        meld = (claim_type, [tile_id] * (needed + 1))
    elif claim_type in ("concealed", "added"):
        # Own-turn kong declarations
        if claim_type == "concealed":
            counts[tile_id] -= 4
            meld = ("concealed_kong", [tile_id] * 4)
        else:
            counts[tile_id] -= 1
            meld = None  # the pong already exists; kind is unchanged
    else:
        return False

    exposed_after = list(hand.exposed) + ([meld] if meld else [])
    after = structural_tai_ceiling(counts, exposed_after, hand.flowers,
                                   seat_index, prevailing_wind, config)
    return after < config.min_tai


def claim_consequence(hand: Hand, seat_index: int, prevailing_wind: int,
                      config: ScoreConfig, tile_id: int, claim_type: str,
                      partners: Optional[Tuple[int, int]] = None) -> Dict:
    """Engine-computed facts about what a claim would do to the hand's
    tai potential — for the /hint payload, the claim card, and the
    coach. A thin formatter over the same arithmetic as the gate."""
    counts = hand.copy_counts()
    exposed_pongs = sum(1 for kind, _ in hand.exposed if kind != "chow")
    claimed = sum(1 for kind, _ in hand.exposed if kind != "concealed_kong")
    pinghu_before = exposed_pongs == 0 and len(hand.exposed) < 4
    concealed_before = claimed == 0

    before = structural_tai_ceiling(counts, hand.exposed, hand.flowers,
                                    seat_index, prevailing_wind, config)
    if claim_type == "chow" and partners is not None:
        counts[partners[0]] -= 1
        counts[partners[1]] -= 1
        meld = ("chow", sorted([tile_id, partners[0], partners[1]]))
    else:
        needed = 2 if claim_type == "pong" else 3
        counts[tile_id] -= needed
        meld = (claim_type, [tile_id] * (needed + 1))
    exposed_after = list(hand.exposed) + [meld]
    after = structural_tai_ceiling(counts, exposed_after, hand.flowers,
                                   seat_index, prevailing_wind, config)

    kills_pinghu = (pinghu_before and claim_type in ("pong", "kong")
                    and not hand.flowers)
    dead_after = after < config.min_tai and not config.allow_chicken_hand

    headline = None
    if dead_after:
        headline = ("Taking this claim leaves no tai source in sight — "
                    "a 0-tai hand cannot win at this table.")
    elif kills_pinghu:
        headline = (f"Taking this {claim_type} forfeits clean ping hu "
                    f"(4 tai) — best achievable drops to about {after} tai.")
    elif concealed_before and claim_type == "chow" and not hand.flowers:
        headline = ("Claiming breaks concealment: the self-drawn 门清 "
                    "+1 is forfeit, and ping hu must now finish on a "
                    "two-tile wait.")

    return {
        "kills_pinghu": kills_pinghu,
        "forfeits_menqing": concealed_before,
        "tai_ceiling_before": before,
        "tai_ceiling_after": after,
        "dead_after": dead_after,
        "headline": headline,
    }
