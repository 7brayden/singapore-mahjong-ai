"""Feature extraction for the learned models.

One source of truth for feature vectors: the data generator, the
training script, and live inference all call these functions, so a
model trained on generated data sees identical features at play time.

Everything here is computed from information VISIBLE to the acting
seat — the same constraint the heuristic agents live under. Labels
(which use perfect information) live in datagen.py, never here.

The four heuristic danger signals are included as features on purpose:
the logistic regression then learns, from real deal-in outcomes, the
weights that defense.py currently hard-codes as 0.25/0.25/0.35/0.15 —
and the raw tile/state features let it correct them where they're wrong.
"""

from typing import Dict, List

from mahjong.tiles import (
    is_honor, is_numbered, rank_of, suit_of, Suit,
    NUM_STANDARD_UNIQUE, WAN_START, TONG_START, SUO_START,
    WIND_START, DRAGON_START, FLOWER_START, ANIMAL_START,
)
from mahjong.defense import estimate_danger_detailed


# ── Danger model: one vector per (state, candidate tile) ─────────────

DANGER_FEATURES = [
    "is_honor",             # 0/1
    "is_terminal",          # 0/1 — numbered rank 1 or 9
    "centrality",           # 0 honors; 0.2 edge .. 1.0 middle (rank 5)
    "copies_visible",       # visible copies of this tile / 4
    "adj1_visible",         # visible copies at rank ±1 / 8 (edges count as dead)
    "adj2_visible",         # visible copies at rank ±2 / 8
    "opp_river_copies",     # copies of this exact tile in opponent rivers / 3
    "sig_visibility",       # the four heuristic signals from defense.py
    "sig_discard_absence",
    "sig_opponent_threat",
    "sig_suit_safety",
    "turn_frac",            # min(1, turn / 60)
    "wall_frac",            # tiles_remaining / 92
    "max_opp_threat",       # opponent_model's strongest-threat estimate
    "opp_max_melds",        # most exposed melds by any opponent / 4
    "opp_total_melds",      # exposed melds across all opponents / 12
]


def danger_features(tile_id: int, player_idx: int, game,
                    visible: List[int], threat_data: Dict) -> List[float]:
    """Feature vector for "how dangerous is discarding this tile here".

    `visible` and `threat_data` are passed in so the caller computes
    them once per decision, not once per candidate tile.
    """
    detail = estimate_danger_detailed(tile_id, player_idx, game, threat_data)
    sig = detail["components"]

    honor = 1.0 if is_honor(tile_id) else 0.0
    if is_numbered(tile_id):
        rank = rank_of(tile_id)
        terminal = 1.0 if rank in (1, 9) else 0.0
        centrality = (5 - abs(rank - 5)) / 5.0
        adj1 = _adjacent_visible(tile_id, rank, visible, 1)
        adj2 = _adjacent_visible(tile_id, rank, visible, 2)
    else:
        terminal = 0.0
        centrality = 0.0
        adj1 = adj2 = 1.0  # honors have no neighbours to worry about

    river_copies = 0
    for p in range(4):
        if p == player_idx:
            continue
        river_copies += sum(1 for t in game.hands[p].discards if t == tile_id)

    opp_melds = [game.hands[p].num_exposed_melds
                 for p in range(4) if p != player_idx]

    return [
        honor,
        terminal,
        centrality,
        visible[tile_id] / 4.0,
        adj1,
        adj2,
        min(1.0, river_copies / 3.0),
        sig["visibility"],
        sig["discard_absence"],
        sig["opponent_threat"],
        sig["suit_safety"],
        min(1.0, game.turn / 60.0),
        min(1.0, game.tiles_remaining / 92.0),
        threat_data["max_threat"],
        max(opp_melds) / 4.0,
        sum(opp_melds) / 12.0,
    ]


def _adjacent_visible(tile_id: int, rank: int, visible: List[int],
                      distance: int) -> float:
    """Visible copies at rank ± distance, normalised to [0, 1].

    Off-the-end ranks count as fully visible (4 copies): a 1-wan has no
    0-wan for an opponent to be waiting through, which is exactly as
    safe as all four copies being dead.
    """
    total = 0
    for delta in (-distance, distance):
        r = rank + delta
        if 1 <= r <= 9:
            total += visible[tile_id + delta]
        else:
            total += 4
    return total / 8.0


# ── Outcome/value model: one vector per POST-DISCARD state ───────────
#
# These describe the 13-tile hand a discard leaves behind: how close it
# is (shanten, acceptance), the table context, and — the Phase A
# addition — what the hand is WORTH if it gets there. The tai-potential
# features encode this table's confirmed rules: ping hu 4 collapses to
# chou ping hu 1 the moment a bonus tile arrives, the concealed bonus
# dies with the first claimed meld, flush and honor-triplet tracks
# multiply value. A value model without them literally cannot tell a
# full flush from a chicken hand.

OUTCOME_FEATURES = [
    "shanten",              # shanten after the discard / 6
    "acceptance",           # tile acceptance after the discard / 30
    "turn_frac",
    "wall_frac",
    "own_melds",            # own exposed melds / 4
    "max_opp_threat",
    "opp_total_melds",
    "is_concealed",         # no claimed melds → 门清平胡 +1 still live
    "has_bonus_tiles",      # any flower/animal → ping hu already degraded
    "bonus_tai",            # tai banked in bonus tiles alone / 8
    "suit_concentration",   # (biggest suit + honors) / all tiles → flush track
    "honor_frac",           # honors / all tiles → half vs full flush
    "dragon_progress",      # dragon pair/triplet trajectory / 3
    "wind_progress",        # seat + prevailing wind triplet trajectory / 2
    "triplet_progress",     # pong-track shape (碰碰胡) / 4
    "run_progress",         # chow-track shape (平胡) / 4
    # ── Phase C additions ────────────────────────────────────────────
    # The GBM ceiling on the 16 features above was R² 0.088 — the
    # information wasn't in the vector. These add what the engine knows
    # but the model couldn't see: which tai tracks are still alive,
    # tai already banked, and the phase interactions a linear model
    # cannot form on its own.
    "pinghu_live",          # no bonus tile AND every meld a chow → 4-tai
                            # ping hu still reachable (the coach's
                            # tai_context, as a number)
    "tenpai",               # shanten == 0 — hand value jumps at ready;
                            # linear shanten/6 can't express the cliff
    "locked_tai",           # tai banked whatever the final shape:
                            # bonus tai + completed dragon/wind triplets / 8
    "flush_purity",         # biggest suit / all tiles — full-flush track,
                            # separate from suit_concentration which
                            # counts honors as flush-compatible
    "shanten_x_turn",       # far hand late in the game ≈ worthless
    "accept_x_wall",        # improving tiles only pay while draws remain
    # ── Phase C2: the law as features ────────────────────────────────
    # A 16-agent audit showed the model priced the ping hu forfeit
    # 2-5x too low and had NO axis for "a 0-tai hand cannot legally
    # win". These come from tai_track — the same engine-truth module
    # the claim gate reads — so training and decisions share one
    # arithmetic and the model can learn to price the law itself.
    "tai_ceiling",          # structural_tai_ceiling / 8 — best achievable
    "dead_hand",            # ceiling < min tai → cannot legally win as-is
    "tenpai_legal",         # at tenpai: does ANY wait complete legally?
                            # (1.0 off-tenpai — no obstruction yet)
    "wait_copies",          # at tenpai: live copies of the wait / 8;
                            # 0 off-tenpai. Exists because `acceptance`
                            # means "improving tiles" at 1+ shanten but
                            # "wait size" at tenpai — one coefficient
                            # cannot serve both, which made the model
                            # rate a wide 1-shanten hand ABOVE a ready
                            # hand (the tenpai-blindness bug).
]

# Progress of one tile type toward a scoring triplet: a pair is worth
# half a triplet in trajectory terms, a lone tile only a seed.
_TRIPLET_PROG = {0: 0.0, 1: 0.15, 2: 0.5, 3: 1.0, 4: 1.0}


# One implementation of banked-bonus arithmetic, shared with the
# decision layer (tai_track) so training and play can never disagree.
from mahjong.tai_track import bonus_tai as _bonus_tai


def _greedy_runs(counts: List[int]) -> int:
    """Complete concealed runs, extracted greedily after triplets.

    An approximation, not a decomposition — fine for a trajectory
    feature, wrong for scoring (scoring uses decompose_winning_hand).
    """
    c = counts[:]
    runs = 0
    for start in (WAN_START, TONG_START, SUO_START):
        for r in range(start, start + 7):
            while c[r] > 0 and c[r + 1] > 0 and c[r + 2] > 0:
                c[r] -= 1
                c[r + 1] -= 1
                c[r + 2] -= 1
                runs += 1
    return runs


def outcome_features(player_idx: int, game, counts_after: List[int],
                     shanten: int, acceptance: int,
                     threat_data: Dict, *,
                     exposed=None, flowers=None) -> List[float]:
    """Feature vector for the state a discard leaves behind.

    counts_after: concealed tile counts AFTER the discard (13-tile
    shape). shanten/acceptance likewise describe the post-discard hand
    — evaluate_discards hands both back per candidate.

    exposed/flowers default to the live hand's; passing them explicitly
    values a HYPOTHETICAL state — the branch evaluation behind learned
    claim decisions ("what is my hand worth if I pong this and discard
    my best tile, versus if I let it pass?").
    """
    hand = game.hands[player_idx]
    if exposed is None:
        exposed = hand.exposed
    if flowers is None:
        flowers = hand.flowers
    num_melds = len(exposed)
    opp_melds = sum(game.hands[p].num_exposed_melds
                    for p in range(4) if p != player_idx)
    seat_index = game.seat_index(player_idx)

    # Exposed meld tiles still belong to the hand for value purposes
    exposed_counts = [0] * NUM_STANDARD_UNIQUE
    exposed_pongs = exposed_chows = 0
    for kind, tiles in exposed:
        if kind == "chow":
            exposed_chows += 1
        else:
            exposed_pongs += 1  # pongs and kongs both fill the pong track
        for t in tiles:
            exposed_counts[t] += 1

    full = [min(4, counts_after[i] + exposed_counts[i])
            for i in range(NUM_STANDARD_UNIQUE)]
    total_tiles = sum(full)

    suit_totals = {Suit.WAN: 0, Suit.TONG: 0, Suit.SUO: 0}
    honors = 0
    for t in range(NUM_STANDARD_UNIQUE):
        if full[t] == 0:
            continue
        if is_honor(t):
            honors += full[t]
        else:
            suit_totals[suit_of(t)] += full[t]
    biggest_suit = max(suit_totals.values()) if total_tiles else 0

    dragon_prog = sum(_TRIPLET_PROG[full[d]]
                      for d in range(DRAGON_START, DRAGON_START + 3))
    seat_wind = WIND_START + seat_index
    wind_prog = (_TRIPLET_PROG[full[seat_wind]]
                 + _TRIPLET_PROG[full[game.prevailing_wind]])

    concealed_triplets = sum(1 for t in range(NUM_STANDARD_UNIQUE)
                             if counts_after[t] >= 3)
    triplet_prog = exposed_pongs + concealed_triplets
    run_prog = exposed_chows + _greedy_runs(counts_after)

    # Phase C features
    bonus_tai = _bonus_tai(flowers, seat_index)
    # Drift fix: a hand with all four melds claimed can never score
    # ping hu (bare-pair wait rule) — the old definition missed it.
    pinghu_live = 1.0 if (not flowers and exposed_pongs == 0
                          and num_melds < 4) else 0.0
    tenpai = 1.0 if shanten <= 0 else 0.0
    locked = bonus_tai
    locked += sum(1 for d in range(DRAGON_START, DRAGON_START + 3)
                  if full[d] >= 3)
    if full[seat_wind] >= 3:
        locked += 1
    if full[game.prevailing_wind] >= 3:
        locked += 1
    turn_frac = min(1.0, game.turn / 60.0)
    wall_frac = min(1.0, game.tiles_remaining / 92.0)

    # Legality features (engine truth via tai_track)
    from mahjong.tai_track import structural_tai_ceiling, legal_win_exists
    cfg = game.score_config
    ceiling = structural_tai_ceiling(counts_after, exposed, flowers,
                                     seat_index, game.prevailing_wind,
                                     cfg, shanten=shanten)
    dead_hand = 1.0 if (ceiling < cfg.min_tai
                        and not cfg.allow_chicken_hand) else 0.0
    if shanten == 0 and not dead_hand:
        tenpai_legal = 1.0 if legal_win_exists(
            counts_after, exposed, flowers, seat_index,
            game.prevailing_wind, cfg) else 0.0
    else:
        tenpai_legal = 0.0 if dead_hand and shanten == 0 else 1.0

    return [
        shanten / 6.0,
        min(1.0, acceptance / 30.0),
        turn_frac,
        wall_frac,
        num_melds / 4.0,
        threat_data["max_threat"],
        opp_melds / 12.0,
        1.0 if num_melds == 0 else 0.0,
        1.0 if flowers else 0.0,
        min(1.0, bonus_tai / 8.0),
        (biggest_suit + honors) / total_tiles if total_tiles else 0.0,
        honors / total_tiles if total_tiles else 0.0,
        min(1.0, dragon_prog / 3.0),
        min(1.0, wind_prog / 2.0),
        min(1.0, triplet_prog / 4.0),
        min(1.0, run_prog / 4.0),
        pinghu_live,
        tenpai,
        min(1.0, locked / 8.0),
        biggest_suit / total_tiles if total_tiles else 0.0,
        (shanten / 6.0) * turn_frac,
        min(1.0, acceptance / 30.0) * wall_frac,
        min(1.0, ceiling / 8.0),
        dead_hand,
        tenpai_legal,
        min(1.0, acceptance / 8.0) if shanten == 0 else 0.0,
    ]
