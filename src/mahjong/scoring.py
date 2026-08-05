"""
Tai (台) scoring for Singapore Mahjong.

Every winning hand is scored as a list of named tai items so the UI can
teach players WHY a hand is worth what it is. The total is capped at
ScoreConfig.tai_cap (limit hands score the cap directly), and the hand's
chip value doubles per tai: value = base_unit * 2^(tai - 1).

House rules vary between tables, so everything contentious lives in
ScoreConfig: the tai value of each rule, the minimum tai to win, the cap,
the payment scheme, and instant bonus payouts. Defaults follow common
Singapore club conventions:

  - Minimum 1 tai to win; chicken hands (0 tai) cannot win by default.
  - Self-draw (tsumo) is worth 1 tai, so any self-drawn hand is legal.
  - Shooter pays: on ron the discarder pays all three shares.
  - Animals and completed flower series pay out instantly when drawn.

Not yet implemented (special hands): thirteen orphans, heaven/earth wins.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mahjong.tiles import (
    NUM_STANDARD_UNIQUE, suit_of, rank_of, is_numbered, is_honor,
    is_wind, is_dragon, is_terminal, Suit,
    WIND_START, DRAGON_START, FLOWER_START, ANIMAL_START,
)
from mahjong.hand import Hand, decompose_winning_hand


# ── Tai table (defaults; every value is a house rule) ─────────────────

DEFAULT_TAI_VALUES: Dict[str, int] = {
    "self_draw": 1,                 # 自摸 — win by own draw
    "no_bonus_tiles": 1,            # 无花 — zero flowers and animals
    "ping_hu": 4,                   # 平胡 — all chows, non-honor pair, no bonus tiles
    "all_triplets": 2,              # 碰碰胡 — every meld a pong/kong
    "half_flush": 2,                # 混一色 — one suit plus honors
    "full_flush": 4,                # 清一色 — one suit only
    "dragon_triplet": 1,            # per dragon pong/kong
    "seat_wind_triplet": 1,         # pong/kong of own seat wind
    "prevailing_wind_triplet": 1,   # pong/kong of the round wind
    "seat_flower": 1,               # per flower matching own seat number
    "animal": 1,                    # per animal tile
    "complete_flower_series": 1,    # all four of F1-F4 or F5-F8
    "little_three_dragons": 2,      # 小三元 — bonus on top of the two dragon pongs
    "kong_draw": 1,                 # 杠上开花 — win on the kong replacement tile
    "rob_kong": 1,                  # 抢杠 — win by robbing an added kong
    "last_tile": 1,                 # 海底捞月 — win on the final live tile/discard
}

# Limit hands score ScoreConfig.tai_cap outright.
LIMIT_HANDS = {
    "big_three_dragons",    # 大三元 — three dragon pongs
    "small_four_winds",     # 小四喜 — three wind pongs + wind pair
    "great_four_winds",     # 大四喜 — four wind pongs
    "all_honors",           # 字一色 — honors only
    "all_terminals",        # 清老头 — terminal pongs + terminal pair
}

# Human-readable labels for the UI / advisor.
TAI_LABELS: Dict[str, str] = {
    "self_draw": "Self-draw (自摸)",
    "no_bonus_tiles": "No flowers or animals (无花)",
    "ping_hu": "Ping Hu — all chows, no bonus tiles (平胡)",
    "all_triplets": "All triplets (碰碰胡)",
    "half_flush": "Half flush (混一色)",
    "full_flush": "Full flush (清一色)",
    "dragon_triplet": "Dragon triplet",
    "seat_wind_triplet": "Seat wind triplet",
    "prevailing_wind_triplet": "Prevailing wind triplet",
    "seat_flower": "Matching seat flower",
    "animal": "Animal",
    "complete_flower_series": "Complete flower series (一套花)",
    "little_three_dragons": "Little three dragons (小三元)",
    "kong_draw": "Win on kong replacement (杠上开花)",
    "rob_kong": "Robbing the kong (抢杠)",
    "last_tile": "Last tile (海底捞月)",
    "big_three_dragons": "Big three dragons (大三元)",
    "small_four_winds": "Small four winds (小四喜)",
    "great_four_winds": "Great four winds (大四喜)",
    "all_honors": "All honors (字一色)",
    "all_terminals": "All terminals (清老头)",
}


@dataclass
class ScoreConfig:
    """House rules for scoring and payments."""

    min_tai: int = 1                 # minimum tai for a legal win
    tai_cap: int = 5                 # limit; payments stop doubling here
    base_unit: int = 1               # chip value of a 1-tai hand
    allow_chicken_hand: bool = False # may a 0-tai hand win?
    shooter_pays_all: bool = True    # ron: discarder pays all three shares
    instant_bonus_payouts: bool = True  # animals / flower series pay when drawn
    instant_kong_payouts: bool = True   # kongs collect chips when declared
                                        # (1 base exposed/added, 2 concealed)
    tai_values: Dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_TAI_VALUES))

    def tai_for(self, rule: str) -> int:
        if rule in LIMIT_HANDS:
            return self.tai_cap
        return self.tai_values.get(rule, 0)


@dataclass
class TaiItem:
    """One named scoring element of a winning hand."""
    rule: str
    tai: int

    @property
    def label(self) -> str:
        return TAI_LABELS.get(self.rule, self.rule)


@dataclass
class HandScore:
    """Score breakdown for a winning hand."""
    items: List[TaiItem]
    total_tai: int      # sum of items, uncapped
    tai: int            # capped at ScoreConfig.tai_cap
    value: int          # chips: base_unit * 2^(tai-1)
    is_limit: bool      # hit the cap

    def describe(self) -> str:
        if not self.items:
            return "Chicken hand (0 tai)"
        parts = [f"{item.label} +{item.tai}" for item in self.items]
        cap_note = " (limit)" if self.is_limit else ""
        return f"{', '.join(parts)} = {self.tai} tai{cap_note}"


def is_legal_win(score: HandScore, config: ScoreConfig) -> bool:
    """A hand may only win if it meets the minimum tai (house rule)."""
    if score.total_tai >= config.min_tai:
        return True
    return config.allow_chicken_hand


# ── Scoring ───────────────────────────────────────────────────────────

def score_win(hand: Hand, win_tile: int, is_tsumo: bool, seat_index: int,
              prevailing_wind: int, config: ScoreConfig, *,
              is_kong_draw: bool = False, is_rob_kong: bool = False,
              is_last_tile: bool = False) -> Optional[HandScore]:
    """Score a winning hand.

    Args:
        hand: the winner's hand; counts must INCLUDE the winning tile
        win_tile: the tile that completed the hand
        is_tsumo: True for self-draw, False for ron
        seat_index: 0-3 relative to the dealer (0 = East = dealer)
        prevailing_wind: tile id of the round wind (27-30)
        config: house rules
        is_kong_draw: won on the replacement tile after declaring a kong
        is_rob_kong: won by robbing an opponent's added kong
        is_last_tile: won on the final live tile (draw or discard)

    Returns:
        HandScore with the best decomposition's breakdown, or None if the
        tiles do not actually form a winning hand.
    """
    melds_needed = 4 - hand.num_exposed_melds
    decompositions = decompose_winning_hand(hand.counts, melds_needed)
    if not decompositions:
        return None

    exposed = [_normalize_meld(kind, tiles) for kind, tiles in hand.exposed]
    seat_wind = WIND_START + seat_index

    best: Optional[HandScore] = None
    for pair_tile, concealed_melds in decompositions:
        melds = exposed + concealed_melds
        items = _score_decomposition(
            hand, melds, pair_tile, is_tsumo, seat_wind, prevailing_wind, config)
        for flag, rule in ((is_kong_draw, "kong_draw"),
                           (is_rob_kong, "rob_kong"),
                           (is_last_tile, "last_tile")):
            tai = config.tai_for(rule)
            if flag and tai > 0:
                items.append(TaiItem(rule, tai))
        score = _finalize(items, config)
        if best is None or score.tai > best.tai or (
                score.tai == best.tai and score.total_tai > best.total_tai):
            best = score
    return best


def _normalize_meld(kind: str, tiles: List[int]) -> Tuple[str, int]:
    """Convert an exposed meld to ("chow", start) / ("pong", tile) form.

    Kongs count as pongs for hand-pattern purposes.
    """
    if kind == "chow":
        return ("chow", min(tiles))
    return ("pong", tiles[0])


def _score_decomposition(hand: Hand, melds: List[Tuple[str, int]], pair_tile: int,
                         is_tsumo: bool, seat_wind: int, prevailing_wind: int,
                         config: ScoreConfig) -> List[TaiItem]:
    items: List[TaiItem] = []

    def add(rule: str):
        tai = config.tai_for(rule)
        if tai > 0:
            items.append(TaiItem(rule, tai))

    pongs = [t for kind, t in melds if kind == "pong"]
    chows = [t for kind, t in melds if kind == "chow"]
    dragon_pongs = [t for t in pongs if is_dragon(t)]
    wind_pongs = [t for t in pongs if is_wind(t)]

    all_tiles = [pair_tile] + pongs + [c + i for c in chows for i in range(3)]
    suits_used = {suit_of(t) for t in all_tiles if is_numbered(t)}
    has_honors = any(is_honor(t) for t in all_tiles)
    has_bonus = len(hand.flowers) > 0

    # ── Limit hands first (they stand alone) ──────────────────────────
    if len(dragon_pongs) == 3:
        add("big_three_dragons")
    if len(wind_pongs) == 4:
        add("great_four_winds")
    elif len(wind_pongs) == 3 and is_wind(pair_tile):
        add("small_four_winds")
    if not suits_used and not chows:
        add("all_honors")
    if (not has_honors and not chows
            and all(is_terminal(t) for t in pongs) and is_terminal(pair_tile)):
        add("all_terminals")

    if items:  # limit hand — no need to stack the small stuff
        return items

    # ── Hand patterns ─────────────────────────────────────────────────
    if len(chows) == 4 and not is_honor(pair_tile) and not has_bonus:
        add("ping_hu")
    if len(pongs) == 4:
        add("all_triplets")
    if len(suits_used) == 1:
        add("half_flush" if has_honors else "full_flush")

    # ── Honor triplets ────────────────────────────────────────────────
    for t in dragon_pongs:
        add("dragon_triplet")
    if len(dragon_pongs) == 2 and is_dragon(pair_tile):
        add("little_three_dragons")
    for t in wind_pongs:
        if t == seat_wind:
            add("seat_wind_triplet")
        if t == prevailing_wind:
            add("prevailing_wind_triplet")

    # ── Bonus tiles ───────────────────────────────────────────────────
    seat_flowers = {FLOWER_START + (seat_wind - WIND_START),
                    FLOWER_START + 4 + (seat_wind - WIND_START)}
    for f in hand.flowers:
        if f in seat_flowers:
            add("seat_flower")
        elif f >= ANIMAL_START:
            add("animal")
    flower_set = set(hand.flowers)
    if flower_set >= set(range(FLOWER_START, FLOWER_START + 4)):
        add("complete_flower_series")
    if flower_set >= set(range(FLOWER_START + 4, FLOWER_START + 8)):
        add("complete_flower_series")
    if not has_bonus and not any(i.rule == "ping_hu" for i in items):
        add("no_bonus_tiles")

    # ── Win conditions ────────────────────────────────────────────────
    if is_tsumo:
        add("self_draw")

    return items


def _finalize(items: List[TaiItem], config: ScoreConfig) -> HandScore:
    total = sum(item.tai for item in items)
    tai = min(total, config.tai_cap)
    if tai >= 1:
        value = config.base_unit * (2 ** (tai - 1))
    else:
        value = config.base_unit  # chicken hand, if the house allows it
    return HandScore(
        items=items,
        total_tai=total,
        tai=tai,
        value=value,
        is_limit=tai >= config.tai_cap,
    )


# ── Payments ──────────────────────────────────────────────────────────

def compute_win_payments(score: HandScore, winner: int,
                         dealt_in_by: Optional[int],
                         config: ScoreConfig) -> List[int]:
    """Chip transfers for a win. Returns per-player deltas (sum = 0).

    Tsumo: every opponent pays the hand value.
    Ron: the shooter pays all three shares (default) or just their own.
    """
    payments = [0, 0, 0, 0]
    value = score.value

    if dealt_in_by is None:
        for p in range(4):
            if p != winner:
                payments[p] -= value
                payments[winner] += value
    else:
        shares = 3 if config.shooter_pays_all else 1
        payments[dealt_in_by] -= value * shares
        payments[winner] += value * shares

    return payments
