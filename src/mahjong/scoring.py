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
  - Self-draw (tsumo) adds no tai — it only widens who pays.
  - Shooter pays: on ron the discarder pays all three shares.
  - Scoring follows the standard Singapore reference rules (ping hu
    wait restrictions, 门清, 小三元 = 3, mixed terminals, and the full
    special-hand set: 十三幺, 花胡/七抢一, 四暗刻, 九连宝灯, 杠杠胡,
    十八罗汉, 天/地/人胡) with this table's house choices on top.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from mahjong.tiles import (
    NUM_STANDARD_UNIQUE, suit_of, rank_of, is_numbered, is_honor,
    is_wind, is_dragon, is_terminal, Suit,
    WIND_START, DRAGON_START, FLOWER_START, ANIMAL_START,
)
from mahjong.hand import (
    Hand, decompose_winning_hand, get_winning_tiles, is_thirteen_wonders,
)


# ── Tai table (defaults; every value is a house rule) ─────────────────

DEFAULT_TAI_VALUES: Dict[str, int] = {
    "self_draw": 0,                 # 自摸 adds NO tai at this table — it only
                                    # changes who pays (all three). Set to 1
                                    # to restore tsumo-as-tai.
    "ping_hu": 4,                   # 平胡 — all chows; see wait restrictions
    "chou_ping_hu": 1,              # 臭平胡 — ping hu shape, but holding flowers/animals
    "fully_concealed": 1,           # 门清 — no claimed melds AND self-drawn win
    "all_triplets": 2,              # 碰碰胡 — every meld a pong/kong
    "half_flush": 2,                # 混一色 — one suit plus honors
    "full_flush": 4,                # 清一色 — one suit only
    "mixed_terminals": 2,           # 混老头 — triplets, all terminals+honors
                                    # (stacks with all_triplets → 4 total)
    "dragon_triplet": 1,            # per dragon pong/kong
    "seat_wind_triplet": 1,         # pong/kong of own seat wind
    "prevailing_wind_triplet": 1,   # pong/kong of the round wind
    "seat_flower": 1,               # per flower matching own seat number
    "animal": 1,                    # per animal tile
    "complete_animals": 1,          # all four animals: +1 on top of the four
    "complete_flower_series": 1,    # all four of F1-F4 or F5-F8
    "little_three_dragons": 1,      # 小三元 — bonus for the dragon pair, on
                                    # top of the two dragon pongs (3 total)
    "small_four_winds": 2,          # 小四喜 — three wind pongs + wind pair;
                                    # NOT a limit hand (wind tai stack on top)
    "kong_draw": 1,                 # 杠上开花 — win on the kong replacement tile
    "flower_draw": 1,               # 花上 — win on a flower replacement tile
    "rob_kong": 1,                  # 抢杠 — win by robbing an added kong
    "last_tile": 1,                 # 海底捞月 — win on the final live tile/discard
                                    # (not via a replacement draw)
}

# Limit hands score ScoreConfig.tai_cap outright.
LIMIT_HANDS = {
    "big_three_dragons",    # 大三元 — three dragon pongs
    "great_four_winds",     # 大四喜 — four wind pongs
    "all_honors",           # 字一色 — honors only
    "all_terminals",        # 清老头 — terminal pongs + terminal pair
    "hidden_treasure",      # 四暗刻 — four concealed triplets, self-drawn
    "nine_gates",           # 九连宝灯 — 1112345678999 + any, concealed
    "thirteen_wonders",     # 十三幺 — one of each orphan + pair (pays double)
    "kong_on_kong",         # 杠杠胡 — win on 2nd consecutive kong replacement
    "eight_flowers",        # 花胡 — all 8 flowers (instant win)
    "eighteen_arhats",      # 十八罗汉 — four kongs
    "heavenly_hand",        # 天胡 — dealer wins on the deal
    "earthly_hand",         # 地胡 — non-dealer: dealer's 1st discard / 1st draw
    "humanly_hand",         # 人胡 — win by discard before your first draw
}

# Human-readable labels for the UI / advisor.
TAI_LABELS: Dict[str, str] = {
    "self_draw": "Self-draw (自摸)",
    "ping_hu": "Ping Hu — all chows, no bonus tiles (平胡)",
    "chou_ping_hu": "Chou ping hu — all chows with bonus tiles (臭平胡)",
    "fully_concealed": "Fully concealed, self-drawn (门清)",
    "all_triplets": "All triplets (碰碰胡)",
    "half_flush": "Half flush (混一色)",
    "full_flush": "Full flush (清一色)",
    "dragon_triplet": "Dragon triplet",
    "seat_wind_triplet": "Seat wind triplet",
    "prevailing_wind_triplet": "Prevailing wind triplet",
    "seat_flower": "Matching seat flower",
    "animal": "Animal",
    "complete_animals": "All four animals",
    "complete_flower_series": "Complete flower series (一套花)",
    "little_three_dragons": "Little three dragons (小三元)",
    "mixed_terminals": "Mixed terminals (混老头)",
    "kong_draw": "Win on kong replacement (杠上开花)",
    "flower_draw": "Win on flower replacement (花上)",
    "rob_kong": "Robbing the kong (抢杠)",
    "last_tile": "Last tile (海底捞月)",
    "big_three_dragons": "Big three dragons (大三元)",
    "hidden_treasure": "Hidden treasure (四暗刻)",
    "nine_gates": "Nine gates (九连宝灯)",
    "thirteen_wonders": "Thirteen wonders (十三幺)",
    "kong_on_kong": "Kong on kong (杠杠胡)",
    "eight_flowers": "Eight flowers (花胡)",
    "eighteen_arhats": "Eighteen arhats (十八罗汉)",
    "heavenly_hand": "Heavenly hand (天胡)",
    "earthly_hand": "Earthly hand (地胡)",
    "humanly_hand": "Humanly hand (人胡)",
    "small_four_winds": "Small four winds (小四喜)",
    "great_four_winds": "Great four winds (大四喜)",
    "all_honors": "All honors (字一色)",
    "all_terminals": "All terminals (清老头)",
}


@dataclass
class ScoreConfig:
    """House rules for scoring and payments."""

    min_tai: int = 1                 # minimum tai for a legal win
    tai_cap: int = 6                 # limit; payments stop doubling here
    base_unit: int = 1               # chip value of a 1-tai hand
    allow_chicken_hand: bool = False # may a 0-tai hand win?
    shooter_pays_all: bool = True    # ron: discarder pays all three shares
    # Tai-only accounting (house decision): instant chip payouts for
    # bonus tiles and kongs are OFF by default. The machinery stays —
    # flip these on when the table starts counting money again.
    instant_bonus_payouts: bool = False  # animals / flower series pay when drawn
    instant_kong_payouts: bool = False   # kongs collect chips when declared
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
    payout_multiplier: int = 1  # 十三幺: everyone pays double

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
              is_last_tile: bool = False, is_flower_draw: bool = False,
              is_kong_on_kong: bool = False, first_turn_flag: str = None
              ) -> Optional[HandScore]:
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
        is_flower_draw: won on the replacement tile after a bonus tile
        is_kong_on_kong: won on the SECOND consecutive kong replacement
        first_turn_flag: "heavenly" | "earthly" | "humanly" | None

    Returns:
        HandScore with the best decomposition's breakdown, or None if the
        tiles do not actually form a winning hand.
    """
    seat_wind = WIND_START + seat_index

    # 海底捞月 does not apply when the last tile arrived via a
    # replacement draw (PDF: 补花/补杠 exception).
    if is_kong_draw or is_flower_draw:
        is_last_tile = False

    # ── Thirteen Wonders: outside the 4-melds-plus-pair shape ────────
    if is_thirteen_wonders(hand.counts, hand.num_exposed_melds):
        items = [TaiItem("thirteen_wonders", config.tai_for("thirteen_wonders"))]
        score = _finalize(items, config)
        score.payout_multiplier = 2  # everyone pays double, tsumo or not
        return score

    melds_needed = 4 - hand.num_exposed_melds
    decompositions = decompose_winning_hand(hand.counts, melds_needed)
    if not decompositions:
        return None

    exposed = [_normalize_meld(kind, tiles) for kind, tiles in hand.exposed]
    claimed_melds = sum(1 for kind, _ in hand.exposed
                        if kind != "concealed_kong")
    kong_count = sum(1 for kind, _ in hand.exposed
                     if kind in ("kong", "concealed_kong"))

    # Wait analysis for the ping hu restriction: how many distinct
    # tiles completed the 13-tile hand this win came from?
    hand.counts[win_tile] -= 1
    waits = get_winning_tiles(hand.counts, hand.num_exposed_melds)
    hand.counts[win_tile] += 1
    single_wait = len(waits) <= 1

    best: Optional[HandScore] = None
    for pair_tile, concealed_melds in decompositions:
        melds = exposed + concealed_melds
        items = _score_decomposition(
            hand, melds, pair_tile, is_tsumo, seat_wind, prevailing_wind,
            config, claimed_melds=claimed_melds, single_wait=single_wait,
            kong_count=kong_count)
        for flag, rule in ((is_kong_draw and not is_kong_on_kong, "kong_draw"),
                           (is_flower_draw, "flower_draw"),
                           (is_rob_kong, "rob_kong"),
                           (is_last_tile, "last_tile")):
            tai = config.tai_for(rule)
            if flag and tai > 0:
                items.append(TaiItem(rule, tai))
        if is_kong_on_kong:
            items.append(TaiItem("kong_on_kong", config.tai_for("kong_on_kong")))
        if first_turn_flag in ("heavenly", "earthly", "humanly"):
            rule = f"{first_turn_flag}_hand"
            items.append(TaiItem(rule, config.tai_for(rule)))
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
                         config: ScoreConfig, *, claimed_melds: int = 0,
                         single_wait: bool = False,
                         kong_count: int = 0) -> List[TaiItem]:
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
    concealed = claimed_melds == 0  # concealed kongs don't break 门清

    # ── Limit hands first (they stand alone) ──────────────────────────
    if len(dragon_pongs) == 3:
        add("big_three_dragons")
    if len(wind_pongs) == 4:
        add("great_four_winds")
    if not suits_used and not chows:
        add("all_honors")
    if (not has_honors and not chows
            and all(is_terminal(t) for t in pongs) and is_terminal(pair_tile)):
        add("all_terminals")
    if len(pongs) == 4 and concealed and is_tsumo:
        add("hidden_treasure")  # 四暗刻 — all triplets drawn by own hand
    if concealed and _is_nine_gates(hand, suits_used, has_honors):
        add("nine_gates")
    if kong_count == 4:
        add("eighteen_arhats")

    if items:  # limit hand — no need to stack the small stuff
        return items

    # ── Hand patterns ─────────────────────────────────────────────────
    # Ping hu family. The clean version — all chows, no bonus tiles —
    # scores full value; holding any flower/animal degrades it to chou
    # ping hu (臭平胡). PDF wait restrictions (both variants): winning
    # by DISCARD requires a wait of 2+ distinct tiles — a middle, edge,
    # or pair wait only earns ping hu on self-draw. With all four chows
    # claimed, the bare pair wait never counts as ping hu at all.
    # Pair rule: any suited tile, or a wind that is neither the seat
    # nor the prevailing wind. Dragons never.
    pair_ok = (not is_honor(pair_tile)
               or (is_wind(pair_tile)
                   and pair_tile not in (seat_wind, prevailing_wind)))
    wait_ok = ((not single_wait and hand.num_exposed_melds < 4)
               or (single_wait and is_tsumo and hand.num_exposed_melds < 4))
    if len(chows) == 4 and pair_ok and wait_ok:
        add("chou_ping_hu" if has_bonus else "ping_hu")
    if len(pongs) == 4:
        add("all_triplets")
        if (all(is_terminal(t) or is_honor(t) for t in pongs + [pair_tile])
                and has_honors
                and any(is_terminal(t) for t in pongs + [pair_tile])):
            add("mixed_terminals")  # 混老头 — stacks to 4 with 碰碰胡
    if len(suits_used) == 1:
        add("half_flush" if has_honors else "full_flush")
    if len(wind_pongs) == 3 and is_wind(pair_tile):
        add("small_four_winds")  # 小四喜 — wind triplet tai stack on top

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
    if flower_set >= set(range(ANIMAL_START, ANIMAL_START + 4)):
        add("complete_animals")  # all four: +1 on top of the animal tai
    if flower_set >= set(range(FLOWER_START, FLOWER_START + 4)):
        add("complete_flower_series")
    if flower_set >= set(range(FLOWER_START + 4, FLOWER_START + 8)):
        add("complete_flower_series")

    # ── Win conditions ────────────────────────────────────────────────
    if concealed and is_tsumo:
        add("fully_concealed")  # 门清 — no claimed melds, self-drawn win
    if is_tsumo:
        add("self_draw")

    return items


def _is_nine_gates(hand: Hand, suits_used, has_honors) -> bool:
    """九连宝灯: concealed 1112345678999 of one suit + any same-suit tile."""
    if has_honors or len(suits_used) != 1 or hand.num_exposed_melds != 0:
        return False
    suit = next(iter(suits_used))
    start = {Suit.WAN: 0, Suit.TONG: 9, Suit.SUO: 18}[suit]
    base = [0] * NUM_STANDARD_UNIQUE
    for r in range(9):
        base[start + r] = 3 if r in (0, 8) else 1
    extras = 0
    for t in range(NUM_STANDARD_UNIQUE):
        d = hand.counts[t] - base[t]
        if d < 0:
            return False
        extras += d
    return extras == 1


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
    value = score.value * score.payout_multiplier

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
