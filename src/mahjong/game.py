"""
Game engine for Singapore Mahjong.

Manages the full game loop:
  1. Build wall (148 tiles), deal 13 tiles to each player
  2. Replace any bonus tiles drawn during deal
  3. Turn loop:
     a. Active player draws from wall
     b. Replace any bonus tiles drawn
     c. Check for win (tsumo / self-draw)
     d. Active player discards a tile
     e. Other players may claim the discard (win > pong/kong > chow)
     f. If claimed, claimer forms meld and discards; otherwise next player
  4. Game ends on: win, wall exhaustion (draw), or no more replacement tiles

GameState plays a single hand with fixed dealer and prevailing wind;
session.py chains hands into full multi-round sessions with dealer
rotation and running scores.

Current simplifications:
  - Kong replacement tiles draw from the front of the wall (same as
    flower replacements) rather than a separate back-of-wall dead wall
"""

import random
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from mahjong.tiles import (
    create_wall, is_bonus, tile_short, hand_to_str,
    NUM_STANDARD_UNIQUE, NUM_TOTAL_TILES,
    is_numbered, suit_of, rank_of, possible_chow_partners,
    WIND_START, FLOWER_START, ANIMAL_START
)
from mahjong.hand import Hand, calculate_shanten, is_winning_hand
from mahjong.scoring import (
    ScoreConfig, HandScore, score_win, is_legal_win, compute_win_payments
)

# Singapore Mahjong: game ends in a draw when 15 tiles remain in the wall.
# These 15 tiles are "dead" and cannot be drawn.
DEAD_WALL_SIZE = 15


# ── Decision requests (the engine's yield points) ─────────────────────
#
# The core game loop is a generator: wherever a player must decide
# something, it yields one of these requests and receives the answer
# back via gen.send(). GameState.play() answers every request from the
# seat's agent; InteractiveGame (interactive.py) pauses on human seats.

@dataclass
class DecisionRequest:
    """Base class for decisions the engine needs a player to make."""
    player: int


@dataclass
class DiscardRequest(DecisionRequest):
    """Choose a tile from hand to discard. Answer: tile_id.

    drawn is the tile just drawn (None when discarding after a claim).
    """
    drawn: Optional[int] = None


@dataclass
class ClaimRequest(DecisionRequest):
    """Claim the discarded tile as a pong or kong? Answer: bool."""
    tile_id: int = -1
    claim_type: str = "pong"


@dataclass
class ChowRequest(DecisionRequest):
    """Pick a chow to form with the discarded tile.

    Answer: one of options as (partner1, partner2), or None to pass.
    """
    tile_id: int = -1
    options: Optional[List[Tuple[int, int]]] = None


@dataclass
class KongRequest(DecisionRequest):
    """Pick a kong to declare on your own turn.

    Answer: one of options as (kind, tile_id), or None to pass.
    """
    options: Optional[List[Tuple[str, int]]] = None


# ── Agent interface ───────────────────────────────────────────────────

class BaseAgent:
    """Base class for all agents. Subclass and override choose_discard."""

    def __init__(self, name: str = "Agent"):
        self.name = name

    def choose_discard(self, player_idx: int, game_state: "GameState") -> int:
        """Given the game state, return the tile_id to discard.

        Args:
            player_idx: which player this agent controls (0-3)
            game_state: current game state (agent should only use
                       visible information for fair play)

        Returns:
            tile_id to discard from the player's hand
        """
        raise NotImplementedError

    def should_claim(self, player_idx: int, tile_id: int,
                     claim_type: str, game_state: "GameState") -> bool:
        """Decide whether to claim a discarded tile for a pong or kong.

        Args:
            player_idx: which player is deciding
            tile_id: the discarded tile available to claim
            claim_type: "pong" or "kong" (kong = 3 copies held; the
                        claimer draws a replacement tile after melding)
            game_state: current game state

        Returns:
            True to claim, False to pass
        """
        return False  # default: never claim

    def choose_kong(self, player_idx: int,
                    options: List[Tuple[str, int]],
                    game_state: "GameState") -> Optional[Tuple[str, int]]:
        """Pick a kong to declare on your own turn, or None to pass.

        Args:
            player_idx: which player is deciding
            options: ("concealed", tile) / ("added", tile) declarations
                     currently available (see Hand.kong_options)
            game_state: current game state

        Returns:
            One of the options, or None to pass. Declaring draws a
            replacement tile; an added kong may be robbed by opponents.
        """
        return None  # default: never declare

    def choose_chow(self, player_idx: int, tile_id: int,
                    options: List[Tuple[int, int]],
                    game_state: "GameState") -> Optional[Tuple[int, int]]:
        """Pick which chow to form with a discarded tile, or None to pass.

        Args:
            player_idx: which player is deciding
            tile_id: the discarded tile available to claim
            options: valid (partner1, partner2) pairs from this player's
                     concealed hand that complete a run with tile_id
            game_state: current game state

        Returns:
            One of the pairs from options, or None to pass.
        """
        return None  # default: never claim


# ── Game result ───────────────────────────────────────────────────────

@dataclass
class GameResult:
    """Outcome of a single game."""
    winner: Optional[int]           # player index (0-3) or None for draw
    win_type: Optional[str]         # "tsumo" / "ron" / None
    turns: int                      # total turns played
    final_shanten: List[int]        # each player's final shanten
    deal_ins: List[int]             # count of deal-ins per player (discarding a winning tile)
    flowers_collected: List[int]    # bonus tiles per player
    tiles_remaining: int            # tiles left in wall
    dealt_in_by: Optional[int] = None  # who discarded the ron-winning tile
    winning_score: Optional[HandScore] = None  # tai breakdown of the winning hand
    payments: Optional[List[int]] = None  # chip deltas per player (win + instant bonuses)
    win_tile: Optional[int] = None  # the tile that completed the winning hand


# ── Game state ────────────────────────────────────────────────────────

class GameState:
    """Full state of a Singapore Mahjong game.

    Visible information (available to all agents):
        - own hand (tiles + counts)
        - all players' exposed melds
        - all players' discards
        - all players' flower counts
        - number of tiles remaining in wall
        - current turn number
        - current active player

    Hidden information:
        - other players' concealed tiles
        - wall contents and order
    """

    def __init__(self, agents: List[BaseAgent], seed: Optional[int] = None,
                 dealer: int = 0, prevailing_wind: int = WIND_START,
                 score_config: Optional[ScoreConfig] = None):
        if len(agents) != 4:
            raise ValueError("Need exactly 4 agents")

        self.agents = agents
        self.rng = random.Random(seed)

        # Scoring context (fixed per hand until dealer rotation arrives)
        self.dealer = dealer
        self.prevailing_wind = prevailing_wind
        self.score_config = score_config or ScoreConfig()

        # Game state
        self.hands: List[Hand] = [Hand() for _ in range(4)]
        self.wall: List[int] = []
        self.wall_idx: int = 0
        self.turn: int = 0
        self.active_player: int = 0
        self.game_over: bool = False
        self.result: Optional[GameResult] = None

        # Chip ledger: instant bonus payouts accumulate here during play;
        # win payments are added when the game ends.
        self.payments: List[int] = [0, 0, 0, 0]

        # Deal-in tracking: deal_ins[p] counts how many times player p
        # discarded a tile that an opponent could legally have won on
        # (meets the minimum tai requirement).
        self.deal_ins: List[int] = [0, 0, 0, 0]

    def seat_index(self, player_idx: int) -> int:
        """Seat relative to the dealer: 0 = East (dealer), 1 = South, ..."""
        return (player_idx - self.dealer) % 4

    # ── Wall and drawing ──────────────────────────────────────────────

    @property
    def tiles_remaining(self) -> int:
        return len(self.wall) - self.wall_idx

    def _draw_tile(self) -> Optional[int]:
        """Draw the next tile from the wall. Returns None if only dead wall remains."""
        if self.tiles_remaining <= DEAD_WALL_SIZE:
            return None
        tile = self.wall[self.wall_idx]
        self.wall_idx += 1
        return tile

    def _deal_tile_to(self, player_idx: int) -> Optional[int]:
        """Draw a tile and add it to a player's hand.

        Handles bonus tiles: if a flower/animal is drawn, it's set aside
        and a replacement is drawn. Repeats until a standard tile is drawn
        or the wall is exhausted.

        Returns the final standard tile added, or None if wall exhausted.
        """
        while True:
            tile = self._draw_tile()
            if tile is None:
                return None

            is_flower = self.hands[player_idx].add_tile(tile)
            if is_flower:
                # Bonus tile — set aside, pay instant bonus, draw replacement
                self._apply_instant_bonus(player_idx, tile)
                continue
            else:
                return tile

    def _apply_instant_bonus(self, player_idx: int, tile: int):
        """Instant payouts for bonus tiles (house rule, on by default).

        Each animal, and each completed flower series (F1-F4 or F5-F8),
        collects one base unit from every other player immediately.
        """
        cfg = self.score_config
        if not cfg.instant_bonus_payouts:
            return

        amount = 0
        if tile >= ANIMAL_START:
            amount += cfg.base_unit

        flowers = set(self.hands[player_idx].flowers)
        for series_start in (FLOWER_START, FLOWER_START + 4):
            series = set(range(series_start, series_start + 4))
            if tile in series and series <= flowers:
                amount += cfg.base_unit

        if amount:
            for q in range(4):
                if q != player_idx:
                    self.payments[q] -= amount
                    self.payments[player_idx] += amount

    # ── Setup ─────────────────────────────────────────────────────────

    def setup(self):
        """Build wall, deal 13 tiles to each player, handle bonus replacements."""
        self.wall = create_wall()
        self.rng.shuffle(self.wall)

        for _ in range(13):
            for p in range(4):
                self._deal_tile_to(p) 

    # ── Visible information helpers (for agents) ──────────────────────

    def get_visible_counts(self, player_idx: int) -> List[int]:
        """Get counts of all tiles visible to a specific player.

        Includes: own hand + all discards + all exposed melds.
        Used by agents to estimate what tiles remain in the wall.
        """
        counts = [0] * NUM_STANDARD_UNIQUE

        # Own concealed hand
        for i in range(NUM_STANDARD_UNIQUE):
            counts[i] += self.hands[player_idx].counts[i]

        # All players' discards
        for p in range(4):
            for tile in self.hands[p].discards:
                if not is_bonus(tile):
                    counts[tile] += 1

        # ALL exposed melds — including the player's own, since claimed
        # tiles were removed from hand.counts when the meld was formed.
        # Concealed kongs are treated as visible (their tiles are dead).
        for p in range(4):
            for meld_type, meld_tiles in self.hands[p].exposed:
                for tile in meld_tiles:
                    if not is_bonus(tile):
                        counts[tile] += 1

        return counts

    def get_all_discards(self) -> List[List[int]]:
        """Get discard piles for all 4 players."""
        return [h.discards[:] for h in self.hands]

    # ── Turn execution ────────────────────────────────────────────────

    def _check_pong_or_kong(self, discarder: int,
                            tile_id: int) -> Optional[Tuple[int, str]]:
        """Check if any opponent claims the discard as a kong or pong.

        A player holding 3 copies is offered the kong first, then the
        pong. Only one player can ever hold enough copies, so there is
        no claim conflict. Priority: closest in turn order.

        Generator: yields ClaimRequests, returns (claimer, "kong"|"pong")
        or None.
        """
        for offset in range(1, 4):
            candidate = (discarder + offset) % 4
            hand = self.hands[candidate]
            if hand.counts[tile_id] >= 3:
                if (yield ClaimRequest(candidate, tile_id, "kong")):
                    return (candidate, "kong")
            if hand.counts[tile_id] >= 2:
                if (yield ClaimRequest(candidate, tile_id, "pong")):
                    return (candidate, "pong")
        return None

    def _check_chow(self, discarder: int, tile_id: int) -> Optional[Tuple[int, List[int]]]:
        """Check if the next player wants to chow the discarded tile.

        Only the player immediately after the discarder can chow.
        Only numbered tiles can form chows. The player chooses which
        combination to form.

        Generator: yields a ChowRequest, returns
        (claimer, [partner1, partner2]) or None.
        """
        if not is_numbered(tile_id):
            return None

        next_player = (discarder + 1) % 4
        hand = self.hands[next_player]

        options = [
            (p1, p2) for p1, p2 in possible_chow_partners(tile_id)
            if hand.counts[p1] >= 1 and hand.counts[p2] >= 1
        ]
        if not options:
            return None

        choice = yield ChowRequest(next_player, tile_id, options)
        if choice is None:
            return None

        choice = tuple(choice)
        if choice not in options:
            raise ValueError(
                f"Agent {self.agents[next_player].name} (P{next_player}) chose "
                f"invalid chow {choice} for {tile_short(tile_id)}; options: {options}"
            )
        return (next_player, list(choice))

    def _execute_claim(self, claimer: int, tile_id: int,
                       claim_type: str, partners: Optional[List[int]] = None):
        """Execute a pong or chow claim.

        The claimer takes the discarded tile, forms the meld,
        then must discard a tile.
        """
        hand = self.hands[claimer]

        if claim_type == "pong":
            # Remove 2 copies from hand, form meld with discarded tile
            hand.counts[tile_id] -= 2
            hand.tiles.remove(tile_id)
            hand.tiles.remove(tile_id)
            hand.add_exposed_meld("pong", [tile_id, tile_id, tile_id])

        elif claim_type == "kong":
            # Remove 3 copies, form the kong, draw a replacement tile.
            # The replacement may win the hand (kong draw).
            hand.declare_kong("exposed", tile_id)
            self._apply_kong_payout(claimer, "exposed")
            replacement = self._deal_tile_to(claimer)
            if replacement is None:
                self._end_game(winner=None, win_type=None)
                return None
            if self._try_tsumo(claimer, replacement, is_kong_draw=True):
                return None

        elif claim_type == "chow" and partners:
            # Remove partner tiles from hand, form meld
            for p_tile in partners:
                hand.counts[p_tile] -= 1
                hand.tiles.remove(p_tile)
            meld_tiles = sorted([tile_id] + partners)
            hand.add_exposed_meld("chow", meld_tiles)

        # After a claim the claimer effectively holds one tile too many
        # (2 concealed tiles left for a 3-tile meld) and must discard.
        discard_tile = yield DiscardRequest(claimer)

        if discard_tile not in hand.tiles:
            raise ValueError(
                f"Agent {self.agents[claimer].name} (P{claimer}) tried to discard "
                f"{tile_short(discard_tile)} after claim, not in hand: "
                f"{hand_to_str(hand.tiles, short=True)}"
            )

        hand.discard(discard_tile)
        return discard_tile

    def _try_tsumo(self, p: int, win_tile: int, is_kong_draw: bool) -> bool:
        """End the game if player p's hand is a legal self-drawn win."""
        hand = self.hands[p]
        if not is_winning_hand(hand.counts, hand.num_exposed_melds):
            return False
        score = score_win(hand, win_tile, True, self.seat_index(p),
                          self.prevailing_wind, self.score_config,
                          is_kong_draw=is_kong_draw,
                          is_last_tile=self.tiles_remaining <= DEAD_WALL_SIZE)
        if score is not None and is_legal_win(score, self.score_config):
            self._end_game(winner=p, win_type="tsumo", score=score,
                           win_tile=win_tile)
            return True
        return False  # complete but below minimum tai — play on

    def _apply_kong_payout(self, player_idx: int, kind: str):
        """Instant chips for declaring a kong (house rule, on by default):
        1 base unit from each player, 2 for a concealed kong."""
        cfg = self.score_config
        if not cfg.instant_kong_payouts:
            return
        amount = cfg.base_unit * (2 if kind == "concealed" else 1)
        for q in range(4):
            if q != player_idx:
                self.payments[q] -= amount
                self.payments[player_idx] += amount

    def _kong_phase(self, p: int) -> bool:
        """Let the active player declare kongs after drawing.

        Each kong grants a replacement draw which may win the hand
        (kong draw); an added kong may first be robbed by an opponent.
        Returns True if the game continues, False if it ended.
        """
        hand = self.hands[p]
        while True:
            options = hand.kong_options()
            if not options:
                return True
            choice = yield KongRequest(p, options)
            if choice is None:
                return True
            choice = tuple(choice)
            if choice not in options:
                raise ValueError(
                    f"Agent {self.agents[p].name} (P{p}) chose invalid kong "
                    f"{choice}; options: {options}")
            kind, tile_id = choice

            if kind == "added":
                rob = self._check_ron(p, tile_id, is_rob_kong=True)
                if rob is not None:
                    winner, score = rob
                    hand.remove_tile(tile_id)
                    self.hands[winner].add_tile(tile_id)
                    self._end_game(winner=winner, win_type="ron",
                                   dealt_in_by=p, score=score,
                                   win_tile=tile_id)
                    return False

            hand.declare_kong(kind, tile_id)
            self._apply_kong_payout(p, kind)

            replacement = self._deal_tile_to(p)
            if replacement is None:
                self._end_game(winner=None, win_type=None)
                return False
            if self._try_tsumo(p, replacement, is_kong_draw=True):
                return False

    def _execute_turn(self) -> bool:
        """Execute one turn. Returns True if game continues, False if over."""
        p = self.active_player
        hand = self.hands[p]

        # 1. Draw
        drawn = self._deal_tile_to(p)
        if drawn is None:
            self._end_game(winner=None, win_type=None)
            return False

        # 2. Check tsumo (self-draw win) — must meet the minimum tai
        if self._try_tsumo(p, drawn, is_kong_draw=False):
            return False

        # 3. Kong declarations (concealed or added), each with a
        #    replacement draw and possible kong-draw win or robbed kong
        if not (yield from self._kong_phase(p)):
            return False

        # 4. Player chooses discard
        discard_tile = yield DiscardRequest(p, drawn=drawn)

        # Validate
        if discard_tile not in hand.tiles:
            raise ValueError(
                f"Agent {self.agents[p].name} (P{p}) tried to discard "
                f"{tile_short(discard_tile)} not in hand: "
                f"{hand_to_str(hand.tiles, short=True)}"
            )

        # 4. Discard
        hand.discard(discard_tile)

        # 5. Resolve claims on the discard (ron > kong/pong > chow,
        #    repeating for each claimer's follow-up discard)
        return (yield from self._resolve_discard(p, discard_tile))

    def _check_ron(self, discarder: int, tile_id: int, *,
                   is_rob_kong: bool = False,
                   is_last_tile: bool = False) -> Optional[Tuple[int, HandScore]]:
        """Check whether any opponent legally wins on the tile.

        Used for discards (ron) and for robbing an added kong. A hand
        must both complete AND meet the minimum tai to win. Every
        opponent with a legal win counts as a deal-in for the discarder
        (or kong declarer); the winner is the closest in turn order.
        """
        winner = None
        winning_score = None
        for offset in range(1, 4):
            opponent = (discarder + offset) % 4
            opp_hand = self.hands[opponent]
            opp_hand.counts[tile_id] += 1
            if is_winning_hand(opp_hand.counts, opp_hand.num_exposed_melds):
                score = score_win(opp_hand, tile_id, False,
                                  self.seat_index(opponent),
                                  self.prevailing_wind, self.score_config,
                                  is_rob_kong=is_rob_kong,
                                  is_last_tile=is_last_tile)
                if score is not None and is_legal_win(score, self.score_config):
                    if winner is None:
                        winner = opponent
                        winning_score = score
                    self.deal_ins[discarder] += 1
            opp_hand.counts[tile_id] -= 1
        if winner is None:
            return None
        return (winner, winning_score)

    def _resolve_discard(self, discarder: int, tile_id: int) -> bool:
        """Resolve claims on a discard with priority ron > kong/pong > chow.

        When a claim is made, the claimer discards and that new discard
        opens a fresh claim window — resolved by looping. A kong claim
        may end the game inside _execute_claim (kong-draw win or wall
        exhausted). Returns True if the game continues, False if over.
        """
        while True:
            is_last = self.tiles_remaining <= DEAD_WALL_SIZE
            ron_result = self._check_ron(discarder, tile_id, is_last_tile=is_last)
            if ron_result is not None:
                ron_winner, score = ron_result
                self.hands[ron_winner].add_tile(tile_id)
                self._end_game(winner=ron_winner, win_type="ron",
                               dealt_in_by=discarder, score=score,
                               win_tile=tile_id)
                return False

            claim = yield from self._check_pong_or_kong(discarder, tile_id)
            if claim is not None:
                claimer, claim_type = claim
                # Remove the tile from the discard pile (it was just added)
                self.hands[discarder].discards.pop()
                tile_id = yield from self._execute_claim(claimer, tile_id, claim_type)
                if tile_id is None:
                    return False  # game ended inside the claim
                discarder = claimer
                continue

            chow_result = yield from self._check_chow(discarder, tile_id)
            if chow_result is not None:
                chow_claimer, partners = chow_result
                self.hands[discarder].discards.pop()
                tile_id = yield from self._execute_claim(chow_claimer, tile_id, "chow", partners)
                discarder = chow_claimer
                continue

            # No claims — turn passes to the player after the last discarder
            self.active_player = (discarder + 1) % 4
            self.turn += 1
            return True

    def _end_game(self, winner: Optional[int], win_type: Optional[str],
                  dealt_in_by: Optional[int] = None,
                  score: Optional[HandScore] = None,
                  win_tile: Optional[int] = None):
        """Record the game result, including scoring and chip payments."""
        self.game_over = True

        payments = self.payments[:]
        if winner is not None and score is not None:
            win_pay = compute_win_payments(score, winner, dealt_in_by,
                                           self.score_config)
            payments = [payments[i] + win_pay[i] for i in range(4)]

        self.result = GameResult(
            winner=winner,
            win_type=win_type,
            turns=self.turn,
            final_shanten=[
                calculate_shanten(h.counts, h.num_exposed_melds)
                for h in self.hands
            ],
            deal_ins=self.deal_ins[:],
            flowers_collected=[len(h.flowers) for h in self.hands],
            tiles_remaining=self.tiles_remaining,
            dealt_in_by=dealt_in_by,
            winning_score=score,
            payments=payments,
            win_tile=win_tile,
        )

    # ── Main game loop ────────────────────────────────────────────────

    def step_game(self, max_turns: int = 200):
        """Master generator: plays the whole game, yielding DecisionRequests.

        Send each answer back with gen.send(); StopIteration.value is the
        GameResult. Builds the wall via setup() if not already done.
        """
        if not self.wall:
            self.setup()
        while not self.game_over and self.turn < max_turns:
            keep_going = yield from self._execute_turn()
            if not keep_going:
                break
        if self.result is None:
            self._end_game(winner=None, win_type=None)
        return self.result

    def dispatch_to_agent(self, request: DecisionRequest):
        """Answer a decision request using the seat's agent."""
        agent = self.agents[request.player]
        if isinstance(request, DiscardRequest):
            return agent.choose_discard(request.player, self)
        if isinstance(request, ClaimRequest):
            return agent.should_claim(request.player, request.tile_id,
                                      request.claim_type, self)
        if isinstance(request, ChowRequest):
            return agent.choose_chow(request.player, request.tile_id,
                                     request.options, self)
        if isinstance(request, KongRequest):
            return agent.choose_kong(request.player, request.options, self)
        raise TypeError(f"Unknown decision request: {request!r}")

    def play(self, max_turns: int = 200, verbose: bool = False) -> GameResult:
        """Play a complete game with every seat driven by its agent.

        Args:
            max_turns: safety limit to prevent infinite loops
            verbose: print game progress
        """
        self.setup()

        if verbose:
            print(f"Game started. Wall: {self.tiles_remaining} tiles.")
            for p in range(4):
                h = self.hands[p]
                print(f"  P{p} ({self.agents[p].name}): "
                      f"{hand_to_str(h.tiles, short=True)}"
                      f" | flowers: {len(h.flowers)}")

        run_decision(self, self.step_game(max_turns))

        if verbose:
            r = self.result
            if r.winner is not None:
                win_msg = (f"\nResult: P{r.winner} ({self.agents[r.winner].name}) "
                          f"wins by {r.win_type} on turn {r.turns}!")
                if r.win_type == "ron" and r.dealt_in_by is not None:
                    win_msg += (f" (P{r.dealt_in_by} "
                               f"({self.agents[r.dealt_in_by].name}) dealt in)")
                print(win_msg)
                if r.winning_score is not None:
                    print(f"Score: {r.winning_score.describe()} "
                          f"→ {r.winning_score.value} chips")
            else:
                print(f"\nResult: Draw after {r.turns} turns. "
                      f"Wall remaining: {r.tiles_remaining}")
            print(f"Final shanten: {r.final_shanten}")
            print(f"Deal-ins: {r.deal_ins}")
            print(f"Flowers: {r.flowers_collected}")
            print(f"Payments: {r.payments}")

        return self.result


def run_decision(game: GameState, gen):
    """Drive a decision generator to completion, answering every request
    from the seats' agents. Returns the generator's return value.

    This is how play() runs the whole game synchronously, and how tests
    can call generator-based internals (e.g. _resolve_discard) directly.
    """
    try:
        request = next(gen)
        while True:
            request = gen.send(game.dispatch_to_agent(request))
    except StopIteration as stop:
        return stop.value


# ══════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════

class _RandomAgent(BaseAgent):
    """Minimal random agent for testing the game loop."""
    def choose_discard(self, player_idx, game_state):
        hand = game_state.hands[player_idx]
        return game_state.rng.choice(hand.tiles)


if __name__ == "__main__":
    print("=== Game Engine Test ===\n")

    # Test 1: Single verbose game
    print("--- Test 1: Single game (verbose) ---")
    agents = [_RandomAgent(f"Random_{i}") for i in range(4)]
    game = GameState(agents, seed=42)
    result = game.play(verbose=True)

    # Test 2: Batch run for stats
    print("\n--- Test 2: 200 games (silent) ---")
    wins = [0, 0, 0, 0]
    draws = 0
    total_turns = 0
    win_turns = []
    ron_count = 0
    tsumo_count = 0
    total_deal_ins = [0, 0, 0, 0]
    total_games = 200

    for i in range(total_games):
        agents = [_RandomAgent(f"R{j}") for j in range(4)]
        game = GameState(agents, seed=i)
        result = game.play()

        if result.winner is not None:
            wins[result.winner] += 1
            win_turns.append(result.turns)
            if result.win_type == "ron":
                ron_count += 1
            else:
                tsumo_count += 1
        else:
            draws += 1
        total_turns += result.turns
        for p in range(4):
            total_deal_ins[p] += result.deal_ins[p]

    total_wins = sum(wins)
    print(f"  Games:       {total_games}")
    print(f"  Wins:        {total_wins} ({100*total_wins/total_games:.1f}%)")
    print(f"    Tsumo:     {tsumo_count}")
    print(f"    Ron:       {ron_count}")
    print(f"  Draws:       {draws} ({100*draws/total_games:.1f}%)")
    print(f"  Avg turns:   {total_turns/total_games:.1f}")
    if win_turns:
        print(f"  Avg win turn: {sum(win_turns)/len(win_turns):.1f}")
    print(f"  Total deal-ins: {total_deal_ins}")
    print(f"  Avg deal-ins/game: {sum(total_deal_ins)/total_games:.2f}")

    print("\n=== Done ===")