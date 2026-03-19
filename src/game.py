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

Simplifications for v1 (Week 1):
  - No chow/pong/kong claiming (pure draw-and-discard)
  - Win by self-draw (tsumo) only
  - No scoring beyond win/loss
  - Kong declaration not yet implemented

These will be added in Week 2 when we need opponent interaction
for the defense/hybrid agents to reason about.
"""

import random
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from tiles import (
    create_wall, is_bonus, tile_short, hand_to_str,
    NUM_STANDARD_UNIQUE, NUM_TOTAL_TILES,
    is_numbered, suit_of, rank_of
)
from hand import Hand, calculate_shanten, is_winning_hand

# Singapore Mahjong: game ends in a draw when 15 tiles remain in the wall.
# These 15 tiles are "dead" and cannot be drawn.
DEAD_WALL_SIZE = 15


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
        """Decide whether to claim a discarded tile for pong or chow.

        Args:
            player_idx: which player is deciding
            tile_id: the discarded tile available to claim
            claim_type: "pong" or "chow"
            game_state: current game state

        Returns:
            True to claim, False to pass
        """
        return False  # default: never claim


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

    def __init__(self, agents: List[BaseAgent], seed: Optional[int] = None):
        if len(agents) != 4:
            raise ValueError("Need exactly 4 agents")

        self.agents = agents
        self.rng = random.Random(seed)

        # Game state
        self.hands: List[Hand] = [Hand() for _ in range(4)]
        self.wall: List[int] = []
        self.wall_idx: int = 0
        self.turn: int = 0
        self.active_player: int = 0
        self.game_over: bool = False
        self.result: Optional[GameResult] = None

        # Deal-in tracking: deal_ins[p] counts how many times player p
        # discarded a tile that would have completed an opponent's hand.
        # This includes the final deal-in that ends the game (ron) AND
        # near-misses where an opponent was tenpai but didn't claim.
        self.deal_ins: List[int] = [0, 0, 0, 0]

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
                # Bonus tile — set aside, draw replacement
                continue
            else:
                return tile

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

        # Own hand
        for i in range(NUM_STANDARD_UNIQUE):
            counts[i] += self.hands[player_idx].counts[i]

        # All players' discards
        for p in range(4):
            for tile in self.hands[p].discards:
                if not is_bonus(tile):
                    counts[tile] += 1

        # Other players' exposed melds (own hand already counted)
        for p in range(4):
            if p == player_idx:
                continue
            for meld_type, meld_tiles in self.hands[p].exposed:
                for tile in meld_tiles:
                    if not is_bonus(tile):
                        counts[tile] += 1

        return counts

    def get_all_discards(self) -> List[List[int]]:
        """Get discard piles for all 4 players."""
        return [h.discards[:] for h in self.hands]

    # ── Turn execution ────────────────────────────────────────────────

    def _check_pong(self, discarder: int, tile_id: int) -> Optional[int]:
        """Check if any opponent wants to pong the discarded tile.

        Any player (not the discarder) with 2+ copies can claim.
        Priority: closest in turn order.
        Returns claimer index or None.
        """
        for offset in range(1, 4):
            candidate = (discarder + offset) % 4
            hand = self.hands[candidate]
            if hand.counts[tile_id] >= 2:
                if self.agents[candidate].should_claim(candidate, tile_id, "pong", self):
                    return candidate
        return None

    def _check_chow(self, discarder: int, tile_id: int) -> Optional[Tuple[int, List[int]]]:
        """Check if the next player wants to chow the discarded tile.

        Only the player immediately after the discarder can chow.
        Only numbered tiles can form chows.
        Returns (claimer, [partner1, partner2]) or None.
        """
        if not is_numbered(tile_id):
            return None

        next_player = (discarder + 1) % 4
        hand = self.hands[next_player]
        tile_rank = rank_of(tile_id)
        tile_suit_start = (tile_id // 9) * 9

        # Find all possible chow combinations with this tile
        possible_chows = []

        # tile is leftmost: [tile, tile+1, tile+2]
        if tile_rank <= 7:
            p1, p2 = tile_id + 1, tile_id + 2
            if suit_of(p1) == suit_of(tile_id) and suit_of(p2) == suit_of(tile_id):
                if hand.counts[p1] >= 1 and hand.counts[p2] >= 1:
                    possible_chows.append([p1, p2])

        # tile is middle: [tile-1, tile, tile+1]
        if tile_rank >= 2 and tile_rank <= 8:
            p1, p2 = tile_id - 1, tile_id + 1
            if suit_of(p1) == suit_of(tile_id) and suit_of(p2) == suit_of(tile_id):
                if hand.counts[p1] >= 1 and hand.counts[p2] >= 1:
                    possible_chows.append([p1, p2])

        # tile is rightmost: [tile-2, tile-1, tile]
        if tile_rank >= 3:
            p1, p2 = tile_id - 2, tile_id - 1
            if suit_of(p1) == suit_of(tile_id) and suit_of(p2) == suit_of(tile_id):
                if hand.counts[p1] >= 1 and hand.counts[p2] >= 1:
                    possible_chows.append([p1, p2])

        if possible_chows:
            if self.agents[next_player].should_claim(next_player, tile_id, "chow", self):
                # Use the first valid combination
                return (next_player, possible_chows[0])

        return None

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

        elif claim_type == "chow" and partners:
            # Remove partner tiles from hand, form meld
            for p_tile in partners:
                hand.counts[p_tile] -= 1
                hand.tiles.remove(p_tile)
            meld_tiles = sorted([tile_id] + partners)
            hand.add_exposed_meld("chow", meld_tiles)

        # Claimer now has 14 - 3 + ... wait, let's think about tile count:
        # Before claim: claimer has 13 tiles (normal hand)
        # We removed 2 (pong) or 2 (chow partners) = 11 tiles
        # The claimed tile is NOT added to hand.tiles — it goes straight to the meld
        # So claimer has 11 tiles + 1 exposed meld = needs to be at 13 - 3 = 10
        # Actually: each exposed meld represents 3 tiles removed from the concealed hand
        # A player with N exposed melds has 13 - 3*N concealed tiles
        # After claiming: concealed = 13 - 3*(melds-1) - 2 = 13 - 3*melds + 3 - 2 = 14 - 3*melds
        # That's one too many — they need to discard!
        # Wait, let me recalculate:
        # Start: 13 concealed tiles, M exposed melds (each = 3 tiles)
        # Total tiles owned = 13 + 3*M
        # After claim: removed 2 from concealed, added 1 new meld (3 tiles)
        # New concealed = 13 - 2 = 11, new melds = M+1
        # Total = 11 + 3*(M+1) = 11 + 3M + 3 = 14 + 3M
        # That's 1 more than the 13 + 3M they should have — need to discard

        # Agent chooses discard
        discard_tile = self.agents[claimer].choose_discard(claimer, self)

        if discard_tile not in hand.tiles:
            raise ValueError(
                f"Agent {self.agents[claimer].name} (P{claimer}) tried to discard "
                f"{tile_short(discard_tile)} after claim, not in hand: "
                f"{hand_to_str(hand.tiles, short=True)}"
            )

        hand.discard(discard_tile)
        return discard_tile

    def _execute_turn(self) -> bool:
        """Execute one turn. Returns True if game continues, False if over."""
        p = self.active_player
        hand = self.hands[p]

        # 1. Draw
        drawn = self._deal_tile_to(p)
        if drawn is None:
            self._end_game(winner=None, win_type=None)
            return False

        # 2. Check tsumo (self-draw win)
        if is_winning_hand(hand.counts, hand.num_exposed_melds):
            self._end_game(winner=p, win_type="tsumo")
            return False

        # 3. Agent chooses discard
        discard_tile = self.agents[p].choose_discard(p, self)

        # Validate
        if discard_tile not in hand.tiles:
            raise ValueError(
                f"Agent {self.agents[p].name} (P{p}) tried to discard "
                f"{tile_short(discard_tile)} not in hand: "
                f"{hand_to_str(hand.tiles, short=True)}"
            )

        # 4. Discard
        hand.discard(discard_tile)

        # 5. Claim resolution — priority: Ron > Pong > Chow
        # 5a. Ron check
        ron_winner = None
        for offset in range(1, 4):
            opponent = (p + offset) % 4
            opp_hand = self.hands[opponent]
            opp_hand.counts[discard_tile] += 1
            if is_winning_hand(opp_hand.counts, opp_hand.num_exposed_melds):
                if ron_winner is None:
                    ron_winner = opponent
                self.deal_ins[p] += 1
            opp_hand.counts[discard_tile] -= 1

        if ron_winner is not None:
            self.hands[ron_winner].add_tile(discard_tile)
            self._end_game(winner=ron_winner, win_type="ron", dealt_in_by=p)
            return False

        # 5b. Pong check (any opponent, priority by turn order)
        pong_claimer = self._check_pong(p, discard_tile)
        if pong_claimer is not None:
            # Remove tile from discard pile (it was just added)
            hand.discards.pop()

            new_discard = self._execute_claim(pong_claimer, discard_tile, "pong")

            # Check ron on the claimer's discard
            ron_winner2 = None
            for offset in range(1, 4):
                opponent = (pong_claimer + offset) % 4
                opp_hand = self.hands[opponent]
                opp_hand.counts[new_discard] += 1
                if is_winning_hand(opp_hand.counts, opp_hand.num_exposed_melds):
                    if ron_winner2 is None:
                        ron_winner2 = opponent
                    self.deal_ins[pong_claimer] += 1
                opp_hand.counts[new_discard] -= 1

            if ron_winner2 is not None:
                self.hands[ron_winner2].add_tile(new_discard)
                self._end_game(winner=ron_winner2, win_type="ron", dealt_in_by=pong_claimer)
                return False

            # Turn passes to player after the claimer
            self.active_player = (pong_claimer + 1) % 4
            self.turn += 1
            return True

        # 5c. Chow check (next player only)
        chow_result = self._check_chow(p, discard_tile)
        if chow_result is not None:
            chow_claimer, partners = chow_result

            # Remove tile from discard pile
            hand.discards.pop()

            new_discard = self._execute_claim(chow_claimer, discard_tile, "chow", partners)

            # Check ron on the claimer's discard
            ron_winner3 = None
            for offset in range(1, 4):
                opponent = (chow_claimer + offset) % 4
                opp_hand = self.hands[opponent]
                opp_hand.counts[new_discard] += 1
                if is_winning_hand(opp_hand.counts, opp_hand.num_exposed_melds):
                    if ron_winner3 is None:
                        ron_winner3 = opponent
                    self.deal_ins[chow_claimer] += 1
                opp_hand.counts[new_discard] -= 1

            if ron_winner3 is not None:
                self.hands[ron_winner3].add_tile(new_discard)
                self._end_game(winner=ron_winner3, win_type="ron", dealt_in_by=chow_claimer)
                return False

            # Turn passes to player after the claimer (which is chow_claimer + 1)
            self.active_player = (chow_claimer + 1) % 4
            self.turn += 1
            return True

        # 6. No claims — next player
        self.active_player = (self.active_player + 1) % 4
        self.turn += 1
        return True

    def _end_game(self, winner: Optional[int], win_type: Optional[str],
                  dealt_in_by: Optional[int] = None):
        """Record the game result."""
        self.game_over = True
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
        )

    # ── Main game loop ────────────────────────────────────────────────

    def play(self, max_turns: int = 200, verbose: bool = False) -> GameResult:
        """Play a complete game and return the result.

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

        while not self.game_over and self.turn < max_turns:
            if verbose and self.turn % 20 == 0 and self.turn > 0:
                print(f"  Turn {self.turn}, wall: {self.tiles_remaining}")

            if not self._execute_turn():
                break

        if self.result is None:
            self._end_game(winner=None, win_type=None)

        if verbose:
            r = self.result
            if r.winner is not None:
                win_msg = (f"\nResult: P{r.winner} ({self.agents[r.winner].name}) "
                          f"wins by {r.win_type} on turn {r.turns}!")
                if r.win_type == "ron" and r.dealt_in_by is not None:
                    win_msg += (f" (P{r.dealt_in_by} "
                               f"({self.agents[r.dealt_in_by].name}) dealt in)")
                print(win_msg)
            else:
                print(f"\nResult: Draw after {r.turns} turns. "
                      f"Wall remaining: {r.tiles_remaining}")
            print(f"Final shanten: {r.final_shanten}")
            print(f"Deal-ins: {r.deal_ins}")
            print(f"Flowers: {r.flowers_collected}")

        return self.result


# ══════════════════════════════════════════════════════════════════════
# QUICK TEST
# ══════════════════════════════════════════════════════════════════════

class _RandomAgent(BaseAgent):
    """Minimal random agent for testing the game loop."""
    def choose_discard(self, player_idx, game_state):
        hand = game_state.hands[player_idx]
        return random.choice(hand.tiles)


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