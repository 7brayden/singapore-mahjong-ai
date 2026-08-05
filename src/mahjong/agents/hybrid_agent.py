"""
Hybrid discard agent — combines offense and defense.

Strategy: for each candidate discard, compute:
  - Offense score: based on resulting shanten and tile acceptance
  - Defense score: based on danger estimation (from defense module)
  - Final score: weighted combination, with weight shifting dynamically

Dynamic weighting logic:
  - If own shanten is low (close to winning) → lean offensive
  - If own shanten is high (far from winning) → lean defensive
  - Early game → more offensive (build your hand)
  - Late game → more defensive (avoid dealing in)

Claiming policy: more selective than greedy — only claims pong/chow
when already close to winning (shanten <= 2), since exposing melds
reveals information to opponents.
"""

from mahjong.game import BaseAgent
from mahjong.hand import evaluate_discards, calculate_shanten, best_chow_option
from mahjong.defense import estimate_danger


class HybridAgent(BaseAgent):
    """Fuses offense and defense scores with dynamic weighting."""

    # Only claim pong/chow at or below this shanten — exposing melds
    # reveals information, so only do it when it matters.
    CLAIM_SHANTEN_LIMIT = 2

    def __init__(self, name: str = "Hybrid",
                 base_offense_weight: float = 0.6,
                 base_defense_weight: float = 0.4):
        super().__init__(name)
        self.base_offense_weight = base_offense_weight
        self.base_defense_weight = base_defense_weight

    def should_claim(self, player_idx: int, tile_id: int,
                     claim_type: str, game_state) -> bool:
        """Claim a pong if it improves shanten, but only when already close
        to winning."""
        if claim_type != "pong":
            return False

        hand = game_state.hands[player_idx]
        counts = hand.copy_counts()
        if counts[tile_id] < 2:
            return False

        current_shanten = calculate_shanten(counts, hand.num_exposed_melds)
        if current_shanten > self.CLAIM_SHANTEN_LIMIT:
            return False

        counts[tile_id] -= 2
        new_shanten = calculate_shanten(counts, hand.num_exposed_melds + 1)
        return new_shanten < current_shanten

    def choose_chow(self, player_idx: int, tile_id: int,
                    options, game_state):
        """Claim the best chow combination, but only when already close
        to winning."""
        hand = game_state.hands[player_idx]
        counts = hand.copy_counts()

        current_shanten = calculate_shanten(counts, hand.num_exposed_melds)
        if current_shanten > self.CLAIM_SHANTEN_LIMIT:
            return None

        best_pair, _ = best_chow_option(counts, hand.num_exposed_melds, options)
        return best_pair

    def choose_discard(self, player_idx: int, game_state) -> int:
        hand = game_state.hands[player_idx]
        visible = game_state.get_visible_counts(player_idx)

        # Get current shanten for dynamic weight adjustment
        current_shanten = calculate_shanten(hand.copy_counts(), hand.num_exposed_melds)

        # Compute dynamic weights
        off_w, def_w = self._dynamic_weights(current_shanten, game_state)

        # Evaluate offense for all candidate discards
        offense_evals = evaluate_discards(hand, visible_counts=visible)

        # Build offense scores: normalize best = 1.0, worst = 0.0
        offense_scores = {}
        if len(offense_evals) > 1:
            best_sh = offense_evals[0]["shanten"]
            worst_sh = offense_evals[-1]["shanten"]
            best_acc = max(e["acceptance"] for e in offense_evals)
            worst_acc = min(e["acceptance"] for e in offense_evals)

            for e in offense_evals:
                if worst_sh == best_sh:
                    sh_score = 1.0
                else:
                    sh_score = 1.0 - (e["shanten"] - best_sh) / (worst_sh - best_sh)

                if best_acc == worst_acc:
                    acc_score = 1.0
                else:
                    acc_score = (e["acceptance"] - worst_acc) / (best_acc - worst_acc)

                # Shanten matters more than acceptance
                offense_scores[e["tile_id"]] = 0.7 * sh_score + 0.3 * acc_score
        else:
            offense_scores[offense_evals[0]["tile_id"]] = 1.0

        # Evaluate defense for all candidate tiles
        defense_scores = {}
        for e in offense_evals:
            tid = e["tile_id"]
            danger = estimate_danger(tid, player_idx, game_state)
            # Invert: low danger = high safety score
            defense_scores[tid] = 1.0 - danger

        # Combine scores
        best_tile = None
        best_combined = -1.0

        for e in offense_evals:
            tid = e["tile_id"]
            combined = (off_w * offense_scores[tid]
                        + def_w * defense_scores[tid])

            if combined > best_combined:
                best_combined = combined
                best_tile = tid

        return best_tile

    def _dynamic_weights(self, current_shanten: int, game_state) -> tuple:
        """Adjust offense/defense weights based on game state.

        Returns (offense_weight, defense_weight) that sum to 1.0.
        """
        off_w = self.base_offense_weight
        def_w = self.base_defense_weight

        # Shanten adjustment: closer to winning → more offensive
        if current_shanten <= 0:
            off_w += 0.25
            def_w -= 0.25
        elif current_shanten == 1:
            off_w += 0.10
            def_w -= 0.10
        elif current_shanten >= 4:
            off_w -= 0.15
            def_w += 0.15

        # Game phase adjustment
        turn = game_state.turn
        if turn > 50:
            off_w -= 0.10
            def_w += 0.10
        elif turn < 15:
            off_w += 0.05
            def_w -= 0.05

        # Clamp and normalize
        off_w = max(0.1, off_w)
        def_w = max(0.1, def_w)
        total = off_w + def_w
        off_w /= total
        def_w /= total

        return off_w, def_w


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from mahjong.game import GameState
    from mahjong.agents.greedy_agent import GreedyAgent

    total = 50

    # Test 1: 4x Hybrid
    print("--- Test 1: 4x Hybrid ---")
    wins = 0
    exposed = 0
    for i in range(total):
        agents = [HybridAgent(f"H{j}") for j in range(4)]
        game = GameState(agents, seed=i)
        result = game.play()
        if result.winner is not None:
            wins += 1
        for p in range(4):
            exposed += game.hands[p].num_exposed_melds
    print(f"  Wins: {wins}/{total}, Avg exposed melds/game: {exposed/total:.1f}")

    # Test 2: 2x Greedy vs 2x Hybrid
    print("\n--- Test 2: 2x Greedy vs 2x Hybrid ---")
    greedy_w = 0
    hybrid_w = 0
    for i in range(total):
        agents = [GreedyAgent("G0"), HybridAgent("H0"),
                  GreedyAgent("G1"), HybridAgent("H1")]
        game = GameState(agents, seed=i)
        result = game.play()
        if result.winner in (0, 2):
            greedy_w += 1
        elif result.winner in (1, 3):
            hybrid_w += 1
    print(f"  Greedy: {greedy_w}, Hybrid: {hybrid_w}, Draw: {total - greedy_w - hybrid_w}")
