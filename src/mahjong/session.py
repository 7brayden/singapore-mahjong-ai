"""
Multi-round session management for Singapore Mahjong.

A full session (雀局) runs four rounds — East, South, West, North —
as the prevailing wind. Within a round each player takes a turn as
dealer. The dealer repeats (连庄) after winning a hand or after a
draw; otherwise the dealership passes to the next player. Once all
four players have held (and lost) the dealership, the prevailing wind
advances. Scores accumulate across hands from each hand's payment
deltas (win payments plus instant bonuses).

Dealer stakes are not doubled — the dealer's reward is the extra
hands they get to play while repeating. Seat winds rotate with the
dealership: the dealer is always East, the next player South, etc.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from mahjong.tiles import WIND_START, WIND_NAMES
from mahjong.game import BaseAgent, GameState, GameResult
from mahjong.scoring import ScoreConfig


@dataclass
class HandRecord:
    """One hand within a session."""
    hand_number: int          # 1-based
    prevailing_wind: int      # wind tile id (27-30)
    dealer: int               # player index holding the dealership
    result: GameResult
    scores_after: List[int]   # running scores once this hand's payments applied


@dataclass
class SessionResult:
    """Outcome of a full session."""
    final_scores: List[int]
    hands_played: int
    records: List[HandRecord]

    @property
    def ranking(self) -> List[int]:
        """Player indices sorted by final score, best first."""
        return sorted(range(4), key=lambda p: -self.final_scores[p])


class Session:
    """Plays a multi-round session of Singapore Mahjong.

    Each hand's wall is seeded from the session RNG, so a session seed
    reproduces the entire session — every hand, every score.
    """

    def __init__(self, agents: List[BaseAgent], seed: Optional[int] = None,
                 score_config: Optional[ScoreConfig] = None,
                 winds: Sequence[int] = tuple(range(WIND_START, WIND_START + 4)),
                 max_hands: int = 100):
        """
        Args:
            agents: the four players, in fixed table order
            seed: session seed (drives every hand's wall)
            score_config: house rules, shared by all hands
            winds: prevailing winds to play (default: all four rounds;
                   pass a shorter sequence for a shorter session)
            max_hands: safety cap on total hands (a dealer on a long
                       winning streak repeats indefinitely otherwise)
        """
        if len(agents) != 4:
            raise ValueError("Need exactly 4 agents")
        self.agents = agents
        self.rng = random.Random(seed)
        self.score_config = score_config or ScoreConfig()
        self.winds = list(winds)
        self.max_hands = max_hands
        self.scores: List[int] = [0, 0, 0, 0]
        self.records: List[HandRecord] = []

    def _play_hand(self, dealer: int, prevailing_wind: int) -> GameResult:
        """Play one hand. Split out so tests can stub game execution."""
        hand_seed = self.rng.getrandbits(32)
        game = GameState(self.agents, seed=hand_seed, dealer=dealer,
                         prevailing_wind=prevailing_wind,
                         score_config=self.score_config)
        return game.play()

    def play(self, verbose: bool = False) -> SessionResult:
        dealer = 0
        for wind in self.winds:
            dealer_passes = 0
            while dealer_passes < 4:
                if len(self.records) >= self.max_hands:
                    if verbose:
                        print(f"Reached max_hands={self.max_hands} — "
                              f"ending session early.")
                    return self._result()

                result = self._play_hand(dealer, wind)
                payments = result.payments or [0, 0, 0, 0]
                for p in range(4):
                    self.scores[p] += payments[p]
                self.records.append(HandRecord(
                    hand_number=len(self.records) + 1,
                    prevailing_wind=wind,
                    dealer=dealer,
                    result=result,
                    scores_after=self.scores[:],
                ))
                if verbose:
                    self._print_hand(self.records[-1])

                # Dealer repeats on their own win or on a draw
                if result.winner is not None and result.winner != dealer:
                    dealer = (dealer + 1) % 4
                    dealer_passes += 1
        return self._result()

    def _result(self) -> SessionResult:
        return SessionResult(
            final_scores=self.scores[:],
            hands_played=len(self.records),
            records=list(self.records),
        )

    def _print_hand(self, rec: HandRecord):
        r = rec.result
        wind_name = WIND_NAMES[rec.prevailing_wind - WIND_START]
        if r.winner is not None:
            desc = (f"P{r.winner} ({self.agents[r.winner].name}) "
                    f"wins by {r.win_type}")
            if r.winning_score is not None:
                desc += f", {r.winning_score.tai} tai"
        else:
            desc = "draw"
        print(f"Hand {rec.hand_number:>2} [{wind_name} round, "
              f"dealer P{rec.dealer}]: {desc} | scores: {rec.scores_after}")


# ══════════════════════════════════════════════════════════════════════
# QUICK DEMO
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from mahjong.agents import GreedyAgent, HybridAgent

    print("=== One East round: 2x Greedy vs 2x Hybrid ===\n")
    agents = [GreedyAgent("G0"), HybridAgent("H0"),
              GreedyAgent("G1"), HybridAgent("H1")]
    session = Session(agents, seed=42, winds=[WIND_START])
    result = session.play(verbose=True)

    print(f"\nFinal scores after {result.hands_played} hands:")
    for rank, p in enumerate(result.ranking, 1):
        print(f"  {rank}. P{p} ({agents[p].name}): {result.final_scores[p]:+d}")
