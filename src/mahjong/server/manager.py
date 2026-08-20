"""In-memory game registry for the API server.

A "game" here is a full SESSION (雀局): four prevailing-wind rounds,
East 1 → North 4, with the dealership rotating per table rules — the
dealer repeats (连庄) on their own win or a draw, otherwise it passes.
Each next-hand call builds a fresh GameState from the session RNG, so
one session seed reproduces every wall of every hand.
"""

import asyncio
import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi import WebSocket

from mahjong.interactive import InteractiveGame
from mahjong.scoring import ScoreConfig
from mahjong.tiles import WIND_START, WIND_NAMES
from mahjong.agents import (
    RandomAgent, GreedyAgent, DefensiveAgent, HybridAgent, LearnedAgent,
)

BOT_TYPES = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "defensive": DefensiveAgent,
    "hybrid": HybridAgent,
    "learned": LearnedAgent,
}


@dataclass
class ManagedGame:
    game_id: str
    interactive: InteractiveGame
    human_seat: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sockets: List[WebSocket] = field(default_factory=list)
    # One coach explanation per decision state — a re-click must not
    # re-bill the API. {"key": fingerprint, "payload": response}
    explain_cache: Dict = field(default_factory=dict)

    # ── Session state (dealership + rounds) ──────────────────────────
    rng: random.Random = field(default_factory=random.Random)
    score_config: Optional[ScoreConfig] = None
    dealer: int = 0
    wind_idx: int = 0          # 0-3 → East, South, West, North round
    dealer_passes: int = 0     # dealerships completed this round
    hand_number: int = 1
    scores: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    session_over: bool = False

    @property
    def round_label(self) -> str:
        """"East 2" = second dealership of the East round."""
        return f"{WIND_NAMES[self.wind_idx]} {self.dealer_passes + 1}"

    def session_view(self) -> Dict:
        return {
            "hand_number": self.hand_number,
            "round_wind": WIND_START + self.wind_idx,
            "round_label": self.round_label,
            "dealer": self.dealer,
            "scores": self.scores[:],
            "session_over": self.session_over,
        }

    def next_hand(self) -> None:
        """Advance the session after a finished hand and deal the next.

        Raises RuntimeError if the current hand is still in progress or
        the session has ended.
        """
        if self.interactive.result is None:
            raise RuntimeError("Current hand is not finished")
        if self.session_over:
            raise RuntimeError("Session is over — North 4 has been played")

        result = self.interactive.result
        payments = result.payments or [0, 0, 0, 0]
        for q in range(4):
            self.scores[q] += payments[q]

        # Dealer repeats (连庄) on their own win or a draw
        if result.winner is not None and result.winner != self.dealer:
            self.dealer = (self.dealer + 1) % 4
            self.dealer_passes += 1
            if self.dealer_passes == 4:
                self.dealer_passes = 0
                self.wind_idx += 1
                if self.wind_idx == 4:
                    self.session_over = True
                    return

        agents = self.interactive.game.agents
        self.interactive = InteractiveGame(
            agents, human_seats={self.human_seat},
            seed=self.rng.getrandbits(32), dealer=self.dealer,
            prevailing_wind=WIND_START + self.wind_idx,
            score_config=self.score_config)
        self.hand_number += 1
        self.explain_cache.clear()
        self.interactive.start()

    def view(self) -> Dict:
        view = self.interactive.view_for(self.human_seat)
        view["session"] = self.session_view()
        return view

    async def broadcast(self):
        """Push the current view to every connected websocket."""
        view = self.view()
        dead = []
        for ws in self.sockets:
            try:
                await ws.send_json(view)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.sockets.remove(ws)


class GameManager:
    """Holds live games in memory, keyed by a short id."""

    def __init__(self):
        self.games: Dict[str, ManagedGame] = {}

    def create(self, seed: Optional[int] = None, human_seat: int = 0,
               bots: Optional[List[str]] = None,
               tai_cap: int = 6, base_unit: int = 1) -> ManagedGame:
        if not 0 <= human_seat <= 3:
            raise ValueError("human_seat must be 0-3")
        if not 1 <= tai_cap <= 13:
            raise ValueError("tai_cap must be between 1 and 13")
        if not 1 <= base_unit <= 1000:
            raise ValueError("base_unit must be between 1 and 1000")
        names = bots or ["hybrid"] * 4
        if len(names) != 4:
            raise ValueError("bots must list exactly 4 agent types")

        agents = []
        for i, name in enumerate(names):
            if name not in BOT_TYPES:
                raise ValueError(f"Unknown bot type {name!r}; "
                                 f"choose from {sorted(BOT_TYPES)}")
            try:
                agents.append(BOT_TYPES[name](f"{name.title()}-{i}"))
            except RuntimeError as exc:  # e.g. learned bot, model not trained
                raise ValueError(str(exc))

        config = ScoreConfig(tai_cap=tai_cap, base_unit=base_unit)
        # The session RNG derives every hand's wall, so one seed
        # reproduces the whole session. Hand 1 keeps the historical
        # behavior of using the seed directly.
        rng = random.Random(seed)
        interactive = InteractiveGame(agents, human_seats={human_seat},
                                      seed=seed, dealer=0,
                                      score_config=config)
        game_id = uuid.uuid4().hex[:12]
        managed = ManagedGame(game_id, interactive, human_seat,
                              rng=rng, score_config=config)
        self.games[game_id] = managed
        interactive.start()
        return managed

    def get(self, game_id: str) -> Optional[ManagedGame]:
        return self.games.get(game_id)

    def remove(self, game_id: str) -> bool:
        return self.games.pop(game_id, None) is not None
