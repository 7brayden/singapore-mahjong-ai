"""In-memory game registry for the API server."""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi import WebSocket

from mahjong.interactive import InteractiveGame
from mahjong.agents import RandomAgent, GreedyAgent, DefensiveAgent, HybridAgent

BOT_TYPES = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "defensive": DefensiveAgent,
    "hybrid": HybridAgent,
}


@dataclass
class ManagedGame:
    game_id: str
    interactive: InteractiveGame
    human_seat: int
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sockets: List[WebSocket] = field(default_factory=list)

    def view(self) -> Dict:
        return self.interactive.view_for(self.human_seat)

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
               bots: Optional[List[str]] = None) -> ManagedGame:
        if not 0 <= human_seat <= 3:
            raise ValueError("human_seat must be 0-3")
        names = bots or ["hybrid"] * 4
        if len(names) != 4:
            raise ValueError("bots must list exactly 4 agent types")

        agents = []
        for i, name in enumerate(names):
            if name not in BOT_TYPES:
                raise ValueError(f"Unknown bot type {name!r}; "
                                 f"choose from {sorted(BOT_TYPES)}")
            agents.append(BOT_TYPES[name](f"{name.title()}-{i}"))

        interactive = InteractiveGame(agents, human_seats={human_seat}, seed=seed)
        game_id = uuid.uuid4().hex[:12]
        managed = ManagedGame(game_id, interactive, human_seat)
        self.games[game_id] = managed
        interactive.start()
        return managed

    def get(self, game_id: str) -> Optional[ManagedGame]:
        return self.games.get(game_id)

    def remove(self, game_id: str) -> bool:
        return self.games.pop(game_id, None) is not None
