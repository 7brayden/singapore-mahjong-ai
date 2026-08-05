"""
Human-in-the-loop controller for Singapore Mahjong.

InteractiveGame drives GameState's decision generator step by step:
bot seats answer their own DecisionRequests via their agents, and when
a request belongs to a human seat the game pauses with `pending` set.
The caller (CLI, FastAPI endpoint, websocket handler) answers it with
submit(), and the game advances to the next human decision or the end.

Views: view_for(seat) serializes only what that seat may see — its own
concealed tiles, everyone's discards/melds/flowers, and opponents'
concealed tile COUNTS, never their tiles. The dict is JSON-ready for a
web frontend.

The agent attached to a human seat is never asked to decide; it can
serve as a hint provider (e.g. "what would the hybrid bot do here?")
via dispatch_to_agent.
"""

from typing import Dict, List, Optional, Set

from mahjong.tiles import WIND_START, tile_short
from mahjong.game import (
    BaseAgent, GameState, GameResult, DecisionRequest,
    DiscardRequest, ClaimRequest, ChowRequest, KongRequest,
)
from mahjong.scoring import ScoreConfig


class InteractiveGame:
    """A single hand with one or more human-controlled seats."""

    def __init__(self, agents: List[BaseAgent], human_seats: Set[int],
                 seed: Optional[int] = None, dealer: int = 0,
                 prevailing_wind: int = WIND_START,
                 score_config: Optional[ScoreConfig] = None,
                 max_turns: int = 200):
        self.game = GameState(agents, seed=seed, dealer=dealer,
                              prevailing_wind=prevailing_wind,
                              score_config=score_config)
        self.human_seats = set(human_seats)
        self.pending: Optional[DecisionRequest] = None
        self.result: Optional[GameResult] = None
        self._gen = self.game.step_game(max_turns)
        self._started = False

    @property
    def game_over(self) -> bool:
        return self.result is not None

    # ── Driving the game ──────────────────────────────────────────────

    def start(self) -> Optional[DecisionRequest]:
        """Deal and advance to the first human decision (or the end).

        Returns the pending request, or None if the game finished
        without needing human input.
        """
        if self._started:
            raise RuntimeError("Game already started")
        self._started = True
        self._advance(first=True)
        return self.pending

    def submit(self, answer) -> Optional[DecisionRequest]:
        """Answer the pending human decision and advance.

        Returns the next pending request, or None if the game is over.
        Raises ValueError for answers the pending request can't accept
        (the game state is untouched, so the caller can retry).
        """
        if self.pending is None:
            raise RuntimeError("No pending decision to answer")
        self._validate(self.pending, answer)
        self.pending = None
        self._advance(answer=answer)
        return self.pending

    def _advance(self, answer=None, first: bool = False):
        try:
            request = next(self._gen) if first else self._gen.send(answer)
            while True:
                if request.player in self.human_seats:
                    self.pending = request
                    return
                request = self._gen.send(self.game.dispatch_to_agent(request))
        except StopIteration as stop:
            self.result = stop.value
            self.pending = None

    def _validate(self, request: DecisionRequest, answer):
        if isinstance(request, DiscardRequest):
            if answer not in self.game.hands[request.player].tiles:
                raise ValueError(
                    f"Cannot discard {answer}: not in hand")
        elif isinstance(request, ClaimRequest):
            if not isinstance(answer, bool):
                raise ValueError("Claim answer must be True or False")
        elif isinstance(request, ChowRequest):
            if answer is not None and \
                    tuple(answer) not in [tuple(o) for o in request.options]:
                raise ValueError(f"Invalid chow choice {answer}; "
                                 f"options: {request.options}")
        elif isinstance(request, KongRequest):
            if answer is not None and \
                    tuple(answer) not in [tuple(o) for o in request.options]:
                raise ValueError(f"Invalid kong choice {answer}; "
                                 f"options: {request.options}")

    # ── Views (redacted, JSON-ready) ──────────────────────────────────

    def view_for(self, seat: int) -> Dict:
        """Serialize the game state as seen from one seat.

        Contains that seat's concealed hand, public information for all
        players (discards, melds, flowers, concealed tile counts), and
        the pending decision if it belongs to this seat.
        """
        g = self.game
        players = []
        for p in range(4):
            h = g.hands[p]
            players.append({
                "seat": p,
                "name": g.agents[p].name,
                "is_human": p in self.human_seats,
                "seat_wind": WIND_START + g.seat_index(p),
                "concealed_count": len(h.tiles),
                "exposed": [[kind, list(tiles)] for kind, tiles in h.exposed],
                "flowers": list(h.flowers),
                "discards": list(h.discards),
                "chips": g.payments[p],
            })
        return {
            "seat": seat,
            "hand": sorted(g.hands[seat].tiles),
            "turn": g.turn,
            "active_player": g.active_player,
            "tiles_remaining": g.tiles_remaining,
            "dealer": g.dealer,
            "prevailing_wind": g.prevailing_wind,
            "players": players,
            "game_over": self.game_over,
            "pending": self._pending_view(seat),
            "result": self._result_view(),
        }

    def _pending_view(self, seat: int) -> Optional[Dict]:
        r = self.pending
        if r is None or r.player != seat:
            return None
        if isinstance(r, DiscardRequest):
            return {"type": "discard", "drawn": r.drawn}
        if isinstance(r, ClaimRequest):
            return {"type": "claim", "tile": r.tile_id,
                    "claim_type": r.claim_type}
        if isinstance(r, ChowRequest):
            return {"type": "chow", "tile": r.tile_id,
                    "options": [list(o) for o in r.options]}
        if isinstance(r, KongRequest):
            return {"type": "kong",
                    "options": [list(o) for o in r.options]}
        return None

    def _result_view(self) -> Optional[Dict]:
        r = self.result
        if r is None:
            return None
        view = {
            "winner": r.winner,
            "win_type": r.win_type,
            "dealt_in_by": r.dealt_in_by,
            "turns": r.turns,
            "payments": list(r.payments) if r.payments else [0, 0, 0, 0],
            "win_tile": r.win_tile,
        }
        if r.winner is not None:
            # The hand is over — revealing the winning hand is standard play
            winner_hand = self.game.hands[r.winner]
            view["winner_hand"] = sorted(winner_hand.tiles)
            view["winner_exposed"] = [[kind, list(tiles)]
                                      for kind, tiles in winner_hand.exposed]
        if r.winning_score is not None:
            s = r.winning_score
            view["score"] = {
                "tai": s.tai,
                "total_tai": s.total_tai,
                "value": s.value,
                "is_limit": s.is_limit,
                "items": [{"rule": i.rule, "label": i.label, "tai": i.tai}
                          for i in s.items],
            }
        return view


# ══════════════════════════════════════════════════════════════════════
# QUICK DEMO — the "human" seat is auto-answered by its own agent
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    from mahjong.agents import HybridAgent

    game = InteractiveGame([HybridAgent(f"H{i}") for i in range(4)],
                           human_seats={0}, seed=42)
    game.start()

    decisions = 0
    while not game.game_over:
        pending = game.pending
        if decisions < 3:  # show the first few decision points
            view = game.view_for(0)
            hand_str = " ".join(tile_short(t) for t in view["hand"])
            print(f"Decision {decisions + 1}: {view['pending']}")
            print(f"  Hand: {hand_str}\n")
        answer = game.game.dispatch_to_agent(pending)  # bot plays the human seat
        game.submit(answer)
        decisions += 1

    print(f"Game over after {decisions} human decisions.")
    print(json.dumps(game.view_for(0)["result"], indent=2))
