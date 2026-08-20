"""FastAPI backend for playing Singapore Mahjong against the bots.

Endpoints:
    GET  /                      server info + available bot types
    GET  /tiles                 tile id → name/short/suit/rank mapping
    POST /games                 create a game (seed, human_seat, bots)
    GET  /games/{id}            redacted view for the human seat
    POST /games/{id}/action     answer the pending decision
    GET  /games/{id}/hint       what the seat's own agent would do
    GET  /games/{id}/analysis   learning stats (shanten, danger, threats)
    POST /games/{id}/explain    coach lesson (Claude if key set, else template)
    DEL  /games/{id}            drop the game
    WS   /games/{id}/ws         view pushed on connect and after actions

Run locally:
    uvicorn mahjong.server.app:app --reload
"""

from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import mahjong
from mahjong.game import DiscardRequest, ClaimRequest, ChowRequest, KongRequest
from mahjong.tiles import NUM_TOTAL_UNIQUE, tile_name, tile_short, suit_of, rank_of
from mahjong.server.manager import GameManager, BOT_TYPES, ManagedGame
from mahjong.server.analysis import analyze_seat
from mahjong.tai_track import claim_consequence
from mahjong.coach import explain_situation

app = FastAPI(title="Singapore Mahjong AI", version=mahjong.__version__)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev; tighten before hosting publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

manager = GameManager()


class CreateGameRequest(BaseModel):
    seed: Optional[int] = None
    human_seat: int = 0
    bots: Optional[List[str]] = None  # 4 entries; human seat's bot gives hints
    tai_cap: int = 6                  # scoring limit (payments double per tai)
    base_unit: int = 1                # chips a 1-tai hand is worth


class ActionRequest(BaseModel):
    answer: Any = None


def _get_or_404(game_id: str) -> ManagedGame:
    managed = manager.get(game_id)
    if managed is None:
        raise HTTPException(404, f"No game with id {game_id!r}")
    return managed


@app.get("/")
def root():
    return {
        "name": "Singapore Mahjong AI",
        "version": mahjong.__version__,
        "bot_types": sorted(BOT_TYPES),
    }


@app.get("/tiles")
def tiles():
    """Static tile metadata so the frontend never hardcodes ids."""
    return {
        tile_id: {
            "short": tile_short(tile_id),
            "name": tile_name(tile_id),
            "suit": suit_of(tile_id).name.lower(),
            "rank": rank_of(tile_id),
        }
        for tile_id in range(NUM_TOTAL_UNIQUE)
    }


@app.post("/games")
async def create_game(req: CreateGameRequest):
    try:
        managed = manager.create(seed=req.seed, human_seat=req.human_seat,
                                 bots=req.bots, tai_cap=req.tai_cap,
                                 base_unit=req.base_unit)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"game_id": managed.game_id, "view": managed.view()}


@app.get("/games/{game_id}")
async def get_view(game_id: str):
    return _get_or_404(game_id).view()


@app.post("/games/{game_id}/action")
async def submit_action(game_id: str, action: ActionRequest):
    managed = _get_or_404(game_id)
    async with managed.lock:
        interactive = managed.interactive
        if interactive.pending is None:
            raise HTTPException(409, "No pending decision (game may be over)")
        answer = _coerce_answer(interactive.pending, action.answer)
        try:
            interactive.submit(answer)
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        await managed.broadcast()
        return managed.view()


@app.post("/games/{game_id}/next-hand")
async def next_hand(game_id: str):
    """Advance the session: apply the finished hand's payments to the
    running scores, move the dealership per table rules (dealer repeats
    on a win or draw), and deal the next wall — or mark the session
    over after North 4."""
    managed = _get_or_404(game_id)
    async with managed.lock:
        try:
            managed.next_hand()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        await managed.broadcast()
        return managed.view()


@app.get("/games/{game_id}/hint")
async def hint(game_id: str):
    """What would the agent seated at the human's position do?

    For claim windows the payload carries claim_context: engine-computed
    facts about what taking the claim does to the hand's tai potential
    (forfeits ping hu, drops below the minimum, ...). The UI shows these
    verdict-independently — the engine is the rulebook even when the
    model is the advisor.
    """
    managed = _get_or_404(game_id)
    interactive = managed.interactive
    if interactive.pending is None:
        raise HTTPException(409, "No pending decision")
    pending = interactive.pending
    suggestion = interactive.game.dispatch_to_agent(pending)
    payload = {
        "pending": interactive.view_for(managed.human_seat)["pending"],
        "suggestion": list(suggestion) if isinstance(suggestion, tuple)
                      else suggestion,
    }
    ctx = _claim_context(interactive.game, pending, suggestion)
    if ctx is not None:
        payload["claim_context"] = ctx
    return payload


def _claim_context(game, request, suggestion):
    """Engine tai facts for a claim/chow window (None otherwise)."""
    if isinstance(request, ClaimRequest):
        partners = None
        claim_type = request.claim_type
    elif isinstance(request, ChowRequest):
        claim_type = "chow"
        partners = (tuple(suggestion) if suggestion
                    else tuple(request.options[0]))
    else:
        return None
    hand = game.hands[request.player]
    return claim_consequence(hand, game.seat_index(request.player),
                             game.prevailing_wind, game.score_config,
                             request.tile_id, claim_type, partners=partners)


@app.get("/games/{game_id}/analysis")
async def analysis(game_id: str):
    managed = _get_or_404(game_id)
    interactive = managed.interactive
    # If the human is choosing a discard, ask their seat's agent what
    # it would throw and star that tile — the coach explains this same
    # pick, so advisor and coach agree by construction.
    agent_pick = None
    pending = interactive.pending
    if (isinstance(pending, DiscardRequest)
            and pending.player == managed.human_seat):
        agent_pick = interactive.game.dispatch_to_agent(pending)
    return analyze_seat(interactive.game, managed.human_seat,
                        agent_pick=agent_pick)


def _suggestion_text(request, suggestion, game=None) -> str:
    """Render the learned advisor's choice as one plain sentence —
    with the engine's tai consequence attached for claim windows."""
    if isinstance(request, DiscardRequest):
        return f"The trained advisor would discard {tile_name(suggestion)}."
    if isinstance(request, ClaimRequest) or isinstance(request, ChowRequest):
        if isinstance(request, ClaimRequest):
            verb = f"take the {request.claim_type} of {tile_name(request.tile_id)}"
            noun = f"the {request.claim_type} of {tile_name(request.tile_id)}"
            if suggestion:
                text = f"The trained advisor would {verb}."
            else:
                text = f"The trained advisor would pass on {noun}."
        else:
            if suggestion:
                a, b = suggestion
                text = (f"The trained advisor would chow the "
                        f"{tile_name(request.tile_id)} with {tile_name(a)} "
                        f"and {tile_name(b)}.")
            else:
                text = (f"The trained advisor would pass on chowing the "
                        f"{tile_name(request.tile_id)}.")
        if game is not None:
            ctx = _claim_context(game, request, suggestion)
            if ctx and ctx.get("headline"):
                text += f" {ctx['headline']}"
        return text
    if isinstance(request, KongRequest):
        if suggestion:
            kind, tile = suggestion
            return (f"The trained advisor would declare the {kind} kong "
                    f"of {tile_name(tile)}.")
        return "The trained advisor would hold off on declaring a kong."
    return ""


@app.post("/games/{game_id}/explain")
async def explain(game_id: str):
    """The coach: engine numbers + retrieved principles → a short lesson.

    Uses the Anthropic API when ANTHROPIC_API_KEY is set (server-side
    env only — the key never reaches the browser); otherwise renders a
    deterministic template over the same numbers. Cached per decision
    state so repeat clicks never re-bill.
    """
    managed = _get_or_404(game_id)
    interactive = managed.interactive
    if interactive.pending is None:
        raise HTTPException(409, "No pending decision")

    pending_view = interactive.view_for(managed.human_seat)["pending"]
    key = f"{interactive.game.turn}:{sorted(pending_view.items())!r}"
    cached = managed.explain_cache
    if cached.get("key") == key:
        return cached["payload"]

    suggestion = interactive.game.dispatch_to_agent(interactive.pending)
    text = _suggestion_text(interactive.pending, suggestion,
                            game=interactive.game)
    payload = await explain_situation(
        interactive.game, managed.human_seat, pending_view, text)
    managed.explain_cache = {"key": key, "payload": payload}
    return payload


@app.delete("/games/{game_id}")
async def delete_game(game_id: str):
    if not manager.remove(game_id):
        raise HTTPException(404, f"No game with id {game_id!r}")
    return {"deleted": game_id}


@app.websocket("/games/{game_id}/ws")
async def game_websocket(websocket: WebSocket, game_id: str):
    managed = manager.get(game_id)
    if managed is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    managed.sockets.append(websocket)
    try:
        await websocket.send_json(managed.view())
        while True:
            await websocket.receive_text()  # keepalive pings; content ignored
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in managed.sockets:
            managed.sockets.remove(websocket)


def _coerce_answer(pending, answer):
    """Convert a JSON answer into what the engine's request expects."""
    if isinstance(pending, DiscardRequest):
        if isinstance(answer, bool) or not isinstance(answer, int):
            raise HTTPException(400, "Discard answer must be a tile id (int)")
        return answer
    if isinstance(pending, ClaimRequest):
        if not isinstance(answer, bool):
            raise HTTPException(400, "Claim answer must be true or false")
        return answer
    if isinstance(pending, (ChowRequest, KongRequest)):
        if answer is None:
            return None
        if isinstance(answer, list) and len(answer) == 2:
            return tuple(answer)
        raise HTTPException(400, "Answer must be null (pass) or a 2-item option")
    raise HTTPException(500, f"Unhandled pending type {type(pending).__name__}")
