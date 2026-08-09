"""Build the situation, retrieve principles, and produce the lesson.

The LLM's contract is narrow on purpose: it receives the engine's
numbers and the retrieved principles, and its ONLY job is prose. The
system prompt forbids inventing numbers; the recommendation it explains
is the learned agent's, computed by the engine before the LLM is ever
involved. With no ANTHROPIC_API_KEY in the environment (or on any API
failure), a deterministic template renders the same content — the coach
degrades to "less fluent", never to "wrong" or "broken".

The key is read server-side from the environment at call time and never
leaves this process. For a future multi-user deployment the call site
is the single place to swap in per-user billing.
"""

import json
import os
from typing import Dict, List, Optional

import httpx

from mahjong.tiles import (
    NUM_STANDARD_UNIQUE, is_honor, suit_of, tile_name,
)
from mahjong.opponent_model import estimate_opponent_threats
from mahjong.server.analysis import analyze_seat
from mahjong.coach.retrieve import retrieve

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = """You are the table coach for a Singapore mahjong training app. You explain one decision to the player, using ONLY the numbers in the SITUATION JSON — never invent probabilities, values, or tile facts. The engine's recommendation is given; explain the trade-off behind it, including what the tempting alternative costs. You advise the player; you do not defend the bot.

House rules at this table (already reflected in the numbers): tai cap 6, minimum 1 tai to win (no chicken hands), self-draw adds no tai (it only makes all three opponents pay), shooter pays all three shares on a ron, ping hu 4 tai only with zero bonus tiles (with any flower/animal it is chou ping hu, 1 tai; +1 if concealed), each seat flower or animal 1 tai, all four animals +1.

Reply in at most 110 words of plain second-person prose. No headers, no lists. Weave in at most two numbers from the situation. End with: Principle: <the one retrieved principle title that best fits>."""


# ── Situation building (engine numbers only) ─────────────────────────

def build_situation(game, seat: int, pending: Optional[Dict],
                    suggestion_text: Optional[str]) -> Dict:
    hand = game.hands[seat]
    analysis = analyze_seat(game, seat)
    threat_data = estimate_opponent_threats(seat, game)

    counts = hand.copy_counts()
    exposed_tiles = [t for _, tiles in hand.exposed for t in tiles]
    total = sum(counts) + len(exposed_tiles)
    suit_tot = {}
    honors = 0
    for t in range(NUM_STANDARD_UNIQUE):
        n = counts[t] + sum(1 for e in exposed_tiles if e == t)
        if n == 0:
            continue
        if is_honor(t):
            honors += n
        else:
            suit_tot[suit_of(t)] = suit_tot.get(suit_of(t), 0) + n
    biggest = max(suit_tot.values()) if suit_tot else 0

    discards = analysis.get("discards", [])[:4]
    top = [{
        "discard": tile_name(d["tile"]),
        "deal_in_prob": d.get("deal_in_prob"),
        "hand_value_pts": d.get("hand_value"),
        "win_prob": d.get("win_prob"),
        "shanten_after": d["shanten_after"],
        "improving_tiles": d["acceptance"],
    } for d in discards]

    opp_melds = sum(game.hands[p].num_exposed_melds
                    for p in range(4) if p != seat)
    situation = {
        "pending_type": pending.get("type") if pending else None,
        "pending": pending,
        "hand": [tile_name(t) for t in sorted(hand.tiles)],
        "exposed_melds": [[kind, [tile_name(t) for t in tiles]]
                          for kind, tiles in hand.exposed],
        "bonus_tiles_held": len(hand.flowers),
        "shanten": analysis["shanten"],
        "waiting_on": [tile_name(t)
                       for t in analysis.get("waiting_on", [])],
        "top_discards": top,
        "engine_recommendation": suggestion_text,
        "opponents": analysis["opponents"],
        "turn": game.turn,
        "tiles_remaining": game.tiles_remaining,
        # derived flags (retrieval + prompt context)
        "is_concealed": hand.num_exposed_melds == 0,
        "has_bonus_tiles": bool(hand.flowers),
        "flush_track": total > 0 and (biggest + honors) / total >= 0.7,
        "chow_shape": sum(1 for k, _ in hand.exposed if k == "chow") >= 2,
        "hand_value": max((d.get("hand_value") or 0.0 for d in discards),
                          default=None) if discards else None,
        "max_deal_in_prob": max((d.get("deal_in_prob") or 0.0
                                 for d in discards), default=0.0),
        "max_opp_threat": threat_data["max_threat"],
        "opp_melds": opp_melds,
        "turn_frac": min(1.0, game.turn / 60.0),
    }
    return situation


# ── Rendering ────────────────────────────────────────────────────────

def fallback_text(situation: Dict, principles: List[Dict]) -> str:
    """Deterministic lesson from the same inputs — used with no API key
    and on any API failure."""
    parts = []
    rec = situation.get("engine_recommendation")
    if rec:
        parts.append(rec)
    top = situation.get("top_discards") or []
    if situation.get("pending_type") == "discard" and top:
        # top_discards ranks by pure efficiency; the recommendation is
        # the EV pick — they can differ, so label this one honestly.
        best = top[0]
        line = (f"Fastest line: discard {best['discard']} — "
                f"{best['improving_tiles']} tiles improve you")
        if best.get("deal_in_prob") is not None:
            line += f", deal-in risk {100 * best['deal_in_prob']:.1f}%"
        if best.get("hand_value_pts") is not None:
            line += f", hand worth about {best['hand_value_pts']:+.1f} pts"
        parts.append(line + ".")
        risky = max(top, key=lambda d: d.get("deal_in_prob") or 0.0)
        if risky is not best and (risky.get("deal_in_prob") or 0) >= 0.03:
            parts.append(f"Watch {risky['discard']}: "
                         f"{100 * risky['deal_in_prob']:.1f}% deal-in risk.")
    if situation.get("shanten") == 0 and situation.get("waiting_on"):
        parts.append("Waiting on: " + ", ".join(situation["waiting_on"]) + ".")
    if principles:
        parts.append(f"Principle: {principles[0]['title']} — "
                     f"{principles[0]['text'].split('. ')[0]}.")
    return " ".join(parts) or "No decision to explain right now."


async def llm_text(situation: Dict, principles: List[Dict],
                   api_key: str) -> str:
    prompt = (
        "SITUATION:\n" + json.dumps(situation, indent=1) +
        "\n\nRETRIEVED PRINCIPLES:\n" +
        "\n".join(f"- {c['title']}: {c['text']}" for c in principles))
    async with httpx.AsyncClient(timeout=25.0) as client:
        resp = await client.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json={
                "model": os.environ.get("COACH_MODEL", DEFAULT_MODEL),
                "max_tokens": 300,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            })
        resp.raise_for_status()
        data = resp.json()
    return "".join(block.get("text", "") for block in data["content"]).strip()


async def explain_situation(game, seat: int, pending: Optional[Dict],
                            suggestion_text: Optional[str]) -> Dict:
    """The full pipeline: situation → retrieval → prose (LLM or template)."""
    situation = build_situation(game, seat, pending, suggestion_text)
    principles = retrieve(situation)
    payload = {
        "principles": [c["title"] for c in principles],
        "recommendation": suggestion_text,
    }
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if api_key:
        try:
            payload["text"] = await llm_text(situation, principles, api_key)
            payload["source"] = "claude"
            payload["model"] = os.environ.get("COACH_MODEL", DEFAULT_MODEL)
            return payload
        except Exception:
            pass  # degrade to the template, never to an error
    payload["text"] = fallback_text(situation, principles)
    payload["source"] = "template"
    return payload
