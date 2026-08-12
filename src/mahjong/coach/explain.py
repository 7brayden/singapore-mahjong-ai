"""Build the situation, retrieve principles, and produce the lesson.

The LLM's contract is narrow on purpose: it receives the engine's
numbers and the retrieved principles, and its ONLY job is prose. The
system prompt forbids inventing numbers; the recommendation it explains
is the learned agent's, computed by the engine before the LLM is ever
involved.

Three backends, selected from the environment by _provider():

    azure      Azure OpenAI chat completions (deployment-addressed)
    anthropic  Anthropic Messages API via the official SDK
    template   deterministic prose over the same numbers

The template is not an error path — with no credentials, or on any API
failure, the coach degrades to "less fluent", never to "wrong" or
"broken". Provider SDKs are imported lazily, so neither is a hard
dependency of the server.

Credentials are read server-side from the environment at call time and
never leave this process; the browser only ever receives rendered text.
For a future multi-user deployment, the two _*_text functions are the
single place to swap in per-user billing.
"""

import json
import os
from typing import Dict, List, Optional

from mahjong.tiles import (
    NUM_STANDARD_UNIQUE, is_honor, suit_of, tile_name,
)
from mahjong.opponent_model import estimate_opponent_threats
from mahjong.server.analysis import analyze_seat
from mahjong.coach.retrieve import retrieve

# Azure addresses a *deployment you named*, not a public model id.
DEFAULT_AZURE_API_VERSION = "2024-10-21"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
MAX_OUTPUT_TOKENS = 2000

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


# ── Providers ────────────────────────────────────────────────────────

def _provider() -> str:
    """Which backend to use, decided from the environment.

    COACH_PROVIDER pins it explicitly (azure | anthropic | template);
    otherwise the first configured credential wins, and "template" is
    the always-available floor.
    """
    pinned = os.environ.get("COACH_PROVIDER", "").strip().lower()
    if pinned:
        return pinned
    if os.environ.get("AZURE_OPENAI_API_KEY", "").strip():
        return "azure"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    return "template"


def _build_prompt(situation: Dict, principles: List[Dict]) -> str:
    return (
        "SITUATION:\n" + json.dumps(situation, indent=1) +
        "\n\nRETRIEVED PRINCIPLES:\n" +
        "\n".join(f"- {c['title']}: {c['text']}" for c in principles))


async def _azure_text(prompt: str) -> str:
    """Azure OpenAI chat completions.

    `model` here is the DEPLOYMENT NAME you chose in the Azure portal,
    not a public model id — the most common source of 404s.
    """
    from openai import AsyncAzureOpenAI  # lazy: optional dependency

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is not set")

    client = AsyncAzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),
        api_key=os.environ["AZURE_OPENAI_API_KEY"].strip(),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION",
                                   DEFAULT_AZURE_API_VERSION).strip(),
        timeout=25.0,
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    try:
        resp = await client.chat.completions.create(
            model=deployment, messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS)
    except Exception as exc:
        # Reasoning-family deployments reject max_tokens and demand
        # max_completion_tokens. The error is confusing enough that it
        # is worth retrying rather than surfacing as a dead coach.
        if "max_completion_tokens" not in str(exc):
            raise
        resp = await client.chat.completions.create(
            model=deployment, messages=messages,
            max_completion_tokens=MAX_OUTPUT_TOKENS)
    return (resp.choices[0].message.content or "").strip()


async def _anthropic_text(prompt: str) -> str:
    """Anthropic Messages API via the official SDK.

    Server-side fallbacks are on: if safety classifiers decline the
    request, the API re-runs it on the recommended model in the same
    call rather than handing back a refusal. A chain that still refuses
    raises, and the caller degrades to the template.
    """
    from anthropic import AsyncAnthropic  # lazy: optional dependency

    client = AsyncAnthropic(timeout=25.0)
    resp = await client.beta.messages.create(
        model=os.environ.get("COACH_MODEL", DEFAULT_ANTHROPIC_MODEL),
        max_tokens=MAX_OUTPUT_TOKENS,
        # Effort low: the model is narrating numbers the engine already
        # computed, not solving the position. max_tokens covers thinking
        # and text together, hence the headroom above.
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("coach request was declined")
    return "".join(block.text for block in resp.content
                   if block.type == "text").strip()


async def explain_situation(game, seat: int, pending: Optional[Dict],
                            suggestion_text: Optional[str]) -> Dict:
    """The full pipeline: situation → retrieval → prose (LLM or template)."""
    situation = build_situation(game, seat, pending, suggestion_text)
    principles = retrieve(situation)
    payload = {
        "principles": [c["title"] for c in principles],
        "recommendation": suggestion_text,
    }

    provider = _provider()
    if provider in ("azure", "anthropic"):
        try:
            prompt = _build_prompt(situation, principles)
            if provider == "azure":
                payload["text"] = await _azure_text(prompt)
                payload["model"] = os.environ.get(
                    "AZURE_OPENAI_DEPLOYMENT", "")
            else:
                payload["text"] = await _anthropic_text(prompt)
                payload["model"] = os.environ.get(
                    "COACH_MODEL", DEFAULT_ANTHROPIC_MODEL)
            payload["source"] = provider
            return payload
        except Exception:
            pass  # degrade to the template, never to an error

    payload["text"] = fallback_text(situation, principles)
    payload["source"] = "template"
    return payload
