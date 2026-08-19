"""Build the situation, retrieve principles, and produce the lesson.

The LLM's contract is narrow on purpose: it receives the engine's
numbers and the retrieved principles, and its ONLY job is prose. The
system prompt forbids inventing numbers; the recommendation it explains
is the learned agent's, computed by the engine before the LLM is ever
involved.

Four backends, selected from the environment by _provider():

    openai     any OpenAI-compatible endpoint named by COACH_BASE_URL —
               OpenAI, Ollama, LM Studio, vLLM, OpenRouter, a gateway
    azure      Azure OpenAI (deployment-addressed; both endpoint shapes)
    anthropic  Anthropic Messages API via the official SDK
    template   deterministic prose over the same numbers

The generic `openai` provider exists because vendors disappear: an
Azure lab sandbox was deleted along with its DNS record, and GitHub
Models entered retirement, both within a week of each other. Which
endpoint answers is configuration, not code.

The template is not an error path — with no credentials, or on any API
failure, the coach degrades to "less fluent", never to "wrong" or
"broken". Provider SDKs are imported lazily, so none is a hard
dependency of the server.

Credentials are read server-side from the environment at call time and
never leave this process; the browser only ever receives rendered text.
For a future multi-user deployment, the _*_text functions are the
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
from mahjong.ml.features import _bonus_tai

# Azure addresses a *deployment you named*, not a public model id.
DEFAULT_AZURE_API_VERSION = "2024-10-21"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
MAX_OUTPUT_TOKENS = 2000

SYSTEM_PROMPT = """You are the table coach for a Singapore mahjong training app. You explain one decision to the player, using ONLY the numbers in the SITUATION JSON — never invent probabilities, values, or tile facts. The engine's recommendation is given; explain the trade-off behind it, including what the tempting alternative costs. You advise the player; you do not defend the bot.

House rules at this table (already reflected in the numbers): tai cap 6, minimum 1 tai to win (no chicken hands), self-draw adds no tai (it only makes all three opponents pay), shooter pays all three shares on a ron, ping hu 4 tai only with zero bonus tiles AND a wait of two or more tiles when winning by discard (a single wait earns ping hu only on self-draw; with any flower/animal the shape is chou ping hu, 1 tai), 门清 +1 for a fully concealed hand won by self-draw, each seat flower or animal 1 tai, all four animals +1.

The situation's tai_context lines come from the scoring engine and are authoritative for THIS hand: they state which tai tracks are live, degraded, or dead right now. Never advise pursuing a track tai_context marks unavailable — in particular, if it says ping hu is degraded to chou ping hu, do not present ping hu (4 tai) as this hand's goal.

Reply in at most 110 words of plain second-person prose. No headers, no lists. Weave in at most two numbers from the situation. End with: Principle: <the one retrieved principle title that best fits>."""


# ── Situation building (engine numbers only) ─────────────────────────

def _tai_context(hand, seat_index: int) -> List[str]:
    """Authoritative per-hand scoring facts, stated by the engine.

    Exists because of an observed coaching failure: with a flower in
    hand the LLM kept presenting ping hu (4 tai) as the goal, even
    though this table degrades any all-chows hand holding bonus tiles
    to chou ping hu (1 tai). The connection between has_bonus_tiles and
    the ping hu track is exactly the kind of inference a small model
    fumbles — so the engine states it outright instead of hoping.
    """
    notes = []
    flowers = hand.flowers
    has_pong_meld = any(kind != "chow" for kind, _ in hand.exposed)
    concealed = hand.num_exposed_melds == 0

    if flowers:
        banked = _bonus_tai(flowers, seat_index)
        notes.append(
            f"You hold {len(flowers)} bonus tile(s) banking {banked} tai. "
            "Ping hu (4 tai) is OFF the table this hand: an all-chows "
            "hand now scores chou ping hu (1 tai).")
    if has_pong_meld:
        notes.append(
            "An exposed pong/kong means this hand can never be all chows — "
            "the ping hu / chou ping hu track is dead; value must come from "
            "triplets, flush, honors, or bonus tiles.")
    elif not flowers:
        notes.append(
            "Clean ping hu (4 tai) is still live — it dies the moment you "
            "draw any flower/animal. To win it by discard you must end on "
            "a wait of two or more tiles; a single wait only scores ping "
            "hu on self-draw.")
    if concealed:
        notes.append(
            "Fully concealed: winning by SELF-DRAW adds 门清 (+1 tai). "
            "Claiming any pong or chow forfeits it.")
    return notes


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
        "tai_context": _tai_context(hand, game.seat_index(seat)),
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
    tai_notes = situation.get("tai_context") or []
    if tai_notes:
        parts.append(tai_notes[0])
    if situation.get("shanten") == 0 and situation.get("waiting_on"):
        parts.append("Waiting on: " + ", ".join(situation["waiting_on"]) + ".")
    if principles:
        parts.append(f"Principle: {principles[0]['title']} — "
                     f"{principles[0]['text'].split('. ')[0]}.")
    return " ".join(parts) or "No decision to explain right now."


# ── Providers ────────────────────────────────────────────────────────

def _provider() -> str:
    """Which backend to use, decided from the environment.

    COACH_PROVIDER pins it explicitly (openai | azure | anthropic |
    template); otherwise the first configured credential wins, and
    "template" is the always-available floor.
    """
    pinned = os.environ.get("COACH_PROVIDER", "").strip().lower()
    if pinned:
        return pinned
    if os.environ.get("COACH_BASE_URL", "").strip():
        return "openai"
    if os.environ.get("AZURE_OPENAI_API_KEY", "").strip():
        return "azure"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    return "template"


def openai_compatible_client():
    """Client for any endpoint speaking the OpenAI chat-completions API.

    Deliberately vendor-neutral. Two backends have already died under
    this project — an Azure lab sandbox deleted with its DNS record, and
    GitHub Models entering retirement — so the endpoint is configuration,
    not code. Point COACH_BASE_URL at OpenAI, Ollama, LM Studio, vLLM,
    OpenRouter, or a gateway; nothing here needs to change.
    """
    base = os.environ.get("COACH_BASE_URL", "").strip().rstrip("/")
    if not base:
        raise RuntimeError("COACH_BASE_URL is not set")
    # Local servers ignore the key, but the SDK insists on a non-empty one.
    key = os.environ.get("COACH_API_KEY", "").strip() or "not-needed"
    from openai import AsyncOpenAI  # lazy: optional dependency
    return AsyncOpenAI(base_url=base, api_key=key, timeout=25.0)


def _build_prompt(situation: Dict, principles: List[Dict]) -> str:
    return (
        "SITUATION:\n" + json.dumps(situation, indent=1) +
        "\n\nRETRIEVED PRINCIPLES:\n" +
        "\n".join(f"- {c['title']}: {c['text']}" for c in principles))


def azure_client():
    """Build a client for whichever Azure surface the endpoint names.

    Azure ships two, and the URL says which:

      classic    https://<resource>.openai.azure.com
                 deployment-addressed, needs an api-version
      v1 compat  https://<resource>.services.ai.azure.com/openai/v1
                 OpenAI-compatible, plain client, no api-version

    Newer Azure AI Foundry resources hand out the second shape, so the
    same credentials that work in one portal fail in the other unless
    the client is chosen from the URL. Returns (client, surface).
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
    if not endpoint or not key:
        raise RuntimeError(
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are both required")

    if endpoint.endswith("/openai/v1") or ".services.ai.azure.com" in endpoint:
        from openai import AsyncOpenAI  # lazy: optional dependency
        base = (endpoint if endpoint.endswith("/openai/v1")
                else f"{endpoint}/openai/v1")
        return AsyncOpenAI(base_url=base, api_key=key, timeout=25.0), "v1"

    from openai import AsyncAzureOpenAI  # lazy: optional dependency
    return AsyncAzureOpenAI(
        azure_endpoint=endpoint,
        api_key=key,
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION",
                                   DEFAULT_AZURE_API_VERSION).strip(),
        timeout=25.0,
    ), "classic"


async def _chat_text(client, model: str, prompt: str) -> str:
    """One OpenAI-compatible chat completion.

    Reasoning-family models reject max_tokens and demand
    max_completion_tokens; the error is confusing enough to be worth
    retrying rather than surfacing as a dead coach.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    # Low temperature on purpose: the coach narrates numbers the engine
    # already computed. Sampling variety only invites embellishment.
    kwargs = {"model": model, "messages": messages,
              "max_tokens": MAX_OUTPUT_TOKENS, "temperature": 0.2}
    for _ in range(3):
        try:
            resp = await client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            text = str(exc)
            if "max_completion_tokens" in text and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            elif "temperature" in text and "temperature" in kwargs:
                # Reasoning-family models reject non-default temperature.
                kwargs.pop("temperature")
            else:
                raise
    raise RuntimeError("chat completion failed after parameter fallbacks")


async def _openai_text(prompt: str) -> str:
    """Any OpenAI-compatible endpoint named by COACH_BASE_URL."""
    model = os.environ.get("COACH_MODEL", "").strip()
    if not model:
        raise RuntimeError("COACH_MODEL is not set (the model this endpoint serves)")
    return await _chat_text(openai_compatible_client(), model, prompt)


async def _azure_text(prompt: str) -> str:
    """Azure OpenAI chat completions.

    `model` here is the DEPLOYMENT NAME you chose in the Azure portal,
    not a public model id — the most common source of 404s.
    """
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
    if not deployment:
        raise RuntimeError("AZURE_OPENAI_DEPLOYMENT is not set")
    client, _surface = azure_client()
    return await _chat_text(client, deployment, prompt)


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
    if provider in ("openai", "azure", "anthropic"):
        try:
            prompt = _build_prompt(situation, principles)
            if provider == "openai":
                payload["text"] = await _openai_text(prompt)
                payload["model"] = os.environ.get("COACH_MODEL", "")
            elif provider == "azure":
                payload["text"] = await _azure_text(prompt)
                payload["model"] = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
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
