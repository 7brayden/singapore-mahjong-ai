"""Score candidate coach models on the contract, not on vibes.

    COACH_BASE_URL=http://localhost:11434/v1 \
      PYTHONPATH=src python3 experiments/coach_bakeoff.py llama3.2 qwen2.5:7b

Does a candidate coach model respect the contract?

The coach's whole premise is that the engine owns the numbers and the
LLM only narrates them. A model that invents figures breaks that
guarantee invisibly — a learner cannot tell a real probability from a
fabricated one. This scores candidates on the contract, not on vibes.
"""
import asyncio, json, os, re, sys, time

from mahjong.game import GameState, advance_turns
from mahjong.agents import GreedyAgent
from mahjong.coach.explain import (
    SYSTEM_PROMPT, build_situation, _build_prompt, _chat_text,
    openai_compatible_client,
)
from mahjong.coach.retrieve import retrieve

MODELS = sys.argv[1:] or ["llama3.2", "qwen2.5:7b"]
SEEDS = [21, 44, 77]


def numbers_in(text: str):
    return {float(x) for x in re.findall(r"\d+\.?\d*", text)}


def allowed_numbers(situation: dict):
    """Every figure the model is entitled to quote, plus the readable
    forms of each (a 0.0154 probability may legitimately appear as
    1.5% or 2%)."""
    ok = set()

    def add(v):
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return
        ok.update({float(v), round(v, 1), round(v, 2), float(round(v))})
        ok.update({round(v * 100, 1), float(round(v * 100)), round(v * 100, 2)})

    def walk(o):
        if isinstance(o, dict):
            for x in o.values():
                walk(x)
        elif isinstance(o, list):
            for x in o:
                walk(x)
        else:
            add(o)
    walk(situation)
    # Tile ranks (1-9) appear inside every tile name; small integers are
    # structural facts (13 tiles, 4 melds, 3 opponents). Neither is a
    # fabricated statistic — flagging them would bury the real signal.
    ok.update(float(i) for i in range(0, 15))
    return ok


def build_cases():
    cases = []
    for seed in SEEDS:
        g = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=seed)
        g.setup()
        advance_turns(g, 30)
        g._deal_tile_to(0)
        s = build_situation(g, 0, {"type": "discard"},
                            "The trained advisor would discard 5 Wan.")
        cases.append((s, retrieve(s)))
    return cases


async def run(model, cases):
    os.environ["COACH_MODEL"] = model
    client = openai_compatible_client()
    rows = []
    for situation, principles in cases:
        titles = [c["title"] for c in principles]
        t = time.time()
        try:
            text = await _chat_text(client, model, _build_prompt(situation, principles))
        except Exception as e:
            rows.append({"err": f"{type(e).__name__}: {e}"})
            continue
        invented = sorted(numbers_in(text) - allowed_numbers(situation))
        rows.append({
            "text": text,
            "secs": time.time() - t,
            "words": len(text.split()),
            "principle_ok": any(text.rstrip(". ").endswith(t_) for t_ in titles),
            "invented": invented,
        })
    return rows


cases = build_cases()
for model in MODELS:
    rows = asyncio.run(run(model, cases))
    ok = [r for r in rows if "err" not in r]
    print("=" * 72)
    print(f"MODEL: {model}")
    if not ok:
        print("  all calls failed:", rows[0].get("err"))
        continue
    print(f"  median latency   {sorted(r['secs'] for r in ok)[len(ok)//2]:.1f}s")
    print(f"  mean words       {sum(r['words'] for r in ok)/len(ok):.0f}  (contract: <=110)")
    print(f"  principle quoted {sum(r['principle_ok'] for r in ok)}/{len(ok)} verbatim")
    bad = sum(1 for r in ok if r["invented"])
    print(f"  INVENTED NUMBERS {bad}/{len(ok)} responses"
          + (f"  {[r['invented'][:4] for r in ok if r['invented']]}" if bad else ""))
    for r in ok:
        if not r["principle_ok"]:
            tail = r["text"].rstrip(". ").split("Principle:")[-1].strip()
            print(f"  bad principle    -> {tail[:70]!r}")
    print(f"\n  --- sample ---\n  {ok[0]['text'][:600]}\n")
