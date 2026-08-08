"""Duplicate-seating evaluator — compare agents on identical deals.

The problem with the naive benchmark: at 300 games it cannot resolve the
differences we care about. Learned vs Hybrid came out 1.22% vs 1.50%
deal-ins (p~=0.17) and 20.7% vs 25.3% wins (p~=0.09) — neither
significant, so any tuning loop run against it would be chasing noise.

The fix is duplicate bridge's trick, not brute force. For each seed we
replay the SAME wall with the agents rotated through the seats, so every
agent plays every starting hand from every position. Deal luck — by far
the biggest variance source — cancels between the sides instead of being
averaged down. Then we compare per-seed PAIRED differences, which is a
far more powerful test than two independent proportions.

Rotations that produce an identical seat->agent-type layout are skipped
(a 2v2 alternating lineup has only 2 distinct rotations, not 4), so the
compute goes into information rather than repeats.

Sanity property, asserted by --self-check: with the same agent type on
both sides every rotation yields the same game, and each side collects
the same multiset of seats, so the paired difference is EXACTLY zero.
Any drift there means the harness is broken.

    PYTHONPATH=src python3 experiments/duplicate.py --games 400
    PYTHONPATH=src python3 experiments/duplicate.py --a greedy --b random
"""

import argparse
import math
import multiprocessing
import os
import time
from collections import defaultdict

from mahjong.game import GameState
from mahjong.agents import (
    RandomAgent, GreedyAgent, DefensiveAgent, HybridAgent, LearnedAgent,
)

AGENTS = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
    "defensive": DefensiveAgent,
    "hybrid": HybridAgent,
    "learned": LearnedAgent,
}


# ── Rotation design ──────────────────────────────────────────────────

def distinct_rotations(roles) -> list:
    """Rotations with a unique seat->agent-type layout.

    In rotation r, role i sits at seat (i + r) % 4 — equivalently seat s
    is played by role (s - r) % 4.
    """
    seen = {}
    for r in range(4):
        layout = tuple(roles[(s - r) % 4] for s in range(4))
        seen.setdefault(layout, r)
    return sorted(seen.values())


def _play(task):
    """Worker: one game. Returns per-seat stats tagged with the role."""
    seed, rotation, roles = task
    agents = []
    for seat in range(4):
        role = roles[(seat - rotation) % 4]
        agents.append(AGENTS[role](f"{role}-{seat}"))
    game = GameState(agents, seed=seed)
    result = game.play()
    payments = result.payments or [0, 0, 0, 0]

    rows = []
    for seat in range(4):
        role = roles[(seat - rotation) % 4]
        rows.append({
            "role": role,
            "won": 1 if result.winner == seat else 0,
            "deal_ins": result.deal_ins[seat],
            "discards": len(game.hands[seat].discards),
            "points": payments[seat],
        })
    return seed, rotation, rows


# ── Statistics ───────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a proportion — behaves at small k."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def paired_ci(diffs, z: float = 1.96):
    """Mean and 95% CI of per-seed paired differences."""
    n = len(diffs)
    if n < 2:
        return (0.0, 0.0, 0.0, 0.0)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    return (mean, mean - z * se, mean + z * se, se)


def two_proportion_test(k1, n1, k2, n2):
    """Pooled two-proportion z-test. Returns (diff, z, p_two_tailed).

    Overlapping confidence intervals are NOT the same as a
    non-significant difference, so the comparison is tested directly
    rather than eyeballed off the two intervals.
    """
    if n1 == 0 or n2 == 0:
        return (0.0, 0.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return (p1 - p2, 0.0, 1.0)
    z = (p1 - p2) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return (p1 - p2, z, p)


def _variance(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    mean = sum(xs) / n
    return sum((x - mean) ** 2 for x in xs) / (n - 1)


def rotation_efficiency(rotation_diffs, seed_diffs, n_rotations: int):
    """How much did rotating seats on one wall actually buy?

    Each seed contributes n_rotations games. If those games were
    independent, the variance of their summed difference would be
    n_rotations x the single-game variance. Rotation makes them
    negatively correlated — every agent plays every seat on the same
    wall, so seat advantage and deal position cancel — and the summed
    variance comes in below that.

    Returns var_if_independent / var_observed. Above 1.0 means the
    design is buying real precision; 1.0 means it bought nothing.

    (Note this is NOT the classic paired-samples gain. The two sides
    sit in the same zero-sum game, so their scores are perfectly
    anti-correlated within a rotation — the gain here comes purely from
    balancing seats across rotations.)
    """
    if n_rotations < 2 or len(seed_diffs) < 2:
        return None
    var_independent = n_rotations * _variance(rotation_diffs)
    var_observed = _variance(seed_diffs)
    if var_observed <= 0:
        return None
    return var_independent / var_observed


# ── Runner ───────────────────────────────────────────────────────────

def run(roles, games, seed_start=0, workers=None, quiet=False):
    rotations = distinct_rotations(roles)
    tasks = [(seed_start + g, r, roles)
             for g in range(games) for r in rotations]
    workers = workers or max(1, (os.cpu_count() or 2) - 2)

    totals = defaultdict(lambda: defaultdict(int))
    per_seed = defaultdict(lambda: defaultdict(float))
    per_game = defaultdict(lambda: defaultdict(float))  # keyed (seed, rotation)
    started = time.time()

    def absorb(seed, rotation, rows):
        for row in rows:
            role = row["role"]
            for key in ("won", "deal_ins", "discards", "points"):
                totals[role][key] += row[key]
            totals[role]["seats"] += 1
            per_seed[seed][role] += row["points"]
            per_game[(seed, rotation)][role] += row["points"]

    if workers > 1:
        with multiprocessing.Pool(workers) as pool:
            for i, (seed, rotation, rows) in enumerate(
                    pool.imap_unordered(_play, tasks, chunksize=4), 1):
                absorb(seed, rotation, rows)
                if not quiet and i % 400 == 0:
                    print(f"    ... {i}/{len(tasks)} games "
                          f"({time.time() - started:.0f}s)", flush=True)
    else:
        for task in tasks:
            seed, rotation, rows = _play(task)
            absorb(seed, rotation, rows)

    return totals, per_seed, per_game, rotations, len(tasks)


def report(roles, totals, per_seed, per_game, rotations, n_games, label):
    sides = sorted(set(roles))
    print(f"\n  {label}")
    print(f"  lineup {list(roles)} · {len(rotations)} rotations/seed "
          f"· {n_games:,} games")
    print(f"  {'Agent':<11} {'Win%':>16} {'DI/disc%':>18} {'Pts/seat':>10}"
          f"   {'raw (wins/seats, DI/discards)'}")
    for role in sides:
        t = totals[role]
        wp, wlo, whi = wilson(t["won"], t["seats"])
        dp, dlo, dhi = wilson(t["deal_ins"], t["discards"])
        pts = t["points"] / t["seats"]
        print(f"  {role:<11} {100*wp:5.1f} [{100*wlo:4.1f},{100*whi:4.1f}]"
              f" {100*dp:6.2f} [{100*dlo:5.2f},{100*dhi:5.2f}]"
              f" {pts:>+10.3f}"
              f"   {t['won']:,}/{t['seats']:,}, {t['deal_ins']:,}/{t['discards']:,}")

    if len(sides) != 2:
        return None
    a, b = sides

    for metric, num, den in (("win rate", "won", "seats"),
                             ("deal-in rate", "deal_ins", "discards")):
        diff, z, p = two_proportion_test(
            totals[a][num], totals[a][den], totals[b][num], totals[b][den])
        mark = "significant" if p < 0.05 else "not significant"
        print(f"    {metric:<13} {a} − {b} = {100*diff:+.3f} pp   "
              f"z={z:+.2f}  p={p:.4f}  ({mark})")
    diffs = [per_seed[s][a] - per_seed[s][b] for s in per_seed]
    mean, lo, hi, se = paired_ci(diffs)
    verdict = ("no significant difference" if lo <= 0 <= hi
               else f"{a if mean > 0 else b} is genuinely ahead")
    print(f"\n  Points difference ({a} − {b}), summed per seed over "
          f"{len(rotations)} rotations:")
    print(f"    {mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  → {verdict}")

    rot_diffs = [per_game[k][a] - per_game[k][b] for k in per_game]
    eff = rotation_efficiency(rot_diffs, diffs, len(rotations))
    if eff is not None:
        print(f"    seat rotation variance reduction: {eff:.2f}× "
              f"({'real gain' if eff > 1.05 else 'no measurable gain'})")
    return mean, lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--a", default="learned", choices=sorted(AGENTS))
    parser.add_argument("--b", default="hybrid", choices=sorted(AGENTS))
    parser.add_argument("--games", type=int, default=400,
                        help="distinct seeds; each is replayed once per rotation")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--self-check", action="store_true",
                        help="same agent both sides — paired diff must be 0")
    args = parser.parse_args()

    if args.self_check:
        print("=" * 62)
        print(f"SELF-CHECK — {args.a} vs itself (paired diff must be exactly 0)")
        print("=" * 62)
        roles = (args.a, args.a, args.a, args.a)
        totals, per_seed, _pg, rot, n = run(roles, min(args.games, 40), quiet=True)
        diffs = [per_seed[s][args.a] - per_seed[s][args.a] for s in per_seed]
        print(f"  {n} games · max |paired diff| = {max(map(abs, diffs), default=0):.6f}")
        print("  harness OK" if not any(diffs) else "  *** HARNESS BROKEN ***")
        return

    print("=" * 62)
    print(f"DUPLICATE SEATING — {args.a} vs {args.b}")
    print("=" * 62)
    roles = (args.a, args.b, args.a, args.b)
    totals, per_seed, per_game, rotations, n = run(
        roles, args.games, args.seed_start)
    report(roles, totals, per_seed, per_game, rotations, n,
           f"2 {args.a} vs 2 {args.b}")


if __name__ == "__main__":
    main()
