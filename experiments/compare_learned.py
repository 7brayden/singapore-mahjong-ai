"""Benchmark the LearnedAgent against the HybridAgent it replaces.

Same metric conventions as run_benchmark.py: Win% and DI/disc%
(deal-ins as a percentage of total discards — per-decision safety,
not per-game, so fast finishers don't look artificially safe).

    PYTHONPATH=src python3 experiments/compare_learned.py [--games 300]
"""

import argparse
import sys
import time

from mahjong.game import GameState
from mahjong.agents import GreedyAgent, HybridAgent, LearnedAgent


def run_match(make_agents, games, seed_start=0, label=""):
    """Play seeded games; aggregate per-NAME-PREFIX stats with seat
    rotation handled by the caller supplying make_agents(i)."""
    stats = {}
    started = time.time()
    for i in range(games):
        agents = make_agents(i)
        game = GameState(agents, seed=seed_start + i)
        result = game.play()
        for p in range(4):
            key = agents[p].name.rstrip("0123456789-")
            s = stats.setdefault(key, {"games": 0, "wins": 0,
                                       "deal_ins": 0, "discards": 0,
                                       "points": 0})
            s["games"] += 1
            s["wins"] += 1 if result.winner == p else 0
            s["deal_ins"] += result.deal_ins[p]
            s["discards"] += len(game.hands[p].discards)
            s["points"] += (result.payments or [0] * 4)[p]
        if (i + 1) % 100 == 0:
            print(f"    ... {i + 1}/{games} "
                  f"({(time.time() - started):.0f}s)", flush=True)

    print(f"\n  {label} — {games} games")
    print(f"  {'Agent':<10} {'Win%':>7} {'DI/disc%':>9} {'Net pts/game':>13}")
    for key, s in sorted(stats.items()):
        win = 100 * s["wins"] / s["games"]
        di = 100 * s["deal_ins"] / s["discards"] if s["discards"] else 0
        pts = s["points"] / s["games"]
        print(f"  {key:<10} {win:>6.1f} {di:>9.2f} {pts:>+13.2f}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=300)
    args = parser.parse_args()

    print("=" * 60)
    print("LEARNED vs HYBRID")
    print("=" * 60)

    # Mirror: 4x Learned (comparable to the hybrid mirror baseline)
    run_match(lambda i: [LearnedAgent(f"Learned-{j}") for j in range(4)],
              args.games, label="Mirror: 4x Learned")

    # Head-to-head, seats alternated every game so neither side owns
    # the dealer advantage.
    def head_to_head(i):
        if i % 2 == 0:
            return [LearnedAgent("Learned-0"), HybridAgent("Hybrid-0"),
                    LearnedAgent("Learned-1"), HybridAgent("Hybrid-1")]
        return [HybridAgent("Hybrid-0"), LearnedAgent("Learned-0"),
                HybridAgent("Hybrid-1"), LearnedAgent("Learned-1")]

    run_match(head_to_head, args.games, seed_start=10_000,
              label="2 Learned vs 2 Hybrid (seats alternate)")

    # Against a loose table: does better defense punish greedy?
    def vs_greedy(i):
        if i % 2 == 0:
            return [LearnedAgent("Learned-0"), GreedyAgent("Greedy-0"),
                    LearnedAgent("Learned-1"), GreedyAgent("Greedy-1")]
        return [GreedyAgent("Greedy-0"), LearnedAgent("Learned-0"),
                GreedyAgent("Greedy-1"), LearnedAgent("Learned-1")]

    run_match(vs_greedy, args.games, seed_start=20_000,
              label="2 Learned vs 2 Greedy (seats alternate)")


if __name__ == "__main__":
    sys.exit(main())
