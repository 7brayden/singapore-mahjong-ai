"""
Benchmark runner for Singapore Mahjong AI agents.

Runs multiple experiment configurations and produces a results table
comparing agent performance across key metrics:
  - Win rate (total, tsumo, ron)
  - Deal-in rate (how often an agent's discard completes an opponent's hand)
  - Draw rate
  - Average turns to win
  - Average final shanten

Usage:
    python experiments/run_benchmark.py
"""

import sys
import os
import time
from typing import List, Dict, Tuple

# Allow running without installing the package (pip install -e .)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mahjong.game import GameState, GameResult, BaseAgent
from mahjong.agents import (
    RandomAgent, GreedyAgent, DefensiveAgent, HybridAgent
)


# ── Experiment runner ─────────────────────────────────────────────────

def run_experiment(agent_configs: List[Tuple[str, type]],
                   num_games: int = 200,
                   seed_offset: int = 0,
                   verbose: bool = False) -> Dict:
    """Run a batch of games with the given agent configuration."""

    # Per-seat tracking
    wins_by_seat = [0] * 4
    tsumo_by_seat = [0] * 4
    ron_by_seat = [0] * 4
    deal_ins_by_seat = [0] * 4
    discards_by_seat = [0] * 4
    shanten_sums = [0] * 4
    flowers_sums = [0] * 4

    # Global tracking
    draws = 0
    total_turns = 0
    win_turns = []
    decisive_games = 0

    start_time = time.time()

    for i in range(num_games):
        agents = [cls(name) for name, cls in agent_configs]
        game = GameState(agents, seed=seed_offset + i)
        result = game.play()

        total_turns += result.turns

        if result.winner is not None:
            wins_by_seat[result.winner] += 1
            win_turns.append(result.turns)
            decisive_games += 1
            if result.win_type == "tsumo":
                tsumo_by_seat[result.winner] += 1
            elif result.win_type == "ron":
                ron_by_seat[result.winner] += 1
        else:
            draws += 1

        for p in range(4):
            deal_ins_by_seat[p] += result.deal_ins[p]
            discards_by_seat[p] += len(game.hands[p].discards)
            shanten_sums[p] += result.final_shanten[p]
            flowers_sums[p] += result.flowers_collected[p]

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            print(f"    {i+1}/{num_games} games ({elapsed:.1f}s)")

    elapsed = time.time() - start_time

    # Aggregate by agent type
    agent_types = {}
    for seat, (name, cls) in enumerate(agent_configs):
        agent_type = cls.__name__
        if agent_type not in agent_types:
            agent_types[agent_type] = {
                "seats": [],
                "wins": 0,
                "tsumo": 0,
                "ron": 0,
                "deal_ins": 0,
                "discards": 0,
                "shanten_sum": 0,
                "flowers_sum": 0,
                "game_count": 0,
            }
        at = agent_types[agent_type]
        at["seats"].append(seat)
        at["wins"] += wins_by_seat[seat]
        at["tsumo"] += tsumo_by_seat[seat]
        at["ron"] += ron_by_seat[seat]
        at["deal_ins"] += deal_ins_by_seat[seat]
        at["discards"] += discards_by_seat[seat]
        at["shanten_sum"] += shanten_sums[seat]
        at["flowers_sum"] += flowers_sums[seat]
        at["game_count"] += num_games

    return {
        "agent_configs": [(name, cls.__name__) for name, cls in agent_configs],
        "num_games": num_games,
        "draws": draws,
        "decisive_games": decisive_games,
        "total_turns": total_turns,
        "win_turns": win_turns,
        "wins_by_seat": wins_by_seat,
        "agent_types": agent_types,
        "elapsed": elapsed,
    }


# ── Results formatting ────────────────────────────────────────────────

def print_experiment_header(title: str, agent_configs: List[Tuple[str, type]]):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    seats = " | ".join(f"P{i}: {cls.__name__}" for i, (_, cls) in enumerate(agent_configs))
    print(f"  Seats: {seats}")
    print()


def print_experiment_results(results: Dict):
    n = results["num_games"]
    draws = results["draws"]
    decisive = results["decisive_games"]
    avg_turns = results["total_turns"] / n
    wt = results["win_turns"]
    avg_win_turn = sum(wt) / len(wt) if wt else 0

    print(f"  Games played:    {n}")
    print(f"  Decisive games:  {decisive} ({100*decisive/n:.1f}%)")
    print(f"  Draws:           {draws} ({100*draws/n:.1f}%)")
    print(f"  Avg game length: {avg_turns:.1f} turns")
    if wt:
        print(f"  Avg win turn:    {avg_win_turn:.1f}")
    print(f"  Time:            {results['elapsed']:.1f}s")

    print(f"\n  Per-agent-type breakdown:")
    print(f"  {'Agent':<15} {'Seats':<8} {'Win%':<7} {'Tsumo':<7} {'Ron':<7} "
          f"{'DI/game':<8} {'DI/disc%':<9} {'AvgShan':<8}")
    print(f"  {'-'*15} {'-'*8} {'-'*7} {'-'*7} {'-'*7} "
          f"{'-'*8} {'-'*9} {'-'*8}")

    for agent_type, data in results["agent_types"].items():
        seats_str = ",".join(f"P{s}" for s in data["seats"])
        num_seats = len(data["seats"])
        total_games = data["game_count"]
        win_pct = 100 * data["wins"] / n if n > 0 else 0
        avg_sh = data["shanten_sum"] / total_games if total_games > 0 else 0
        di_per_game = data["deal_ins"] / n if n > 0 else 0
        di_per_discard = (100 * data["deal_ins"] / data["discards"]
                         if data["discards"] > 0 else 0)

        print(f"  {agent_type:<15} {seats_str:<8} {win_pct:<7.1f} "
              f"{data['tsumo']:<7} {data['ron']:<7} "
              f"{di_per_game:<8.2f} {di_per_discard:<9.2f} {avg_sh:<8.2f}")


def print_summary_table(all_results: List[Tuple[str, Dict]]):
    print(f"\n{'='*105}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*105}")
    print(f"  {'Experiment':<28} {'Agent':<15} {'Win%':<7} {'Tsumo%':<8} {'Ron%':<7} "
          f"{'DI/game':<8} {'DI/disc%':<9} {'AvgShan':<8} {'Decisive%'}")
    print(f"  {'-'*28} {'-'*15} {'-'*7} {'-'*8} {'-'*7} "
          f"{'-'*8} {'-'*9} {'-'*8} {'-'*9}")

    for title, results in all_results:
        n = results["num_games"]
        decisive_pct = 100 * results["decisive_games"] / n

        first = True
        for agent_type, data in results["agent_types"].items():
            total_games = data["game_count"]
            win_pct = 100 * data["wins"] / n
            tsumo_pct = 100 * data["tsumo"] / n
            ron_pct = 100 * data["ron"] / n
            avg_sh = data["shanten_sum"] / total_games
            di_per_game = data["deal_ins"] / n
            di_per_discard = (100 * data["deal_ins"] / data["discards"]
                             if data["discards"] > 0 else 0)

            exp_name = title if first else ""
            dec_str = f"{decisive_pct:.1f}%" if first else ""

            print(f"  {exp_name:<28} {agent_type:<15} {win_pct:<7.1f} "
                  f"{tsumo_pct:<8.1f} {ron_pct:<7.1f} "
                  f"{di_per_game:<8.2f} {di_per_discard:<9.2f} {avg_sh:<8.2f} {dec_str}")
            first = False


# ── Experiment definitions ────────────────────────────────────────────

def main():
    NUM_GAMES = 200
    all_results = []

    print("Singapore Mahjong AI — Agent Benchmark (v2: with Ron & Deal-in)")
    print(f"Running {NUM_GAMES} games per experiment...\n")

    # ── Exp 1: 4x Random (baseline floor) ─────────────────────────────
    title = "4x Random"
    configs = [("R0", RandomAgent), ("R1", RandomAgent),
               ("R2", RandomAgent), ("R3", RandomAgent)]
    print_experiment_header(title, configs)
    results = run_experiment(configs, NUM_GAMES, seed_offset=0, verbose=True)
    print_experiment_results(results)
    all_results.append((title, results))

    # ── Exp 2: 4x Greedy (offense baseline) ───────────────────────────
    title = "4x Greedy"
    configs = [("G0", GreedyAgent), ("G1", GreedyAgent),
               ("G2", GreedyAgent), ("G3", GreedyAgent)]
    print_experiment_header(title, configs)
    results = run_experiment(configs, NUM_GAMES, seed_offset=0, verbose=True)
    print_experiment_results(results)
    all_results.append((title, results))

    # ── Exp 3: 4x Defensive (defense baseline) ────────────────────────
    title = "4x Defensive"
    configs = [("D0", DefensiveAgent), ("D1", DefensiveAgent),
               ("D2", DefensiveAgent), ("D3", DefensiveAgent)]
    print_experiment_header(title, configs)
    results = run_experiment(configs, NUM_GAMES, seed_offset=0, verbose=True)
    print_experiment_results(results)
    all_results.append((title, results))

    # ── Exp 4: 4x Hybrid ──────────────────────────────────────────────
    title = "4x Hybrid"
    configs = [("H0", HybridAgent), ("H1", HybridAgent),
               ("H2", HybridAgent), ("H3", HybridAgent)]
    print_experiment_header(title, configs)
    results = run_experiment(configs, NUM_GAMES, seed_offset=0, verbose=True)
    print_experiment_results(results)
    all_results.append((title, results))

    # ── Exp 5: Head-to-head — Greedy vs Hybrid ────────────────────────
    title = "2 Greedy vs 2 Hybrid"
    configs = [("G0", GreedyAgent), ("H0", HybridAgent),
               ("G1", GreedyAgent), ("H1", HybridAgent)]
    print_experiment_header(title, configs)
    results = run_experiment(configs, NUM_GAMES, seed_offset=0, verbose=True)
    print_experiment_results(results)
    all_results.append((title, results))

    # ── Exp 6: Mixed field — one of each ──────────────────────────────
    title = "Mixed: G vs H vs D vs R"
    configs = [("Greedy", GreedyAgent), ("Hybrid", HybridAgent),
               ("Defensive", DefensiveAgent), ("Random", RandomAgent)]
    print_experiment_header(title, configs)
    results = run_experiment(configs, NUM_GAMES, seed_offset=0, verbose=True)
    print_experiment_results(results)
    all_results.append((title, results))

    # ── Summary ───────────────────────────────────────────────────────
    print_summary_table(all_results)

    print(f"\n{'='*105}")
    print("  Key metrics:")
    print("    Win%     = percentage of games this agent type wins")
    print("    Tsumo%   = wins by self-draw")
    print("    Ron%     = wins by claiming opponent's discard")
    print("    DI/game  = deal-ins per game (discards that complete an opponent's hand)")
    print("    DI/disc% = deal-ins as % of total discards (normalized for game length)")
    print("    AvgShan  = average final shanten (lower = closer to winning)")
    print(f"{'='*105}")


if __name__ == "__main__":
    main()