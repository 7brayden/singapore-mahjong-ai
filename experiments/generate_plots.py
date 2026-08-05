"""
Generate benchmark plots for the Singapore Mahjong AI project.

Produces:
  1. Win rate comparison (bar chart across all agents)
  2. Deal-in rate comparison
  3. Win type breakdown (tsumo vs ron)
  4. Head-to-head: Greedy vs Hybrid (grouped bar)

Usage:
    python experiments/generate_plots.py
"""

import sys
import os
import time

# Allow running without installing the package (pip install -e .)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from mahjong.game import GameState
from mahjong.agents import (
    RandomAgent, GreedyAgent, DefensiveAgent, HybridAgent
)


# ── Color palette ─────────────────────────────────────────────────────

COLORS = {
    "RandomAgent": "#9E9E9E",
    "GreedyAgent": "#E53935",
    "DefensiveAgent": "#1E88E5",
    "HybridAgent": "#43A047",
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


# ── Data collection ───────────────────────────────────────────────────

def collect_data(num_games=200):
    """Run all experiments and return structured data for plotting."""
    print(f"Running {num_games} games per experiment...")

    experiments = {}

    # 4x mirror matches
    for name, agent_cls in [("Random", RandomAgent), ("Greedy", GreedyAgent),
                             ("Defensive", DefensiveAgent), ("Hybrid", HybridAgent)]:
        print(f"  4x {name}...", end=" ", flush=True)
        start = time.time()
        stats = {"wins": 0, "tsumo": 0, "ron": 0, "deal_ins": 0,
                 "shanten_sum": 0, "turns_sum": 0, "win_turns": []}

        for i in range(num_games):
            agents = [agent_cls(f"{name[0]}{j}") for j in range(4)]
            game = GameState(agents, seed=i)
            result = game.play()

            stats["turns_sum"] += result.turns
            if result.winner is not None:
                stats["wins"] += 1
                stats["win_turns"].append(result.turns)
                if result.win_type == "tsumo":
                    stats["tsumo"] += 1
                else:
                    stats["ron"] += 1
            for p in range(4):
                stats["deal_ins"] += result.deal_ins[p]
                stats["shanten_sum"] += result.final_shanten[p]

        stats["num_games"] = num_games
        experiments[f"4x {name}"] = stats
        print(f"{time.time()-start:.1f}s")

    # Head-to-head: 2x Greedy vs 2x Hybrid
    print("  2 Greedy vs 2 Hybrid...", end=" ", flush=True)
    start = time.time()
    h2h = {"greedy_wins": 0, "hybrid_wins": 0, "draws": 0,
           "greedy_di": 0, "hybrid_di": 0,
           "greedy_tsumo": 0, "greedy_ron": 0,
           "hybrid_tsumo": 0, "hybrid_ron": 0}

    for i in range(num_games):
        agents = [GreedyAgent("G0"), HybridAgent("H0"),
                  GreedyAgent("G1"), HybridAgent("H1")]
        game = GameState(agents, seed=i)
        result = game.play()

        if result.winner is None:
            h2h["draws"] += 1
        elif result.winner in (0, 2):
            h2h["greedy_wins"] += 1
            if result.win_type == "tsumo":
                h2h["greedy_tsumo"] += 1
            else:
                h2h["greedy_ron"] += 1
        else:
            h2h["hybrid_wins"] += 1
            if result.win_type == "tsumo":
                h2h["hybrid_tsumo"] += 1
            else:
                h2h["hybrid_ron"] += 1

        h2h["greedy_di"] += result.deal_ins[0] + result.deal_ins[2]
        h2h["hybrid_di"] += result.deal_ins[1] + result.deal_ins[3]

    h2h["num_games"] = num_games
    experiments["h2h"] = h2h
    print(f"{time.time()-start:.1f}s")

    # Mixed field
    print("  Mixed field...", end=" ", flush=True)
    start = time.time()
    mixed = {name: {"wins": 0, "tsumo": 0, "ron": 0, "deal_ins": 0}
             for name in ["Greedy", "Hybrid", "Defensive", "Random"]}

    for i in range(num_games):
        agents = [GreedyAgent("Greedy"), HybridAgent("Hybrid"),
                  DefensiveAgent("Defensive"), RandomAgent("Random")]
        game = GameState(agents, seed=i)
        result = game.play()

        names = ["Greedy", "Hybrid", "Defensive", "Random"]
        if result.winner is not None:
            winner_name = names[result.winner]
            mixed[winner_name]["wins"] += 1
            if result.win_type == "tsumo":
                mixed[winner_name]["tsumo"] += 1
            else:
                mixed[winner_name]["ron"] += 1
        for p in range(4):
            mixed[names[p]]["deal_ins"] += result.deal_ins[p]

    mixed["num_games"] = num_games
    experiments["mixed"] = mixed
    print(f"{time.time()-start:.1f}s")

    return experiments


# ── Plot 1: Win rate comparison (mirror matches) ─────────────────────

def plot_win_rates(data, num_games):
    fig, ax = plt.subplots(figsize=(8, 5))

    agents = ["Random", "Greedy", "Defensive", "Hybrid"]
    win_rates = []
    colors = []

    for name in agents:
        key = f"4x {name}"
        wr = 100 * data[key]["wins"] / num_games
        win_rates.append(wr)
        colors.append(COLORS[f"{name}Agent"])

    bars = ax.bar(agents, win_rates, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f"{wr:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Win Rate by Agent Type (4x Mirror Match)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "win_rates.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved win_rates.png")


# ── Plot 2: Deal-in rate comparison (mirror matches) ─────────────────

def plot_deal_in_rates(data, num_games):
    fig, ax = plt.subplots(figsize=(8, 5))

    agents = ["Random", "Greedy", "Defensive", "Hybrid"]
    di_rates = []
    colors = []

    for name in agents:
        key = f"4x {name}"
        di = data[key]["deal_ins"] / num_games
        di_rates.append(di)
        colors.append(COLORS[f"{name}Agent"])

    bars = ax.bar(agents, di_rates, color=colors, width=0.6, edgecolor="white", linewidth=1.5)

    for bar, di in zip(bars, di_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{di:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Deal-ins per Game", fontsize=12)
    ax.set_title("Deal-in Rate by Agent Type (4x Mirror Match)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(di_rates) * 1.3 + 0.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "deal_in_rates.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved deal_in_rates.png")


# ── Plot 3: Win type breakdown (tsumo vs ron) ────────────────────────

def plot_win_types(data, num_games):
    fig, ax = plt.subplots(figsize=(8, 5))

    agents = ["Greedy", "Hybrid"]
    tsumo_pcts = []
    ron_pcts = []

    for name in agents:
        key = f"4x {name}"
        total = data[key]["wins"] if data[key]["wins"] > 0 else 1
        tsumo_pcts.append(100 * data[key]["tsumo"] / total)
        ron_pcts.append(100 * data[key]["ron"] / total)

    x = np.arange(len(agents))
    width = 0.35

    bars1 = ax.bar(x - width/2, tsumo_pcts, width, label="Tsumo (self-draw)",
                   color="#66BB6A", edgecolor="white", linewidth=1.5)
    bars2 = ax.bar(x + width/2, ron_pcts, width, label="Ron (from discard)",
                   color="#EF5350", edgecolor="white", linewidth=1.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 1,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=11)

    ax.set_ylabel("Percentage of Wins", fontsize=12)
    ax.set_title("Win Type Breakdown (4x Mirror Match)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "win_types.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved win_types.png")


# ── Plot 4: Head-to-head Greedy vs Hybrid ─────────────────────────────

def plot_head_to_head(data, num_games):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    h2h = data["h2h"]

    # Left: win rate
    ax = axes[0]
    agents = ["Greedy", "Hybrid", "Draw"]
    values = [100 * h2h["greedy_wins"] / num_games,
              100 * h2h["hybrid_wins"] / num_games,
              100 * h2h["draws"] / num_games]
    colors_h2h = [COLORS["GreedyAgent"], COLORS["HybridAgent"], "#BDBDBD"]

    bars = ax.bar(agents, values, color=colors_h2h, width=0.6, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Percentage of Games", fontsize=12)
    ax.set_title("Win Rate: Greedy vs Hybrid", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: deal-in rate
    ax = axes[1]
    agents_di = ["Greedy", "Hybrid"]
    di_values = [h2h["greedy_di"] / num_games,
                 h2h["hybrid_di"] / num_games]
    colors_di = [COLORS["GreedyAgent"], COLORS["HybridAgent"]]

    bars = ax.bar(agents_di, di_values, color=colors_di, width=0.6, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, di_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_ylabel("Deal-ins per Game", fontsize=12)
    ax.set_title("Deal-in Rate: Greedy vs Hybrid", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(di_values) * 1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "head_to_head.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved head_to_head.png")


# ── Plot 5: Mixed field results ───────────────────────────────────────

def plot_mixed_field(data, num_games):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    mixed = data["mixed"]

    agents = ["Greedy", "Hybrid", "Defensive", "Random"]
    agent_colors = [COLORS[f"{a}Agent"] for a in agents]

    # Left: win rate
    ax = axes[0]
    win_rates = [100 * mixed[a]["wins"] / num_games for a in agents]
    bars = ax.bar(agents, win_rates, color=agent_colors, width=0.6, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Win Rate (%)", fontsize=12)
    ax.set_title("Mixed Field: Win Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(win_rates) * 1.3 + 2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: deal-in rate
    ax = axes[1]
    di_rates = [mixed[a]["deal_ins"] / num_games for a in agents]
    bars = ax.bar(agents, di_rates, color=agent_colors, width=0.6, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, di_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{v:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax.set_ylabel("Deal-ins per Game", fontsize=12)
    ax.set_title("Mixed Field: Deal-in Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(di_rates) * 1.4 + 0.02)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "mixed_field.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved mixed_field.png")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    NUM_GAMES = 200
    data = collect_data(NUM_GAMES)

    print("\nGenerating plots...")
    plot_win_rates(data, NUM_GAMES)
    plot_deal_in_rates(data, NUM_GAMES)
    plot_win_types(data, NUM_GAMES)
    plot_head_to_head(data, NUM_GAMES)
    plot_mixed_field(data, NUM_GAMES)

    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()