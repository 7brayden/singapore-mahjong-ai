"""
Interactive Discard Advisor for Singapore Mahjong.

Input your hand of 14 tiles and get ranked discard recommendations
with offense (shanten/acceptance) and defense (danger) analysis.

Usage:
    python src/advisor.py

Tile input format:
    Numbered: 1w-9w (wan), 1t-9t (tong), 1s-9s (suo)
    Winds:    ew, sw, ww, nw (east/south/west/north)
    Dragons:  rd, gd, wd (red/green/white)

Example:
    Enter hand: 1w 2w 3w 4t 4t 5t 6t 7s 8s 9s ew ew rd

You can also specify discards already on the table to improve
danger estimation accuracy.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tiles import (
    NUM_STANDARD_UNIQUE, tile_short, tile_name, suit_of, rank_of,
    is_numbered, is_honor, is_bonus, hand_to_str, Suit
)
from hand import (
    Hand, calculate_shanten, evaluate_discards,
    tile_acceptance, get_winning_tiles, is_winning_hand
)


# ── Tile parsing ──────────────────────────────────────────────────────

TILE_MAP = {}

# Build reverse lookup: short string → tile_id
for i in range(9):
    TILE_MAP[f"{i+1}w"] = i          # 1w-9w
    TILE_MAP[f"{i+1}t"] = i + 9      # 1t-9t
    TILE_MAP[f"{i+1}s"] = i + 18     # 1s-9s

TILE_MAP["ew"] = 27   # East Wind
TILE_MAP["sw"] = 28   # South Wind
TILE_MAP["ww"] = 29   # West Wind
TILE_MAP["nw"] = 30   # North Wind
TILE_MAP["rd"] = 31   # Red Dragon
TILE_MAP["gd"] = 32   # Green Dragon
TILE_MAP["wd"] = 33   # White Dragon


def parse_tile(s: str) -> int:
    """Parse a tile string to tile_id. Returns -1 if invalid."""
    s = s.lower().strip()
    return TILE_MAP.get(s, -1)


def parse_hand(input_str: str):
    """Parse a space-separated hand string into a list of tile_ids."""
    tokens = input_str.lower().strip().split()
    tiles = []
    errors = []
    for token in tokens:
        tid = parse_tile(token)
        if tid == -1:
            errors.append(token)
        else:
            tiles.append(tid)
    return tiles, errors


# ── Analysis ──────────────────────────────────────────────────────────

def analyze_hand(tile_ids, discard_ids=None):
    """Analyze a hand and return discard recommendations.

    Args:
        tile_ids: list of 14 tile IDs (after drawing)
        discard_ids: optional list of visible discards (for better danger estimation)
    """
    # Build hand
    hand = Hand()
    for t in tile_ids:
        hand.add_tile(t)

    # Build visible counts (own hand + any known discards)
    visible = [0] * NUM_STANDARD_UNIQUE
    for t in tile_ids:
        if not is_bonus(t):
            visible[t] += 1
    if discard_ids:
        for t in discard_ids:
            if not is_bonus(t):
                visible[t] += 1

    # Current state
    current_shanten = calculate_shanten(hand.copy_counts(), hand.num_exposed_melds)

    # Check for winning hand
    if current_shanten == -1:
        return {
            "status": "winning",
            "shanten": -1,
            "message": "This hand is already a winning hand!",
            "discards": [],
        }

    # Check for tenpai
    winning_tiles = []
    if current_shanten == 0:
        winning_tiles = get_winning_tiles(hand.copy_counts(), hand.num_exposed_melds)

    # Evaluate all discards
    evals = evaluate_discards(hand, visible_counts=visible)

    # Build results
    results = []
    for e in evals:
        tid = e["tile_id"]

        # After discarding this tile, what's the hand state?
        counts_after = hand.copy_counts()
        counts_after[tid] -= 1
        new_shanten = e["shanten"]

        # What tiles improve the hand after this discard?
        improving = e["improving_tiles"]

        # Danger estimation (simple version without game state)
        # Use visibility-based danger: fewer copies visible = more dangerous
        copies_visible = visible[tid]
        danger_visibility = max(0.0, (4 - copies_visible) / 4)

        results.append({
            "tile_id": tid,
            "tile_name": tile_name(tid),
            "tile_short": tile_short(tid),
            "shanten": new_shanten,
            "acceptance": e["acceptance"],
            "improving_tiles": [tile_short(t) for t in improving],
            "danger_visibility": danger_visibility,
            "shanten_change": new_shanten - current_shanten,
        })

    return {
        "status": "tenpai" if current_shanten == 0 else "playing",
        "shanten": current_shanten,
        "winning_tiles": [tile_short(t) for t in winning_tiles],
        "discards": results,
    }


# ── Display ───────────────────────────────────────────────────────────

def display_analysis(analysis):
    """Pretty-print the analysis results."""
    sh = analysis["shanten"]

    if analysis["status"] == "winning":
        print(f"\n  *** WINNING HAND! (Tsumo) ***\n")
        return

    # Status header
    status_labels = {
        "tenpai": "TENPAI (ready to win!)",
        "playing": f"Shanten: {sh} ({sh} tile{'s' if sh > 1 else ''} away from tenpai)",
    }
    print(f"\n  Status: {status_labels[analysis['status']]}")

    if analysis["winning_tiles"]:
        print(f"  Waiting on: {', '.join(analysis['winning_tiles'])}")

    # Discard recommendations
    print(f"\n  {'Rank':<5} {'Discard':<12} {'Shanten':<9} {'Accept':<8} {'Improving tiles'}")
    print(f"  {'-'*5} {'-'*12} {'-'*9} {'-'*8} {'-'*40}")

    for i, d in enumerate(analysis["discards"]):
        rank = i + 1
        change = d["shanten_change"]
        change_str = f"{d['shanten']}"
        if change < 0:
            change_str += " ↓"
        elif change > 0:
            change_str += " ↑"

        improving_str = ", ".join(d["improving_tiles"])

        marker = " ★" if rank == 1 else ""
        print(f"  {rank:<5} {d['tile_name']:<12} {change_str:<9} "
              f"{d['acceptance']:<8} {improving_str}{marker}")

    # Best discard summary
    best = analysis["discards"][0]
    print(f"\n  ★ Recommended discard: {best['tile_name']} ({best['tile_short']})")
    print(f"    → shanten {best['shanten']}, "
          f"{best['acceptance']} tiles improve your hand")


# ── Interactive loop ──────────────────────────────────────────────────

def interactive():
    """Run the interactive advisor."""
    print("=" * 60)
    print("  Singapore Mahjong Discard Advisor")
    print("=" * 60)
    print()
    print("  Tile format:")
    print("    Numbered: 1w-9w (wan), 1t-9t (tong), 1s-9s (suo)")
    print("    Winds:    ew sw ww nw")
    print("    Dragons:  rd gd wd")
    print()
    print("  Enter 14 tiles (after drawing) separated by spaces.")
    print("  Type 'quit' to exit, 'help' for examples.")
    print()

    while True:
        try:
            user_input = input("  Hand > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        if user_input.lower() == "help":
            print()
            print("  Examples:")
            print("    1w 2w 3w 4t 4t 5t 6t 7s 8s 9s ew ew rd rd")
            print("    1w 1w 1w 2w 3w 4w 5w 6w 7w 8w 9w 9w 9w rd")
            print("    3t 4t 5t 6t 7t 8t 1s 2s 3s ew ew ew rd rd")
            print()
            continue

        tiles, errors = parse_hand(user_input)

        if errors:
            print(f"  Unknown tiles: {', '.join(errors)}")
            print("  Use format like: 1w 5t 3s ew rd")
            continue

        if len(tiles) != 14:
            print(f"  Need exactly 14 tiles, got {len(tiles)}")
            continue

        # Validate tile counts (max 4 of each)
        from collections import Counter
        counts = Counter(tiles)
        invalid = [(tile_short(t), c) for t, c in counts.items() if c > 4]
        if invalid:
            for name, c in invalid:
                print(f"  Too many {name}: {c} (max 4)")
            continue

        # Show the hand
        print(f"\n  Your hand: {hand_to_str(tiles, short=True)}")

        # Analyze
        analysis = analyze_hand(tiles)
        display_analysis(analysis)
        print()


# ── Quick demo mode ───────────────────────────────────────────────────

def demo():
    """Run a quick demo with sample hands."""
    print("=== Discard Advisor Demo ===\n")

    # Demo 1: Tenpai hand
    print("--- Demo 1: Tenpai hand ---")
    tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 31]  # 1-9wan + 1t×3 + 2t + Rd
    print(f"  Hand: {hand_to_str(tiles, short=True)}")
    analysis = analyze_hand(tiles)
    display_analysis(analysis)

    # Demo 2: Mid-game hand
    print("\n--- Demo 2: Mid-game hand ---")
    tiles = [0, 0, 1, 3, 4, 9, 10, 11, 18, 19, 27, 28, 31, 32]
    print(f"  Hand: {hand_to_str(tiles, short=True)}")
    analysis = analyze_hand(tiles)
    display_analysis(analysis)

    # Demo 3: Winning hand
    print("\n--- Demo 3: Winning hand ---")
    tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 10]
    print(f"  Hand: {hand_to_str(tiles, short=True)}")
    analysis = analyze_hand(tiles)
    display_analysis(analysis)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        interactive()