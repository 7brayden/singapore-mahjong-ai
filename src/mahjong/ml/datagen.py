"""Generate supervised training data from simulated games.

Runs seeded games with a rotating mix of agent lineups and records every
discard decision from all four seats. Because the simulator has perfect
information, each candidate tile gets a ground-truth label — not just
the tile the agent happened to throw:

    waited        some opponent is tenpai waiting on this tile
    waited_legal  ...and their hand meets the minimum tai, so discarding
                  it would ACTUALLY deal in (mirrors _check_ron exactly)
    chosen        the agent discarded this tile
    dealt_in      chosen AND the game ended in a ron on it

`waited_legal` is the label the danger model trains on: it marks every
hot tile at every decision, giving dense supervision instead of the
~2%-sparse observed deal-ins. The observed outcome (dealt_in) is a
strict subset — a hot tile that was never thrown still teaches.

A second table records one row per decision with hand-progress features
and win/net-point labels for the outcome (win probability) model.

Usage:
    PYTHONPATH=src python3 -m mahjong.ml.datagen --games 2000 --out data/
"""

import argparse
import csv
import gzip
import multiprocessing
import os
import time

from mahjong.game import GameState, DiscardRequest, DEAD_WALL_SIZE
from mahjong.hand import evaluate_discards, get_winning_tiles
from mahjong.scoring import score_win, is_legal_win
from mahjong.opponent_model import estimate_opponent_threats
from mahjong.agents import GreedyAgent, HybridAgent, DefensiveAgent, RandomAgent
from mahjong.ml.features import (
    DANGER_FEATURES, OUTCOME_FEATURES, danger_features, outcome_features,
)

# Varied lineups so the models see many table textures: exposed-heavy
# greedy games, cautious defensive rivers, chaotic random discards.
LINEUPS = [
    lambda: [HybridAgent(f"H{i}") for i in range(4)],
    lambda: [GreedyAgent(f"G{i}") for i in range(4)],
    lambda: [GreedyAgent("G0"), HybridAgent("H1"),
             GreedyAgent("G2"), HybridAgent("H3")],
    lambda: [GreedyAgent("G"), HybridAgent("H"),
             DefensiveAgent("D"), RandomAgent("R")],
    lambda: [HybridAgent("H0"), DefensiveAgent("D"),
             HybridAgent("H1"), GreedyAgent("G")],
]

# decision_id is a per-game counter shared by every row of one decision,
# so the two tables join on (game_id, decision_id) and per-decision
# analysis needs no fragile reconstruction from feature values.
DANGER_COLUMNS = (["game_id", "decision_id", "lineup", "seat", "tile"]
                  + DANGER_FEATURES
                  + ["waited", "waited_legal", "chosen", "dealt_in"])
OUTCOME_COLUMNS = (["game_id", "decision_id", "lineup", "seat"]
                   + OUTCOME_FEATURES + ["won", "net_points"])


def _opponent_waits(game: GameState, player_idx: int, cache: dict) -> dict:
    """Perfect-information wait sets for each opponent, {seat: set(tiles)}.

    Cached on (counts, melds) — an opponent's concealed hand only changes
    on their own turn, so between their turns this is a dict lookup.
    """
    waits = {}
    for opp in range(4):
        if opp == player_idx:
            continue
        hand = game.hands[opp]
        if len(hand.tiles) % 3 != 1:  # not a 13-tile shape (mid-kong etc.)
            continue
        key = (tuple(hand.counts), hand.num_exposed_melds)
        if key not in cache:
            cache[key] = frozenset(
                get_winning_tiles(hand.copy_counts(), hand.num_exposed_melds))
        if cache[key]:
            waits[opp] = cache[key]
    return waits


def _ron_would_be_legal(game: GameState, tile_id: int, waits: dict,
                        is_last: bool) -> bool:
    """Would discarding tile_id right now deal in? Mirrors _check_ron:
    the hand must complete AND meet the minimum tai."""
    for opp, wait_set in waits.items():
        if tile_id not in wait_set:
            continue
        hand = game.hands[opp]
        hand.counts[tile_id] += 1
        score = score_win(hand, tile_id, False, game.seat_index(opp),
                          game.prevailing_wind, game.score_config,
                          is_last_tile=is_last)
        hand.counts[tile_id] -= 1
        if score is not None and is_legal_win(score, game.score_config):
            return True
    return False


def generate_game(game_id: int, seed: int, lineup_fn,
                  danger_rows: list, outcome_rows: list,
                  lineup_idx: int = 0) -> dict:
    """Play one seeded game, appending labeled rows to both tables."""
    game = GameState(lineup_fn(), seed=seed)
    gen = game.step_game()
    wait_cache: dict = {}
    outcome_start = len(outcome_rows)
    last_chosen = None  # (row_index, seat, tile) of the latest chosen discard
    decision_id = 0

    request = None
    try:
        request = next(gen)
        while True:
            answer = game.dispatch_to_agent(request)
            if isinstance(request, DiscardRequest):
                seat = request.player
                hand = game.hands[seat]
                visible = game.get_visible_counts(seat)
                threat = estimate_opponent_threats(seat, game)
                waits = _opponent_waits(game, seat, wait_cache)
                is_last = game.tiles_remaining <= DEAD_WALL_SIZE
                evals = evaluate_discards(hand, visible_counts=visible)

                chosen_eval = None
                for e in evals:
                    tile = e["tile_id"]
                    waited = any(tile in w for w in waits.values())
                    legal = waited and _ron_would_be_legal(
                        game, tile, waits, is_last)
                    chosen = 1 if tile == answer else 0
                    if chosen:
                        last_chosen = (len(danger_rows), seat, tile)
                        chosen_eval = e
                    danger_rows.append(
                        [game_id, decision_id, lineup_idx, seat, tile]
                        + danger_features(tile, seat, game, visible, threat)
                        + [int(waited), int(legal), chosen, 0])

                # Outcome row: the state the CHOSEN discard leaves behind
                # — its realized net points become the value label.
                if chosen_eval is not None:
                    counts_after = hand.copy_counts()
                    counts_after[answer] -= 1
                    outcome_rows.append(
                        [game_id, decision_id, lineup_idx, seat]
                        + outcome_features(seat, game, counts_after,
                                           chosen_eval["shanten"],
                                           chosen_eval["acceptance"], threat)
                        + [0, 0])
                decision_id += 1
            request = gen.send(answer)
    except StopIteration as stop:
        result = stop.value

    # ── Backfill outcome labels now the result is known ──────────────
    if result.win_type == "ron" and last_chosen is not None:
        row_idx, seat, tile = last_chosen
        if seat == result.dealt_in_by and tile == result.win_tile:
            danger_rows[row_idx][-1] = 1  # dealt_in
    payments = result.payments or [0, 0, 0, 0]
    seat_col = OUTCOME_COLUMNS.index("seat")
    for i in range(outcome_start, len(outcome_rows)):
        seat = outcome_rows[i][seat_col]
        outcome_rows[i][-2] = 1 if result.winner == seat else 0
        outcome_rows[i][-1] = payments[seat]

    return {"win_type": result.win_type, "turns": result.turns}


def _run_one(task):
    """Worker entry: play one game, return its rows (multiprocessing)."""
    game_id, seed, lineup_idx = task
    danger_rows: list = []
    outcome_rows: list = []
    stats = generate_game(game_id, seed, LINEUPS[lineup_idx],
                          danger_rows, outcome_rows, lineup_idx)
    return danger_rows, outcome_rows, stats["win_type"]


def _write_gz(path: str, header: list, rows: list) -> None:
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--out", default="data")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    danger_rows: list = []
    outcome_rows: list = []
    wins = {"ron": 0, "tsumo": 0, None: 0}
    started = time.time()

    tasks = [(i, args.seed_start + i, i % len(LINEUPS))
             for i in range(args.games)]

    def _accumulate(done: int, result) -> None:
        d_rows, o_rows, win_type = result
        danger_rows.extend(d_rows)
        outcome_rows.extend(o_rows)
        wins[win_type] += 1
        if done % args.progress_every == 0:
            elapsed = time.time() - started
            hot = sum(r[-3] for r in danger_rows)  # waited_legal
            di = sum(r[-1] for r in danger_rows)   # dealt_in
            print(f"[{done}/{args.games}] {elapsed:.0f}s "
                  f"({done / elapsed:.1f} games/s) — "
                  f"danger rows: {len(danger_rows):,} "
                  f"(hot: {hot:,}, dealt-in: {di:,}) — "
                  f"ron {wins['ron']} / tsumo {wins['tsumo']} / "
                  f"draw {wins[None]}", flush=True)

    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            for done, result in enumerate(
                    pool.imap_unordered(_run_one, tasks, chunksize=4), 1):
                _accumulate(done, result)
    else:
        for done, task in enumerate(tasks, 1):
            _accumulate(done, _run_one(task))

    danger_path = os.path.join(args.out, "danger.csv.gz")
    outcome_path = os.path.join(args.out, "outcome.csv.gz")
    _write_gz(danger_path, DANGER_COLUMNS, danger_rows)
    _write_gz(outcome_path, OUTCOME_COLUMNS, outcome_rows)

    hot = sum(r[-3] for r in danger_rows)
    print(f"\nWrote {danger_path}: {len(danger_rows):,} rows, "
          f"{hot:,} hot ({100 * hot / max(1, len(danger_rows)):.2f}%)")
    print(f"Wrote {outcome_path}: {len(outcome_rows):,} rows, "
          f"{sum(r[-2] for r in outcome_rows):,} winning-seat rows")


if __name__ == "__main__":
    main()
