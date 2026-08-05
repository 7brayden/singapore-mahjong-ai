# Singapore Mahjong AI

An agent that plays Singapore Mahjong, built from scratch. It weighs hand efficiency against the risk of feeding opponents when deciding what to discard, and ships with a command-line advisor you can run during real games.

## What is this?

Mahjong is a 4-player tile game. You see your own hand and everyone's discards, but not what opponents hold. Each turn you draw a tile and discard one, working toward a winning hand of 4 melds plus a pair. The catch is that opponents can claim your discards to complete their own hands, so every discard trades off building your hand against handing someone else the win.

This project builds agents that make that tradeoff, pits them against each other, and measures who actually plays better.

## Terminology

If you don't play, here's the jargon used below:

| Term | Meaning |
|---|---|
| **Shanten** | Tile swaps away from winning. 0 = one tile away ("tenpai"). -1 = already won. |
| **Tsumo** | Winning off your own draw. |
| **Ron** | Winning off an opponent's discard. The discarder "dealt in." |
| **Deal-in** | Your discard completed an opponent's hand. |
| **Pong** | Claiming a discard to form a triplet. Any player can pong. |
| **Chow** | Claiming a discard to form a sequence (e.g. 3-4-5). Only the next player can chow. |
| **Tile acceptance** | How many drawable tiles would improve your hand. Higher = more flexible. |

## Quick start

```bash
pip install -e .
mahjong-advisor             # interactive discard advisor
mahjong-advisor --demo      # example hands
```

Or without installing:

```bash
PYTHONPATH=src python3 -m mahjong.advisor
```

The advisor takes your 14 tiles and tells you what to discard:

```
Hand > 1w 2w 3w 4t 4t 5t 6t 7s 8s 9s ew ew rd rd

Status: TENPAI (ready to win!)
Waiting on: 2t, 3t, Rd

Rank  Discard      Shanten   Accept   Improving tiles
1     Red Dragon   0         7        2t, 3t ★
2     2 Tong       0         3        Rd
...
```

Type tiles in any order. Format:

- `1w`–`9w` wan, `1t`–`9t` tong, `1s`–`9s` suo
- `ew` / `sw` / `ww` / `nw` winds
- `rd` / `gd` / `wd` dragons

One caveat: the advisor only optimises for hand efficiency. It doesn't track opponent discards, so judge defense yourself in the late game.

## Project structure

```
src/mahjong/
├── tiles.py               # Tile encoding (integers 0-33), suits, wall creation
├── hand.py                # Hand management, shanten, discard evaluation
├── game.py                # Game loop: draw, discard, tsumo, ron, pong/chow
├── defense.py             # Danger scoring for discard candidates
├── opponent_model.py      # Estimates how close each opponent is to winning
├── scoring.py             # Tai scoring, payments, house-rule config
├── advisor.py             # Interactive tool — input your hand, get advice
└── agents/
    ├── random_agent.py    # Discards randomly. Never wins.
    ├── greedy_agent.py    # Minimises shanten. Wins a lot, deals in often.
    ├── defensive_agent.py # Minimises danger. Safe, never wins.
    └── hybrid_agent.py    # Balances offense and defense with dynamic weights.

tests/                     # pytest suite: shanten, claims, determinism, golden games

experiments/
├── run_benchmark.py       # Runs 6 experiments, prints results
└── generate_plots.py      # Bar charts from benchmark data (needs matplotlib)

results/
└── *.png                  # Benchmark plots
```

## How it works
Melds - completed sets of 3 or 4 tiles.

**Shanten calculation.** Given 13 tiles, find the minimum swaps to a winning hand. Implemented as recursive backtracking: try each tile as the pair, extract complete melds, then count partial melds capped so a hand can't claim more blocks than it needs. Fast enough for interactive use (~20 ms to fully evaluate a 14-tile hand); a memoized per-suit lookup table is planned for when Monte Carlo rollouts need it.

**Discard evaluation.** For a 14-tile hand, remove each tile in turn and score shanten plus tile acceptance on the remaining 13. Rank by lowest shanten, break ties by highest acceptance. This is the greedy agent's whole strategy.

**Defense.** Four signals score how dangerous each discard is:

- *Visibility* — fewer visible copies means opponents may be holding the rest.
- *Discard absence* — if nobody's discarded this tile or its neighbours, someone may be collecting it.
- *Opponent threat* — per-opponent closeness to winning, weighted by which suits they seem to want.
- *Suit safety* — if opponents are dumping a suit, that suit is probably safe.

**Opponent modelling.** Estimates each opponent's threat from what you can see: exposed melds, whether they keep their draws (discard efficiency), whether they're clearing safe tiles late (a tenpai tell), and which suits they avoid discarding.

**Hybrid agent.** Scores each discard on offense (shanten + acceptance) and defense (danger), then combines them with weights that shift by context — aggressive when close to winning, cautious when far, more defensive late. It also claims pong/chow selectively, only when shanten ≤ 2 and the claim genuinely helps. Greedy claims everything; the hybrid avoids exposing information it doesn't need to.

## Scoring (tai)

Every win is scored as a list of named tai items (`scoring.py`), so the UI can show players exactly why a hand is worth what it is. Chip value doubles per tai: `base_unit × 2^(tai−1)`, capped at 5 tai (the limit).

| Rule | Tai |
|---|---|
| Self-draw (自摸) | 1 |
| No flowers or animals (无花) | 1 |
| Matching seat flower / animal | 1 each |
| Complete flower series (一套花) | 1 |
| Dragon triplet / seat wind / prevailing wind | 1 each |
| All triplets (碰碰胡) | 2 |
| Half flush (混一色) | 2 |
| Little three dragons (小三元) | +2 over the dragon pongs |
| Full flush (清一色) | 4 |
| Ping Hu — all chows, no bonus tiles (平胡) | 4 |
| Big three dragons, four winds, all honors, all terminals | limit (5) |

House rules live in `ScoreConfig` and are all adjustable: minimum 1 tai to win (chicken hands blocked by default), shooter pays all three shares on a ron, tsumo collects from everyone, and animals/completed flower series pay out instantly when drawn. Kong-related tai, thirteen orphans, and heaven/earth wins arrive with kong mechanics.

## Benchmark results

200 games per experiment, deterministic seeds.

> Note: these tables predate the engine fixes in v0.2 (fully seeded RNG, correct chow selection, nested claim windows). Numbers will be regenerated together with tai scoring, which changes what "winning" means anyway.

### Mirror matches

| Agent | Win% | DI/disc% | Avg Shanten |
|---|---|---|---|
| Random | 0.0 | 0.00 | 3.79 |
| Greedy | 100.0 | 2.21 | 0.41 |
| Defensive | 0.0 | 0.00 | 3.10 |
| Hybrid | 97.0 | 1.69 | 0.48 |

Greedy wins every game but deals in most per discard, at 2.21%. Hybrid is 24% safer per discard and still wins 97%.

### 2 Greedy vs 2 Hybrid

| Agent | Win% | DI/disc% |
|---|---|---|
| Greedy | 59.5 | 2.17 |
| Hybrid | 40.5 | 2.03 |

Greedy wins more by reaching tenpai faster and ending games before defense matters. Per discard, Hybrid is still safer.

### Mixed field (one of each)

| Agent | Win% | DI/disc% |
|---|---|---|
| Greedy | 49.5 | 0.75 |
| Hybrid | 38.5 | 1.09 |
| Defensive | 0.0 | 0.41 |
| Random | 0.0 | 3.54 |

Against weak opponents, speed beats safety: there's nobody good enough to punish a loose discard, so Greedy runs away with it.

### Why DI/disc% and not DI/game

Deal-ins per game made Greedy look safest, but only because it ends games fastest — fewer total discards means fewer chances to deal in. Normalising by total discards measures how safe each individual decision is, which is the thing we actually care about.

## Running everything

Python 3.9+, no dependencies for the core code. pytest for the test suite, matplotlib for plots.

```bash
pip install -e ".[dev]"
pytest                              # engine test suite

PYTHONPATH=src python3 -m mahjong.tiles   # tile encoding demo
PYTHONPATH=src python3 -m mahjong.hand    # shanten sanity checks
PYTHONPATH=src python3 -m mahjong.game    # game loop demo

python3 experiments/run_benchmark.py      # full benchmark (~5-10 min)
python3 experiments/generate_plots.py     # result charts (needs matplotlib)
```

## Future work

- Kong mechanics (exposed/concealed/added, replacement draws, robbing the kong) plus kong tai, thirteen orphans, heaven/earth wins
- Seat/prevailing winds, dealer rotation, multi-round sessions
- Human-in-the-loop play: FastAPI backend + React UI with live stats (shanten, danger, opponent threat) so players learn while they play
- LLM-powered move explanations grounded in the engine's analysis
- Learned danger model trained on simulation data
- Lookahead: sample future draws, estimate discard value
- Tune hybrid weights to detected opponent strength

## What this project covers

- State representation and feature engineering
- Heuristic search with pruning and block-capping
- Multi-objective decision making (offense vs defense)
- Opponent modelling from observable information
- Experiment design with controlled baselines and normalised metrics
- Catching a misleading metric (DI/game vs DI/disc%)