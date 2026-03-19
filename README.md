# Singapore Mahjong AI

An AI agent that plays Singapore Mahjong, built from scratch. It combines hand-efficiency optimisation with opponent threat modelling to decide what to discard. Includes a command-line advisor you can use during real games.

## What is this?

Mahjong is a 4-player tile game where you can see your own hand and what's been discarded, but not what opponents hold. Every turn you draw a tile and discard one, trying to form a winning hand (4 melds + 1 pair). The interesting part: your discards can be claimed by opponents to complete their hand — so every discard is a risk/reward tradeoff between building your hand and accidentally helping someone else win.

This project builds agents that make that tradeoff, benchmarks them against each other, and measures the results.

## Terminology

If you don't play Mahjong, here's the jargon:

| Term | Meaning |
|---|---|
| **Shanten** | How many tile swaps away from a winning hand. 0 = one tile away ("tenpai"). -1 = already won. |
| **Tsumo** | Winning by drawing the tile yourself |
| **Ron** | Winning by claiming someone else's discard. The discarder "dealt in." |
| **Deal-in** | Your discard completed an opponent's hand. Bad. |
| **Pong** | Claiming a discard to form a triplet (3 identical tiles). Any player can do this. |
| **Chow** | Claiming a discard to form a sequence (e.g. 3-4-5). Only the next player can do this. |
| **Tile acceptance** | How many drawable tiles would improve your hand. Higher = more flexible. |

## Project structure

```
src/
├── tiles.py               # Tile encoding (integers 0-33), suits, wall creation
├── hand.py                # Hand management, shanten calculation, discard evaluation
├── game.py                # Game loop: draw, discard, tsumo, ron, pong/chow claiming
├── defense.py             # Danger scoring for discard candidates
├── opponent_model.py      # Estimates how close each opponent is to winning
├── advisor.py             # Interactive tool — input your hand, get discard advice
└── agents/
    ├── random_agent.py    # Discards randomly. Never wins.
    ├── greedy_agent.py    # Minimises shanten. Wins a lot but deals in frequently.
    ├── defensive_agent.py # Minimises danger. Safe but never wins.
    └── hybrid_agent.py    # Balances offense and defense with dynamic weights.

experiments/
├── run_benchmark.py       # Runs 6 experiments, prints results
└── generate_plots.py      # Bar charts from benchmark data (needs matplotlib)

results/
└── *.png                  # Benchmark plots
```

## How it works

### Shanten calculation

The core algorithm. Given 13 tiles, it figures out the minimum number of swaps to reach a winning hand. The approach: decompose by suit (melds can't cross suits), recursively try all ways to extract melds within each suit, count leftover partial melds, then combine across suits. A suit-based decomposition gives a 14× speedup over naive backtracking — important because this function gets called thousands of times per game.

### Discard evaluation

For a 14-tile hand, try removing each tile and calculate shanten + tile acceptance on the remaining 13. Rank by lowest shanten, break ties by highest acceptance. This is what the greedy agent does.

### Defense

Four signals estimate how dangerous each discard is:

- **Visibility** — tiles with fewer visible copies are riskier (opponents might be holding them)
- **Discard absence** — if nobody's discarded this tile or its neighbours, someone might be collecting
- **Opponent threat** — per-opponent estimate of how close they are to winning, combined with which suits they seem to be collecting
- **Suit safety** — if opponents are dumping a suit, that suit is probably safe

### Opponent modelling

Estimates each opponent's threat level from what you can observe: how many exposed melds they have, whether they're keeping drawn tiles (discard efficiency), whether they're clearing safe tiles late (suggests tenpai), and which suits they're avoiding in discards.

### Hybrid agent

Scores each discard on both offense (shanten + acceptance) and defense (danger), then combines them with weights that shift based on context. Close to winning? Go aggressive. Far away? Play safe. Late game? More defensive.

Also claims pong/chow selectively — only when shanten ≤ 2 and the claim actually improves the hand. The greedy agent claims everything; the hybrid is more careful about exposing information.

## Benchmark results

200 games per experiment, deterministic seeds.

### Mirror matches

| Agent | Win% | DI/disc% | Avg Shanten |
|---|---|---|---|
| Random | 0.0 | 0.00 | 3.79 |
| Greedy | 100.0 | 2.21 | 0.41 |
| Defensive | 0.0 | 0.00 | 3.10 |
| Hybrid | 97.0 | 1.69 | 0.48 |

Greedy wins every game but has the highest per-discard deal-in rate at 2.21%. Hybrid is 24% safer per discard while still winning 97% of games.

### Head-to-head: 2 Greedy vs 2 Hybrid

| Agent | Win% | DI/disc% |
|---|---|---|
| Greedy | 59.5 | 2.17 |
| Hybrid | 40.5 | 2.03 |

Greedy wins more through speed — it reaches tenpai faster and ends games before defense matters. But per discard, Hybrid is still the safer player.

### Mixed field (one of each agent)

| Agent | Win% | DI/disc% |
|---|---|---|
| Greedy | 49.5 | 0.75 |
| Hybrid | 38.5 | 1.09 |
| Defensive | 0.0 | 0.41 |
| Random | 0.0 | 3.54 |

Against weak opponents, Greedy dominates because speed beats safety when there's nobody threatening to punish your discards.

### What DI/disc% reveals

We originally tracked deal-ins per game, but that metric was misleading — Greedy appeared safest because it ends games fastest (fewer total discards = fewer deal-in opportunities). DI/disc% normalises by total discards, giving a fairer picture of how safe each individual discard decision is.

## Interactive advisor

Use during real games:

```bash
cd src
python3 advisor.py
```

```
Hand > 1w 2w 3w 4t 4t 5t 6t 7s 8s 9s ew ew rd rd

Status: TENPAI (ready to win!)
Waiting on: 2t, 3t, Rd

Rank  Discard      Shanten   Accept   Improving tiles
1     Red Dragon   0         7        2t, 3t ★
2     2 Tong       0         3        Rd
...
```

Type your 14 tiles in any order. Format: `1w`-`9w` (wan), `1t`-`9t` (tong), `1s`-`9s` (suo), `ew`/`sw`/`ww`/`nw` (winds), `rd`/`gd`/`wd` (dragons).

Note: the advisor is purely offensive — it picks the best discard for hand efficiency. It doesn't know what opponents have discarded, so use your own judgment for defense in late-game situations.

## Running everything

Python 3.8+, no dependencies for core code. matplotlib for plots, scikit-learn for the ML experiments.

```bash
cd src
python3 tiles.py           # tile encoding test
python3 hand.py            # shanten tests
python3 game.py            # game loop test
python3 advisor.py         # interactive advisor
python3 advisor.py --demo  # example hands

cd experiments
python3 run_benchmark.py   # full benchmark (~5-10 min)
pip3 install matplotlib
python3 generate_plots.py  # generate result charts
```

## Future work

- Scoring system with fan counting (hand value)
- Learned danger model trained on simulation data
- Simulation-based lookahead (sample future draws, estimate discard value)
- Tune hybrid weights based on detected opponent strength
- Kong mechanics

## What this project covers

- State representation and feature engineering
- Heuristic search with algorithmic optimisation (14× speedup)
- Multi-objective decision making (offense vs defense)
- Opponent modelling from observable information
- Experiment design with controlled baselines and normalised metrics
- Identifying misleading metrics (DI/game vs DI/disc%)