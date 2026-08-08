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
├── game.py                # Game loop: draw, discard, tsumo, ron, pong/chow/kong
├── defense.py             # Danger scoring for discard candidates
├── opponent_model.py      # Estimates how close each opponent is to winning
├── scoring.py             # Tai scoring, payments, house-rule config
├── session.py             # Multi-round sessions: dealer rotation, running scores
├── interactive.py         # Human-in-the-loop controller with redacted views
├── advisor.py             # Interactive tool — input your hand, get advice
├── server/                # FastAPI backend: games, actions, hints, analysis, websocket
├── ml/                    # Learned models: datagen → train → pure-Python inference
│   ├── features.py        # Feature extraction shared by training and play
│   ├── datagen.py         # Simulates games, dumps labeled decision records
│   ├── train.py           # Fits models, prints metrics vs the heuristic (needs sklearn)
│   ├── model.py           # Evaluates exported JSON weights — zero dependencies
│   └── danger_model.json  # Trained deal-in model (committed, human-readable)
└── agents/
    ├── random_agent.py    # Discards randomly. Never wins.
    ├── greedy_agent.py    # Minimises shanten. Wins a lot, deals in often.
    ├── defensive_agent.py # Minimises danger. Safe, never wins.
    ├── hybrid_agent.py    # Balances offense and defense with dynamic weights.
    └── learned_agent.py   # Hybrid offense + danger model trained on simulations.

tests/                     # pytest suite: shanten, claims, determinism, golden games

experiments/
├── run_benchmark.py       # Runs 6 experiments, prints results
├── compare_learned.py     # Learned vs Hybrid head-to-head
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

## Learned danger model

The heuristic danger weights above are hand-tuned guesses. The `mahjong/ml/` pipeline replaces them with weights fitted to actual outcomes:

1. **Generate data** — `datagen.py` plays thousands of seeded games with mixed lineups and records every discard decision from all four seats. Because the simulator sees all hands, every *candidate* tile gets a ground-truth label — `waited_legal`: would discarding this tile deal in right now? That's dense supervision (every hot tile at every decision), not just the ~2% of discards that actually got ronned.
2. **Train** — `train.py` fits a logistic regression on those labels (a gradient-boosting ceiling check confirms how much a fancier model would add), evaluates on held-out games against the hand-tuned baseline, and prints the coefficient table: what the data says each signal is actually worth.
3. **Play** — the model is exported as JSON (features, means, scales, coefficients) and evaluated in pure Python at play time; no numpy/sklearn at inference. `LearnedAgent` is the hybrid with its defense swapped for the model's calibrated deal-in probability, and the `/analysis` endpoint reports that probability per candidate discard (`deal_in_prob`) so the UI can say "this tile deals in 4% of the time" instead of an abstract danger score.

```bash
pip install -e ".[ml]"                                  # numpy + scikit-learn (training only)
PYTHONPATH=src python3 -m mahjong.ml.datagen --games 3000 --out data/
PYTHONPATH=src python3 -m mahjong.ml.train --data data/
PYTHONPATH=src python3 experiments/compare_learned.py   # benchmark vs Hybrid
```

### What the data said (3,000 games, 1.4M candidate discards, held-out 20% of games)

| Ranking hot tiles | ROC AUC | PR AUC |
|---|---|---|
| Hand-tuned heuristic (defense.py) | 0.561 | 0.040 |
| Logistic regression | **0.868** | **0.136** |
| Gradient boosting (ceiling check) | 0.872 | 0.140 |

The model's strongest signals are ones the heuristic underweights: *it's late*
(+0.71), *the neighbouring ranks are dead* (−0.59), *this exact tile is already
in an opponent's river* (−0.57), *melds are exposed* (+0.50). The four heuristic
signals it was built to re-weight all landed near zero — including
opponent_threat, the old blend's 0.35-weight star. A win-probability model
(AUC 0.719 vs 0.648 for shanten alone) is trained by the same script and feeds
the EV agent.

Two caveats worth stating plainly, both found by auditing this pipeline:

- **Global AUC flatters the model.** It pools rows across every decision, which
  punishes a scorer that ranks correctly *within a hand* but whose absolute
  scale drifts between situations — and within-hand ranking is the only job the
  heuristic ever had. Measured fairly, on decisions containing both a hot and a
  cold tile, it is **0.72 for the model vs 0.61 for the heuristic**. A clear
  win, but not the coin-flip gap the pooled number suggests.
- **The train/test split was once broken.** `datagen` assigns lineups by
  `game_id % 5` and the split held out `game_id % 5 == 4`, so the test set was
  exactly one lineup that training never saw. Now a seeded random split of
  games, with lineup balance asserted on every run. (The headline AUC survived
  the fix — 0.867 → 0.870 — but that was luck, not design.)

### The integration lesson (honest numbers)

Better prediction did not instantly make a better player:

| 300 games each, seats alternate | Win% | DI/disc% |
|---|---|---|
| Mirror: 4× Learned | 23.8/seat | 1.44 |
| vs Hybrid — Hybrid / **Learned** | 26.2 / **21.8** | 1.30 / **1.90** |
| vs Greedy — Greedy / **Learned** | 29.5 / **19.7** | 2.08 / **1.69** |

Against concealed hands the model's best features go quiet (nothing exposed,
nothing proven safe), its calibrated ~2% probabilities flatten the defense term,
and the agent drifts toward pure offense — dealing in *more* than the crudely
fearful heuristic it replaced. A calibrated probability squashed into a slot
built for an inflated 0–1 danger score loses the caution the blend was balanced
around.

### The fix: decide in points, not scores

The current LearnedAgent scores every candidate discard in actual points:

```
EV(tile) = P(win | hand after discard) × 8.63  −  P(deal-in on tile) × 8.55
```

— both probabilities from the trained models, both stakes measured empirically
over seeded games. No normalisation, no dynamic weights, no squash. Folding
emerges naturally: with a hopeless hand every candidate's P(win) is flat and
tiny, so the risk term decides and the agent discards its safest tile without
an explicit "defense mode".

Those two stakes are nearly equal under this table's rules, and that is
informative: with instant chip payouts off, a ron is a straight transfer — the
shooter pays exactly what the winner collects — so risking a deal-in costs
almost precisely what winning pays.

### The verdict: not yet better than the heuristic

Measured on the duplicate-seating evaluator (`experiments/duplicate.py`),
1,000 games, every agent playing every seat on every wall:

| | Win% | DI/disc% | Pts/seat |
|---|---|---|---|
| Hybrid | 24.0 [22.2, 25.9] | 1.59 [1.42, 1.77] | +0.130 |
| **Learned (EV)** | 22.1 [20.4, 24.0] | **1.36 [1.21, 1.53]** | −0.130 |

```
win rate      hybrid − learned = +1.850 pp   z=+1.39  p=0.1650  (not significant)
deal-in rate  hybrid − learned = +0.223 pp   z=+1.89  p=0.0585  (not significant)
points diff   +1.040  95% CI [−1.336, +3.416]           (not significant)
```

**Nothing here is significant.** The EV agent is the safer discarder and the
weaker winner, and the two cancel: on net points the confidence interval
comfortably straddles zero. The honest summary is that a model which is
overwhelmingly better at *predicting* deal-ins produces an agent that is
merely *equal* at playing — the gap between a good world model and a good
policy, which is what the next phase attacks by learning the decision itself
rather than hand-writing the formula that consumes the predictions.

## Scoring (tai)

Every win is scored as a list of named tai items (`scoring.py`), so the UI can show players exactly why a hand is worth what it is. Point value doubles per tai (`2^(tai−1)`), capped at the tai limit — set per game in the setup screen (5-10, default 6).

| Rule | Tai |
|---|---|
| Self-draw (自摸) | 0 — changes who pays, not the tai |
| No flowers or animals (无花) | 1 |
| Matching seat flower / animal | 1 each |
| All four animals | +1 on top of the four |
| Complete flower series (一套花) | 1 |
| Dragon triplet / seat wind / prevailing wind | 1 each |
| Kong replacement win (杠上开花) / robbing the kong (抢杠) / last tile (海底捞月) | 1 each |
| All triplets (碰碰胡) | 2 |
| Half flush (混一色) | 2 |
| Little three dragons (小三元) | +2 over the dragon pongs |
| Full flush (清一色) | 4 |
| Ping Hu — all chows, non-honor pair, zero bonus tiles (平胡) | 4 |
| Chou ping hu — same shape but holding flowers/animals (臭平胡) | 1 |
| Concealed ping hu — either variant with no claimed melds (门清平胡) | +1 |
| Big three dragons, four winds, all honors, all terminals | limit (6 by default) |

House rules live in `ScoreConfig` and are all adjustable: minimum 1 tai to win (chicken hands blocked — confirmed table rule), shooter pays all three shares on a ron, tsumo collects from everyone. Self-draw adds **no tai** at this table — its reward is collecting from all three players, a payment-structure edge the advisor will weight when payouts return. Accounting is **tai-only for now**: instant chip payouts for animals, flower series, and kongs are off by default (the machinery remains one config flag away). Special hands (thirteen orphans, heaven/earth wins) are not yet implemented.

**Wall mechanics.** Normal turn draws come off the front of the wall; replacement draws — after a flower, an animal, or a kong — come off the back, as at a real table. The hand ends in a draw when 15 live tiles remain, regardless of which end they left from. East (the dealer) always draws first.

## Sessions

`session.py` chains hands into a full game: four prevailing-wind rounds, dealer repeats on a win or draw (连庄) and rotates otherwise, seat winds follow the dealership, and payments accumulate into running scores. One seed reproduces an entire session.

```bash
PYTHONPATH=src python3 -m mahjong.session   # demo: one East round, 2 Greedy vs 2 Hybrid
```

## Playing against the bots (engine API)

The game loop is a generator that yields decision requests, so a human can sit in any seat. `InteractiveGame` pauses whenever a human seat must act and exposes a JSON-ready, redacted view of the table — your tiles, everyone's discards and melds, and opponents' tile *counts* only:

```python
from mahjong.interactive import InteractiveGame
from mahjong.agents import HybridAgent

game = InteractiveGame([HybridAgent(f"Bot{i}") for i in range(4)],
                       human_seats={0}, seed=1)
game.start()                      # runs until it's your decision
while not game.game_over:
    view = game.view_for(0)       # what seat 0 is allowed to see
    game.submit(answer)           # answer game.pending; bots then play on
```

This is the surface the upcoming web UI talks to. The agent attached to a human seat can still be consulted for hints ("what would the hybrid bot do here?") via `game.game.dispatch_to_agent(game.pending)`.

## Playing in the browser

```bash
docker compose up --build
# → app at http://localhost:5173, API at http://localhost:8000 (docs at /docs)
```

The React frontend (`frontend/`) implements the "Mahjong Trainer" design (see `design_handoff_mahjong_trainer/`): a felt table with the three bots, your clickable hand with per-tile danger heat, claim/kong prompts with an auto-pass countdown, and the coach sidebar — shanten meter, discard advisor with the four-signal danger breakdown, opponent threat gauges, and a hint button. The game-end screen shows the winner's revealed hand and the tai receipt with named scoring items. The Analysis toggle (or "no training wheels") hides all coaching for serious play.

For frontend development:

```bash
cd frontend && npm install && npm run dev   # Vite dev server on :5173
```

## Web API

The FastAPI backend wraps `InteractiveGame`:

```bash
pip install -e ".[server]"
uvicorn mahjong.server.app:app --reload
```

| Endpoint | Purpose |
|---|---|
| `POST /games` | create a game (`seed`, `human_seat`, `bots` — 4 of random/greedy/defensive/hybrid) |
| `GET /games/{id}` | redacted view for the human seat |
| `POST /games/{id}/action` | answer the pending decision (discard / claim / chow / kong) |
| `GET /games/{id}/hint` | what the seat's own bot would do |
| `GET /games/{id}/analysis` | learning stats: shanten, per-discard acceptance + danger breakdown, opponent threats |
| `WS /games/{id}/ws` | view pushed on connect and after every action |
| `GET /tiles` | tile id → name/suit/rank metadata |

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

- Special hands: thirteen orphans, heaven/earth wins
- Post-game review screen ("Review game with coach") over a hand event log
- Multi-hand browser sessions backed by session.py (dealer rotation in the UI)
- LLM move explanations (`/explain`) grounded in the analysis endpoint + RAG strategy corpus
- Value-aware win model (predict points, not just P(win)) so the EV agent pushes big hands
- Lookahead: sample future draws, estimate discard value
- Tune hybrid weights to detected opponent strength

## What this project covers

- State representation and feature engineering
- Heuristic search with pruning and block-capping
- Multi-objective decision making (offense vs defense)
- Opponent modelling from observable information
- Supervised learning on simulation data: perfect-information labels,
  group-wise train/test splits, calibration, interpretable models
- Experiment design with controlled baselines and normalised metrics
- Catching a misleading metric (DI/game vs DI/disc%)