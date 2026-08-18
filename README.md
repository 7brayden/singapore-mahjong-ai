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

The LearnedAgent scores every candidate discard in actual points:

```
EV(tile) = (1 − P(deal-in on tile)) × V(hand after discard)  −  P(deal-in on tile) × 8.55
```

P(deal-in) comes from the danger model. V comes from the **hand-value model**
(Phase A): expected net points of the post-discard state, fitted on realized
outcomes over 10,000 games with tai-potential features — flush concentration,
dragon/wind triplet trajectories, chow-vs-pong shape, banked bonus tai, and
the two gates this table makes expensive (`is_concealed`, `has_bonus_tiles`).
An earlier version priced every win at one constant, which is exactly why it
folded hands worth fighting for.

The value model's coefficients recover the house rules from outcomes alone:
`bonus_tai` +1.78 (banked flowers/animals pay), `has_bonus_tiles` **−0.87**
(net of banked tai, merely holding a bonus tile costs value — the 臭平胡
degradation, learned, never coded), `run_progress` +0.93 vs `triplet_progress`
+0.12 (ping hu 4 vs all-triplets 2), `is_concealed` +0.36 (门清平胡). Its R²
is an honest 0.066 (GBM ceiling 0.088): from one mid-hand snapshot, a hand's
final ledger is mostly future luck — the model's job is ranking candidate
states, not prophecy.

### The verdict: a genuinely different player, not yet a better one

Duplicate-seating evaluation (`experiments/duplicate.py`), 1,000 games, every
agent playing every seat on every wall:

| | Win% | DI/disc% | Pts/seat |
|---|---|---|---|
| Hybrid | 24.5 [22.7, 26.4] | 1.65 [1.49, 1.82] | −0.062 |
| **Learned (value-aware EV)** | 20.4 [18.7, 22.3] | **1.14 [1.01, 1.29]** | **+0.062** |

```
deal-in rate  hybrid − learned = +0.506 pp   z=+4.51  p<0.0001  (significant)
win rate      hybrid − learned = +4.050 pp   z=+3.07  p=0.0022  (significant)
points diff   learned +0.500/seed  95% CI [−1.836, +2.836]      (not significant)
```

The first statistically established differences in the project: the
value-aware agent deals in far less (p<0.0001) and wins genuinely fewer
hands (p=0.0002) — a real style, not noise.

### Phase B: the claims learned too

Pong/chow decisions are no longer `if shanten <= 2` rules. There is
deliberately no supervised "claim model" — the heuristic agents that
generated the data only claim when already close to winning, so observed
claims are hopelessly confounded. Instead every claim window is a branch
comparison under the same value model: *V(meld formed, then best-EV discard —
its forced discard carrying the usual deal-in risk)* versus *V(hand as it
stands)*. Claim iff the melded branch is worth more points. The concealment
bonus, the ping hu forfeit, the shape change — all priced by coefficients
fitted to outcomes, none of it hand-coded.

The result contradicted the obvious prediction: value-priced claims got MORE
frequent, not less (4.4 vs 3.5 melds/game in mirror play). A claim converts a
partial into a meld without spending a draw, and the model prices that
tempo at roughly the concealment tax — so after the first meld, claims are
nearly free. The style paid:

| 1,000 games, duplicate seating | Win% | DI/disc% | Pts/seat |
|---|---|---|---|
| Hybrid | 25.3 [23.4, 27.3] | 1.81 [1.64, 2.00] | −0.181 |
| **Learned (value discards + claims)** | 20.3 [18.6, 22.1] | **1.24 [1.10, 1.40]** | **+0.181** |

```
deal-in rate  p<0.0001 (significant)      win rate  p=0.0002 (significant)
points diff   learned +1.448/seed  95% CI [−0.951, +3.847]  (not yet significant)
```

At 1,000 games the learned side led on points (+1.45/seed, not significant),
so we ran the properly powered test: 3,000 fresh seeds, 6,000 games. It
reversed the sign:

```
6,000 games — points diff: hybrid +1.275/seed  95% CI [+0.285, +2.265]
→ HYBRID is genuinely ahead (~0.16 pts/seat/game)
deal-in:  learned 1.22% vs 1.68%  p<0.0001 — learned confirmed safer
win rate: learned 20.4% vs 24.9%  p<0.0001 — and confirmed slower
```

The two earlier 1,000-game leads were noise — the exact trap the evaluator
exists to catch, caught. The arithmetic of the loss is instructive: deal-ins
are rare (~1.5 per 100 discards), so cutting them 27% saves only ~0.03
pts/seat, while conceding 4.5 pp of win rate costs ~0.40. The agent
over-folds — a predictable consequence of pairing a sharp risk model
(deal-in AUC 0.87) with a fuzzy value model (R² 0.066): the EV comparison
trusts its precise term and under-weights its noisy one. The learned player
is statistically the safest and most value-selective at the table, and
statistically the weaker points earner. Fixing the value signal — richer
features, a stronger regressor, or self-play data — is where the next
strength comes from.

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

## The coach

`POST /games/{id}/explain` turns a live decision into a short lesson. The
split is deliberate and load-bearing: **the engine computes every number**
(shanten, deal-in probability, hand value, opponent threats, and the learned
agent's recommendation), **retrieval picks the principles** that fit the
situation from `mahjong/coach/corpus.py` — a small tagged knowledge base
encoding *this table's* house rules alongside strategy measured in this
project's own simulations — and **the LLM only writes prose**. Its system
prompt forbids inventing numbers, and it explains trade-offs rather than
defending the bot: the recommended move is just the argmax of the same
numbers shown to the player.

Retrieval is lexical tag-matching over ~15 curated chunks, not embeddings —
at this corpus size an index would be ceremony. Revisit past ~50 chunks.

### Backends

Selected from the environment; the first configured credential wins, and
`COACH_PROVIDER` (`azure` | `anthropic` | `template`) pins one explicitly.

| Provider | Environment |
|---|---|
| **OpenAI-compatible** | `COACH_BASE_URL`, `COACH_MODEL`, optional `COACH_API_KEY` |
| **Azure OpenAI** | `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`, optional `AZURE_OPENAI_API_VERSION` |
| **Anthropic** | `ANTHROPIC_API_KEY`, optional `COACH_MODEL` (default `claude-opus-5`) |
| **Template** | none — always available |

The generic `openai` provider takes any endpoint speaking the OpenAI
chat-completions API — OpenAI itself, Ollama, LM Studio, vLLM, OpenRouter, a
gateway. It exists because vendors disappear: an Azure lab sandbox was deleted
with its DNS record, and GitHub Models entered retirement, within a week of
each other. Which endpoint answers is configuration, not code.

```
# OpenAI
COACH_BASE_URL=https://api.openai.com/v1
COACH_API_KEY=sk-...
COACH_MODEL=gpt-4o-mini

# Ollama on your own machine (free, no key)
COACH_BASE_URL=http://localhost:11434/v1     # host.docker.internal from Docker
COACH_MODEL=llama3.2
```

`AZURE_OPENAI_DEPLOYMENT` is the deployment *name you chose in the Azure
portal*, not a public model id; using a model id there is the usual cause of
a 404. Put the values in a `.env` file beside `docker-compose.yml`
(gitignored) — they stay server-side and never reach the browser.

The template backend is not an error path. With no credentials, an
unreachable endpoint, a refused request, or an SDK that isn't installed, the
coach renders the same content deterministically — it degrades to *less
fluent*, never to *broken*. Explanations are cached per decision state, so
re-clicking never re-bills.

That silence is right during play and wrong while setting up — a mistyped
deployment name looks exactly like no credentials. Check the configuration
directly instead of guessing:

```bash
pip install -e ".[server,coach]"
PYTHONPATH=src python3 -m mahjong.coach.check
```

It prints the selected provider and every variable (keys masked), makes one
tiny real call, and on failure translates the error — a 404 into "that's a
model id, not your deployment name", a 401 into "wrong key for this
resource". Exits non-zero on failure, so it works in CI too.

**Before any public deployment**, replace the single shared server-side key
with per-user credentials or metered access: as written, every visitor's
explanations bill one account.

## Future work

- Special hands: thirteen orphans, heaven/earth wins
- Post-game review screen ("Review game with coach") over a hand event log
- Multi-hand browser sessions backed by session.py (dealer rotation in the UI)
- Per-user LLM billing before any public deployment (see The coach, below)
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