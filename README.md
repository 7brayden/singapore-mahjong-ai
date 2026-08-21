# Singapore Mahjong AI

I built this to get better at Singapore mahjong. It grew into three things:

- a rules-accurate game engine for Singapore mahjong (tai scoring, flowers, animals, the works)
- bots to play against, including one trained on outcomes from 10,000 simulated games
- an AI coach that explains every decision while you play in the browser

Everything runs locally. No accounts, no API keys, nothing to download — the trained models ship in the repo.

## Play it

You'll need [Docker](https://docs.docker.com/get-docker/).

```bash
git clone https://github.com/7brayden/singapore-mahjong-ai.git
cd singapore-mahjong-ai
docker compose up --build
```

Open **http://localhost:5173**. Pick your three opponents, and play a full session — East 1 through North 4, with the dealership rotating like at a real table. The coach sidebar shows your shanten, which discards are safe, how threatening each opponent looks, and will explain any decision if you ask. Hide it when you want to play for real.

If you don't want Docker:

```bash
pip install -e ".[server]"
PYTHONPATH=src uvicorn mahjong.server.app:app --port 8000   # terminal 1
cd frontend && npm install && npm run dev                    # terminal 2
```

### Give the coach a voice (optional, free)

Out of the box the coach writes its explanations from a fixed template — correct numbers, plain wording. Point it at a local LLM and it writes actual prose. The easiest way is [Ollama](https://ollama.com):

```bash
ollama pull qwen2.5:7b
OLLAMA_KEEP_ALIVE=30s ollama serve
```

Then make a `.env` file next to `docker-compose.yml`:

```
COACH_BASE_URL=http://host.docker.internal:11434/v1
COACH_MODEL=qwen2.5:7b
```

Restart with `docker compose up -d` and check the wiring with `docker compose exec api python -m mahjong.coach.check`. Cloud providers (OpenAI, Azure, Anthropic) work too — the coach speaks to anything with an OpenAI-compatible API. Keys stay server-side and never reach the browser.

One warning if you host this publicly: every visitor's explanations bill whatever key you configured. Don't put a shared key behind a public URL.

## The rules

Scoring follows the standard Singapore reference rules, with the house choices my table plays: minimum 1 tai to win, self-draw adds no tai (it just makes everyone pay), shooter pays all three shares, ping hu needs a two-tile wait to win by discard. All the special hands are in — thirteen wonders, eight flowers, hidden treasure, nine gates, kong on kong, heavenly/earthly/humanly hands. Every house rule lives in one config object (`ScoreConfig`), so if your table plays differently, change it there.

If you don't play mahjong: you're trying to build a winning hand of four sets plus a pair, opponents can steal your discards, and hands are scored in "tai" — doubling points for harder patterns. A hand worth zero tai can't win at all, which turns out to matter a lot (see below).

## The bots

There are five, from `random` (loses every game) up to `learned`. The learned one uses three small models trained on simulated games:

- a danger model: how likely is this discard to hand an opponent the win? (it's right about 87% of the time when ranking a hot tile against a safe one)
- a win model: how likely is this hand to win from here?
- a value model: how many points is this hand actually worth?

Every discard and every claim is then just expected value: chance of winning times what the hand pays, minus the chance of feeding someone. The models are plain logistic/ridge regressions exported to JSON — you can open the files and read every weight. No neural nets, no GPU, and honestly the interpretability has caught more bugs than cleverness ever would have.

The most useful lesson in the whole project came from a bug: the claim advisor once told me to pong my own pair out of a live ping hu — trading a 4-tai hand for one that couldn't legally win. The model had priced that mistake too cheap, and nothing in the code knew the legality rule at all. The fix wasn't a smarter model. It was admitting that "a 0-tai hand cannot win" is law, not statistics: the engine now computes what every hand can still score (`tai_track.py`), refuses claims that make a hand legally dead, and the UI tells you the cost of a claim no matter what the bot thinks. Then the models were retrained with the law as features, and training now refuses to export any model that believes dead hands win. Engine as rulebook, model as beliefs.

How good is the learned bot? After both it and the strongest heuristic bot got the legality gate, they're in a statistical dead heat on points over 6,000 paired games — but the learned one deals in noticeably less (1.2% of discards vs 1.7%). It plays like a careful player who picks its fights.

## The coach

The coach never invents anything. The engine computes the numbers (shanten, deal-in odds, which tai tracks are still alive), a small tagged corpus supplies the strategy principles, and the LLM only turns that into sentences. Its system prompt forbids making up statistics, and the engine's facts are marked authoritative — so when you're holding a flower, it won't dangle a 4-tai ping hu that stopped being possible ten turns ago.

## Poking around

```
src/mahjong/
├── tiles.py, hand.py      tile encoding, shanten calculation, win detection
├── game.py                the game loop: draws, claims, kongs, wins
├── scoring.py             tai scoring and payments (all house rules here)
├── tai_track.py           what can this hand still score? (the legality gate)
├── agents/                the five bots
├── ml/                    features → data generation → training → inference
├── coach/                 the explainer: corpus, retrieval, LLM providers
└── server/                FastAPI backend for the web app

frontend/                  React app (the table)
experiments/               benchmarks, incl. the duplicate-seating evaluator
tests/                     pytest suite (~180 tests)
```

Run the tests:

```bash
pip install -e ".[dev]"
pytest
```

Retrain the models from scratch (only needed if you change the features or the rules — takes a couple of hours):

```bash
pip install -e ".[ml]"
PYTHONPATH=src python3 -m mahjong.ml.datagen --games 10000 --out data/
PYTHONPATH=src python3 -m mahjong.ml.train --data data/
PYTHONPATH=src python3 experiments/duplicate.py --games 3000   # did it get better?
```

The evaluator plays every matchup twice with the seats swapped on identical walls, so luck cancels out and you get real confidence intervals instead of vibes.

There's also a little CLI advisor you can use at a real table:

```bash
pip install -e .
mahjong-advisor
```

Type your fourteen tiles (`1w 2w 3w 4t 4t ...`) and it tells you what to discard.

## License

[MIT](LICENSE). Use it, learn from it, build on it.
