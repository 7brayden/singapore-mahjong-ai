"""The coach: engine numbers + retrieved principles → a short lesson.

Division of labour, fixed by design:
  - The ENGINE computes every number (shanten, deal-in probabilities,
    hand values, threats) and the recommended action (the learned
    agent's argmax).
  - RETRIEVAL picks the strategy/rules passages relevant to the
    situation from a corpus that encodes THIS TABLE's house rules.
  - The LLM only turns those into prose. It never picks moves, never
    invents numbers, and its output is advice about trade-offs — not a
    defence of whatever the bot would do.

Works without an API key: explain.py falls back to a deterministic
template over the same numbers, so the product degrades to "less
fluent", never to "broken".
"""

from mahjong.coach.explain import explain_situation  # noqa: F401
