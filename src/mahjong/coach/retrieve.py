"""Situation-aware retrieval over the coach corpus.

Deliberately lexical, not vector: the corpus is ~15 curated chunks, so
the honest retrieval problem is "which principles apply to this game
state", and the game state already knows its own structure. We derive
tags from the live analysis payload (tenpai? claim window? flush track?
hot table?) and rank chunks by tag overlap. Deterministic, dependency-
free, and debuggable — an embedding index over 15 documents would be
ceremony. Revisit if the corpus grows past ~50 chunks.
"""

from typing import Dict, List

from mahjong.coach.corpus import CHUNKS


def situation_tags(situation: Dict) -> List[str]:
    """Derive retrieval tags from the situation dict built in explain.py."""
    tags = ["scoring"]
    pending = situation.get("pending_type")
    if pending in ("claim", "chow"):
        tags.append("claim_window")
    if pending == "kong":
        tags.append("kong")

    shanten = situation.get("shanten")
    if shanten is not None:
        if shanten <= 0:
            tags += ["tenpai", "push"]
        elif shanten >= 3:
            tags += ["far", "efficiency"]
        else:
            tags.append("efficiency")

    if situation.get("is_concealed"):
        tags.append("concealed")
    if situation.get("has_bonus_tiles"):
        tags.append("bonus_tiles")
    if situation.get("flush_track"):
        tags.append("flush_track")
    if situation.get("chow_shape"):
        tags.append("chow_shape")
    if situation.get("hand_value") is not None and situation["hand_value"] < 2.0:
        tags.append("cheap_hand")

    turn_frac = situation.get("turn_frac", 0.0)
    if turn_frac > 0.6:
        tags += ["late_game", "defense", "danger"]
    max_threat = situation.get("max_opp_threat", 0.0)
    max_deal_in = situation.get("max_deal_in_prob", 0.0)
    if max_threat > 0.35 or max_deal_in > 0.04 or situation.get("opp_melds", 0) >= 2:
        tags += ["hot_table", "danger", "defense", "opponent_threat"]
    if pending == "discard":
        tags.append("discard")
    return tags


def retrieve(situation: Dict, k: int = 4) -> List[Dict]:
    """Top-k chunks by tag overlap; stable order for determinism."""
    tags = situation_tags(situation)
    weights = {}
    for t in tags:  # repeated tags weigh heavier
        weights[t] = weights.get(t, 0) + 1

    scored = []
    for idx, chunk in enumerate(CHUNKS):
        score = sum(weights.get(t, 0) for t in chunk["tags"])
        if score > 0:
            scored.append((-score, idx, chunk))
    scored.sort()
    return [chunk for _, _, chunk in scored[:k]]
