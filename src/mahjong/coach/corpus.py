"""The coach's knowledge base: this table's rules and the strategy
principles behind the engine's numbers.

Every chunk is tagged; retrieval scores tag overlap against the tags of
the live situation (see retrieve.py). The rules chunks encode the HOUSE
rules confirmed by the table owner — a generic Singapore mahjong text
would confidently teach rules this table does not play by (self-draw
tai, for one). Several strategy chunks state findings measured in THIS
project's simulations rather than folklore; those cite their numbers.

Kept as Python data, not markdown files: the corpus is small, versioned
with the code that builds situations from it, and grep-able.
"""

CHUNKS = [
    # ── House rules ──────────────────────────────────────────────────
    {
        "id": "rules-ping-hu-family",
        "title": "Ping hu, chou ping hu, and the concealed bonus",
        "tags": ["chow_shape", "concealed", "bonus_tiles", "claim_window",
                 "scoring"],
        "text": (
            "At this table an all-chows hand with a non-honor pair scores "
            "4 tai (平胡) ONLY while it holds zero flowers and animals. "
            "Pick up any bonus tile and the same shape collapses to chou "
            "ping hu (臭平胡), worth 1 tai. A ping hu of either kind with "
            "no claimed melds adds 1 more tai (门清). Practical upshot: a "
            "concealed clean ping hu is a 5-tai hand, and a single chow "
            "claim or flower can cost most of that value."),
    },
    {
        "id": "rules-tsumo-payment",
        "title": "Self-draw pays wider, scores nothing",
        "tags": ["tenpai", "scoring", "push"],
        "text": (
            "Self-draw (自摸) adds NO tai at this table — it changes who "
            "pays, not what the hand is worth: on a tsumo all three "
            "opponents pay, on a ron the shooter alone pays all three "
            "shares. A hand with zero tai cannot win even by self-draw, "
            "because of the 1-tai minimum."),
    },
    {
        "id": "rules-bonus-tiles",
        "title": "Flowers and animals",
        "tags": ["bonus_tiles", "scoring"],
        "text": (
            "Each flower matching your seat scores 1 tai (both series "
            "count — East matches flower #1 in each). Each animal is 1 "
            "tai; collecting all four animals adds 1 more (4+1). A "
            "complete flower series (#1-#4 of one series) adds 1. A hand "
            "with no bonus tiles at all scores 1 tai (无花) — which is "
            "why drawing a flower can genuinely REDUCE a hand's value: "
            "it kills 无花 and degrades ping hu to chou ping hu."),
    },
    {
        "id": "rules-minimum-tai",
        "title": "The chicken-hand gate",
        "tags": ["scoring", "tenpai", "cheap_hand"],
        "text": (
            "A complete hand needs at least 1 tai to declare a win — "
            "chicken hands (鸡胡) cannot win here. If your only route to "
            "completion is a 0-tai shape, you are drawing dead: build a "
            "tai source (seat flower already banked, a dragon pong, half "
            "flush, or keep the hand a clean ping hu) or treat the hand "
            "as a folding hand."),
    },
    {
        "id": "rules-kong",
        "title": "Kongs",
        "tags": ["kong", "claim_window"],
        "text": (
            "Kongs come three ways: concealed (four in hand), added (the "
            "fourth tile onto your exposed pong), exposed (claiming a "
            "discard with three in hand). Every kong draws a replacement "
            "from the BACK of the wall; winning on that replacement "
            "scores 杠上开花. An added kong can be robbed (抢杠) — an "
            "opponent waiting on that tile wins off you the moment you "
            "declare. Declare added kongs only when nobody looks tenpai."),
    },

    # ── Strategy principles (several measured in this project) ───────
    {
        "id": "principle-shanten",
        "title": "Shanten first",
        "tags": ["far", "efficiency", "discard"],
        "text": (
            "Shanten — how many tile exchanges from ready — is the "
            "strongest single predictor of winning: in our simulations "
            "it dwarfs every other factor. Early game, take the discard "
            "that lowers shanten or, at equal shanten, keeps the most "
            "ways to improve. Defense earns its keep later; a hand that "
            "never approaches tenpai wins nothing."),
    },
    {
        "id": "principle-acceptance",
        "title": "Keep your outs wide",
        "tags": ["efficiency", "discard", "far"],
        "text": (
            "Between discards that leave equal shanten, prefer the one "
            "accepting more distinct tiles. Acceptance is your draw "
            "flexibility: a two-sided run piece (5-6 waiting 4 or 7) is "
            "roughly twice the outs of an edge or closed wait. This is "
            "what the acceptance number in the advisor counts."),
    },
    {
        "id": "principle-river-repeats",
        "title": "Repeated tiles are safer",
        "tags": ["danger", "hot_table", "discard", "defense"],
        "text": (
            "A tile already sitting in opponents' rivers is much less "
            "likely to deal in — our deal-in model's strongest safety "
            "signal is exactly 'copies of this tile already discarded'. "
            "Note the reason is statistical, not a rule: this table has "
            "no furiten, so a past discard does not LOCK an opponent out "
            "— it just means nobody wanted it when it mattered."),
    },
    {
        "id": "principle-adjacency",
        "title": "Dead neighbours make safe tiles",
        "tags": ["danger", "defense", "discard", "hot_table"],
        "text": (
            "Most rons complete a run, so a tile whose neighbours (±1, "
            "±2) are heavily visible has few live waits running through "
            "it — if the 4s and 6s are nearly all gone, 5s has little "
            "left to complete. Terminals and honors are safest of all "
            "late: only pairs and triplets want them."),
    },
    {
        "id": "principle-lateness",
        "title": "Late tiles are hot tiles",
        "tags": ["late_game", "danger", "defense"],
        "text": (
            "Danger rises steeply with the turn count — lateness is the "
            "single biggest risk factor our deal-in model found. By the "
            "last third of the wall, assume at least one opponent is "
            "tenpai: every fresh tile you discard (one no player has "
            "shown is safe) is a real risk, and with this table's "
            "shooter-pays-all, one deal-in outweighs several small wins."),
    },
    {
        "id": "principle-exposed-melds",
        "title": "Read exposed melds as a clock",
        "tags": ["danger", "defense", "opponent_threat", "hot_table"],
        "text": (
            "Each meld an opponent exposes means fewer tiles they still "
            "need — two or more exposed melds is a loud tenpai warning, "
            "and exposed-meld count is a top danger signal in our "
            "model. Watch WHICH suit they melded: feeding the suit a "
            "half-flush hand is collecting multiplies the damage."),
    },
    {
        "id": "principle-concealment",
        "title": "Concealment is worth real points",
        "tags": ["concealed", "claim_window", "chow_shape"],
        "text": (
            "Staying concealed is worth roughly 0.7 points of hand value "
            "in our measurements — the concealed ping hu bonus plus the "
            "information you deny opponents. The first claim spends that "
            "premium forever. It is not a rule against claiming; it is a "
            "price tag: the claim must buy at least that much progress."),
    },
    {
        "id": "principle-claim-tempo",
        "title": "A claim is a free draw — priced in exposure",
        "tags": ["claim_window", "efficiency", "chow_shape"],
        "text": (
            "Claiming converts a partial set into a fixed meld without "
            "spending a draw — pure tempo. Our value model prices that "
            "tempo at about one shanten step, roughly the same as the "
            "concealment premium, which is why the first claim of a hand "
            "is the expensive one and later claims are nearly free. But "
            "a claim also forces an immediate discard: on a hot table "
            "that forced throw can cost more than the tempo buys."),
    },
    {
        "id": "principle-push-fold",
        "title": "Push or fold is arithmetic",
        "tags": ["defense", "danger", "late_game", "push", "cheap_hand"],
        "text": (
            "Compare in points, not vibes: chance of completing times "
            "what the hand pays, against deal-in chance times the "
            "shooter's bill (about 8.5 points here, since the shooter "
            "pays all three shares). A cheap hand justifies almost no "
            "risk — folding a 1-tai hand costs little. A 5-tai concealed "
            "ping hu justifies pushing through moderate danger. When "
            "every discard is hot and the hand is poor, throw your "
            "safest tile and stop advancing."),
    },
    {
        "id": "principle-fat-hands",
        "title": "Fewer, fatter wins",
        "tags": ["scoring", "cheap_hand", "chow_shape", "flush_track",
                 "push"],
        "text": (
            "Because value doubles per tai (2^(tai−1)), one 4-tai hand "
            "outpays four 1-tai hands — and our strongest agent wins "
            "FEWER hands than the fast ones yet more points, by folding "
            "cheap hands early and pushing valuable ones. Before "
            "committing a hand, ask what it will actually pay if it "
            "wins; 'complete anything' is a losing plan at this table."),
    },
    {
        "id": "principle-flush-track",
        "title": "Flush tracks multiply value",
        "tags": ["flush_track", "scoring", "efficiency"],
        "text": (
            "A hand leaning heavily into one suit is on the half-flush "
            "(2 tai) or full-flush (4 tai) track — often the cheapest "
            "big-hand upgrade available, because it needs no rare tiles, "
            "just discipline about which draws to keep. The cost is "
            "telegraphing: opponents watch you dump two suits and stop "
            "feeding the third, so flush hands get harder to finish "
            "late. Commit early or not at all."),
    },
    {
        "id": "principle-honor-pongs",
        "title": "Dragons and your winds are compact tai",
        "tags": ["scoring", "efficiency", "bonus_tiles"],
        "text": (
            "A dragon pong is 1 tai in three tiles, stackable with "
            "everything; seat and prevailing wind pongs likewise (both "
            "at once if they coincide). A pair of dragons early is "
            "worth holding a few extra turns — completing it converts a "
            "chicken-risk hand into a legal one. Lone honors, by "
            "contrast, are your safest early discards."),
    },
]
