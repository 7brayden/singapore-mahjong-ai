# Handoff: Singapore Mahjong Trainer — web UI

## Overview
A desktop-first web app where one human plays Singapore Mahjong against 3 AI bots while a
live analysis sidebar coaches them. Core layout idea: **play on the left, learn on the right.**

Four states are designed: the table (your-turn), the claim prompt, the kong offer, the
bots-thinking wait, the game-end overlay, and the setup screen.

Tone: premium board-game app, friendly coach — *not* a casino. Clarity over flash.

## About the Design Files
`Mahjong Trainer.dc.html` in this bundle is a **design reference created in HTML** — a
prototype showing the intended look, layout, and behavior. It is **not production code to
copy**. It uses a small custom streaming-template runtime and inline styles only, which you
should not port.

The task is to **recreate these designs in the target codebase's existing environment**
(React, Vue, Svelte, etc.) using its established component patterns, styling solution, and
state management. If no frontend exists yet, pick the framework that fits the project and
implement there. Read the HTML for exact values; structure the real implementation your own way.

To view it: open the file in a browser. The top bar has a **States** switcher
(Your turn / Claim / Kong / Waiting / Game end / Setup) that jumps between every designed state.

## Fidelity
**High-fidelity.** Colors, typography, spacing, radii, shadows, copy, and motion are final.
Recreate pixel-faithfully. The only intentionally loose parts:
- Bot names, chip totals, seed, and hand contents are sample data.
- Game logic is scripted (3 canned advisor states), not a real engine.

---

## Design Tokens

### Color
| Token | Hex | Use |
|---|---|---|
| `bg/app` | `#0b1216` | Page background |
| `bg/topbar` | `#0e171c` | Top bar |
| `bg/sidebar` | `#0f181d` | Analysis sidebar |
| `bg/card` | `#111b20` | Setup card, game-end card |
| `bg/overlay-card` | `#101b1f` | Claim / kong prompts |
| `felt/base` | `#123029` | Table fallback fill |
| `felt/gradient` | `radial-gradient(115% 85% at 50% 40%, #26604c 0%, #174034 55%, #102b25 100%)` | Table surface |
| `felt/inset-well` | `rgba(8,20,17,.40)` → `.44` | Center pod, seat panels |
| `text/primary` | `#e9e3d5` | Body text on dark |
| `text/secondary` | `#8b9aa1` | Sub copy |
| `text/muted` | `#6f8189` | Meta, labels |
| `text/label` | `#7f8f96` | Uppercase section labels |
| `text/on-felt-secondary` | `#8fb0a0` / `#9dc0ae` | Copy inside the felt |
| `accent/500` | `#e8ac3d` | Primary actions, dealer chip, recommendation |
| `accent/600` | `#d0901f` | Button gradient bottom stop |
| `accent/700` | `#d99a2b` | Eyebrows, small accent text |
| `accent/tint-text` | `#f0d9a6` | Accent text on dark |
| `accent/tint-text-soft` | `#e3cfa5` | Coach body copy |
| `accent/tint-bg` | `rgba(232,172,61,.09)` – `.18` | Coach cards, active states |
| `accent/tint-border` | `rgba(232,172,61,.22)` – `.60` | Coach card + active borders |
| `tile/face` | `linear-gradient(#fdfaf1, #efe8d5)` | Tile face (hand, prompts) |
| `tile/face-river` | `linear-gradient(#fbf7ec, #ece4cf)` | Smaller tiles |
| `tile/edge` | `#cec4aa` | Tile bottom edge (`0 3px 0`) |
| `tile/back` | `linear-gradient(#2f6a56, #1f4a3c)` | Face-down slivers |
| `tile/ink` | `#25333a` | Numerals + neutral honor glyphs |
| `suit/wan 萬` | `#b4483c` | Character suit color (also 中) |
| `suit/tong 筒` | `#2f6fa8` | Circle suit color (also 白) |
| `suit/suo 索` | `#2f7d52` | Bamboo suit color (also 發) |
| `honor/sub` | `#8d949a` | Latin sub-label on honors |
| `safe/500` | `#6fae7f` | Low danger, wall bar, tenpai |
| `safe/text` | `#8fd3a2` / `#a6d3b2` | Tenpai headline / body |
| `warn/500` | `#d6ac47` | Mid danger |
| `danger/500` | `#c6564b` | High danger |
| `danger/text` | `#d99a8f` | Negative chip deltas |
| `hairline` | `rgba(255,255,255,.06)` – `.07` | Section dividers |
| `stroke/subtle` | `rgba(255,255,255,.08)` – `.16` | Card + button borders |
| `surface/subtle` | `rgba(255,255,255,.03)` – `.05` | Inactive chips, ghost buttons |

**Danger heat ramp** (used for tile heat strips, danger %, threat bars, breakdown bars) —
piecewise-linear RGB interpolation on a 0→1 value:
`0.0 = rgb(111,174,127)` → `0.5 = rgb(214,172,71)` → `1.0 = rgb(198,86,75)`.

```js
function heat(v) {
  const stops = [[0,111,174,127],[0.5,214,172,71],[1,198,86,75]];
  let a = stops[0], b = stops[1];
  if (v > 0.5) { a = stops[1]; b = stops[2]; }
  const f = (v - a[0]) / (b[0] - a[0]);
  return `rgb(${[1,2,3].map(i => Math.round(a[i] + (b[i]-a[i]) * f)).join(',')})`;
}
```

### Typography
Google Fonts: `DM Sans` (400/500/700), `Newsreader` (400/500), `DM Mono` (400/500),
`Noto Sans SC` (400/500/700).

- **UI / body**: DM Sans. Tile numerals use the stack `'DM Sans','Noto Sans SC',sans-serif`
  so Latin digits come from DM Sans and CJK glyphs fall back to Noto Sans SC.
- **Display**: Newsreader (serif) — app wordmark 19px, screen titles 30–34px, sidebar
  headline 27px, prompt title 22px, "Coach" 17px.
- **Numerals / seeds / chips**: DM Mono.
- **Chinese**: Noto Sans SC. Chinese scoring terms appear beside English as flavor.
- **Section label**: 10px / `letter-spacing:.14em` / uppercase / `#7f8f96`.
  Eyebrow variant: 10px / `.16em` / uppercase / `#d99a2b`.
- Sizes in use: 9px, 9.5px, 10px, 10.5px, 11px, 11.5px, 12px, 12.5px, 13px, 13.5px, 14px,
  15px, 16px, 17px, 19px, 20px, 22px, 27px, 30px, 34px, 38px.
- Line-height: 1 for numerals/headlines, 1.4–1.6 for prose.

### Spacing / radii / shadows
- Spacing scale in use: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40 px.
- Radii: 2 (heat strip), 3–4 (meld/river tiles), 5–7 (small tiles, chips), 8–9 (buttons, hand tiles),
  10–12 (cards, primary buttons), 14 (seat panels, center pod), 16 (claim card),
  20 (setup/end card), 22 (felt), 999 (pills, switches).
- Shadows:
  - Tile rest: `0 3px 0 #cec4aa, 0 8px 16px rgba(0,0,0,.32)`
  - Tile recommended: `0 0 0 2px #e8ac3d, 0 12px 26px rgba(232,172,61,.4)`
  - River newest tile: `0 0 0 2px rgba(232,172,61,.55)`; claimed tile `0 0 0 2px #e8ac3d, 0 6px 16px rgba(232,172,61,.45)`
  - Felt: `inset 0 0 0 1px rgba(255,255,255,.07), inset 0 60px 120px rgba(0,0,0,.25), 0 24px 60px rgba(0,0,0,.4)`
  - Cards: `0 20px 46px rgba(0,0,0,.45)` (prompt), `0 30px 70px rgba(0,0,0,.45)` (setup), `0 40px 90px rgba(0,0,0,.55)` (end)
  - Primary button: `0 10px 26px rgba(217,154,43,.28)`

---

## Tile rendering (shared primitive)

Every tile is a cream rounded rect with a **large top glyph** and a **small bottom sub-label**.
No sprite sheets, no Unicode mahjong block (inconsistent across platforms).

- Number suits: top = Arabic numeral `1–9` in `tile/ink`, weight 700; sub = suit character
  `萬` / `筒` / `索` in that suit's color.
- Winds: top = `東 南 西 北` in `tile/ink`; sub = `E S W N` in `honor/sub`.
- Dragons: top = `中` (red `#b4483c`) / `發` (green `#2f7d52`) / `白` (slate `#5b7386`);
  sub = `RED` / `GRN` / `WHT` in `honor/sub`.

Sizes:
| Context | Size | Radius | Top / sub font |
|---|---|---|---|
| Human hand + drawn tile | 54 × 78 | 7 | 27 / 14 |
| Claim prompt subject tile | 54 × 74 | 8 | 27 / 14 |
| Game-end winning hand | 40 × 54 | 6 | 20 / 10 |
| Advisor candidate | 38 × 50 | 6 | 19 / 10 |
| Sidebar wait tiles | 34 × 44 | 5 | 17 / 9 |
| Kong banner | 30 × 40 | 5 | 15 / 8 |
| Discard river | 26 × 34 | 4 | 13 / 8 |
| Chow combination | 26 × 34 | 4 | 13 / 8 |
| Exposed melds | 20 × 27 | 3 | 11 / 7 |
| Face-down sliver | 9 × 22 | 2 | — (`tile/back` + `inset 0 0 0 1px rgba(255,255,255,.12)`) |

**Heat strip** (hand tiles + advisor candidates only, only on your turn): absolutely
positioned 4px (3px on advisor) bar, inset 8px left/right, 7px from the bottom, radius 2,
filled with `heat(danger)`. When heat is disabled it becomes `rgba(0,0,0,.06)`.

**Flower / animal chips**: 19 × 23, radius 3, CJK glyph 10px.
Flower = `rgba(251,247,236,.9)` bg / `rgba(255,255,255,.3)` border / `#2f7d52` glyph.
Animal = `rgba(232,172,61,.9)` bg / `rgba(255,255,255,.35)` border / `#3a2708` glyph.

---

## Screens / Views

### 0. App shell (all screens)
- Root: `min-width:1380px; min-height:900px; height:100vh`, column flex, `bg/app`.
- **Top bar**, 56px, `bg/topbar`, bottom hairline, 0 22px padding, space-between:
  - Left: 22 × 28 cream tile mark with `中` in `#1d5c47`; wordmark
    "Singapore Mahjong Trainer" (Newsreader 19px); 1 × 22 divider `rgba(255,255,255,.1)`;
    meta row 12px `#8b9aa1`: `東` · "East round" · "Hand 3 of 16" · `seed 7A2K-91` (DM Mono),
    separated by `·` at 40% opacity.
  - Right: **States switcher** — pill container `rgba(255,255,255,.05)`, 4px padding, radius 999,
    label "STATES" 9px `.13em`, then 6 buttons (12px, 6/11 padding, radius 999); the active one is
    `#e8ac3d` bg with `#23180a` text, others transparent with `#9aa8ae`.
    *This is a demo affordance — drop it or hide it behind a dev flag in production.*
  - Right end: **Analysis toggle** — outlined pill, 30 × 17 track (radius 999) with a
    13px cream knob at `left:2px` off / `left:15px` on, 180ms ease; track `#d99a2b` on,
    `rgba(255,255,255,.16)` off; label "Analysis".

### 1. Table — your turn (primary screen)
`data-screen-label="Table"`. Row flex: felt area `flex:1` (padding `20px 20px 20px 22px`),
sidebar 404px fixed.

**Felt**: `flex:1`, radius 22, 18px padding, `overflow:hidden`, absolute gradient layer +
inset shadow underneath the content layer.

**Felt grid**: `grid-template-columns: 196px minmax(0,1fr) 196px`,
`grid-template-rows: 128px minmax(0,1fr) auto`, `gap:14px`.
- North seat → col 2 / row 1, centered, `align-items:flex-start` (card hugs content — do not stretch).
- West seat → col 1 / row 2, vertically centered.
- East seat → col 3 / row 2, vertically centered.
- Center pod + rivers → col 2 / row 2.
- Human area → col 1 / span 3, row 3.

**Seat panels** (radius 14, 12px padding):
- Normal: bg `rgba(8,20,17,.4)`, border `rgba(255,255,255,.09)`.
- Dealer / high-threat (East): bg `rgba(232,172,61,.09)`, border `rgba(232,172,61,.28)`.
- Content: seat character (Noto SC 14px `#cfe6d8`; `#f0d9a6` for East) + name (13px/500) +
  persona badge (9.5px `.1em` uppercase `#9dc0ae`, 1px `rgba(255,255,255,.16)` border,
  radius 999, 2/7 padding) + dealer chip (`#e8ac3d` bg, `#23180a` text, 9px uppercase) +
  chip count (DM Mono 12px `#ecd9ae`).
- Face-down count: 4 tile-back slivers (2px gap) + `× 7` in DM Mono 11px `#a9c6b7`.
- Exposed melds: groups of 3–4 20 × 27 tiles, 2px gap inside a group, 6–8px between groups.
- Flowers/animals: chip row, right-aligned on the north panel, wrapped on side panels.
- Optional 10.5px "tell" line: `#8fb0a0` normally, `#d8b478` on the East panel
  ("Two 筒 melds and pushing hard — she is 1 away.").
- North panel is 340px wide; side panels fill their 196px column.

**Center** — a 3 × 3 grid (`1fr 190px 1fr` / `1fr auto 1fr`), items centered:
- Rivers at N (top-center, max 190px), W (middle-left, max 150px, right-aligned),
  E (middle-right, max 150px), S (bottom-center, max 190px). Each is a 3px-gap wrap of
  26 × 34 tiles, grouped per player. The **last** tile in each river gets the accent ring;
  during the claim state the East river's last tile gets the stronger glow ring.
- **Center pod**: 176px wide, radius 14, 14px padding, `rgba(8,20,17,.44)` bg,
  `rgba(255,255,255,.09)` border, column, 8px gap, centered:
  1. Wall count — DM Mono 30px `#f4efe1` + "tiles left" 11px `#9dc0ae`.
  2. Wall bar — 4px track `rgba(255,255,255,.12)`, fill `#6fae7f` at `wallRemaining/144`.
  3. Round + dealer — 30px accent-tinted square with `東`, then "East round" 11px /
     "Dealer: Auntie Lim" 10px `#8fb0a0`.
  4. Hairline.
  5. **Turn pill** — 6/12 padding, radius 999. Your turn: `rgba(232,172,61,.18)` bg,
     `rgba(232,172,61,.5)` border, `#f0d9a6` text, animation `turnPulse 2.2s ease-out infinite`.
     Waiting: `rgba(255,255,255,.06)` / `rgba(255,255,255,.12)` / `#cfe6d8`, no pulse, with
     three 4px dots animating `thinkDot 1.2s infinite` staggered 0 / .2s / .4s.
     Labels: "Your turn" · "Mei Ling is thinking" · "Claim window" · "Your turn — kong offered".

**Human area** (row 3, column, 10px gap, centered):
- Info row (full width, space-between):
  - Left: `南` 15px `#f0d9a6`, "You · South" 13px/500, "CONCEALED" badge, "128 chips" DM Mono,
    flower/animal chips.
  - Right: status text 11px `#8fb0a0` ("Click a tile to discard" / "Bots are playing") +
    **Hint button** — `rgba(232,172,61,.12)` bg, `rgba(232,172,61,.45)` border, `#f0d9a6`,
    radius 9, 8/14 padding, 12.5px, prefixed with `師`.
- Hand row: flex, `align-items:flex-end`, `gap:6px`, `padding:20px 0 12px`
  (the top padding is the room for the ★ marker), `opacity:1` on your turn / `.58` otherwise.
  13 tiles, then a **16px spacer**, then the drawn tile.
  - Recommended tile: `translateY(-12px)`, accent ring shadow, and a 12px `★` `#f0d9a6`
    at `top:-15px`, centered. Only the *first* matching tile is marked.
  - Drawn tile: "DRAWN" label 9px `.1em` uppercase `#9dc0ae` at `top:-15px`,
    animation `tileIn .22s ease`.
  - Tiles are buttons; `cursor:pointer` only on your turn. Click = discard.
  - Total width at 54px tiles + 6px gaps + 16px spacer = 850px; fits the 1380px min-width.
    If you make tiles larger, verify against the narrowest supported width.

### 2. Table — claim prompt
Full-felt scrim `rgba(6,14,13,.5)` (rounded 22 to match), content bottom-aligned with
`padding-bottom:200px`. Card 620px, radius 16, `#101b1f`, border `rgba(232,172,61,.3)`,
`animation: riseIn .22s ease`.
- Header (18/22 padding, bottom hairline): the discarded tile at 54 × 74 with accent ring and
  `softGlow 1.8s ease-in-out infinite`; title "Pong this 5 Tong?" (Newsreader 22px);
  sub "Auntie Lim (East) discarded it · you hold two 5 筒" (12px `#8fb0a0`);
  right-aligned "auto-pass in 6s" (DM Mono 11px `#6f8189`) — countdown.
- Action row: three equal buttons, 14px padding, radius 10.
  **Pong 碰** = accent gradient `linear-gradient(#e8ac3d,#d0901f)` / `#23180a` / 700.
  **Kong 杠** = disabled look: `rgba(255,255,255,.05)` bg, `rgba(255,255,255,.14)` border, `#6f8189`.
  **Pass** = same shell but `#cfd8da` text.
- Chow block: label "OR CHOW 吃 — PICK A COMBINATION"; three selectable buttons, each showing
  the 3-tile combination (3-4·5, 4·5·6, 6-7·5 — the claimed tile is highlighted with
  `linear-gradient(#fff5d8,#f4e6bd)`). Selected: `rgba(232,172,61,.12)` bg,
  `rgba(232,172,61,.6)` border. Unselected: `rgba(255,255,255,.03)` / `rgba(255,255,255,.1)`.
- Coach note: `rgba(232,172,61,.09)` bg, `rgba(232,172,61,.22)` border, radius 11;
  24px `師` badge in solid accent; 12.5px `#e3cfa5` copy, lead word in `#f0d9a6`.
  Copy: "**Pass.** Ponging 5 筒 breaks your 5-6-7 run and leaves you 2 away instead of 1.
  Your hand is already concealed and quiet — keep it that way and stay flexible."

### 3. Table — kong offer (your turn)
Inline banner, no scrim: absolutely positioned, `bottom:200px`, centered. Row card,
radius 14, `#101b1f`, `rgba(232,172,61,.3)` border, `0 20px 46px rgba(0,0,0,.45)`,
`riseIn .22s`. Contents: four 30 × 40 `東` tiles (the 4th ringed accent);
title "You drew the fourth 東 — declare Kong?" 15px; coach line 11.5px `#8fb0a0`;
**Declare Kong** accent button + **Not yet** ghost button.

### 4. Table — waiting for bots
No overlay. Turn pill switches to the dots + "{Bot} is thinking"; hand drops to `.58` opacity
and stops accepting clicks; heat strips and the ★ hide; the sidebar advisor header shows
"Paused — resumes on your turn" in `#6f8189`. Deliberately quiet — one small dot animation,
no spinners.

### 5. Analysis sidebar (404px, right, `bg/sidebar`, left hairline, scrolls)
Sections are separated by hairlines with `20px 22px` padding.
- **Header** (16/22): accent 22px `師` badge + "Coach" (Newsreader 17px);
  right link-button "Hide · no training wheels" 11.5px `#6f8189` → hides the sidebar entirely.
- **Hand progress**
  - Headline Newsreader 27px: "1 away from ready" (`#f0d9a6`) / "Ready!" (`#8fd3a2`) plus
    `听牌` 13px `#6fae7f` at tenpai.
  - Sub 12px `#8b9aa1`, e.g. "Three runs, an East pair and one floater. One good draw from ready."
  - 4-segment meter, 5px gap: segments `3 away / 2 away / 1 away / Ready`; filled count =
    `4 − shanten`; filled color `#e8ac3d` (`#6fae7f` at tenpai); the leading filled segment
    gets `box-shadow: 0 0 12px rgba(232,172,61,.5)`; labels 9.5px centered, filled
    `#e3cfa5` / empty `#5f7178`.
  - Tenpai celebration card: `rgba(111,174,127,.1)` bg, `rgba(111,174,127,.28)` border,
    radius 11; "Waiting on — 5 of 8 still live" 11.5px `#a6d3b2`; the wait tiles at 34 × 44.
- **Discard advisor** — label row: "DISCARD ADVISOR" + right note ("17 tiles improve you" in
  `#6fae7f`, or the paused note in `#6f8189`).
  Three candidate rows (8px gap), each radius 12; recommended = `rgba(232,172,61,.08)` bg /
  `rgba(232,172,61,.42)` border, others `rgba(255,255,255,.03)` / `rgba(255,255,255,.08)`.
  Row button (12/14 padding): 38 × 50 tile with heat strip · title 13.5px + "★ BEST" chip
  (accent bg, `#23180a`, 9px uppercase) · acceptance line 11.5px `#8b9aa1`
  ("17 tiles improve your hand") · right column: danger % in DM Mono 13px colored by
  `heat(danger)` over a 9.5px "danger" caption.
  Expanded body (accordion, one open at a time, recommended open by default):
  a note card (11.5px `#a4b1b6`, `rgba(255,255,255,.04)`, radius 9), then the **4-bar danger
  breakdown** — rows of `108px label (11px #8b9aa1)` + 6px track `rgba(255,255,255,.07)` with
  a `heat(v)` fill + right-aligned percentage (DM Mono 10.5px `#7f8f96`).
  Bars, in order: **Visibility, Discard pattern, Opponent threat, Suit safety**.
- **Opponent threat** — three gauges, 16px gap. Each: seat char + name 13px + persona 10px
  `#6f8189` on the left, percentage DM Mono 13px in `heat(pct/100)` on the right;
  7px bar (track `rgba(255,255,255,.07)`, fill `heat`); below it a **per-suit strip** of four
  22 × 4 mini bars labelled `萬 筒 索 字` (label 10px `#7f8f96`) plus a tell line 11px in the
  gauge's color. Sample data: Auntie Lim 72% (30/88/45/60) "Avoid 筒 — two melds showing";
  Kumar 41% (66/34/30/44) "Watch upper 萬"; Mei Ling 18% (20/14/26/10) "Folding — safe to push".
- **Hint** — full-width accent-outline button, radius 11, 14px padding, prefixed `師`;
  label toggles "Show me what you would do" / "Hide the coach's line".
  Open panel: `rgba(232,172,61,.09)` bg, `rgba(232,172,61,.24)` border, radius 12,
  `riseIn .2s`; 20px `師` badge + "WHAT I WOULD PLAY" 11px `.1em` `#d99a2b`;
  body 13px `#e3cfa5`; footnote 11px `#a3947a`
  ("Simulated 4,000 hands · this line wins 1.7× more often than ponging.").

When the sidebar is hidden the felt area takes the full width — no layout gap, no placeholder.

### 6. Game end overlay
`data-screen-label="Game end"`. Fixed inset 0, `rgba(7,13,16,.78)` + `backdrop-filter: blur(6px)`,
centered, 40px padding, `z-index:50`. Card 760px, radius 20, `#111b20`,
`rgba(255,255,255,.09)` border, `0 40px 90px rgba(0,0,0,.55)`, `riseIn .3s`,
`max-height:100%` + internal scroll.
- **Header** (28/32 padding, `linear-gradient(180deg, rgba(232,172,61,.14), rgba(232,172,61,0))`,
  bottom hairline):
  - Left: eyebrow "HAND 3 · WON BY YOU"; title "You win — self-draw" (Newsreader 34px);
    sub "Drew **8 筒** from the wall on turn 14 · **自摸** pays from all three".
  - Right: "4 tai" (DM Mono 38px `#f0d9a6`) + "8 chips per player" 12px.
  - Winning hand: 14 tiles at 40 × 54, 4px gap; the winning tile gets the accent ring.
- **Body**: two columns `1fr 260px`, 30px gap.
  - **Scoring receipt** — rows with `1px dashed rgba(255,255,255,.1)` bottom borders, 11px
    vertical padding: English name 14px + Chinese term 13px `#9aa8ae` + right-aligned tai
    (DM Mono 14px `#f0d9a6`). Items: Half Flush 混一色 +2 · Animal — Rooster 公鸡 +1 ·
    Self-draw 自摸 +1 · Flower (own seat) 花 +0. Then a borderless **Total** row:
    "4 tai → 8 chips" DM Mono 18px.
  - **Chip payments** — four rows, radius 10, 10/12 padding: seat char, name, delta
    (DM Mono 14px; winner `#8fd3a2`, payers `#d99a8f`), running total DM Mono 11px `#6f8189`.
    Winner row is tinted `rgba(111,174,127,.12)` with a `rgba(111,174,127,.3)` border;
    others `rgba(255,255,255,.03)` / `rgba(255,255,255,.08)`.
    Sample: You +24 → 152 · Auntie Lim −8 → 88 · Kumar −8 → 96 · Mei Ling −8 → 64.
- **Footer**: **Next hand** (accent gradient, `flex:1`, 16px padding, radius 12, 16px/700) ·
  **Review game with coach** (ghost, `flex:1`, same metrics) · **End session** (bare text
  button, 12.5px `#6f8189`).

### 7. Setup screen
`data-screen-label="Setup"`. Replaces the table area entirely. Background
`radial-gradient(1100px 620px at 50% 0%, #14262a 0%, #0b1216 70%)`, centered, 40px padding.
Card 880px, radius 20, `#111b20`, `rgba(255,255,255,.08)` border,
`0 30px 70px rgba(0,0,0,.45)`, `riseIn .35s`.
- **Header** (26/32/20 padding, bottom hairline, baseline space-between): eyebrow "NEW GAME";
  title "Set the table" (Newsreader 30px); sub "Singapore rules · 16 hands · flowers &
  animals on"; right ghost button "Randomize everything" (12px, radius 8).
- **Body**: two columns `1fr 1fr`, 30px gap, 24/32 padding.
  - **Left column** (22px gap):
    1. *Your seat* — 4-up grid, 8px gap. Each option: seat char 20px, English name 11px
       `#9aa8ae`, and a 9px uppercase tag row (East shows "DEALER" in `#d99a2b`; others render
       an invisible `·` to keep heights equal). Selected: `rgba(232,172,61,.12)` bg,
       `rgba(232,172,61,.6)` border, `#f0d9a6` char.
    2. *Stakes* — 4 equal segmented buttons, DM Mono 12px: "1 chip / tai", "2 chips",
       "5 chips", "10 chips".
    3. *Seed* — label with an inline hint "— optional, replay the same shuffle"; a bordered
       field (radius 9, 10/12) holding the seed in DM Mono 13px, a vertical divider, and
       "blank = random shuffle" 12px `#6f8189`. Should be a real text input.
    4. *Coach sidebar* toggle card — `rgba(217,154,43,.08)` bg, `rgba(217,154,43,.22)` border,
       radius 10: "Coach sidebar" 13px `#f0dcb5` + "Turn it off any time for
       no-training-wheels play" 11px `#a3947a`, with a 38 × 21 switch (15px knob,
       `left:3px` / `left:20px`).
  - **Right column** — "YOUR OPPONENTS" label, then three bot cards (radius 12,
    `rgba(255,255,255,.03)`, `rgba(255,255,255,.08)` border, 12/14 padding): seat glyph in a
    24px rounded square, bot name 13px, relative position 11px `#6f8189` ("right of you",
    "across", "left of you"); below, a 4-up grid of personality chips (11px, radius 8) —
    Aggressive / Balanced / Defensive / Random, same selected styling as the seat picker.
    Below the cards, above a top hairline: the personality legend, one 11.5px line each —
    name in `#d99a2b` (74px column) + description in `#8b9aa1`:
    - Aggressive — "Melds early, pushes for tai, rarely folds."
    - Balanced — "Plays value against risk — a solid sparring partner."
    - Defensive — "Folds fast, feeds you almost nothing."
    - Random — "Unpredictable. Good for practising reads, bad for reads."
- **Footer**: one full-width **Deal me in** button — accent gradient, radius 12, 18px padding,
  17px/700, `0 10px 26px rgba(217,154,43,.28)`.

---

## Interactions & Behavior

### Flows
- Setup → "Deal me in" → table (your turn).
- Your turn → click any hand tile (or the drawn tile) → tile animates into your river →
  turn pill switches to "{Bot} is thinking" → hand dims → after the bots resolve, your next
  draw appears with `tileIn` and the advisor refreshes. (The prototype fakes this with a
  1900ms timeout and a 3-entry script; wire it to your real engine.)
- Bot discards a tile you can claim → claim prompt with a 6s auto-pass countdown.
  Pong / Kong / selected Chow → meld and return to your discard step; Pass → resume bots.
- You draw the 4th copy of a tile you hold as a triplet → kong banner. Declare → meld +
  replacement draw; "Not yet" → normal discard.
- Hand ends → game-end overlay → "Next hand" (deal again, keep chip totals) /
  "Review game with coach" (post-game analysis view, **not designed yet**) /
  "End session" → setup.

### Motion (all subtle — no bouncing, no confetti)
```css
@keyframes turnPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(217,154,43,.45); } 50% { box-shadow: 0 0 0 8px rgba(217,154,43,0); } }
@keyframes softGlow  { 0%,100% { opacity:.55; } 50% { opacity:1; } }
@keyframes tileIn    { from { transform: translateY(-14px) scale(.94); opacity:0; } to { transform:none; opacity:1; } }
@keyframes thinkDot  { 0%,80%,100% { opacity:.25; } 40% { opacity:1; } }
@keyframes riseIn    { from { transform: translateY(12px); opacity:0; } to { transform:none; opacity:1; } }
```
- Recommended-tile lift: `transform .16s ease`.
- Switch knob: `left .18s ease`.
- Cards/prompts enter with `riseIn` (.22s prompts, .3s end overlay, .35s setup).
- Discarded tiles should slide from the hand into the river (~200ms, ease-out) — the prototype
  only fakes the arrival; implement the travel if cheap.
- Honor `prefers-reduced-motion`: drop `turnPulse`, `softGlow`, and slide/enter animations,
  keep opacity changes.

### Hover / focus (specify in your implementation — the prototype only shows rest states)
- Hand tiles on your turn: lift ~4px + slightly stronger shadow.
- Buttons: accent buttons brighten one step; ghost buttons go to `rgba(255,255,255,.08)`.
- Advisor rows: border to `rgba(255,255,255,.16)`.
- Everything clickable needs a visible keyboard focus ring (accent, 2px offset).

### Accessibility
- Tiles are real `<button>`s with accessible names ("Discard 5 Tong, danger 30%").
- Never encode danger by color alone — the numeric percentage is always shown; keep it.
- Announce turn changes and claim windows via a polite live region.
- The claim window is a timed decision: pause the countdown on keyboard focus.

## State Management
Game state (owned by the engine, not the view):
- `hand: Tile[]` (13) + `drawn: Tile | null`
- `players[4]`: `{ seat, name, personality, chips, hiddenCount, melds: Meld[], bonus: Tile[], river: Tile[] }`
- `wallRemaining: number`, `roundWind`, `dealerSeat`, `turnSeat`
- `phase: 'setup' | 'your-turn' | 'claim' | 'kong-offer' | 'bots' | 'hand-over'`
- `claim: { tile, fromSeat, options: { pong, kong, chows: Tile[][] }, expiresAt }`
- `result: { winnerSeat, winType: 'self-draw' | 'discard', receipt: {en, zh, tai}[], totalTai, chipsPerPlayer, payments }`

Analysis state (derived from the hand + visible tiles):
- `shanten: number`, `waits: Tile[]`, `liveCount`
- `candidates: { tile, acceptance, danger, breakdown: { visibility, discardPattern, opponentThreat, suitSafety }, note }[]`
  — sorted best-first; index 0 drives the ★ on the table.
- `threats: { seat, pct, suits: [wan, tong, suo, honors], tell }[]`
- Compute on the client for instant feedback; if the advisor is server-side, keep the last
  result rendered and dim the note rather than emptying the panel.

UI state: `analysisVisible`, `tileHeatVisible`, `hintOpen`, `expandedCandidate`,
plus the setup form (`seat`, `bots[3]`, `seed`, `stakes`).

The prototype also exposes three props for demoing: `startView`, `showAnalysis`, `tileHeat`.

### Responsive
Desktop-only by design; `min-width:1380px`. Below that, the intended (undesigned) behavior is
to collapse the sidebar into a slide-over panel — ask before building it.

## Assets
None. No images, no icon fonts, no SVG. Everything is CSS + text glyphs (CJK characters,
`★`, `·`, `×`, `−`). Fonts come from Google Fonts — self-host them in production.
If your codebase already has a design system, map the tokens above onto it rather than
introducing a second palette.

## Files
- `screenshots/` — one capture per designed state:
  `01-table-your-turn`, `02-claim-prompt`, `03-kong-offer`, `04-waiting-for-bots`,
  `05-game-end-overlay`, `06-setup`.
- `Mahjong Trainer.dc.html` — the full design reference; all six states, driven by the
  **States** switcher in the top bar. Read it for exact style values.
