// Tile metadata derived from the engine's id encoding (see mahjong/tiles.py):
// 0-8 wan, 9-17 tong, 18-26 suo, 27-30 winds ESWN, 31-33 dragons RGW,
// 34-41 flowers, 42-45 animals.

export interface TileFace {
  glyph: string;       // large top glyph
  sub: string;         // small bottom sub-label
  glyphColor: string;  // CSS color for the top glyph
  subColor: string;    // CSS color for the sub-label
  label: string;       // accessible name, e.g. "5 Tong"
}

const INK = "#25333a";
const WAN = "#b4483c";
const TONG = "#2f6fa8";
const SUO = "#2f7d52";
const WHITE_DRAGON = "#5b7386";
const HONOR_SUB = "#8d949a";

const WIND_GLYPHS = ["東", "南", "西", "北"];
const WIND_SUBS = ["E", "S", "W", "N"];
const WIND_NAMES = ["East Wind", "South Wind", "West Wind", "North Wind"];
const DRAGONS: Array<[string, string, string, string]> = [
  ["中", WAN, "RED", "Red Dragon"],
  ["發", SUO, "GRN", "Green Dragon"],
  ["白", WHITE_DRAGON, "WHT", "White Dragon"],
];
const SUITS: Array<[string, string, string]> = [
  ["萬", WAN, "Wan"],
  ["筒", TONG, "Tong"],
  ["索", SUO, "Suo"],
];

export function tileFace(id: number): TileFace {
  if (id >= 0 && id <= 26) {
    const suit = Math.floor(id / 9);
    const rank = (id % 9) + 1;
    const [sub, color, name] = SUITS[suit];
    return { glyph: String(rank), sub, glyphColor: INK, subColor: color, label: `${rank} ${name}` };
  }
  if (id >= 27 && id <= 30) {
    const i = id - 27;
    return { glyph: WIND_GLYPHS[i], sub: WIND_SUBS[i], glyphColor: INK, subColor: HONOR_SUB, label: WIND_NAMES[i] };
  }
  if (id >= 31 && id <= 33) {
    const [glyph, color, sub, label] = DRAGONS[id - 31];
    return { glyph, sub, glyphColor: color, subColor: HONOR_SUB, label };
  }
  return { glyph: "?", sub: "", glyphColor: INK, subColor: HONOR_SUB, label: `tile ${id}` };
}

// Bonus tiles render as small chips, not full tiles.
// Flowers are two numbered series — red 1-4 (34-37) and blue 1-4
// (38-41). The number is the identity: seat 1 (East) matches the two
// 1-flowers, seat 2 (South) the 2s, and so on (正花 by number).
const ANIMAL_GLYPHS = ["猫", "鼠", "鸡", "蜈"];

export const isAnimal = (id: number) => id >= 42;
export const isBonus = (id: number) => id >= 34;
export const isRedFlower = (id: number) => id >= 34 && id <= 37;
export const isBlueFlower = (id: number) => id >= 38 && id <= 41;

/** Flower number 1-4 within its series, or 0 for animals. */
export const flowerNumber = (id: number) =>
  id >= 34 && id <= 41 ? ((id - 34) % 4) + 1 : 0;

export function bonusGlyph(id: number): string {
  if (id >= 42) return ANIMAL_GLYPHS[id - 42] ?? "?";
  if (id >= 34) return String(flowerNumber(id));
  return "?";
}

export function bonusLabel(id: number): string {
  if (id >= 42) return ["Cat", "Rat", "Rooster", "Centipede"][id - 42] ?? "?";
  if (id >= 38) return `Blue ${flowerNumber(id)}`;
  if (id >= 34) return `Red ${flowerNumber(id)}`;
  return "?";
}

/** Seat character for a seat-wind tile id (27-30). */
export const seatChar = (seatWind: number) => WIND_GLYPHS[seatWind - 27] ?? "?";
export const seatName = (seatWind: number) =>
  ["East", "South", "West", "North"][seatWind - 27] ?? "?";

/** Short human-readable name, e.g. "5 筒" for prose lines. */
export function tileShortZh(id: number): string {
  if (id <= 26) {
    const suit = Math.floor(id / 9);
    return `${(id % 9) + 1} ${SUITS[suit][0]}`;
  }
  return tileFace(id).glyph;
}
