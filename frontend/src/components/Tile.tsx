import { tileFace, bonusGlyph, isAnimal } from "../tiles";
import { heat } from "../heat";

export type TileSize =
  | "hand" | "prompt" | "end" | "advisor" | "wait" | "kong" | "river" | "meld";

export type TileRing = "ring" | "ring-soft" | "ring-glow";

interface TileProps {
  id: number;
  size: TileSize;
  ring?: TileRing;
  riverFace?: boolean;
  highlightFace?: boolean;
  danger?: number;       // 0-1; renders the heat strip when provided
  heatOff?: boolean;     // strip present but neutral
  entrance?: "land";     // one-shot mount animation (newest river tile)
}

export function TileView({ id, size, ring, riverFace, highlightFace, danger, heatOff, entrance }: TileProps) {
  const face = tileFace(id);
  const classes = ["tile", `size-${size}`];
  if (entrance === "land") classes.push("tile-land");
  if (ring) classes.push(ring);
  if (riverFace) classes.push("river-face");
  if (highlightFace) classes.push("highlight-face");
  return (
    <span className={classes.join(" ")} role="img" aria-label={face.label}>
      <span className="glyph" style={{ color: face.glyphColor }}>{face.glyph}</span>
      <span className="sub" style={{ color: face.subColor }}>{face.sub}</span>
      {danger !== undefined && (
        <span
          className="heat-strip"
          style={heatOff ? undefined : { background: heat(danger) }}
        />
      )}
    </span>
  );
}

export function BonusChip({ id }: { id: number }) {
  return (
    <span
      className={`bonus-chip${isAnimal(id) ? " animal" : ""}`}
      title={isAnimal(id) ? "Animal" : "Flower"}
    >
      {bonusGlyph(id)}
    </span>
  );
}

export function FaceDownSlivers({ count = 4 }: { count?: number }) {
  return (
    <span className="sliver-row" aria-hidden>
      {Array.from({ length: count }, (_, i) => (
        <span key={i} className="tile-back-sliver" />
      ))}
    </span>
  );
}
