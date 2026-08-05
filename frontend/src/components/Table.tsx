import type { ReactNode } from "react";
import type { GameView, PlayerView } from "../api";
import { seatChar, seatName, tileFace } from "../tiles";
import { fmtSigned } from "../money";
import { TileView, BonusChip, FaceDownSlivers } from "./Tile";

// Visual placement relative to the human seat: the previous player sits
// to the right, the player across at the top, the next player left.
export function seatLayout(humanSeat: number) {
  return {
    right: (humanSeat + 3) % 4,
    top: (humanSeat + 2) % 4,
    left: (humanSeat + 1) % 4,
  };
}

interface SeatPanelProps {
  player: PlayerView;
  displayName: string;
  persona: string;
  isDealer: boolean;
  chips: number;
  tell: string | null;
  north?: boolean;
}

function SeatPanel({ player, displayName, persona, isDealer, chips, tell, north }: SeatPanelProps) {
  const head = (
    <div className="seat-head">
      <span className="seat-char">{seatChar(player.seat_wind)}</span>
      <span className="seat-name">{displayName}</span>
      <span className="persona-badge">{persona}</span>
      {isDealer && <span className="dealer-chip">Dealer</span>}
      <span className="seat-chips">{fmtSigned(chips)}</span>
    </div>
  );
  const facedown = (
    <div className="facedown-row">
      <FaceDownSlivers />
      <span className="facedown-count">× {player.concealed_count}</span>
    </div>
  );
  const melds = player.exposed.length > 0 && (
    <div className="meld-row">
      {player.exposed.map(([, tiles], i) => (
        <span key={i} className="meld-group">
          {tiles.map((t, j) => (
            <TileView key={j} id={t} size="meld" />
          ))}
        </span>
      ))}
    </div>
  );
  const bonus = player.flowers.length > 0 && (
    <div className="bonus-row">
      {player.flowers.map((f, i) => (
        <BonusChip key={i} id={f} />
      ))}
    </div>
  );
  return (
    <div className={`seat-panel${isDealer ? " dealer" : ""}`}>
      {head}
      {north ? (
        <div className="seat-body">
          {facedown}
          {melds}
          {bonus}
        </div>
      ) : (
        <>
          {facedown}
          {melds}
          {bonus}
        </>
      )}
      {tell && <div className="tell-line">{tell}</div>}
    </div>
  );
}

function DiscardRiver({ player, position, claimTile }: {
  player: PlayerView;
  position: "north" | "east" | "south" | "west";
  claimTile: number | null;
}) {
  const last = player.discards.length - 1;
  return (
    <div className={`river ${position}`}>
      {player.discards.map((t, i) => {
        const isLast = i === last;
        const isClaimed = isLast && claimTile !== null && t === claimTile;
        return (
          <TileView
            key={i}
            id={t}
            size="river"
            riverFace
            ring={isClaimed ? "ring-glow" : isLast ? "ring-soft" : undefined}
          />
        );
      })}
    </div>
  );
}

interface CenterPodProps {
  view: GameView;
  pillLabel: string;
  yourTurn: boolean;
  thinking: boolean;
  dealerName: string;
}

function CenterPod({ view, pillLabel, yourTurn, thinking, dealerName }: CenterPodProps) {
  return (
    <div className="center-pod">
      <div className="wall-count">{view.tiles_remaining}</div>
      <div className="wall-caption">tiles left</div>
      <div className="wall-bar">
        <div className="fill" style={{ width: `${(view.tiles_remaining / 148) * 100}%` }} />
      </div>
      <div className="round-row">
        <span className="round-glyph">{seatChar(view.prevailing_wind)}</span>
        <span>
          <div className="round-name">{seatName(view.prevailing_wind)} round</div>
          <div className="round-dealer">Dealer: {dealerName}</div>
        </span>
      </div>
      <div className="pod-hairline" />
      <div className={`turn-pill ${yourTurn ? "yours" : "waiting"}`}>
        {pillLabel}
        {thinking && (
          <>
            <span className="think-dot" />
            <span className="think-dot" />
            <span className="think-dot" />
          </>
        )}
      </div>
    </div>
  );
}

interface HumanAreaProps {
  view: GameView;
  chips: number;
  yourDiscardTurn: boolean;
  drawnTile: number | null;
  dangerByTile: Map<number, number> | null;
  recommendedTile: number | null;
  heatVisible: boolean;
  statusText: string;
  onDiscard: (tile: number) => void;
  onHint: () => void;
}

function HumanArea({ view, chips, yourDiscardTurn, drawnTile, dangerByTile,
                     recommendedTile, heatVisible, statusText, onDiscard, onHint }: HumanAreaProps) {
  const me = view.players[view.seat];
  // Split the drawn tile out of the sorted hand for the 16px spacer gap.
  let base = [...view.hand];
  if (drawnTile !== null) {
    const idx = base.indexOf(drawnTile);
    if (idx >= 0) base.splice(idx, 1);
  }
  let starPlaced = false;

  const renderTile = (t: number, key: string, drawn: boolean): ReactNode => {
    const isStar =
      yourDiscardTurn && !starPlaced && recommendedTile !== null && t === recommendedTile;
    if (isStar) starPlaced = true;
    const danger = yourDiscardTurn && dangerByTile ? dangerByTile.get(t) : undefined;
    const face = tileFace(t);
    return (
      <button
        key={key}
        className={`hand-tile${isStar ? " recommended" : ""}${drawn ? " tile-in" : ""}`}
        onClick={() => yourDiscardTurn && onDiscard(t)}
        disabled={!yourDiscardTurn}
        aria-label={
          danger !== undefined
            ? `Discard ${face.label}, danger ${Math.round(danger * 100)}%`
            : face.label
        }
      >
        {isStar && <span className="star-marker">★</span>}
        {drawn && <span className="drawn-label">Drawn</span>}
        <TileView
          id={t}
          size="hand"
          danger={yourDiscardTurn && dangerByTile ? danger ?? 0 : undefined}
          heatOff={!heatVisible}
        />
      </button>
    );
  };

  return (
    <div className="human-area">
      <div className="human-wrap">
        <div className="human-info">
          <div className="human-info-left">
            <span className="human-seat-char">{seatChar(me.seat_wind)}</span>
            <span className="human-name">You · {seatName(me.seat_wind)}</span>
            {me.exposed.length === 0 && <span className="concealed-badge">Concealed</span>}
            <span className="human-chips">{fmtSigned(chips)}</span>
            {me.exposed.length > 0 && (
              <span className="meld-row">
                {me.exposed.map(([, tiles], i) => (
                  <span key={i} className="meld-group">
                    {tiles.map((t, j) => (
                      <TileView key={j} id={t} size="meld" />
                    ))}
                  </span>
                ))}
              </span>
            )}
            <span className="bonus-row">
              {me.flowers.map((f, i) => (
                <BonusChip key={i} id={f} />
              ))}
            </span>
          </div>
          <div className="human-info-right">
            <span className="human-status">{statusText}</span>
            <button className="hint-button" onClick={onHint}>
              <span className="zh">師</span> Hint
            </button>
          </div>
        </div>
        <div className={`hand-row${yourDiscardTurn ? " active" : " dimmed"}`}>
          {base.map((t, i) => renderTile(t, `h${i}`, false))}
          {drawnTile !== null && (
            <>
              <span className="hand-spacer" />
              {renderTile(drawnTile, "drawn", true)}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export interface TableProps {
  view: GameView;
  displayNames: string[];
  personas: string[];
  chipsFor: (seat: number) => number;
  tells: (string | null)[];
  pillLabel: string;
  yourTurn: boolean;
  thinking: boolean;
  claimTile: number | null;
  yourDiscardTurn: boolean;
  drawnTile: number | null;
  dangerByTile: Map<number, number> | null;
  recommendedTile: number | null;
  heatVisible: boolean;
  statusText: string;
  onDiscard: (tile: number) => void;
  onHint: () => void;
  overlay?: ReactNode;
}

export function Table(props: TableProps) {
  const { view } = props;
  const layout = seatLayout(view.seat);
  const seatProps = (seat: number, north = false) => ({
    player: view.players[seat],
    displayName: props.displayNames[seat],
    persona: props.personas[seat],
    isDealer: view.dealer === seat,
    chips: props.chipsFor(seat),
    tell: props.tells[seat],
    north,
  });
  const riverFor = (seat: number, position: "north" | "east" | "south" | "west") => (
    <DiscardRiver player={view.players[seat]} position={position} claimTile={props.claimTile} />
  );

  return (
    <div className="felt-wrap">
      <div className="felt">
        <div className="felt-grid">
          <div className="seat-north">
            <SeatPanel {...seatProps(layout.top, true)} />
          </div>
          <div className="seat-west">
            <SeatPanel {...seatProps(layout.left)} />
          </div>
          <div className="seat-east">
            <SeatPanel {...seatProps(layout.right)} />
          </div>
          <div className="felt-center">
            <div className="center-grid">
              {riverFor(layout.top, "north")}
              {riverFor(layout.left, "west")}
              <CenterPod
                view={view}
                pillLabel={props.pillLabel}
                yourTurn={props.yourTurn}
                thinking={props.thinking}
                dealerName={props.displayNames[view.dealer]}
              />
              {riverFor(layout.right, "east")}
              {riverFor(view.seat, "south")}
            </div>
          </div>
          <HumanArea
            view={view}
            chips={props.chipsFor(view.seat)}
            yourDiscardTurn={props.yourDiscardTurn}
            drawnTile={props.drawnTile}
            dangerByTile={props.dangerByTile}
            recommendedTile={props.recommendedTile}
            heatVisible={props.heatVisible}
            statusText={props.statusText}
            onDiscard={props.onDiscard}
            onHint={props.onHint}
          />
        </div>
        {props.overlay}
      </div>
    </div>
  );
}
