import type { GameView, ResultView } from "../api";
import { seatChar, tileFace } from "../tiles";
import { TileView } from "./Tile";

// Engine labels look like "Half flush (混一色)" — split for the receipt.
function splitLabel(label: string): [string, string] {
  const match = label.match(/^(.*?)\s*\((.*)\)\s*$/);
  return match ? [match[1], match[2]] : [label, ""];
}

interface GameEndProps {
  view: GameView;
  result: ResultView;
  displayNames: string[];
  handNumber: number;
  stakes: number;
  ledgerAfter: number[];
  onNextHand: () => void;
  onEndSession: () => void;
}

export function GameEnd({ view, result, displayNames, handNumber, stakes,
                          ledgerAfter, onNextHand, onEndSession }: GameEndProps) {
  const humanWon = result.winner === view.seat;
  const isDraw = result.winner === null;
  const winnerName = result.winner !== null ? displayNames[result.winner] : "";
  const winTileFace = result.win_tile !== null ? tileFace(result.win_tile) : null;

  const eyebrow = isDraw
    ? `Hand ${handNumber} · Nobody wins`
    : `Hand ${handNumber} · Won by ${humanWon ? "you" : winnerName}`;

  const title = isDraw
    ? "Draw — wall exhausted"
    : `${humanWon ? "You win" : `${winnerName} wins`} — ${
        result.win_type === "tsumo" ? "self-draw" : "off a discard"}`;

  const sub = isDraw
    ? "The live wall ran out with nobody ready to claim a win. Payments below are instant bonuses only."
    : result.win_type === "tsumo"
      ? `Drew ${winTileFace?.label ?? "the winning tile"} from the wall on turn ${result.turns} · 自摸 pays from all three`
      : `Took ${result.dealt_in_by !== null ? displayNames[result.dealt_in_by] : "a"}${
          result.dealt_in_by !== null ? "'s" : ""} ${winTileFace?.label ?? "discard"} on turn ${
          result.turns} · shooter pays all three shares`;

  // Winner's revealed hand: concealed tiles + exposed melds, win tile ringed once.
  const winnerHand = result.winner_hand ?? [];
  const exposedTiles = (result.winner_exposed ?? []).flatMap(([, tiles]) => tiles);
  let winMarked = false;
  const markWin = (t: number) => {
    if (!winMarked && result.win_tile !== null && t === result.win_tile) {
      winMarked = true;
      return true;
    }
    return false;
  };

  const score = result.score;
  const chipsWon = score ? score.value * stakes : 0;

  return (
    <div className="end-overlay">
      <div className="end-card">
        <div className="end-header">
          <div className="end-header-top">
            <div>
              <div className="eyebrow">{eyebrow}</div>
              <div className="end-title">{title}</div>
              <div className="end-sub">{sub}</div>
            </div>
            {score && (
              <div className="end-tai">
                <div className="big">{score.tai} tai{score.is_limit ? " ·限" : ""}</div>
                <div className="small">
                  {chipsWon} chip{chipsWon === 1 ? "" : "s"}
                  {result.win_type === "tsumo" ? " per player" : " from the shooter"}
                </div>
              </div>
            )}
          </div>
          {!isDraw && winnerHand.length > 0 && (
            <div className="end-hand">
              {winnerHand.map((t, i) => (
                <TileView key={`c${i}`} id={t} size="end" ring={markWin(t) ? "ring" : undefined} />
              ))}
              {exposedTiles.map((t, i) => (
                <TileView key={`e${i}`} id={t} size="end" riverFace />
              ))}
            </div>
          )}
        </div>
        <div className="end-body">
          <div>
            <div className="section-label" style={{ marginBottom: 6 }}>Scoring</div>
            {score && score.items.length > 0 ? (
              <>
                {score.items.map((item, i) => {
                  const [en, zh] = splitLabel(item.label);
                  return (
                    <div key={i} className="receipt-row">
                      <span>
                        <span className="receipt-name">{en}</span>
                        {zh && <span className="receipt-zh">{zh}</span>}
                      </span>
                      <span className="receipt-tai">+{item.tai}</span>
                    </div>
                  );
                })}
                <div className="receipt-total">
                  <span>Total</span>
                  <span>{score.tai} tai → {chipsWon} chips</span>
                </div>
              </>
            ) : (
              <div className="end-sub">
                {isDraw ? "No winning hand this round." : "Chicken hand — no scoring elements."}
              </div>
            )}
          </div>
          <div>
            <div className="section-label">Chip payments</div>
            <div className="payments-list">
              {view.players.map((p) => {
                const delta = (result.payments[p.seat] ?? 0) * stakes;
                const isWinner = result.winner === p.seat;
                return (
                  <div key={p.seat} className={`payment-row${isWinner ? " winner" : ""}`}>
                    <span className="seat-char zh">{seatChar(p.seat_wind)}</span>
                    <span className="payment-name">{displayNames[p.seat]}</span>
                    <span className={`payment-delta ${delta >= 0 ? "pos" : "neg"}`}>
                      {delta >= 0 ? "+" : "−"}{Math.abs(delta)}
                    </span>
                    <span className="payment-total">{ledgerAfter[p.seat]}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
        <div className="end-footer">
          <button className="btn-accent" onClick={onNextHand}>Next hand</button>
          <button className="btn-ghost" disabled title="Coming soon">
            Review game with coach
          </button>
          <button className="bare" onClick={onEndSession}>End session</button>
        </div>
      </div>
    </div>
  );
}
