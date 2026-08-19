import type { GameView, ResultView } from "../api";
import { tileFace } from "../tiles";
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
  onNextHand: () => void;
  onEndSession: () => void;
}

export function GameEnd({ view, result, displayNames, handNumber,
                          onNextHand, onEndSession }: GameEndProps) {
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
        result.win_type === "tsumo" ? "self-draw"
        : result.win_type === "flowers" ? "eight flowers 花胡"
        : "off a discard"}`;

  const sub = isDraw
    ? "The live wall ran out with nobody ready to claim a win. No tai are scored on a draw."
    : result.win_type === "flowers"
      ? (result.dealt_in_by !== null
          ? `Robbed the eighth flower from ${displayNames[result.dealt_in_by]} (七抢一) — an instant limit win on the flowers alone`
          : "Collected all eight flowers — an instant limit win, whatever the hand held")
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
                  {result.win_type === "tsumo"
                    ? "collects from all three players"
                    : "shooter pays all three shares"}
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
                  <span>{score.tai} tai</span>
                </div>
              </>
            ) : (
              <div className="end-sub">
                {isDraw ? "No winning hand this round." : "Chicken hand — no scoring elements."}
              </div>
            )}
          </div>
        </div>
        <div className="end-footer">
          <button className="btn-accent" autoFocus onClick={onNextHand}>Next hand</button>
          <button className="btn-ghost" disabled title="Coming soon">
            Review game with coach
          </button>
          <button className="bare" onClick={onEndSession}>End session</button>
        </div>
      </div>
    </div>
  );
}
