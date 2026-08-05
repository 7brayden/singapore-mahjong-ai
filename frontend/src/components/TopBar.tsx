import { seatChar, seatName } from "../tiles";

interface TopBarProps {
  prevailingWind: number | null;
  handNumber: number;
  seedText: string;
  analysisOn: boolean;
  onToggleAnalysis: () => void;
  inGame: boolean;
}

export function TopBar({ prevailingWind, handNumber, seedText,
                         analysisOn, onToggleAnalysis, inGame }: TopBarProps) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <span className="topbar-mark">中</span>
        <span className="topbar-wordmark">Singapore Mahjong Trainer</span>
        <span className="topbar-divider" />
        <span className="topbar-meta">
          {inGame && prevailingWind !== null ? (
            <>
              <span className="zh">{seatChar(prevailingWind)}</span>
              <span>{seatName(prevailingWind)} round</span>
              <span className="dot">·</span>
              <span>Hand {handNumber}</span>
              {seedText && (
                <>
                  <span className="dot">·</span>
                  <span className="mono">seed {seedText}</span>
                </>
              )}
            </>
          ) : (
            <span>New game</span>
          )}
        </span>
      </div>
      <div className="topbar-right">
        <button className="switch-row" onClick={onToggleAnalysis} aria-pressed={analysisOn}>
          <span>Analysis</span>
          <span className={`switch${analysisOn ? " on" : ""}`}>
            <span className="knob" />
          </span>
        </button>
      </div>
    </header>
  );
}
