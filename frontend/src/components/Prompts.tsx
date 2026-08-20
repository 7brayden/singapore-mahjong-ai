import { useEffect, useState } from "react";
import type { ClaimContext, Explanation, GameView, PendingView } from "../api";
import { tileFace, tileShortZh } from "../tiles";
import { TileView } from "./Tile";

interface ClaimPromptProps {
  view: GameView;
  pending: PendingView;   // type "claim" or "chow"
  discarderName: string;
  coachLine: string | null;
  claimContext: ClaimContext | null;
  explanation: Explanation | null;
  explainLoading: boolean;
  onExplain: () => void;
  onClaim: (accept: boolean) => void;          // for pong/kong claims
  onChow: (option: [number, number] | null) => void;
}

export function ClaimPrompt({ view, pending, discarderName, coachLine,
                              claimContext, explanation, explainLoading,
                              onExplain, onClaim, onChow }: ClaimPromptProps) {
  const tile = pending.tile ?? -1;
  const isChow = pending.type === "chow";
  const claimType = pending.claim_type ?? "pong";
  const chowOptions = (isChow ? (pending.options as Array<[number, number]>) : []) ?? [];
  const [selectedChow, setSelectedChow] = useState(0);

  const pass = () => (isChow ? onChow(null) : onClaim(false));

  // Escape passes — the card is the pending decision, so keyboard
  // players need a way out that isn't hunting for the ghost button.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") pass();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const held = view.hand.filter((t) => t === tile).length;
  const face = tileFace(tile);
  const verb = isChow ? "Chow" : claimType === "kong" ? "Kong" : "Pong";
  const holdNote = isChow
    ? "it completes a run for you"
    : `you hold ${held === 3 ? "three" : "two"} ${tileShortZh(tile)}`;

  return (
    <div className="felt-scrim">
      <div className="claim-card">
        <div className="claim-header">
          <TileView id={tile} size="prompt" ring="ring-glow" />
          <div className="claim-titles">
            <div className="claim-title">{verb} this {face.label}?</div>
            <div className="claim-sub">
              {discarderName} discarded it · {holdNote}
            </div>
          </div>

        </div>
        {claimContext && (claimContext.kills_pinghu || claimContext.dead_after) && (
          <div className={`tai-cost${claimContext.dead_after ? " dead" : ""}`}>
            {claimContext.dead_after
              ? `Taking this leaves no tai in sight — best achievable drops ${
                  claimContext.tai_ceiling_before} → ${claimContext.tai_ceiling_after
                }, below the 1-tai minimum. A 0-tai hand cannot win.`
              : `This forfeits ping hu: best achievable drops ${
                  claimContext.tai_ceiling_before} → ${claimContext.tai_ceiling_after} tai.`}
          </div>
        )}
        <div className="claim-actions">
          {!isChow && (
            <button className="btn-accent" autoFocus onClick={() => onClaim(true)}>
              {verb} <span className="zh">{claimType === "kong" ? "杠" : "碰"}</span>
            </button>
          )}
          <button className="btn-ghost" onClick={pass}>
            Pass
          </button>
          <button
            className="btn-coach-ask"
            onClick={onExplain}
            disabled={explainLoading || explanation !== null}
          >
            師 {explanation ? "Coach has answered" : explainLoading ? "Coach is writing…" : "Ask the coach"}
          </button>
        </div>
        {isChow && chowOptions.length > 0 && (
          <div className="chow-block">
            <div className="section-label chow-label">
              Chow <span className="zh">吃</span> — pick a combination
            </div>
            <div className="chow-options">
              {chowOptions.map((opt, i) => {
                const combo = [...opt, tile].sort((a, b) => a - b);
                return (
                  <button
                    key={i}
                    className={`chow-option${selectedChow === i ? " selected" : ""}`}
                    onClick={() => setSelectedChow(i)}
                    onDoubleClick={() => onChow(opt)}
                  >
                    {combo.map((t, j) => (
                      <TileView key={j} id={t} size="river" highlightFace={t === tile} />
                    ))}
                  </button>
                );
              })}
            </div>
            <div className="claim-actions" style={{ padding: "12px 0 0" }}>
              <button className="btn-accent" autoFocus onClick={() => onChow(chowOptions[selectedChow])}>
                Chow the selected run
              </button>
            </div>
          </div>
        )}
        {(explanation || explainLoading || coachLine) && (
          <div className="coach-note" aria-live="polite">
            <span className="coach-badge">師</span>
            {explanation ? (
              <span className="coach-copy">{explanation.text}</span>
            ) : explainLoading ? (
              <span className="coach-copy">
                {coachLine && <>{coachLine}{" "}</>}
                <span className="coach-writing">Writing the full read…</span>
              </span>
            ) : (
              <span className="coach-copy">{coachLine}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

interface KongBannerProps {
  pending: PendingView;   // type "kong"
  onDeclare: (option: [string, number]) => void;
  onPass: () => void;
}

export function KongBanner({ pending, onDeclare, onPass }: KongBannerProps) {
  const options = (pending.options as Array<[string, number]>) ?? [];
  if (options.length === 0) return null;
  const [kind, tile] = options[0];
  const face = tileFace(tile);
  const title =
    kind === "added"
      ? `You drew the fourth ${face.label} — declare Kong?`
      : `You hold all four ${face.label} — declare Kong?`;
  return (
    <div className="kong-banner">
      <span className="kong-tiles">
        {[0, 1, 2, 3].map((i) => (
          <TileView key={i} id={tile} size="kong" ring={i === 3 ? "ring" : undefined} />
        ))}
      </span>
      <span className="kong-texts">
        <div className="kong-title">{title}</div>
        <div className="kong-sub">
          A kong draws a replacement tile from the back of the wall — free
          value when the fourth copy serves nothing else.
        </div>
      </span>
      <span className="kong-actions">
        {options.map((opt, i) => (
          <button key={i} className="btn-accent" autoFocus={i === 0} onClick={() => onDeclare(opt)}>
            Declare Kong{options.length > 1 ? ` (${tileFace(opt[1]).label})` : ""}
          </button>
        ))}
        <button className="btn-ghost" onClick={onPass}>
          Not yet
        </button>
      </span>
    </div>
  );
}
