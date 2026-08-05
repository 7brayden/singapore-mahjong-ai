import { useEffect, useRef, useState } from "react";
import type { GameView, PendingView } from "../api";
import { tileFace, tileShortZh } from "../tiles";
import { TileView } from "./Tile";

const AUTO_PASS_SECONDS = 6;

interface ClaimPromptProps {
  view: GameView;
  pending: PendingView;   // type "claim" or "chow"
  discarderName: string;
  coachLine: string | null;
  onClaim: (accept: boolean) => void;          // for pong/kong claims
  onChow: (option: [number, number] | null) => void;
}

export function ClaimPrompt({ view, pending, discarderName, coachLine,
                              onClaim, onChow }: ClaimPromptProps) {
  const tile = pending.tile ?? -1;
  const isChow = pending.type === "chow";
  const claimType = pending.claim_type ?? "pong";
  const chowOptions = (isChow ? (pending.options as Array<[number, number]>) : []) ?? [];
  const [selectedChow, setSelectedChow] = useState(0);
  const [secondsLeft, setSecondsLeft] = useState(AUTO_PASS_SECONDS);
  const paused = useRef(false);

  const pass = () => (isChow ? onChow(null) : onClaim(false));

  useEffect(() => {
    const timer = setInterval(() => {
      if (paused.current) return;
      setSecondsLeft((s) => s - 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (secondsLeft <= 0) pass();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [secondsLeft]);

  const held = view.hand.filter((t) => t === tile).length;
  const face = tileFace(tile);
  const verb = isChow ? "Chow" : claimType === "kong" ? "Kong" : "Pong";
  const holdNote = isChow
    ? "it completes a run for you"
    : `you hold ${held === 3 ? "three" : "two"} ${tileShortZh(tile)}`;

  return (
    <div className="felt-scrim">
      <div
        className="claim-card"
        onMouseEnter={() => (paused.current = true)}
        onMouseLeave={() => (paused.current = false)}
        onFocusCapture={() => (paused.current = true)}
        onBlurCapture={() => (paused.current = false)}
      >
        <div className="claim-header">
          <TileView id={tile} size="prompt" ring="ring-glow" />
          <div className="claim-titles">
            <div className="claim-title">{verb} this {face.label}?</div>
            <div className="claim-sub">
              {discarderName} discarded it · {holdNote}
            </div>
          </div>
          <div className="claim-countdown" aria-live="polite">
            auto-pass in {Math.max(0, secondsLeft)}s
          </div>
        </div>
        <div className="claim-actions">
          <button
            className={!isChow && claimType === "pong" ? "btn-accent" : "btn-disabled"}
            disabled={isChow || claimType !== "pong"}
            onClick={() => onClaim(true)}
          >
            Pong <span className="zh">碰</span>
          </button>
          <button
            className={!isChow && claimType === "kong" ? "btn-accent" : "btn-disabled"}
            disabled={isChow || claimType !== "kong"}
            onClick={() => onClaim(true)}
          >
            Kong <span className="zh">杠</span>
          </button>
          <button className="btn-ghost" onClick={pass}>
            Pass
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
              <button className="btn-accent" onClick={() => onChow(chowOptions[selectedChow])}>
                Chow the selected run
              </button>
            </div>
          </div>
        )}
        {coachLine && (
          <div className="coach-note">
            <span className="coach-badge">師</span>
            <span className="coach-copy">{coachLine}</span>
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
          A kong draws a replacement tile and pays out instantly — free value
          when the fourth copy serves nothing else.
        </div>
      </span>
      <span className="kong-actions">
        {options.map((opt, i) => (
          <button key={i} className="btn-accent" onClick={() => onDeclare(opt)}>
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
