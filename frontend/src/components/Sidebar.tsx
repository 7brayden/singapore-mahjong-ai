import { useEffect, useState } from "react";
import type { Analysis, DiscardAnalysis, Explanation, GameView } from "../api";
import { heat } from "../heat";
import { tileFace } from "../tiles";
import { TileView } from "./Tile";

const SEG_LABELS = ["3 away", "2 away", "1 away", "Ready"];
const BREAKDOWN: Array<[keyof DiscardAnalysis["danger_components"], string]> = [
  ["visibility", "Visibility"],
  ["discard_absence", "Discard pattern"],
  ["opponent_threat", "Opponent threat"],
  ["suit_safety", "Suit safety"],
];
const SUIT_ORDER: Array<[string, string]> = [
  ["wan", "萬"], ["tong", "筒"], ["suo", "索"], ["honor", "字"],
];

function liveWaitCount(view: GameView, waits: number[]): [number, number] {
  const visible = new Map<number, number>();
  const bump = (t: number) => visible.set(t, (visible.get(t) ?? 0) + 1);
  view.hand.forEach(bump);
  for (const p of view.players) {
    p.discards.forEach(bump);
    p.exposed.forEach(([, tiles]) => tiles.forEach(bump));
  }
  let live = 0;
  for (const w of waits) live += Math.max(0, 4 - (visible.get(w) ?? 0));
  return [live, waits.length * 4];
}

function candidateNote(c: DiscardAnalysis, isBest: boolean): string {
  const after = c.shanten_after;
  const stage =
    after < 0 ? "completes your hand" :
    after === 0 ? "leaves you ready to win" :
    `keeps you ${after} away`;
  const danger = Math.round(c.danger * 100);
  const risk = danger >= 60 ? "a risky throw right now" :
               danger >= 35 ? "a moderate risk" : "a safe throw";
  const base = `${isBest ? "Best line — this" : "This"} ${stage} with ${c.acceptance} improving ${
    c.acceptance === 1 ? "tile" : "tiles"}, and it's ${risk} (${danger}%).`;
  if (c.deal_in_prob === undefined) return base;
  const p = c.deal_in_prob * 100;
  const pTxt = p >= 10 ? p.toFixed(0) : p.toFixed(1);
  const winTxt = c.win_prob !== undefined
    ? `, wins ${(c.win_prob * 100).toFixed(0)}% of hands`
    : "";
  const valueTxt = c.hand_value !== undefined
    ? `, and the hand it keeps is worth ${c.hand_value >= 0 ? "+" : ""}${c.hand_value.toFixed(1)} pts`
    : "";
  return `${base} Trained model: deals in ${pTxt}% of the time${winTxt}${valueTxt}.`;
}

function tellFor(threat: number, suits: Record<string, number>): [string, string] {
  const honor = Math.max(suits["wind"] ?? 0, suits["dragon"] ?? 0);
  const named: Array<[string, number]> = [
    ["萬", suits["wan"] ?? 0], ["筒", suits["tong"] ?? 0],
    ["索", suits["suo"] ?? 0], ["字", honor],
  ];
  named.sort((a, b) => b[1] - a[1]);
  const worst = named[0][0];
  if (threat >= 0.6) return [`Pushing hard — avoid ${worst}`, "hot"];
  if (threat >= 0.3) return [`Building — watch ${worst}`, "warm"];
  return ["Quiet — safe to push", "cool"];
}

interface SidebarProps {
  view: GameView;
  analysis: Analysis | null;
  displayNames: string[];
  personas: string[];
  paused: boolean;         // not the human's discard turn
  onHide: () => void;
  explanation: Explanation | null;
  explainLoading: boolean;
  interim: string | null;  // engine's instant pick while the LLM writes
  onExplain: () => void;
}

export function Sidebar({ view, analysis, displayNames, personas, paused,
                          onHide, explanation, explainLoading, interim,
                          onExplain }: SidebarProps) {
  const [expanded, setExpanded] = useState(0);
  // The local model's first answer can take ~10s while it loads into
  // memory. Past 4s, say so — silence reads as "broken".
  const [slowLoad, setSlowLoad] = useState(false);
  useEffect(() => {
    if (!explainLoading) { setSlowLoad(false); return; }
    const t = window.setTimeout(() => setSlowLoad(true), 4000);
    return () => window.clearTimeout(t);
  }, [explainLoading]);
  const shanten = analysis?.shanten ?? null;
  const filled = shanten === null ? 0 : Math.max(0, Math.min(4, 4 - Math.max(shanten, 0)));
  const tenpai = shanten !== null && shanten <= 0;
  const candidates = (analysis?.discards ?? []).slice(0, 3);
  const best = candidates[0];
  const waits = analysis?.waiting_on ?? null;

  const progressSub = (() => {
    if (shanten === null) return "Sizing up your hand…";
    if (tenpai && waits) return "Hold steady — every discard should keep the wait alive.";
    if (best) {
      const face = tileFace(best.tile);
      return `Best line: let go of ${face.label} and ${best.acceptance} tiles still improve you.`;
    }
    return "Waiting for your next draw.";
  })();

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-header-left">
          <span className="coach-badge">師</span>
          <span className="sidebar-title">Coach</span>
        </div>
        <button className="link-button" onClick={onHide}>
          Hide · no training wheels
        </button>
      </div>

      <section className="sidebar-section">
        <div className={`progress-headline${tenpai ? " tenpai" : ""}`}>
          {shanten === null ? "—"
            : tenpai ? <>Ready!<span className="zh-note">听牌</span></>
            : `${shanten} away from ready`}
        </div>
        <div className="progress-sub">{progressSub}</div>
        <div className="shanten-meter">
          {SEG_LABELS.map((label, i) => {
            const isFilled = i < filled;
            const isLeading = isFilled && i === filled - 1;
            return (
              <span
                key={label}
                className={`shanten-seg${isFilled ? " filled" : ""}${
                  isFilled && tenpai ? " tenpai-fill" : ""}${isLeading && !tenpai ? " leading" : ""}`}
              >
                <span className="bar" />
                <span className="seg-label">{label}</span>
              </span>
            );
          })}
        </div>
        {tenpai && waits && waits.length > 0 && (() => {
          const [live, total] = liveWaitCount(view, waits);
          return (
            <div className="tenpai-card">
              <div className="caption">
                Waiting on — {live} of {total} still live
              </div>
              <div className="wait-tiles">
                {waits.map((t) => (
                  <TileView key={t} id={t} size="wait" />
                ))}
              </div>
            </div>
          );
        })()}
      </section>

      <section className="sidebar-section">
        <div className="label-row">
          <span className="section-label">Ask the coach</span>
          {explanation && (
            <span className="advisor-note-right">
              {explanation.source === "template"
                ? "engine summary"
                : `coached by ${explanation.model || explanation.source}`}
            </span>
          )}
        </div>
        <div aria-live="polite">
          {explanation ? (
            <div className="coach-note">
              <p>{explanation.text}</p>
              {explanation.principles.length > 0 && (
                <div className="principle-chips">
                  {explanation.principles.map((p) => (
                    <span key={p} className="principle-chip">{p}</span>
                  ))}
                </div>
              )}
            </div>
          ) : explainLoading ? (
            <div className="coach-note">
              {interim && <p className="coach-interim">{interim}</p>}
              <p className="coach-writing">
                {slowLoad
                  ? "Warming up the local model — the first ask takes the longest…"
                  : "Writing the full read…"}
              </p>
            </div>
          ) : (
            <button className="hint-button coach-explain" onClick={onExplain}>
              師 Explain this decision
            </button>
          )}
        </div>
      </section>

      <section className="sidebar-section">
        <div className="label-row">
          <span className="section-label">Discard advisor</span>
          {paused || !best ? (
            <span className="advisor-note-right paused">Paused — resumes on your turn</span>
          ) : (
            <span className="advisor-note-right">{best.acceptance} tiles improve you</span>
          )}
        </div>
        {!paused && candidates.length > 0 && (
          <div className="candidate-list">
            {candidates.map((c, i) => {
              const face = tileFace(c.tile);
              const pct = Math.round(c.danger * 100);
              const open = expanded === i;
              return (
                <div key={c.tile} className={`candidate${i === 0 ? " recommended" : ""}`}>
                  <button className="candidate-row" onClick={() => setExpanded(open ? -1 : i)}>
                    <TileView id={c.tile} size="advisor" danger={c.danger} />
                    <span className="candidate-main">
                      <span className="candidate-title">
                        Discard {face.label}
                        {i === 0 && <span className="best-chip">★ Best</span>}
                      </span>
                      <span className="candidate-acceptance">
                        {c.acceptance} {c.acceptance === 1 ? "tile improves" : "tiles improve"} your hand
                      </span>
                    </span>
                    <span className="candidate-danger">
                      <div className="pct" style={{ color: heat(c.danger) }}>{pct}%</div>
                      <div className="caption">danger</div>
                    </span>
                  </button>
                  {open && (
                    <div className="candidate-body">
                      <div className="candidate-note">{candidateNote(c, i === 0)}</div>
                      {BREAKDOWN.map(([key, label]) => {
                        const v = c.danger_components[key];
                        return (
                          <div key={key} className="breakdown-row">
                            <span className="breakdown-label">{label}</span>
                            <span className="breakdown-track">
                              <span
                                className="breakdown-fill"
                                style={{ width: `${v * 100}%`, background: heat(v) }}
                              />
                            </span>
                            <span className="breakdown-pct">{Math.round(v * 100)}%</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </section>

      <section className="sidebar-section">
        <div className="label-row">
          <span className="section-label">Opponent threat</span>
        </div>
        <div className="threat-list">
          {(analysis?.opponents ?? []).map((opp) => {
            const pct = Math.round(opp.threat * 100);
            const color = heat(opp.threat);
            const [tell] = tellFor(opp.threat, opp.suit_danger);
            const honor = Math.max(opp.suit_danger["wind"] ?? 0, opp.suit_danger["dragon"] ?? 0);
            const suitValue = (key: string) =>
              key === "honor" ? honor : opp.suit_danger[key] ?? 0;
            return (
              <div key={opp.seat}>
                <div className="threat-head">
                  <span className="threat-who">
                    <span className="threat-name">{displayNames[opp.seat]}</span>
                    <span className="threat-persona">{personas[opp.seat]}</span>
                  </span>
                  <span className="threat-pct" style={{ color }}>{pct}%</span>
                </div>
                <div className="threat-bar">
                  <span className="fill" style={{ transform: `scaleX(${opp.threat})`, background: color }} />
                </div>
                <div className="suit-strip">
                  {SUIT_ORDER.map(([key, glyph]) => (
                    <span key={key} className="suit-cell">
                      <span className="suit-label">{glyph}</span>
                      <span className="mini-bar">
                        <span
                          className="mini-fill"
                          style={{ transform: `scaleX(${suitValue(key)})`, background: heat(suitValue(key)) }}
                        />
                      </span>
                    </span>
                  ))}
                </div>
                <div className="threat-tell" style={{ color }}>{tell}</div>
              </div>
            );
          })}
        </div>
      </section>

    </aside>
  );
}
