import { useState } from "react";

export interface SetupConfig {
  humanSeat: number;          // 0 = East (dealer), 1 = South, ...
  stakes: number;             // chips per base unit
  seedText: string;
  personas: string[];         // by table seat, human's entry is the hint bot
  coachOn: boolean;
}

export const PERSONAS: Array<{ id: string; label: string; desc: string }> = [
  { id: "aggressive", label: "Aggressive", desc: "Melds early, pushes for tai, rarely folds." },
  { id: "balanced", label: "Balanced", desc: "Plays value against risk — a solid sparring partner." },
  { id: "defensive", label: "Defensive", desc: "Folds fast, feeds you almost nothing." },
  { id: "random", label: "Random", desc: "Unpredictable. Good for practising reads, bad for reads." },
];

export const PERSONA_TO_BOT: Record<string, string> = {
  aggressive: "greedy",
  balanced: "hybrid",
  defensive: "defensive",
  random: "random",
};

export const BOT_NAMES = ["Auntie Lim", "Kumar", "Mei Ling"];

const SEATS = [
  { char: "東", name: "East", tag: "Dealer" },
  { char: "南", name: "South", tag: null },
  { char: "西", name: "West", tag: null },
  { char: "北", name: "North", tag: null },
];
const STAKES = [1, 2, 5, 10];

/** Bot seats in visual order right / across / left of the human. */
export function botSeats(humanSeat: number): number[] {
  return [(humanSeat + 3) % 4, (humanSeat + 2) % 4, (humanSeat + 1) % 4];
}
const POSITIONS = ["right of you", "across", "left of you"];

interface SetupProps {
  initial: SetupConfig;
  error: string | null;
  busy: boolean;
  onDeal: (config: SetupConfig) => void;
}

export function Setup({ initial, error, busy, onDeal }: SetupProps) {
  const [humanSeat, setHumanSeat] = useState(initial.humanSeat);
  const [stakes, setStakes] = useState(initial.stakes);
  const [seedText, setSeedText] = useState(initial.seedText);
  const [coachOn, setCoachOn] = useState(initial.coachOn);
  // Personas for the three bot cards (visual order right/across/left)
  const [botPersonas, setBotPersonas] = useState<string[]>(["aggressive", "balanced", "defensive"]);

  const randomize = () => {
    setHumanSeat(Math.floor(Math.random() * 4));
    setStakes(STAKES[Math.floor(Math.random() * STAKES.length)]);
    setSeedText("");
    setBotPersonas([0, 1, 2].map(() => PERSONAS[Math.floor(Math.random() * PERSONAS.length)].id));
  };

  const deal = () => {
    const personas = ["balanced", "balanced", "balanced", "balanced"];
    botSeats(humanSeat).forEach((seat, i) => {
      personas[seat] = botPersonas[i];
    });
    onDeal({ humanSeat, stakes, seedText: seedText.trim(), personas, coachOn });
  };

  return (
    <div className="setup-screen">
      <div className="setup-card">
        <div className="setup-header">
          <div>
            <div className="eyebrow">New game</div>
            <div className="setup-title">Set the table</div>
            <div className="setup-sub">Singapore rules · flowers &amp; animals on · kong pays instantly</div>
          </div>
          <button className="btn-ghost" style={{ padding: "8px 14px", borderRadius: 8, fontSize: 12 }}
                  onClick={randomize}>
            Randomize everything
          </button>
        </div>
        <div className="setup-body">
          <div className="setup-col">
            <div>
              <div className="section-label field-label">Your seat</div>
              <div className="seat-grid">
                {SEATS.map((seat, i) => (
                  <button
                    key={seat.name}
                    className={`option-card${humanSeat === i ? " selected" : ""}`}
                    onClick={() => setHumanSeat(i)}
                  >
                    <span className="big-char">{seat.char}</span>
                    <span className="opt-name">{seat.name}</span>
                    <span className={`opt-tag${seat.tag ? "" : " hidden-tag"}`}>
                      {seat.tag ?? "·"}
                    </span>
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div className="section-label field-label">Stakes</div>
              <div className="stakes-grid">
                {STAKES.map((s) => (
                  <button
                    key={s}
                    className={`option-card${stakes === s ? " selected" : ""}`}
                    onClick={() => setStakes(s)}
                  >
                    {s} chip{s === 1 ? " / tai" : "s"}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <div className="section-label field-label">
                Seed <span className="field-hint">— optional, replay the same shuffle</span>
              </div>
              <div className="seed-field">
                <input
                  value={seedText}
                  onChange={(e) => setSeedText(e.target.value)}
                  placeholder="e.g. 7A2K-91"
                  spellCheck={false}
                />
                <span className="divider" />
                <span className="note">blank = random shuffle</span>
              </div>
            </div>
            <button className="coach-toggle-card" onClick={() => setCoachOn(!coachOn)}
                    aria-pressed={coachOn}>
              <span style={{ textAlign: "left" }}>
                <div className="title">Coach sidebar</div>
                <div className="sub">Turn it off any time for no-training-wheels play</div>
              </span>
              <span className={`switch large${coachOn ? " on" : ""}`}>
                <span className="knob" />
              </span>
            </button>
          </div>
          <div>
            <div className="section-label field-label">Your opponents</div>
            {botSeats(humanSeat).map((seat, i) => (
              <div key={seat} className="bot-card">
                <div className="bot-card-head">
                  <span className="bot-glyph">{SEATS[seat].char}</span>
                  <span className="bot-name">{BOT_NAMES[i]}</span>
                  <span className="bot-pos">{POSITIONS[i]}</span>
                </div>
                <div className="persona-grid">
                  {PERSONAS.map((p) => (
                    <button
                      key={p.id}
                      className={`persona-option${botPersonas[i] === p.id ? " selected" : ""}`}
                      onClick={() =>
                        setBotPersonas(botPersonas.map((cur, j) => (j === i ? p.id : cur)))
                      }
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
            <div className="persona-legend">
              {PERSONAS.map((p) => (
                <div key={p.id} className="legend-row">
                  <span className="legend-name">{p.label}</span>
                  <span className="legend-desc">{p.desc}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        {error && <div className="error-banner">{error}</div>}
        <div className="setup-footer">
          <button className="btn-accent deal-button" onClick={deal} disabled={busy}>
            {busy ? "Shuffling…" : "Deal me in"}
          </button>
        </div>
      </div>
    </div>
  );
}
