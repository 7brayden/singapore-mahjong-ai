import { useCallback, useEffect, useRef, useState } from "react";
import {
  Analysis, Explanation, GameView, createGame, getAnalysis, getHint,
  openGameSocket, postAction, postExplain,
} from "./api";
import { tileFace } from "./tiles";
import { TopBar } from "./components/TopBar";
import { Table } from "./components/Table";
import { ClaimPrompt, KongBanner } from "./components/Prompts";
import { Sidebar } from "./components/Sidebar";
import { GameEnd } from "./components/GameEnd";
import {
  Setup, SetupConfig, PERSONAS, PERSONA_TO_BOT, BOT_NAMES, botSeats,
} from "./components/Setup";

// Balances start at zero and track net win/loss in cents.
const BOT_BEAT_MS = 750;

function parseSeed(text: string): number | null {
  if (!text) return null;
  if (/^\d+$/.test(text)) return parseInt(text, 10) % 2 ** 31;
  let hash = 0;
  for (const ch of text) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return hash;
}

export default function App() {
  const [config, setConfig] = useState<SetupConfig>({
    humanSeat: 1, taiCap: 6, seedText: "", personas: [], coachOn: true,
  });
  const [phase, setPhase] = useState<"setup" | "playing">("setup");
  const [gameId, setGameId] = useState<string | null>(null);
  const [view, setView] = useState<GameView | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [ledger, setLedger] = useState<number[]>([0, 0, 0, 0]);
  const [handNumber, setHandNumber] = useState(1);
  const [coachVisible, setCoachVisible] = useState(true);
  const [botBeat, setBotBeat] = useState(false);
  const [claimCoachLine, setClaimCoachLine] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  // Instant local recommendation shown while the LLM writes its prose
  const [coachInterim, setCoachInterim] = useState<string | null>(null);
  // Guards late responses from a decision the player already left
  const explainSeq = useRef(0);
  const [setupError, setSetupError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [announcement, setAnnouncement] = useState("");

  const socketRef = useRef<WebSocket | null>(null);
  const ledgerAppliedFor = useRef<string | null>(null);
  const beatTimer = useRef<number | null>(null);

  const displayNames = (() => {
    const names = ["", "", "", ""];
    if (view) {
      names[view.seat] = "You";
      botSeats(view.seat).forEach((seat, i) => (names[seat] = BOT_NAMES[i]));
    }
    return names;
  })();

  const personaLabels = (() => {
    const labels = ["", "", "", ""];
    config.personas.forEach((id, seat) => {
      labels[seat] = PERSONAS.find((p) => p.id === id)?.label ?? id;
    });
    if (view) labels[view.seat] = "Human";
    return labels;
  })();

  // ── Game lifecycle ───────────────────────────────────────────────

  const startHand = useCallback(async (cfg: SetupConfig, hand: number) => {
    setBusy(true);
    setSetupError(null);
    try {
      // The human seat's agent never plays — it answers /hint requests,
      // so it should be the strongest advisor available: the learned
      // agent (value-aware discards AND claims).
      const bots = cfg.personas.map((p, seat) =>
        seat === cfg.humanSeat ? "learned" : PERSONA_TO_BOT[p] ?? "hybrid");
      const baseSeed = parseSeed(cfg.seedText);
      const created = await createGame({
        seed: baseSeed === null ? null : baseSeed + (hand - 1),
        human_seat: cfg.humanSeat,
        bots,
        tai_cap: cfg.taiCap,
      });
      socketRef.current?.close();
      ledgerAppliedFor.current = null;
      setGameId(created.game_id);
      setView(created.view);
      setAnalysis(null);
      explainSeq.current += 1;
      setClaimCoachLine(null);
      setExplanation(null);
      setExplainLoading(false);
      setCoachInterim(null);
      setBotBeat(false);
      setPhase("playing");
      socketRef.current = openGameSocket(created.game_id, setView);
    } catch (err) {
      setSetupError(err instanceof Error ? err.message : "Could not reach the game server");
      setPhase("setup");
    } finally {
      setBusy(false);
    }
  }, []);

  const onDeal = (cfg: SetupConfig) => {
    setConfig(cfg);
    setCoachVisible(cfg.coachOn);
    setLedger([0, 0, 0, 0]);
    setHandNumber(1);
    startHand(cfg, 1);
  };

  const onNextHand = () => {
    const next = handNumber + 1;
    setHandNumber(next);
    startHand(config, next);
  };

  const onEndSession = () => {
    socketRef.current?.close();
    setPhase("setup");
    setGameId(null);
    setView(null);
  };

  // ── Actions ──────────────────────────────────────────────────────

  const act = useCallback(async (answer: unknown) => {
    if (!gameId) return;
    try {
      const next = await postAction(gameId, answer);
      explainSeq.current += 1;
      setClaimCoachLine(null);
      setExplanation(null);
      setExplainLoading(false);
      setCoachInterim(null);
      if (!next.game_over) {
        setBotBeat(true);
        if (beatTimer.current) window.clearTimeout(beatTimer.current);
        beatTimer.current = window.setTimeout(() => setBotBeat(false), BOT_BEAT_MS);
      }
      setView(next);
    } catch (err) {
      setAnnouncement(err instanceof Error ? err.message : "That move was rejected");
    }
  }, [gameId]);

  const onDiscard = (tile: number) => act(tile);
  const onClaim = (accept: boolean) => act(accept);
  const onChow = (option: [number, number] | null) => act(option);
  const onKong = (option: [string, number] | null) => act(option);

  // ── Analysis + hints ─────────────────────────────────────────────

  const pendingKey = view?.pending ? `${view.pending.type}:${view.turn}` : `none:${view?.turn}`;

  useEffect(() => {
    if (!gameId || !view || view.game_over || !coachVisible) return;
    let stale = false;
    getAnalysis(gameId)
      .then((a) => { if (!stale) setAnalysis(a); })
      .catch(() => undefined);
    return () => { stale = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gameId, pendingKey, coachVisible]);

  const formatHint = useCallback((suggestion: unknown): string => {
    const pending = view?.pending;
    if (!pending) return "";
    if (pending.type === "discard") {
      return typeof suggestion === "number"
        ? `I would discard ${tileFace(suggestion).label} here.`
        : "I would keep the hand as it is.";
    }
    if (pending.type === "claim") {
      return suggestion
        ? `I would take the ${pending.claim_type} — it genuinely improves your hand.`
        : "I would pass. Claiming here exposes your hand without making it faster.";
    }
    if (pending.type === "chow") {
      if (Array.isArray(suggestion)) {
        const [a, b] = suggestion as [number, number];
        return `I would chow it with ${tileFace(a).label} and ${tileFace(b).label}.`;
      }
      return "I would pass — none of these runs actually help you.";
    }
    if (pending.type === "kong") {
      if (Array.isArray(suggestion)) {
        const [kind, t] = suggestion as [string, number];
        return `I would declare the ${kind} kong of ${tileFace(t).label} — the replacement draw is free value.`;
      }
      return "I would hold off — that fourth tile still serves your hand.";
    }
    return "";
  }, [view]);

  const requestExplanation = useCallback(async () => {
    if (!gameId || explainLoading || explanation) return;
    const seq = ++explainSeq.current;
    setExplainLoading(true);
    // Stage 1 — the engine's own pick, instantly, so the wait has substance
    getHint(gameId)
      .then((hint) => {
        if (explainSeq.current === seq) setCoachInterim(formatHint(hint.suggestion));
      })
      .catch(() => undefined);
    // Stage 2 — the full prose read
    try {
      const result = await postExplain(gameId);
      if (explainSeq.current === seq) setExplanation(result);
    } catch {
      /* no pending decision or server hiccup — button stays available */
    } finally {
      if (explainSeq.current === seq) {
        setExplainLoading(false);
        setCoachInterim(null);
      }
    }
  }, [gameId, explainLoading, explanation, formatHint]);

  // Coach line for claim prompts (auto-fetched when the window opens)
  useEffect(() => {
    const pending = view?.pending;
    if (!gameId || !pending || !coachVisible) return;
    if (pending.type !== "claim" && pending.type !== "chow") return;
    let stale = false;
    getHint(gameId)
      .then((hint) => { if (!stale) setClaimCoachLine(formatHint(hint.suggestion)); })
      .catch(() => undefined);
    return () => { stale = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gameId, pendingKey, coachVisible]);

  // Announce turn changes politely
  useEffect(() => {
    const pending = view?.pending;
    if (!pending) return;
    const messages: Record<string, string> = {
      discard: "Your turn — choose a tile to discard",
      claim: "Claim window open",
      chow: "Chow window open",
      kong: "You may declare a kong",
    };
    setAnnouncement(messages[pending.type] ?? "");
  }, [pendingKey, view]);

  useEffect(() => () => socketRef.current?.close(), []);

  // ── Derived display state ────────────────────────────────────────

  if (phase === "setup" || !view) {
    return (
      <div className="app">
        <TopBar prevailingWind={null} handNumber={handNumber} seedText={config.seedText}
                analysisOn={coachVisible} onToggleAnalysis={() => setCoachVisible(!coachVisible)}
                inGame={false} />
        <Setup initial={config} error={setupError} busy={busy} onDeal={onDeal} />
      </div>
    );
  }

  const pending = botBeat ? null : view.pending;
  const yourDiscardTurn = pending?.type === "discard";
  const drawnTile = yourDiscardTurn ? pending?.drawn ?? null : null;
  const claimTile =
    pending && (pending.type === "claim" || pending.type === "chow")
      ? pending.tile ?? null : null;

  const pillLabel = view.game_over
    ? "Hand over"
    : pending == null
      ? "Bots are playing"
      : pending.type === "discard"
        ? "Your turn"
        : pending.type === "kong"
          ? "Your turn — kong offered"
          : "Claim window";

  const dangerByTile = analysis?.discards
    ? new Map(analysis.discards.map((d) => [d.tile, d.danger]))
    : null;
  const recommendedTile = analysis?.discards?.[0]?.tile ?? null;

  const discarderName = (() => {
    if (claimTile === null) return "An opponent";
    for (const p of view.players) {
      if (p.seat === view.seat) continue;
      if (p.discards[p.discards.length - 1] === claimTile) return displayNames[p.seat];
    }
    return "An opponent";
  })();

  // Engine payments are in points (base unit 1)
  const chipsFor = (seat: number) => ledger[seat] + (view.players[seat].chips ?? 0);

  const ledgerAfter = view.result
    ? ledger.map((chips, seat) => chips + (view.result!.payments[seat] ?? 0))
    : ledger;

  const applyLedgerAndAdvance = () => {
    if (gameId && ledgerAppliedFor.current !== gameId) {
      ledgerAppliedFor.current = gameId;
      setLedger(ledgerAfter);
    }
  };

  const tells: (string | null)[] = [null, null, null, null];

  const overlay = (
    <>
      {pending && (pending.type === "claim" || pending.type === "chow") && (
        <ClaimPrompt view={view} pending={pending} discarderName={discarderName}
                     coachLine={coachVisible ? claimCoachLine : null}
                     explanation={coachVisible ? explanation : null}
                     explainLoading={coachVisible && explainLoading}
                     onExplain={requestExplanation}
                     onClaim={onClaim} onChow={onChow} />
      )}
      {pending?.type === "kong" && (
        <KongBanner pending={pending} onDeclare={(opt) => onKong(opt)} onPass={() => onKong(null)} />
      )}
    </>
  );

  return (
    <div className="app">
      <TopBar prevailingWind={view.prevailing_wind} handNumber={handNumber}
              seedText={config.seedText}
              analysisOn={coachVisible} onToggleAnalysis={() => setCoachVisible(!coachVisible)}
              inGame />
      <div className="table-row">
        <Table
          view={view}
          displayNames={displayNames}
          personas={personaLabels}
          chipsFor={chipsFor}
          tells={tells}
          pillLabel={pillLabel}
          yourTurn={pending != null}
          thinking={pending == null && !view.game_over}
          claimTile={claimTile}
          yourDiscardTurn={yourDiscardTurn}
          drawnTile={drawnTile}
          dangerByTile={dangerByTile}
          recommendedTile={coachVisible ? recommendedTile : null}
          heatVisible={coachVisible}
          statusText={yourDiscardTurn ? "Click a tile to discard" : "Bots are playing"}
          onDiscard={onDiscard}
          onHint={requestExplanation}
          overlay={overlay}
        />
        {coachVisible && (
          <Sidebar
            view={view}
            analysis={analysis}
            displayNames={displayNames}
            personas={personaLabels}
            paused={!yourDiscardTurn}
            onHide={() => setCoachVisible(false)}
            explanation={explanation}
            explainLoading={explainLoading}
            interim={coachInterim}
            onExplain={requestExplanation}
          />
        )}
      </div>
      {view.game_over && view.result && (
        <GameEnd
          view={view}
          result={view.result}
          displayNames={displayNames}
          handNumber={handNumber}
          onNextHand={() => { applyLedgerAndAdvance(); onNextHand(); }}
          onEndSession={() => { applyLedgerAndAdvance(); onEndSession(); }}
        />
      )}
      <div className="sr-only" aria-live="polite">{announcement}</div>
    </div>
  );
}
