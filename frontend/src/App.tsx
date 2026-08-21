import { useCallback, useEffect, useRef, useState } from "react";
import {
  Analysis, ClaimContext, Explanation, GameView, createGame, getAnalysis, getHint,
  openGameSocket, postAction, postExplain, postNextHand,
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
  const [handNumber, setHandNumber] = useState(1);
  const [coachVisible, setCoachVisible] = useState(true);
  // Bot-turn replay: the server resolves every bot turn between your
  // decisions and sends ONE view, so up to three discards would snap
  // into the rivers at once. The sequencer replays them as beats —
  // acting seat pulses, tile lands, next seat — so the table reads
  // like a table instead of a teleport.
  const [actingSeat, setActingSeat] = useState<number | null>(null);
  const [latestDiscard, setLatestDiscard] =
    useState<{ seat: number; index: number } | null>(null);
  const seqTimer = useRef<number | null>(null);
  const targetView = useRef<GameView | null>(null);
  const [claimCoachLine, setClaimCoachLine] = useState<string | null>(null);
  // Engine-computed tai consequence of the pending claim — rendered on
  // the claim card verdict-INDEPENDENTLY (the rulebook speaks even
  // when the advisor is wrong). null between claim windows.
  const [claimContext, setClaimContext] = useState<ClaimContext | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  // Instant local recommendation shown while the LLM writes its prose
  const [coachInterim, setCoachInterim] = useState<string | null>(null);
  // Guards late responses from a decision the player already left
  const explainSeq = useRef(0);
  const [setupError, setSetupError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [announcement, setAnnouncement] = useState("");
  // Visible error surface — the aria-live region alone is invisible to
  // sighted players, which made rejected moves look like dead clicks.
  const [errorToast, setErrorToast] = useState<string | null>(null);
  const toastTimer = useRef<number | null>(null);
  const actBusy = useRef(false);

  const showError = useCallback((text: string) => {
    setErrorToast(text);
    setAnnouncement(text);
    if (toastTimer.current) window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setErrorToast(null), 4000);
  }, []);

  const socketRef = useRef<WebSocket | null>(null);

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

  const ingestView = useCallback((next: GameView) => {
    const prev = targetView.current;
    targetView.current = next;
    if (seqTimer.current) {
      window.clearTimeout(seqTimer.current);
      seqTimer.current = null;
    }
    // Fresh hand / first view / anything that rewinds a river: no replay
    const reset = !prev || next.players.some(
      (pl, seatIdx) => pl.discards.length < prev.players[seatIdx].discards.length);
    if (reset) {
      setView(next);
      setActingSeat(null);
      setLatestDiscard(null);
      return;
    }
    // New bot discards since the last displayed view, in turn order
    // from whoever acted last (approximate across claims, exact within
    // each river — legibility, not forensics).
    const steps: Array<{ seat: number; index: number }> = [];
    for (let off = 0; off < 4; off++) {
      const seatIdx = (prev.active_player + off) % 4;
      if (seatIdx === next.seat) continue;
      for (let i = prev.players[seatIdx].discards.length;
           i < next.players[seatIdx].discards.length; i++) {
        steps.push({ seat: seatIdx, index: i });
      }
    }
    if (steps.length === 0) {
      setView(next);
      return;
    }
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const stepMs = reduce ? 40 : steps.length > 5 ? 260 : 430;
    const run = (k: number) => {
      if (targetView.current !== next) return; // superseded by a newer view
      if (k >= steps.length) {
        setActingSeat(null);
        setView(next);
        return;
      }
      const upto = steps.slice(0, k + 1);
      setActingSeat(steps[k].seat);
      setLatestDiscard(steps[k]);
      setView({
        ...next,
        pending: null,  // your prompt waits until the table catches up
        players: next.players.map((pl, seatIdx) => {
          if (seatIdx === next.seat) return pl;
          const shown = prev.players[seatIdx].discards.length
            + upto.filter((st) => st.seat === seatIdx).length;
          return { ...pl, discards: pl.discards.slice(0, shown) };
        }),
      });
      seqTimer.current = window.setTimeout(() => run(k + 1), stepMs);
    };
    run(0);
  }, []);


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
      if (socketRef.current) socketRef.current.onclose = null;
      socketRef.current?.close();
      setGameId(created.game_id);
      targetView.current = null;   // new session: nothing to replay
      ingestView(created.view);
      setAnalysis(null);
      explainSeq.current += 1;
      setClaimCoachLine(null);
      setClaimContext(null);
      setExplanation(null);
      setExplainLoading(false);
      setCoachInterim(null);
      setPhase("playing");
      socketRef.current = openGameSocket(created.game_id, ingestView, () =>
        showError("Live connection dropped — if the table stops updating, refresh."));
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
    setHandNumber(1);
    startHand(cfg, 1);
  };

  const onNextHand = async () => {
    if (!gameId) return;
    try {
      const next = await postNextHand(gameId);
      ingestView(next);
      setAnalysis(null);
      explainSeq.current += 1;
      setClaimCoachLine(null);
      setClaimContext(null);
      setExplanation(null);
      setExplainLoading(false);
      setCoachInterim(null);
    } catch (err) {
      showError(err instanceof Error ? err.message : "Could not deal the next hand");
    }
  };

  const onEndSession = () => {
    if (socketRef.current) socketRef.current.onclose = null;
    socketRef.current?.close();
    if (seqTimer.current) window.clearTimeout(seqTimer.current);
    targetView.current = null;
    setPhase("setup");
    setGameId(null);
    setView(null);
    setActingSeat(null);
    setLatestDiscard(null);
  };

  // ── Actions ──────────────────────────────────────────────────────

  const act = useCallback(async (answer: unknown) => {
    if (!gameId || actBusy.current) return;  // one move at a time
    actBusy.current = true;
    try {
      const next = await postAction(gameId, answer);
      explainSeq.current += 1;
      setClaimCoachLine(null);
      setClaimContext(null);
      setExplanation(null);
      setExplainLoading(false);
      setCoachInterim(null);
      ingestView(next);
    } catch (err) {
      showError(err instanceof Error ? err.message : "That move was rejected");
    } finally {
      actBusy.current = false;
    }
  }, [gameId, showError]);

  const onDiscard = (tile: number) => { setLatestDiscard(null); act(tile); };
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

  // The one-liner states the verdict plus, for claims, the ENGINE's
  // computed consequence. No fabricated rationales: an earlier version
  // hardcoded "it genuinely improves your hand" onto whatever boolean
  // the agent returned — asserting a reason nothing had computed.
  const formatHint = useCallback((suggestion: unknown, ctx?: ClaimContext): string => {
    const pending = view?.pending;
    if (!pending) return "";
    const consequence = ctx?.headline ? ` ${ctx.headline}` : "";
    if (pending.type === "discard") {
      return typeof suggestion === "number"
        ? `I would discard ${tileFace(suggestion).label} here.`
        : "I would keep the hand as it is.";
    }
    if (pending.type === "claim") {
      return (suggestion
        ? `I would take the ${pending.claim_type}.`
        : `I would pass on the ${pending.claim_type}.`) + consequence;
    }
    if (pending.type === "chow") {
      if (Array.isArray(suggestion)) {
        const [a, b] = suggestion as [number, number];
        return `I would chow it with ${tileFace(a).label} and ${tileFace(b).label}.` + consequence;
      }
      return "I would pass on the chow." + consequence;
    }
    if (pending.type === "kong") {
      if (Array.isArray(suggestion)) {
        const [kind, t] = suggestion as [string, number];
        return `I would declare the ${kind} kong of ${tileFace(t).label} — the replacement draw comes from the back of the wall.`;
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
        if (explainSeq.current === seq) {
          setCoachInterim(formatHint(hint.suggestion, hint.claim_context));
          setClaimContext(hint.claim_context ?? null);
        }
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
      .then((hint) => {
        if (stale) return;
        setClaimCoachLine(formatHint(hint.suggestion, hint.claim_context));
        setClaimContext(hint.claim_context ?? null);
      })
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

  useEffect(() => () => {
    if (socketRef.current) socketRef.current.onclose = null;
    socketRef.current?.close();
  }, []);

  // ── Derived display state ────────────────────────────────────────

  if (phase === "setup" || !view) {
    return (
      <div className="app">
        <TopBar prevailingWind={null} roundLabel={null} handNumber={handNumber} seedText={config.seedText}
                analysisOn={coachVisible} onToggleAnalysis={() => setCoachVisible(!coachVisible)}
                inGame={false} />
        <Setup initial={config} error={setupError} busy={busy} onDeal={onDeal} />
      </div>
    );
  }

  const pending = view.pending;
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

  // Running session scores live on the server; during a finished hand
  // the result's payments are shown on top so the end screen already
  // reflects what the next hand will bank.
  const chipsFor = (seat: number) =>
    (view.session?.scores[seat] ?? 0) +
    (view.result ? view.result.payments[seat] ?? 0 : 0);

  const tells: (string | null)[] = [null, null, null, null];

  const overlay = (
    <>
      {pending && (pending.type === "claim" || pending.type === "chow") && (
        <ClaimPrompt view={view} pending={pending} discarderName={discarderName}
                     coachLine={coachVisible ? claimCoachLine : null}
                     claimContext={claimContext}
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
      <TopBar prevailingWind={view.prevailing_wind}
              roundLabel={view.session?.round_label ?? null}
              handNumber={view.session?.hand_number ?? handNumber}
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
          showExplain={coachVisible}
          actingSeat={actingSeat}
          latestDiscard={latestDiscard}
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
            canExplain={pending != null}
            onExplain={requestExplanation}
          />
        )}
      </div>
      {view.game_over && view.result && (
        <GameEnd
          view={view}
          result={view.result}
          displayNames={displayNames}
          handNumber={view.session?.hand_number ?? handNumber}
          session={view.session}
          onNextHand={onNextHand}
          onEndSession={onEndSession}
        />
      )}
      {errorToast && (
        <div className="toast" role="status">{errorToast}</div>
      )}
      <div className="sr-only" aria-live="polite">{announcement}</div>
    </div>
  );
}
