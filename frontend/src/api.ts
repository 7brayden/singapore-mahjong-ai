// REST + websocket client for the FastAPI backend.

export interface PlayerView {
  seat: number;
  name: string;
  is_human: boolean;
  seat_wind: number;
  concealed_count: number;
  exposed: Array<[string, number[]]>;
  flowers: number[];
  discards: number[];
  chips: number;
}

export interface PendingView {
  type: "discard" | "claim" | "chow" | "kong";
  drawn?: number | null;
  tile?: number;
  claim_type?: string;
  options?: Array<[number, number]> | Array<[string, number]>;
}

export interface ScoreItemView {
  rule: string;
  label: string;
  tai: number;
}

export interface ResultView {
  winner: number | null;
  win_type: string | null;
  dealt_in_by: number | null;
  turns: number;
  payments: number[];
  win_tile: number | null;
  winner_hand?: number[];
  winner_exposed?: Array<[string, number[]]>;
  score?: {
    tai: number;
    total_tai: number;
    value: number;
    is_limit: boolean;
    items: ScoreItemView[];
  };
}

export interface GameView {
  seat: number;
  hand: number[];
  turn: number;
  active_player: number;
  tiles_remaining: number;
  dealer: number;
  prevailing_wind: number;
  players: PlayerView[];
  game_over: boolean;
  pending: PendingView | null;
  result: ResultView | null;
}

export interface DiscardAnalysis {
  tile: number;
  shanten_after: number;
  acceptance: number;
  improving_tiles: number[];
  danger: number;
  danger_components: {
    visibility: number;
    discard_absence: number;
    opponent_threat: number;
    suit_safety: number;
  };
  /** Calibrated P(deal-in) from the trained model; absent if untrained. */
  deal_in_prob?: number;
  /** Calibrated P(win from here) from the trained model. */
  win_prob?: number;
}

export interface OpponentAnalysis {
  seat: number;
  threat: number;
  suit_danger: Record<string, number>;
}

export interface Analysis {
  seat: number;
  shanten: number;
  opponents: OpponentAnalysis[];
  discards?: DiscardAnalysis[];
  waiting_on?: number[];
}

const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://localhost:8000";

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function createGame(body: {
  seed?: number | null;
  human_seat?: number;
  bots?: string[];
  tai_cap?: number;
  base_unit?: number;
}): Promise<{ game_id: string; view: GameView }> {
  return request("POST", "/games", body);
}

export const getView = (gameId: string) => request<GameView>("GET", `/games/${gameId}`);

export const postAction = (gameId: string, answer: unknown) =>
  request<GameView>("POST", `/games/${gameId}/action`, { answer });

export const getHint = (gameId: string) =>
  request<{ pending: PendingView; suggestion: unknown }>("GET", `/games/${gameId}/hint`);

export const getAnalysis = (gameId: string) =>
  request<Analysis>("GET", `/games/${gameId}/analysis`);

export function openGameSocket(gameId: string, onView: (view: GameView) => void): WebSocket {
  const wsBase = API_BASE.replace(/^http/, "ws");
  const socket = new WebSocket(`${wsBase}/games/${gameId}/ws`);
  socket.onmessage = (event) => {
    try {
      onView(JSON.parse(event.data));
    } catch {
      /* ignore malformed frames */
    }
  };
  return socket;
}
