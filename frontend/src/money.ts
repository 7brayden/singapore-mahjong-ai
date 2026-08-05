// The engine scores in integer units; the UI treats those units as cents.
// A "rate" of 20 means a 20-cent game: a 1-tai win pays $0.20.

export function fmtMoney(cents: number): string {
  const sign = cents < 0 ? "−" : "";
  const abs = Math.abs(cents);
  const amount = abs % 100 === 0 ? String(abs / 100) : (abs / 100).toFixed(2);
  return `${sign}$${amount}`;
}

/** Signed form for balances: +$2.40 / −$0.80 / $0 */
export function fmtSigned(cents: number): string {
  return cents > 0 ? `+${fmtMoney(cents)}` : fmtMoney(cents);
}
