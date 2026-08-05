/** Signed point display: +3 / −8 / 0. Engine payment units are points. */
export function fmtPoints(points: number): string {
  if (points > 0) return `+${points}`;
  if (points < 0) return `−${Math.abs(points)}`;
  return "0";
}
