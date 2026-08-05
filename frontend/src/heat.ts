// Danger heat ramp from the design handoff:
// 0.0 = green (111,174,127) -> 0.5 = amber (214,172,71) -> 1.0 = red (198,86,75)

const STOPS: Array<[number, number, number, number]> = [
  [0, 111, 174, 127],
  [0.5, 214, 172, 71],
  [1, 198, 86, 75],
];

export function heat(value: number): string {
  const v = Math.min(1, Math.max(0, value));
  let a = STOPS[0];
  let b = STOPS[1];
  if (v > 0.5) {
    a = STOPS[1];
    b = STOPS[2];
  }
  const f = (v - a[0]) / (b[0] - a[0]);
  const rgb = [1, 2, 3].map((i) => Math.round(a[i] + (b[i] - a[i]) * f));
  return `rgb(${rgb.join(",")})`;
}
