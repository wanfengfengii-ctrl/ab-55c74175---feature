import { describe, expect, it } from 'vitest';
import { solve, optimalTimesBudgeted } from '../src/core/solver.js';
import { SolveError } from '../src/core/types.js';
import type { PacketInput } from '../src/core/types.js';
import { bruteSolveJitter, makeRng } from './helpers/brute.js';

// Deterministic planted chain (modulus 7, interval 4..7, nominal 7):
//   id 0 -> 1 -> 2 -> 3 -> 4 -> 5
// packets are listed in scrambled download order.
const planted: PacketInput[] = [
  { id: 2, remainder: 0, timeLower: 25, timeUpper: 27 },
  { id: 0, remainder: 4, timeLower: 10, timeUpper: 12 },
  { id: 5, remainder: 4, timeLower: 50, timeUpper: 52 },
  { id: 1, remainder: 6, timeLower: 18, timeUpper: 20 },
  { id: 3, remainder: 2, timeLower: 39, timeUpper: 41 },
  { id: 4, remainder: 3, timeLower: 46, timeUpper: 48 },
];
const base = { modulus: 7, countLower: 0, countUpper: 19, minInterval: 4, maxInterval: 7 };

describe('jitter budget: successful responses', () => {
  it('recovers the chain at the smallest fitting budget and reports the exact spend', () => {
    const r = solve(planted, base.modulus, base.countLower, base.countUpper,
      base.minInterval, base.maxInterval, { nominalInterval: 7, totalJitterBudget: 7 });
    expect(r.order).toEqual([0, 1, 2, 3, 4, 5]);
    expect(r.jitterBudget).toEqual({
      nominalInterval: 7,
      budget: 7,
      used: 7,
      remaining: 0,
      exhausted: true,
    });
    // Per-adjacency evidence: nominal gap, jitter and running cumulative spend.
    expect(r.adjacency).toHaveLength(5);
    let running = 0;
    for (const ev of r.adjacency) {
      expect(ev.nominalTimeGap).toBe(ev.countGap * 7);
      expect(ev.jitter).toBe(Math.abs(ev.timeGap - ev.nominalTimeGap!));
      running += ev.jitter!;
      expect(ev.cumulativeJitter).toBe(running);
    }
    expect(r.adjacency[r.adjacency.length - 1].cumulativeJitter).toBe(7);
    expect(r.adjacency[r.adjacency.length - 1].cumulativeJitter).toBe(r.jitterBudget!.used);
  });

  it('accepts a budget that is exactly exhausted', () => {
    const r = solve(planted, base.modulus, base.countLower, base.countUpper,
      base.minInterval, base.maxInterval, { nominalInterval: 7, totalJitterBudget: 7 });
    expect(r.jitterBudget!.used).toBe(7);
    expect(r.jitterBudget!.exhausted).toBe(true);
  });

  it('uses no more jitter than necessary under a loose budget', () => {
    const r = solve(planted, base.modulus, base.countLower, base.countUpper,
      base.minInterval, base.maxInterval, { nominalInterval: 7, totalJitterBudget: 100 });
    expect(r.jitterBudget!.used).toBe(9);
    expect(r.jitterBudget!.remaining).toBe(91);
    expect(r.jitterBudget!.exhausted).toBe(false);
    // A loose budget still returns the gap-minimal primary order.
    expect(r.missingCountTotal).toBe(2);
  });

  it('omits every jitter field when neither parameter is supplied', () => {
    const r = solve(planted, base.modulus, base.countLower, base.countUpper,
      base.minInterval, base.maxInterval);
    expect(r.jitterBudget).toBeUndefined();
    for (const ev of r.adjacency) {
      expect(ev.nominalTimeGap).toBeUndefined();
      expect(ev.jitter).toBeUndefined();
      expect(ev.cumulativeJitter).toBeUndefined();
    }
  });

  it('honors nominalInterval equal to either interval bound', () => {
    for (const nominal of [4, 7]) {
      const r = solve(planted, base.modulus, base.countLower, base.countUpper,
        base.minInterval, base.maxInterval, { nominalInterval: nominal, totalJitterBudget: 1000 });
      for (const ev of r.adjacency) expect(ev.jitter!).toBeGreaterThanOrEqual(0);
      let total = 0;
      for (const ev of r.adjacency) total += ev.jitter!;
      expect(total).toBe(r.jitterBudget!.used);
    }
  });
});

describe('jitter budget: joint solving rather than post-hoc filtering', () => {
  it('reroutes the search: tight budgets are reached through non-gap-minimal chains', () => {
    // Across planted instances, at least some must be feasible only because
    // the optimizer leaves the primary gap-minimal chain (which exceeds the
    // budget) for a slightly longer chain that fits. Such cases cannot be
    // produced by solving first and filtering afterwards.
    const rnd = makeRng(31337);
    let rerouted = 0;
    for (let iter = 0; iter < 120 && rerouted < 3; iter++) {
      const cfg = plant(rnd);
      let plain: ReturnType<typeof solve>;
      try {
        plain = solve(cfg.packets, cfg.modulus, cfg.countLower, cfg.countUpper,
          cfg.minInterval, cfg.maxInterval);
      } catch {
        continue;
      }
      let plainJitter = 0;
      for (let k = 1; k < plain.assignments.length; k++) {
        const d = plain.assignments[k].absoluteCount - plain.assignments[k - 1].absoluteCount;
        plainJitter += Math.abs(plain.assignments[k].time - plain.assignments[k - 1].time - d * cfg.nominal);
      }
      for (const budget of [0, 1, 2]) {
        if (plainJitter <= budget) continue;
        try {
          const r = solve(cfg.packets, cfg.modulus, cfg.countLower, cfg.countUpper,
            cfg.minInterval, cfg.maxInterval,
            { nominalInterval: cfg.nominal, totalJitterBudget: budget });
          // Feasible under the budget even though the plain optimum was not:
          // the joint search found another order/count/timestamp combination.
          expect(r.jitterBudget!.used).toBeLessThanOrEqual(budget);
          rerouted++;
          break;
        } catch (e) {
          if (!(e instanceof SolveError)) throw e;
        }
      }
    }
    expect(rerouted).toBeGreaterThan(0);
  });
});

describe('jitter budget: first-blocking evidence', () => {
  it('reports an extension-stage blocker with used, minimumAdditional and budget', () => {
    let caught: SolveError | null = null;
    try {
      solve(planted, base.modulus, base.countLower, base.countUpper,
        base.minInterval, base.maxInterval, { nominalInterval: 7, totalJitterBudget: 4 });
    } catch (e) {
      caught = e as SolveError;
    }
    expect(caught).not.toBeNull();
    expect(caught!.code).toBe('NO_CONSISTENT_INTERPRETATION');
    expect(caught!.evidence?.stage).toBe('extension');
    expect(caught!.evidence?.detail?.cause).toBe('JITTER_BUDGET');
    expect(caught!.evidence?.detail?.jitter).toEqual({
      nominalInterval: 7,
      budget: 4,
      used: 4,
      minimumAdditional: 1,
    });
  });

  it('reports a complete-chain shortfall at the leaf blocker', () => {
    let caught: SolveError | null = null;
    try {
      solve(planted, base.modulus, base.countLower, base.countUpper,
        base.minInterval, base.maxInterval, { nominalInterval: 7, totalJitterBudget: 6 });
    } catch (e) {
      caught = e as SolveError;
    }
    expect(caught).not.toBeNull();
    expect(caught!.evidence?.partialLength).toBe(6);
    expect(caught!.evidence?.detail?.cause).toBe('JITTER_BUDGET');
    expect(caught!.evidence?.detail?.jitter).toEqual({
      nominalInterval: 7,
      budget: 6,
      used: 7,
      minimumAdditional: 1,
    });
  });

  it('keeps the original structural error class when time/count constraints fail', () => {
    let caught: SolveError | null = null;
    try {
      solve(planted, base.modulus, base.countLower, base.countUpper,
        base.minInterval, base.maxInterval, { nominalInterval: 7, totalJitterBudget: 0 });
    } catch (e) {
      caught = e as SolveError;
    }
    expect(caught).not.toBeNull();
    expect(caught!.code).toBe('NO_CONSISTENT_INTERPRETATION');
    expect(['TIME_GAP', 'COUNT_WINDOW', 'CONGRUENCE']).toContain(
      caught!.evidence?.detail?.cause,
    );
  });
});

describe('jitter budget: differential testing against brute force', () => {
  it('matches the exhaustive reference on planted chains', () => {
    let feasiblePairs = 0;
    let totalRuns = 0;
    for (const seedStart of [101, 202, 303]) {
      const rnd = makeRng(seedStart);
      for (let iter = 0; iter < 40; iter++) {
        const cfg = plant(rnd);
        for (const budget of [0, 1, 2, 5, 1000]) {
          totalRuns++;
          let got: {
            missing: number;
            deviation: number;
            idSeq: (string | number)[];
            jitter: number;
          } | null = null;
          let gotError = false;
          try {
            const r = solve(cfg.packets, cfg.modulus, cfg.countLower, cfg.countUpper,
              cfg.minInterval, cfg.maxInterval,
              { nominalInterval: cfg.nominal, totalJitterBudget: budget });
            let deviation = 0;
            let jitter = 0;
            for (const a of r.assignments) {
              const p = cfg.packets.find((pp) => pp.id === a.id)!;
              deviation += Math.abs(2 * a.time - (p.timeLower + p.timeUpper));
            }
            for (const ev of r.adjacency) jitter += ev.jitter!;
            expect(jitter).toBeLessThanOrEqual(budget);
            got = { missing: r.missingCountTotal, deviation, idSeq: r.order, jitter };
          } catch (e) {
            if (!(e instanceof SolveError)) throw e;
            gotError = true;
          }

          const ref = bruteSolveJitter(cfg.packets, cfg.modulus, cfg.countLower, cfg.countUpper,
            cfg.minInterval, cfg.maxInterval, cfg.nominal, budget);
          expect(gotError).toBe(ref === null);
          if (got && ref) {
            expect(got.missing).toBe(ref.missing);
            expect(got.deviation).toBe(ref.deviation);
            expect(got.idSeq).toEqual(ref.idSeq);
            expect(got.jitter).toBe(ref.jitter);
            feasiblePairs++;
          }
        }
      }
    }
    // Ensure the differential actually exercised feasible budgeted cases.
    expect(feasiblePairs).toBeGreaterThan(60);
    expect(totalRuns).toBe(600);
  });
});

describe('optimalTimesBudgeted: fixed-chain unit behavior', () => {
  const packets = [
    { lo: 12, hi: 16 }, { lo: 15, hi: 19 }, { lo: 19, hi: 23 },
    { lo: 23, hi: 27 }, { lo: 29, hi: 33 }, { lo: 30, hi: 34 },
  ].map((w) => ({ ...w, mid2: w.lo + w.hi })) as Parameters<typeof optimalTimesBudgeted>[0];

  it('returns null when no timestamp vector meets the budget', () => {
    const order = [0, 1, 2, 3, 4, 5];
    const gaps = [1, 1, 7, 2, 1];
    const windows = [
      { lo: 12, hi: 16 }, { lo: 15, hi: 19 }, { lo: 19, hi: 20 },
      { lo: 26, hi: 27 }, { lo: 29, hi: 33 }, { lo: 30, hi: 34 },
    ];
    // That chain needs at least 2 total jitter (its big gap forces a skew).
    expect(optimalTimesBudgeted(packets, order, windows, gaps, 1, 4, 1, 0)).toBeNull();
    const atOne = optimalTimesBudgeted(packets, order, windows, gaps, 1, 4, 1, 1);
    expect(atOne).toBeNull();
  });

  it('hits the budget exactly with a deviation-optimal vector', () => {
    const order = [0, 1, 2, 3, 4, 5];
    const gaps = [1, 6, 2, 2, 1];
    const windows = [
      { lo: 12, hi: 16 }, { lo: 15, hi: 17 }, { lo: 21, hi: 23 },
      { lo: 23, hi: 27 }, { lo: 29, hi: 33 }, { lo: 30, hi: 34 },
    ];
    const sol = optimalTimesBudgeted(packets, order, windows, gaps, 1, 4, 1, 2);
    expect(sol).not.toBeNull();
    expect(sol!.jitter).toBeLessThanOrEqual(2);
    expect(sol!.times).toEqual([16, 17, 23, 25, 29, 30]);
  });
});

describe('optimalTimesBudgeted: wide-window spanning-set exactness', () => {
  /** Exhaustive minimum-deviation vector under the jitter budget. */
  function brute(
    wins: { lo: number; hi: number; mid2: number }[],
    gaps: number[],
    minI: number,
    maxI: number,
    nom: number,
    budget: number,
  ): { dev: number; ts: number[]; jitter: number } | null {
    const n = wins.length;
    const ts = new Array<number>(n);
    let best: { dev: number; ts: number[]; jitter: number } | null = null;
    const lexLess = (a: number[], b: number[]): boolean => {
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i];
      return false;
    };
    const rec = (k: number): void => {
      if (k === n) {
        let jitter = 0;
        for (let e = 0; e < n - 1; e++) {
          jitter += Math.abs(ts[e + 1] - ts[e] - gaps[e] * nom);
        }
        if (jitter > budget) return;
        let dev = 0;
        for (let i = 0; i < n; i++) dev += Math.abs(2 * ts[i] - wins[i].mid2);
        if (best === null || dev < best.dev || (dev === best.dev && lexLess(ts, best.ts))) {
          best = { dev, ts: [...ts], jitter };
        }
        return;
      }
      for (let t = wins[k].lo; t <= wins[k].hi; t++) {
        if (k > 0) {
          const dt = t - ts[k - 1];
          if (dt < gaps[k - 1] * minI || dt > gaps[k - 1] * maxI) continue;
        }
        ts[k] = t;
        rec(k + 1);
      }
    };
    rec(0);
    return best;
  }

  it('matches brute force over wide windows and varied budgets', () => {
    const rnd = makeRng(0xc0ffee);
    let checked = 0;
    for (let it = 0; it < 140; it++) {
      const n = 2 + Math.floor(rnd() * 5);
      const minI = 1 + Math.floor(rnd() * 4);
      const maxI = minI + 1 + Math.floor(rnd() * 5);
      const nom = minI + Math.floor(rnd() * (maxI - minI + 1));
      const gaps = Array.from({ length: n - 1 }, () => 1 + Math.floor(rnd() * 3));
      // Wide windows: tens to hundreds of integers, sometimes asymmetric.
      const centers = new Array<number>(n);
      centers[0] = 20 + Math.floor(rnd() * 40);
      for (let k = 1; k < n; k++) {
        centers[k] = centers[k - 1] + gaps[k - 1] * nom + (Math.floor(rnd() * 9) - 4);
      }
      const wins = centers.map((m) => {
        const half = 12 + Math.floor(rnd() * 40);
        return { lo: m - half, hi: m + half, mid2: (m - half) + (m + half) };
      });
      const packets = wins.map((w) => ({ lo: w.lo, hi: w.hi, mid2: w.mid2 })) as Parameters<
        typeof optimalTimesBudgeted
      >[0];
      const order = wins.map((_, i) => i);
      for (const budget of [0, 1, 3, 8, 25]) {
        const got = optimalTimesBudgeted(packets, order, wins.map((w) => ({ ...w })), gaps, minI, maxI, nom, budget);
        const want = brute(wins, gaps, minI, maxI, nom, budget);
        if ((got === null) !== (want === null) || (got && want && (got.deviation2 !== want.dev || JSON.stringify(got.times) !== JSON.stringify(want.ts)))) {
          throw new Error(`wide mismatch: ${JSON.stringify({ n, gaps, wins, minI, maxI, nom, budget, got, want })}`);
        }
        checked++;
      }
    }
    expect(checked).toBe(700);
  });
});

/** Plant a guaranteed-feasible ground-truth chain for differential testing. */
function plant(rnd: () => number) {
  const n = 6;
  const modulus = 2 + Math.floor(rnd() * 6);
  const minInterval = 1 + Math.floor(rnd() * 4);
  const maxInterval = minInterval + 1 + Math.floor(rnd() * 4);
  const nominal = minInterval + Math.floor(rnd() * (maxInterval - minInterval + 1));
  const counts: number[] = [];
  let c = Math.floor(rnd() * modulus);
  for (let i = 0; i < n; i++) {
    counts.push(c);
    c += 1 + Math.floor(rnd() * 2);
  }
  const countLower = 0;
  const countUpper = c + modulus;
  const baseTimes: number[] = [];
  let t = 5 + Math.floor(rnd() * 10);
  for (let i = 0; i < n; i++) {
    if (i === 0) {
      baseTimes.push(t);
    } else {
      const d = counts[i] - counts[i - 1];
      const loNoise = d * (minInterval - nominal);
      const hiNoise = d * (maxInterval - nominal);
      const noise = loNoise + Math.floor(rnd() * (hiNoise - loNoise + 1));
      t = baseTimes[i - 1] + d * nominal + noise;
      baseTimes.push(t);
    }
  }
  const hw = Math.floor(rnd() * 3);
  const packetsList: PacketInput[] = counts.map((cc, i) => ({
    id: i,
    remainder: ((cc % modulus) + modulus) % modulus,
    timeLower: baseTimes[i] - hw,
    timeUpper: baseTimes[i] + hw,
  }));
  for (let i = packetsList.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [packetsList[i], packetsList[j]] = [packetsList[j], packetsList[i]];
  }
  return { modulus, countLower, countUpper, minInterval, maxInterval, nominal, packets: packetsList };
}
