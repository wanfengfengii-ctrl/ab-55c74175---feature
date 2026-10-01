import { describe, expect, it } from 'vitest';
import { optimalTimes } from '../src/core/solver.js';

/**
 * The tight-edge-chain candidate enumeration inside optimalTimes must span
 * an optimum. These tests compare it with a DP over the ENTIRE integer
 * coordinate domain (no candidate-generation assumptions), using wide
 * windows so the claim is exercised beyond the narrow fuzz intervals.
 */

interface P {
  lo: number;
  hi: number;
  mid2: number;
}

/** Reference: full-domain integer DP + lex-smallest greedy reconstruction. */
function fullDomainOptimal(
  packets: P[],
  windows: { lo: number; hi: number }[],
  gaps: number[],
  minInterval: number,
  maxInterval: number,
): { times: number[]; deviation2: number } {
  const n = windows.length;
  const L = gaps.map((d) => d * minInterval);
  const U = gaps.map((d) => d * maxInterval);
  let domLo = Infinity;
  let domHi = -Infinity;
  for (const w of windows) {
    domLo = Math.min(domLo, w.lo);
    domHi = Math.max(domHi, w.hi);
  }
  // Every feasible value lies inside [domLo, domHi]; use it as the domain.
  const coords: number[] = [];
  for (let v = domLo; v <= domHi; v++) coords.push(v);
  const inWin = (k: number, v: number) => v >= windows[k].lo && v <= windows[k].hi;
  const dev = (k: number, v: number) => Math.abs(2 * v - packets[k].mid2);

  const suffix: Map<number, number>[] = new Array(n);
  suffix[n - 1] = new Map(coords.filter((v) => inWin(n - 1, v)).map((v) => [v, dev(n - 1, v)]));
  for (let k = n - 2; k >= 0; k--) {
    suffix[k] = new Map();
    for (const v of coords) {
      if (!inWin(k, v)) continue;
      let best = Infinity;
      for (const u of coords) {
        if (!inWin(k + 1, u)) continue;
        const g = u - v;
        if (g < L[k] || g > U[k]) continue;
        const c = suffix[k + 1].get(u);
        if (c !== undefined && c < best) best = c;
      }
      if (Number.isFinite(best)) suffix[k].set(v, best + dev(k, v));
    }
  }
  let globalBest = Infinity;
  for (const c of suffix[0].values()) globalBest = Math.min(globalBest, c);

  const times: number[] = [];
  let target = globalBest;
  let prev = NaN;
  for (let k = 0; k < n; k++) {
    let pick = NaN;
    for (const v of coords) {
      if (!inWin(k, v)) continue;
      if (k > 0) {
        const g = v - prev;
        if (g < L[k - 1] || g > U[k - 1]) continue;
      }
      if (suffix[k].get(v) === target) {
        pick = v;
        break;
      }
    }
    if (Number.isNaN(pick)) throw new Error('reference reconstruction failed');
    times.push(pick);
    target -= dev(k, pick);
    prev = pick;
  }
  return { times, deviation2: globalBest };
}

let seed = 20260930;
const rnd = () => {
  seed = (seed * 1103515245 + 12345) % 2147483648;
  return seed / 2147483648;
};

describe('optimalTimes candidate completeness', () => {
  it('matches the full-domain DP on 200 random wide-window chains', () => {    for (let iter = 0; iter < 200; iter++) {
      const n = 2 + Math.floor(rnd() * 6); // 2..7
      const packets: P[] = [];
      const center: number[] = [];
      for (let k = 0; k < n; k++) {
        center.push(-20 + Math.floor(rnd() * 60));
        const half = Math.floor(rnd() * 9);
        const lo = center[k] - half;
        const hi = center[k] + half;
        packets.push({ lo, hi, mid2: lo + hi });
      }
      const gaps = Array.from({ length: n - 1 }, () => 1 + Math.floor(rnd() * 3));
      const minInterval = 1 + Math.floor(rnd() * 4);
      const maxInterval = minInterval + Math.floor(rnd() * 8);

      // Tighten exactly like the solver does.
      const windows = packets.map((p) => ({ lo: p.lo, hi: p.hi }));
      let feasible = true;
      for (let k = 1; k < n; k++) {
        windows[k].lo = Math.max(windows[k].lo, windows[k - 1].lo + gaps[k - 1] * minInterval);
        windows[k].hi = Math.min(windows[k].hi, windows[k - 1].hi + gaps[k - 1] * maxInterval);
        if (windows[k].lo > windows[k].hi) {
          feasible = false;
          break;
        }
      }
      if (!feasible) continue;
      for (let k = n - 2; k >= 0; k--) {
        windows[k].lo = Math.max(windows[k].lo, windows[k + 1].lo - gaps[k] * maxInterval);
        windows[k].hi = Math.min(windows[k].hi, windows[k + 1].hi - gaps[k] * minInterval);
        if (windows[k].lo > windows[k].hi) {
          feasible = false;
          break;
        }
      }
      if (!feasible) continue;

      // optimalTimes expects the solver's Packet shape; only mid2 is read.
      const order = packets.map((_, i) => i);
      const asPackets = packets as unknown as Parameters<typeof optimalTimes>[0];
      const got = optimalTimes(asPackets, order, windows, gaps, minInterval, maxInterval);
      const ref = fullDomainOptimal(packets, windows, gaps, minInterval, maxInterval);

      expect(got.deviation2).toBe(ref.deviation2);
      expect(got.times).toEqual(ref.times);
    }
  });

  it('matches the full-domain DP with a cumulative jitter budget (300 random chains)', () => {
    for (let iter = 0; iter < 300; iter++) {
      const n = 2 + Math.floor(rnd() * 5); // 2..6
      const packets: P[] = [];
      for (let k = 0; k < n; k++) {
        const center = -20 + Math.floor(rnd() * 60);
        const half = Math.floor(rnd() * 9);
        const lo = center - half;
        const hi = center + half;
        packets.push({ lo, hi, mid2: lo + hi });
      }
      const gaps = Array.from({ length: n - 1 }, () => 1 + Math.floor(rnd() * 3));
      const minInterval = 1 + Math.floor(rnd() * 4);
      const maxInterval = minInterval + Math.floor(rnd() * 8);
      const nominal = minInterval + Math.floor(rnd() * (maxInterval - minInterval + 1));

      const feasibleWins = packets.map((p) => ({ lo: p.lo, hi: p.hi }));
      let feasible = true;
      for (let k = 1; k < n; k++) {
        feasibleWins[k].lo = Math.max(feasibleWins[k].lo, feasibleWins[k - 1].lo + gaps[k - 1] * minInterval);
        feasibleWins[k].hi = Math.min(feasibleWins[k].hi, feasibleWins[k - 1].hi + gaps[k - 1] * maxInterval);
        if (feasibleWins[k].lo > feasibleWins[k].hi) {
          feasible = false;
          break;
        }
      }
      if (!feasible) continue;
      for (let k = n - 2; k >= 0; k--) {
        feasibleWins[k].lo = Math.max(feasibleWins[k].lo, feasibleWins[k + 1].lo - gaps[k] * maxInterval);
        feasibleWins[k].hi = Math.min(feasibleWins[k].hi, feasibleWins[k + 1].hi - gaps[k] * minInterval);
        if (feasibleWins[k].lo > feasibleWins[k].hi) {
          feasible = false;
          break;
        }
      }
      if (!feasible) continue;

      const order = packets.map((_, i) => i);
      const asPackets = packets as unknown as Parameters<typeof optimalTimes>[0];

      // Uncapped feasibility: minimum jitter attainable.
      const minJ = optimalTimes(
        asPackets, order, feasibleWins, gaps, minInterval, maxInterval,
        { nominal, total: Number.POSITIVE_INFINITY }, 'feasibility',
      ).jitter;
      if (!Number.isFinite(minJ)) continue;

      // Reference over the ENTIRE integer domain, budget enforced jointly.
      let domLo = Infinity;
      let domHi = -Infinity;
      for (const w of feasibleWins) {
        domLo = Math.min(domLo, w.lo);
        domHi = Math.max(domHi, w.hi);
      }
      const coords: number[] = [];
      for (let v = domLo; v <= domHi; v++) coords.push(v);
      const inWin = (k: number, v: number) => v >= feasibleWins[k].lo && v <= feasibleWins[k].hi;
      const devCost = (k: number, v: number) => Math.abs(2 * v - packets[k].mid2);
      const L = gaps.map((d) => d * minInterval);
      const U = gaps.map((d) => d * maxInterval);
      const N = gaps.map((d) => d * nominal);

      for (const budget of [0, minJ, minJ + Math.floor(rnd() * 5)]) {
        if (budget < 0) continue;
        // Reference over the full integer domain. Per (timestamp, jitter)
        // state keep the minimum deviation; ties keep the lex-smallest
        // prefix vector, so the surviving optimal state's vector is exactly
        // the lex-smallest global optimum.
        type St = { v: number; jit: number; cost: number; vec: number[] };
        const merge = (map: Map<string, St>, st: St): void => {
          const key = `${st.v}|${st.jit}`;
          const old = map.get(key);
          if (
            old === undefined ||
            st.cost < old.cost ||
            (st.cost === old.cost && lexVec(st.vec, old.vec) < 0)
          ) {
            map.set(key, st);
          }
        };
        const lexVec = (a: number[], b: number[]): number => {
          for (let i = 0; i < a.length; i++) {
            if (a[i] !== b[i]) return a[i] - b[i];
          }
          return 0;
        };
        let layerMap = new Map<string, St>();
        for (const v of coords) {
          if (!inWin(0, v)) continue;
          merge(layerMap, { v, jit: 0, cost: devCost(0, v), vec: [v] });
        }
        for (let k = 1; k < n; k++) {
          const nextMap = new Map<string, St>();
          for (const st of layerMap.values()) {
            for (const u of coords) {
              if (!inWin(k, u)) continue;
              const dt = u - st.v;
              if (dt < L[k - 1] || dt > U[k - 1]) continue;
              const jit = st.jit + Math.abs(dt - N[k - 1]);
              if (jit > budget) continue;
              merge(nextMap, {
                v: u,
                jit,
                cost: st.cost + devCost(k, u),
                vec: [...st.vec, u],
              });
            }
          }
          layerMap = nextMap;
        }
        const layer = [...layerMap.values()];
        if (layer.length === 0) {
          const got = optimalTimes(
            asPackets, order, feasibleWins, gaps, minInterval, maxInterval,
            { nominal, total: budget }, 'deviation',
          );
          expect(Number.isFinite(got.deviation2)).toBe(false);
          const gotFeas = optimalTimes(
            asPackets, order, feasibleWins, gaps, minInterval, maxInterval,
            { nominal, total: budget }, 'feasibility',
          );
          expect(Number.isFinite(gotFeas.jitter)).toBe(false);
          continue;
        }
        let refCost = Infinity;
        let refJit = Infinity;
        for (const st of layer) {
          if (st.cost < refCost || (st.cost === refCost && st.jit < refJit)) {
            refCost = st.cost;
            refJit = st.jit;
          }
        }
        const refVec = layer
          .filter((st) => st.cost === refCost && st.jit === refJit)
          .map((st) => st.vec)
          .reduce((a, b) => (lexVec(a, b) <= 0 ? a : b));

        const got = optimalTimes(
          asPackets, order, feasibleWins, gaps, minInterval, maxInterval,
          { nominal, total: budget }, 'deviation',
        );
        expect(got.deviation2).toBe(refCost);
        expect(got.jitter).toBe(refJit);
        expect(got.times).toEqual(refVec);

        // Feasibility mode ignores deviation: it must return the minimum
        // attainable jitter and the lex-smallest vector attaining it.
        const feasJitter = Math.min(...layer.map((st) => st.jit));
        const feasVec = layer
          .filter((st) => st.jit === feasJitter)
          .map((st) => st.vec)
          .reduce((a, b) => (lexVec(a, b) <= 0 ? a : b));
        const gotFeas = optimalTimes(
          asPackets, order, feasibleWins, gaps, minInterval, maxInterval,
          { nominal, total: budget }, 'feasibility',
        );
        expect(gotFeas.jitter).toBe(feasJitter);
        expect(gotFeas.times).toEqual(feasVec);
      }
    }
  });
});
