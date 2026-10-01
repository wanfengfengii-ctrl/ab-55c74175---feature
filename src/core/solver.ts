import type {
  AdjacencyEvidence,
  AssignedPacket,
  JitterConfig,
  MissingSegment,
  PacketInput,
  ConstraintFailureEvidence,
  SolveResult,
} from './types.js';
import { SolveError } from './types.js';

interface Packet {
  index: number;
  id: string | number;
  remainder: number;
  lo: number;
  hi: number;
  /** Twice the interval midpoint (lo + hi); deviation uses |2t - mid2|. */
  mid2: number;
  /** Smallest absolute count >= countLower congruent to remainder. */
  baseCount: number;
  /** Largest absolute count <= countUpper congruent to remainder. */
  topCount: number;
  /** Rank among packets with identical (remainder, lo, hi), by id. */
  symRank: number;
  /** Identifier of the identical-packet symmetry group. */
  symGroup: number;
}

/** Intrinsic feasibility of an adjacency i -> j: congruent d in [dLo, dHi]. */
interface PairFeas {
  /** Required residue d ≡ delta (mod modulus); 0 means a multiple. */
  delta: number;
  dLo: number;
  dHi: number;
}

interface DeadState {
  depth: number;
  placed: number[];
  gaps: number[];
  last: number;
  S: number;
  c0lo: number;
  c0hi: number;
  tLo: number;
  tHi: number;
  /** Forced lower bound on prefix jitter at the dead state (budget enabled). */
  jitterLB: number;
}

function modNonNeg(a: number, m: number): number {
  return ((a % m) + m) % m;
}

/** Smallest value >= bound congruent to residue mod m. */
function ceilResidue(bound: number, residue: number, m: number): number {
  return bound + modNonNeg(residue - bound, m);
}

/** Largest value <= bound congruent to residue mod m. */
function floorResidue(bound: number, residue: number, m: number): number {
  return bound - modNonNeg(bound - residue, m);
}

/**
 * Canonical id comparison for the tertiary tie-break:
 * numbers by numeric value, then strings by UTF-16 code unit order.
 */
function compareId(a: string | number, b: string | number): number {
  if (typeof a === 'number' && typeof b === 'number') return a < b ? -1 : a > b ? 1 : 0;
  if (typeof a === 'number') return -1;
  if (typeof b === 'number') return 1;
  return a < b ? -1 : a > b ? 1 : 0;
}

/** Minimum |2t - mid2| for an integer t inside [lo, hi]. */
function minDeviation2(lo: number, hi: number, mid2: number): number {
  const low = Math.floor(mid2 / 2);
  const high = mid2 % 2 === 0 ? low : low + 1;
  const t = high < lo ? lo : low > hi ? hi : Math.max(low, lo);
  return Math.abs(2 * t - mid2);
}

/**
 * Tighten timestamp windows for a fixed complete order and gap sequence.
 * Forward pass intersects [t_prev + L, t_prev + U]; backward pass intersects
 * [t_next - U, t_next - L]. Nonempty forward windows already imply global
 * feasibility of the difference-constraint chain; the backward pass only
 * shrinks domains for the deviation optimizer. Null = defensively infeasible.
 */
function tightenWindows(
  packets: Packet[],
  order: number[],
  gaps: number[],
  minInterval: number,
  maxInterval: number,
): { lo: number; hi: number }[] | null {
  const n = order.length;
  const win = new Array<{ lo: number; hi: number }>(n);
  win[0] = { lo: packets[order[0]].lo, hi: packets[order[0]].hi };
  for (let k = 1; k < n; k++) {
    const p = packets[order[k]];
    const lo = Math.max(p.lo, win[k - 1].lo + gaps[k - 1] * minInterval);
    const hi = Math.min(p.hi, win[k - 1].hi + gaps[k - 1] * maxInterval);
    if (lo > hi) return null;
    win[k] = { lo, hi };
  }
  for (let k = n - 2; k >= 0; k--) {
    const lo = Math.max(win[k].lo, win[k + 1].lo - gaps[k] * maxInterval);
    const hi = Math.min(win[k].hi, win[k + 1].hi - gaps[k] * minInterval);
    if (lo > hi) return null;
    win[k] = { lo, hi };
  }
  return win;
}

/**
 * Minimize Σ |2 t_k - mid2_k| over integer timestamps subject to
 * t_k ∈ window_k and L_e ≤ t_{e+1} - t_e ≤ U_e, for a FIXED order.
 *
 * Exported for direct differential testing against a full-domain DP.
 *
 * Difference-constraint L1 program on a path. At an integral optimum every
 * variable is pinned — directly or through a chain of tight lower/upper edge
 * constraints — to a pivot: an interval bound or one of the two integers
 * adjacent to its midpoint. Propagating every pivot along every lower/upper
 * pin chain gives O(n · 2^n) candidate values per position (n ≤ 14). A
 * backward shortest-path DP with monotone sliding-window minima computes the
 * optimum; the forward greedy reconstruction returns the lexicographically
 * smallest optimal timestamp vector.
 */
export function optimalTimes(
  packets: Packet[],
  order: number[],
  windows: { lo: number; hi: number }[],
  gaps: number[],
  minInterval: number,
  maxInterval: number,
): { times: number[]; deviation2: number } {
  const n = order.length;
  const L = gaps.map((d) => d * minInterval);
  const U = gaps.map((d) => d * maxInterval);

  const candSets: Set<number>[] = windows.map(() => new Set<number>());
  const add = (k: number, v: number): void => {
    if (Number.isSafeInteger(v) && v >= windows[k].lo && v <= windows[k].hi) {
      candSets[k].add(v);
    }
  };

  const pivotsOf = (k: number): number[] => {
    const p = packets[order[k]];
    const f = Math.floor(p.mid2 / 2);
    const c = p.mid2 % 2 === 0 ? f : f + 1;
    return [p.lo, p.hi, f, c];
  };

  // Forward tight-edge chains.
  for (let j = 0; j < n; j++) {
    for (const s of pivotsOf(j)) add(j, s);
    if (j === n - 1) continue;
    const visit = (pos: number, v: number): void => {
      add(pos, v);
      if (pos < n - 1) {
        visit(pos + 1, v + L[pos]);
        visit(pos + 1, v + U[pos]);
      }
    };
    for (const s of pivotsOf(j)) {
      visit(j + 1, s + L[j]);
      visit(j + 1, s + U[j]);
    }
  }
  // Backward tight-edge chains.
  for (let j = 1; j < n; j++) {
    const visit = (pos: number, v: number): void => {
      add(pos, v);
      if (pos > 0) {
        visit(pos - 1, v - L[pos - 1]);
        visit(pos - 1, v - U[pos - 1]);
      }
    };
    for (const s of pivotsOf(j)) {
      visit(j - 1, s - L[j - 1]);
      visit(j - 1, s - U[j - 1]);
    }
  }

  const candidates = candSets.map((set) => [...set].sort((a, b) => a - b));
  const dev = (k: number, t: number): number => Math.abs(2 * t - packets[order[k]].mid2);

  // suffix[k][c] = minimal cost on positions k..n-1 with t_k = cand[k][c].
  const suffix: number[][] = new Array(n);
  suffix[n - 1] = candidates[n - 1].map((t) => dev(n - 1, t));
  for (let k = n - 2; k >= 0; k--) {
    const prev = suffix[k + 1];
    const nextCands = candidates[k + 1];
    const cur: number[] = new Array(candidates[k].length);
    // Feasible t_{k+1} for t is [t + L_e, t + U_e]; monotone deque minimum.
    const deque: number[] = [];
    let head = 0;
    let pushed = -1;
    for (let c = 0; c < candidates[k].length; c++) {
      const t = candidates[k][c];
      const low = t + L[k];
      const high = t + U[k];
      while (head < deque.length && nextCands[deque[head]] < low) head++;
      while (pushed + 1 < nextCands.length && nextCands[pushed + 1] <= high) {
        pushed++;
        while (deque.length > head && prev[deque[deque.length - 1]] >= prev[pushed]) deque.pop();
        deque.push(pushed);
      }
      const best = head < deque.length ? prev[deque[head]] : Infinity;
      cur[c] = best + dev(k, t);
    }
    suffix[k] = cur;
  }

  const globalBest = Math.min(...suffix[0]);
  // Lexicographically smallest optimal vector: smallest t keeping the
  // remaining optimum attainable at every position.
  const times = new Array<number>(n);
  let target = globalBest;
  let prevTime = Number.NaN;
  for (let k = 0; k < n; k++) {
    let chosen = -1;
    for (let c = 0; c < candidates[k].length; c++) {
      const t = candidates[k][c];
      if (k > 0 && (t - prevTime < L[k - 1] || t - prevTime > U[k - 1])) continue;
      if (suffix[k][c] !== target) continue;
      chosen = c;
      break;
    }
    times[k] = candidates[k][chosen];
    target -= dev(k, times[k]);
    prevTime = times[k];
  }
  return { times, deviation2: globalBest };
}

// ---------------------------------------------------------------------------
// Cumulative-jitter machinery (optional nominalInterval/totalJitterBudget).
//
// For a FIXED order and gap sequence the per-edge jitter is
// |(t_{e+1} - t_e) - d_e * nominal| and their sum must not exceed the budget.
// Two exact time solvers are used:
//   - minChainJitter:       pure minimum total jitter (budget feasibility and
//                           phase-A pruning), evaluated with two monotone
//                           sliding-window minima per layer;
//   - optimalTimesBudgeted: lexicographically smallest timestamp vector that
//                           minimizes midpoint deviation subject to the jitter
//                           budget, via Pareto labels.
//
// Both rely on a finite candidate set per node containing an optimum of each
// L1 program: at an integral optimum every variable is pinned through a chain
// of tight edge constraints, which here may be the sampling lower gap L_e,
// the upper gap U_e, or the zero-jitter gap N_e = d_e * nominal. Pivots are
// the interval bounds and the two integers adjacent to each midpoint; the
// closure propagates them along every tight chain until a fixed point.
// ---------------------------------------------------------------------------

function jitterCandidates(
  packets: Packet[],
  order: number[],
  windows: { lo: number; hi: number }[],
  gaps: number[],
  minInterval: number,
  maxInterval: number,
  nominal: number,
): number[][] {
  const n = order.length;
  const L = gaps.map((d) => d * minInterval);
  const U = gaps.map((d) => d * maxInterval);
  const N = gaps.map((d) => d * nominal);
  const sets: Set<number>[] = windows.map(() => new Set<number>());
  const add = (k: number, v: number): boolean => {
    if (Number.isSafeInteger(v) && v >= windows[k].lo && v <= windows[k].hi && !sets[k].has(v)) {
      sets[k].add(v);
      return true;
    }
    return false;
  };
  for (let k = 0; k < n; k++) {
    const p = packets[order[k]];
    add(k, windows[k].lo);
    add(k, windows[k].hi);
    const f = Math.floor(p.mid2 / 2);
    add(k, f);
    add(k, p.mid2 % 2 === 0 ? f : f + 1);
  }
  // Propagate tight lower / nominal / upper pins both directions. Each pass
  // at least advances the reach by one edge; chains have at most n-1 edges.
  for (let sweep = 0; sweep < n; sweep++) {
    let changed = false;
    for (let k = 0; k < n - 1; k++) {
      for (const v of [...sets[k]]) {
        if (add(k + 1, v + L[k])) changed = true;
        if (add(k + 1, v + N[k])) changed = true;
        if (add(k + 1, v + U[k])) changed = true;
      }
      for (const v of [...sets[k + 1]]) {
        if (add(k, v - L[k])) changed = true;
        if (add(k, v - N[k])) changed = true;
        if (add(k, v - U[k])) changed = true;
      }
    }
    if (!changed) break;
  }
  return sets.map((set) => [...set].sort((a, b) => a - b));
}

/**
 * Minimum total jitter tables for a fixed chain.
 *
 * suffix[k][c] = minimum sum of edge jitters on edges k..n-2 given
 * t_k = candidates[k][c]. Recurrence with center c0 = t + N_k:
 *   f_k(t) = min(  min_{x in [t+L_k, c0]} f_{k+1}(x) + (c0 - x),
 *                  min_{x in [c0, t+U_k]} f_{k+1}(x) + (x - c0) )
 * Each branch is a sliding-window minimum with a monotone deque:
 *   left  (x <= c0): key f_{k+1}(x) - x, window [t + L_k, c0]
 *   right (x >= c0): key f_{k+1}(x) + x, window [c0, t + U_k].
 */
function jitterSuffixTables(
  candidates: number[][],
  gaps: number[],
  minInterval: number,
  maxInterval: number,
  nominal: number,
): number[][] {
  const n = candidates.length;
  const L = gaps.map((d) => d * minInterval);
  const U = gaps.map((d) => d * maxInterval);
  const N = gaps.map((d) => d * nominal);

  const suffix: number[][] = new Array(n);
  suffix[n - 1] = new Array(candidates[n - 1].length).fill(0);
  for (let k = n - 2; k >= 0; k--) {
    const candsNext = candidates[k + 1];
    const prev = suffix[k + 1];
    const gMinus = candsNext.map((x, ix) => (Number.isFinite(prev[ix]) ? prev[ix] - x : Infinity));
    const gPlus = candsNext.map((x, ix) => (Number.isFinite(prev[ix]) ? prev[ix] + x : Infinity));

    // Both deques advance monotonically while t (hence window bounds) rises.
    const dequeLo: number[] = [];
    const dequeHi: number[] = [];
    let headLo = 0;
    let headHi = 0;
    let pushedLo = -1;
    let pushedHi = -1;
    const cur = new Array(candidates[k].length).fill(Infinity);

    for (let c = 0; c < candidates[k].length; c++) {
      const t = candidates[k][c];
      const center = t + N[k];
      const low = t + L[k];
      const high = t + U[k];

      // Left branch window x in [low, min(center, high)].
      const leftHigh = Math.min(center, high);
      while (pushedLo + 1 < candsNext.length && candsNext[pushedLo + 1] <= leftHigh) {
        pushedLo++;
        if (!Number.isFinite(gMinus[pushedLo])) continue;
        while (dequeLo.length > headLo && gMinus[dequeLo[dequeLo.length - 1]] >= gMinus[pushedLo]) {
          dequeLo.pop();
        }
        dequeLo.push(pushedLo);
      }
      while (headLo < dequeLo.length && candsNext[dequeLo[headLo]] < low) headLo++;

      // Right branch window x in [max(low, center), high].
      while (pushedHi + 1 < candsNext.length && candsNext[pushedHi + 1] <= high) {
        pushedHi++;
        if (!Number.isFinite(gPlus[pushedHi])) continue;
        while (dequeHi.length > headHi && gPlus[dequeHi[dequeHi.length - 1]] >= gPlus[pushedHi]) {
          dequeHi.pop();
        }
        dequeHi.push(pushedHi);
      }
      const rightLow = Math.max(low, center);
      while (headHi < dequeHi.length && candsNext[dequeHi[headHi]] < rightLow) headHi++;

      let best = Infinity;
      if (headLo < dequeLo.length) best = Math.min(best, gMinus[dequeLo[headLo]] + center);
      if (headHi < dequeHi.length) best = Math.min(best, gPlus[dequeHi[headHi]] - center);
      cur[c] = best;
    }
    suffix[k] = cur;
  }
  return suffix;
}

/** Minimum total jitter of a fixed chain; Infinity if no finite assignment. */
function minChainJitter(
  candidates: number[][],
  gaps: number[],
  minInterval: number,
  maxInterval: number,
  nominal: number,
): number {
  const suffix = jitterSuffixTables(candidates, gaps, minInterval, maxInterval, nominal);
  const v = Math.min(...suffix[0]);
  return Number.isFinite(v) ? v : Infinity;
}

/**
 * Lower bound on the jitter of one edge implied by tightened endpoint
 * windows: distance of the feasible closed time-gap range
 * [loNext - hiPrev, hiNext - loPrev] intersected with [d*min, d*max] to the
 * zero-jitter gap d * nominal. Infinity when no gap survives.
 */
function edgeJitterLowerBound(
  d: number,
  prevLo: number,
  prevHi: number,
  nextLo: number,
  nextHi: number,
  minInterval: number,
  maxInterval: number,
  nominal: number,
): number {
  const nGap = d * nominal;
  const a = Math.max(nextLo - prevHi, d * minInterval);
  const b = Math.min(nextHi - prevLo, d * maxInterval);
  if (a > b) return Infinity;
  return nGap < a ? a - nGap : nGap > b ? nGap - b : 0;
}

interface BudgetedSolution {
  times: number[];
  /** Total midpoint deviation in doubled units. */
  deviation2: number;
  /** Total jitter of the returned timestamp vector. */
  jitter: number;
}

/** Pareto label: suffix cumulative jitter -> suffix minimum deviation2. */
interface Label {
  j: number;
  d: number;
}

/** Keep strictly improving points sorted by ascending jitter. */
function mergeLabels(labels: Label[]): Label[] {
  if (labels.length === 0) return [];
  labels.sort((a, b) => a.j - b.j || a.d - b.d);
  const out: Label[] = [];
  for (const lab of labels) {
    if (out.length > 0 && out[out.length - 1].j === lab.j) continue; // smallest d first
    if (out.length > 0 && out[out.length - 1].d <= lab.d) continue; // dominated
    out.push(lab);
  }
  return out;
}

/**
 * Lexicographically smallest integer timestamp vector minimizing the total
 * midpoint deviation subject to total jitter <= budget for a FIXED chain.
 * Returns null when no timestamp vector meets the budget.
 *
 * Per (position, candidate timestamp) we keep the nondominated Pareto
 * frontier of (suffix jitter, suffix deviation2); joining adjacent positions
 * shifts suffix jitter by the chosen edge jitter. Reconstruction greedily
 * fixes the smallest timestamp that still admits a deviation-optimal,
 * budget-feasible tail.
 */
export function optimalTimesBudgeted(
  packets: Packet[],
  order: number[],
  windows: { lo: number; hi: number }[],
  gaps: number[],
  minInterval: number,
  maxInterval: number,
  nominal: number,
  budget: number,
): BudgetedSolution | null {
  const n = order.length;
  const L = gaps.map((d) => d * minInterval);
  const U = gaps.map((d) => d * maxInterval);
  const N = gaps.map((d) => d * nominal);
  const dev = (k: number, t: number): number => Math.abs(2 * t - packets[order[k]].mid2);

  // Shortcut: if the unconstrained deviation optimum already fits the budget,
  // it is also the deviation optimum under the budget (budget never helps the
  // deviation objective).
  const unconstrained = optimalTimes(packets, order, windows, gaps, minInterval, maxInterval);
  let unconstrainedJitter = 0;
  for (let k = 0; k < n - 1; k++) {
    unconstrainedJitter += Math.abs(unconstrained.times[k + 1] - unconstrained.times[k] - N[k]);
  }
  if (unconstrainedJitter <= budget) {
    return { times: unconstrained.times, deviation2: unconstrained.deviation2, jitter: unconstrainedJitter };
  }

  // The budget binds. Transform to (t_0, δ) with t_k = t_0 + baseN_k + δ_k,
  // δ_0 = 0. Then edge jitter is |δ_{e+1} - δ_e| (total variation of δ) and
  // every budget-feasible vector satisfies |δ_k| ≤ budget. At a constrained
  // optimum t_0 sits at a window/midpoint kink t_0 = pivot_j - baseN_j - δ_j
  // for some node j with |δ_j| ≤ budget, hence each optimal timestamp
  //   t_k = t_0 + baseN_k + δ_k
  // lies within 2·budget of pivot_j - baseN_j + baseN_k for some pivot_j
  // (interval bound or midpoint-adjacent integer of node j). The union of
  // those O(n²) integer intervals is a finite spanning set independent of the
  // timestamp-window width; the Pareto DP below enforces the true edge L/U
  // bounds and the exact cumulative-jitter ≤ budget constraint.
  const baseN = new Array<number>(n).fill(0);
  for (let k = 1; k < n; k++) baseN[k] = baseN[k - 1] + N[k - 1];

  const structuralSets = jitterCandidates(packets, order, windows, gaps, minInterval, maxInterval, nominal);
  const sets: Set<number>[] = structuralSets.map((s) => new Set(s));
  const addRange = (i: number, lo: number, hi: number): void => {
    const a = Math.max(lo, windows[i].lo);
    const b = Math.min(hi, windows[i].hi);
    for (let v = a; v <= b; v++) {
      if (Number.isSafeInteger(v)) sets[i].add(v);
    }
  };

  for (let j = 0; j < n; j++) {
    const p = packets[order[j]];
    const f = Math.floor(p.mid2 / 2);
    const c = p.mid2 % 2 === 0 ? f : f + 1;
    for (const pivot of [windows[j].lo, windows[j].hi, f, c]) {
      const projected = pivot - baseN[j];
      for (let k = 0; k < n; k++) {
        addRange(k, projected + baseN[k] - 2 * budget, projected + baseN[k] + 2 * budget);
      }
    }
  }
  const candidates = sets.map((set) => [...set].sort((a, b) => a - b));
  // front[k][c] = Pareto labels for positions k..n-1 given t_k = cand, using
  // jitter on edges k..n-2 and deviation on nodes k..n-1.
  const front: Label[][][] = new Array(n);
  front[n - 1] = candidates[n - 1].map((t) => [{ j: 0, d: dev(n - 1, t) }]);
  for (let k = n - 2; k >= 0; k--) {
    front[k] = candidates[k].map((t) => {
      const gathered: Label[] = [];
      for (let ix = 0; ix < candidates[k + 1].length; ix++) {
        const x = candidates[k + 1][ix];
        const dt = x - t;
        if (dt < L[k] || dt > U[k]) continue;
        const edge = Math.abs(dt - N[k]);
        for (const lab of front[k + 1][ix]) {
          const j = lab.j + edge;
          if (j <= budget) gathered.push({ j, d: lab.d });
        }
      }
      const dk = dev(k, t);
      return mergeLabels(gathered).map((lab) => ({ j: lab.j, d: lab.d + dk }));
    });
  }

  let Dstar = Infinity;
  for (const labels of front[0]) {
    for (const lab of labels) if (lab.d < Dstar) Dstar = lab.d;
  }
  if (!Number.isFinite(Dstar)) return null;

  const times = new Array<number>(n);
  let remDev = Dstar;
  let remBudget = budget;
  let prevTime = Number.NaN;
  for (let k = 0; k < n; k++) {
    let chosenT = Number.NaN;
    let chosenLabel: Label | null = null;
    for (let c = 0; c < candidates[k].length; c++) {
      const t = candidates[k][c];
      let edge = 0;
      if (k > 0) {
        const dt = t - prevTime;
        if (dt < L[k - 1] || dt > U[k - 1]) continue;
        edge = Math.abs(dt - N[k - 1]);
      }
      for (const lab of front[k][c]) {
        if (lab.d !== remDev) continue;
        if (lab.j <= remBudget - edge) {
          chosenT = t;
          chosenLabel = lab;
          break;
        }
      }
      if (chosenLabel) break;
    }
    if (chosenLabel === null) return null; // defensive
    times[k] = chosenT;
    if (k > 0) remBudget -= Math.abs(chosenT - prevTime - N[k - 1]);
    remDev -= dev(k, chosenT);
    prevTime = chosenT;
  }

  let jitter = 0;
  for (let k = 0; k < n - 1; k++) jitter += Math.abs(times[k + 1] - times[k] - N[k]);
  return { times, deviation2: Dstar, jitter };
}

interface Move {
  j: number;
  d: number;
  c0lo: number;
  c0hi: number;
  tLo: number;
  tHi: number;
}

/**
 * Jointly recover transmission order, wrap-crossing absolute counters and
 * transmit timestamps.
 *
 * Optimization is lexicographic:
 *   1. missing packet count between first/last observed packet
 *   2. total deviation of chosen times from interval midpoints
 *   3. the recovered packet-id sequence (lexicographic)
 *
 * Implemented as three exhaustive branch-and-bound phases over the same
 * state space:
 *   A — minimum total counter gap (primary value),
 *   B — minimum midpoint deviation subject to primary = optimum,
 *   C — lexicographically smallest id sequence subject to both (greedy
 *       position fixing with a memoized feasibility oracle).
 * Packets sharing (remainder, time interval) are exact symmetry twins: they
 * may only be consumed in ascending id order, which never removes the
 * lex-min solution but collapses permutation families.
 *
 * Throws SolveError(NO_CONSISTENT_INTERPRETATION) with first-failure evidence.
 */
export function solve(
  inputs: PacketInput[],
  modulus: number,
  countLower: number,
  countUpper: number,
  minInterval: number,
  maxInterval: number,
  jitter?: JitterConfig,
): SolveResult {
  const n = inputs.length;
  const W = countUpper - countLower;
  const jc = jitter;
  const nominal = jc?.nominalInterval;
  const jitterBudget = jc?.totalJitterBudget ?? Infinity;

  // Group identical (remainder, lo, hi) packets for symmetry breaking.
  const groups = new Map<string, number[]>();
  for (let i = 0; i < n; i++) {
    const p = inputs[i];
    const key = `${p.remainder}|${p.timeLower}|${p.timeUpper}`;
    const g = groups.get(key);
    if (g) g.push(i);
    else groups.set(key, [i]);
  }
  const symRank = new Array<number>(n);
  const symGroup = new Array<number>(n);
  let groupId = 0;
  for (const members of groups.values()) {
    members.sort((a, b) => compareId(inputs[a].id, inputs[b].id));
    members.forEach((ix, rank) => {
      symRank[ix] = rank;
      symGroup[ix] = groupId;
    });
    groupId++;
  }

  const packets: Packet[] = inputs.map((p, index) => ({
    index,
    id: p.id,
    remainder: p.remainder,
    lo: p.timeLower,
    hi: p.timeUpper,
    mid2: p.timeLower + p.timeUpper,
    baseCount: ceilResidue(countLower, p.remainder, modulus),
    topCount: floorResidue(countUpper, p.remainder, modulus),
    symRank: symRank[index],
    symGroup: symGroup[index],
  }));

  // Intrinsic adjacency feasibility = congruent-gap RANGE per ordered pair.
  // Time: L_d ≤ t_j - t_i ≤ U_d with t_i∈I_i, t_j∈I_j:
  //   d ≥ ceil((lo_j - hi_i)/maxInterval), d ≤ floor((hi_j - lo_i)/minInterval).
  // Counts: c_i, c_j = c_i + d both inside the search window:
  //   base_j - top_i ≤ d ≤ top_j - base_i.
  const pair: PairFeas[][] = packets.map((pi) =>
    packets.map((pj): PairFeas => {
      const delta = modNonNeg(pj.remainder - pi.remainder, modulus);
      const d0 = pi.index === pj.index ? Infinity : delta === 0 ? modulus : delta;
      const dLo = Math.max(
        d0,
        Math.ceil((pj.lo - pi.hi) / maxInterval),
        pj.baseCount - pi.topCount,
      );
      const dHi = Math.min(
        W,
        Math.floor((pj.hi - pi.lo) / minInterval),
        pj.topCount - pi.baseCount,
      );
      return { delta, dLo, dHi };
    }),
  );

  const minGap: number[][] = packets.map((pi) =>
    packets.map((pj) => {
      if (pi.index === pj.index) return Infinity;
      const pf = pair[pi.index][pj.index];
      return pf.dLo <= pf.dHi ? pf.dLo : Infinity;
    }),
  );

  // cont[mask][j] = minimum gap sum of a path starting at j visiting all
  // nodes of `mask` (j ∉ mask). Exact admissible completion bound, O(2^n n²).
  const full = (1 << n) - 1;
  const cont: number[][] = Array.from({ length: 1 << n }, () => new Array<number>(n).fill(Infinity));
  for (let j = 0; j < n; j++) cont[0][j] = 0;
  for (let mask = 1; mask <= full; mask++) {
    for (let j = 0; j < n; j++) {
      if (mask & (1 << j)) continue;
      let best = Infinity;
      for (let x = 0; x < n; x++) {
        if (!(mask & (1 << x))) continue;
        const v = minGap[j][x] + cont[mask ^ (1 << x)][x];
        if (v < best) best = v;
      }
      cont[mask][j] = best;
    }
  }

  const seedOrder = packets
    .filter((p) => p.baseCount <= Math.min(p.topCount, countUpper - n + 1))
    .sort((a, b) => compareId(a.id, b.id));
  if (seedOrder.length === 0) {
    throw new SolveError(
      'NO_CONSISTENT_INTERPRETATION',
      'no globally consistent interpretation exists within the search window',
      {
        stage: 'seed',
        partialLength: 0,
        partialOrder: [],
        candidateId: packets[0].id,
        reason:
          `no packet can be seeded inside [${countLower}, ${countUpper}] while leaving ` +
          `room for ${n - 1} further strictly increasing absolute counters`,
      },
    );
  }
  const globalPrimaryLB = Math.min(
    ...seedOrder.map((p) => cont[full ^ (1 << p.index)][p.index]),
  );

  // Per-position arrays shared by the recursive searches.
  const orderArr = new Array<number>(n);
  const tLoArr = new Array<number>(n);
  const tHiArr = new Array<number>(n);
  const gapsArr = new Array<number>(n - 1);
  const used = new Uint8Array(n);
  let bestDead: DeadState | null = null;

  const usedMask = (): number => {
    let bits = 0;
    for (let i = 0; i < n; i++) if (used[i]) bits |= 1 << i;
    return bits;
  };

  /**
   * Exact prefix signature: full packet sequence, fixed gaps and the
   * forward-tightened per-position windows. When the jitter budget is enabled
   * two paths sharing only the search frontier are not interchangeable
   * (prefix jitter consumption differs), so the phase-A/oracle memos must key
   * on the complete prefix instead of just (mask, last, frontier).
   */
  const prefixKey = (
    tag: string,
    depth: number,
    last: number,
    S: number,
    c0lo: number,
    c0hi: number,
    tLo: number,
    tHi: number,
  ): string => {
    let s = `${tag}|${depth}|${last}|${S}|${c0lo}|${c0hi}|${tLo}|${tHi}`;
    for (let k = 0; k < depth; k++) {
      s += `>${orderArr[k]}:${k > 0 ? gapsArr[k - 1] : 0}:${tLoArr[k]},${tHiArr[k]}`;
    }
    return s;
  };

  /**
   * Lower bound on the jitter of one edge implied by tightened endpoint
   * windows; thin closure over the request's interval parameters.
   */
  const edgeJitterLB = (
    d: number,
    prevLo: number,
    prevHi: number,
    nextLo: number,
    nextHi: number,
  ): number =>
    edgeJitterLowerBound(d, prevLo, prevHi, nextLo, nextHi, minInterval, maxInterval, nominal!);

  /** Symmetry leader: within a twin group only the smallest-ranked still
   * unused member may be picked next. Relabeling identical twins never
   * changes the objectives, and the lex-min order always consumes them in
   * ascending id rank. */
  const isSymmetryAllowed = (j: number, mask: number): boolean => {
    const pj = packets[j];
    if (pj.symRank === 0) return true;
    for (let i = 0; i < n; i++) {
      const pi = packets[i];
      if (pi.symGroup === pj.symGroup && pi.symRank < pj.symRank && !(mask & (1 << i))) {
        return false;
      }
    }
    return true;
  };

  /** Successors in canonical order: every congruent feasible gap per target,
   * sorted by smallest gap then smallest target id. When the jitter budget is
   * enabled, gaps whose edge alone forces the cumulative lower bound past the
   * budget are dropped; pass jlbLB = Infinity to enumerate without the filter
   * (used for first-blocker evidence). */
  const enumerateMoves = (
    depth: number,
    last: number,
    S: number,
    c0lo: number,
    c0hi: number,
    tLo: number,
    tHi: number,
    mask: number,
    jlb = Infinity,
  ): Move[] => {
    const slotsAfter = n - 1 - depth;
    const moves: Move[] = [];
    for (let j = 0; j < n; j++) {
      if (mask & (1 << j)) continue;
      if (!isSymmetryAllowed(j, mask)) continue;
      const pj = packets[j];
      const pf = pair[last][j];
      const dLo = Math.max(
        pf.dLo,
        pj.baseCount - S - c0hi,
        Math.ceil((pj.lo - tHi) / maxInterval),
      );
      const dHi0 = Math.min(
        pf.dHi,
        pj.topCount - S - c0lo,
        countUpper - slotsAfter - S - c0lo,
        Math.floor((pj.hi - tLo) / minInterval),
      );
      if (dLo > dHi0) continue;
      const dMin = ceilResidue(dLo, pf.delta, modulus);
      if (dMin > dHi0) continue;

      const consider = (d: number): Move | null => {
        const njLo = Math.max(c0lo, pj.baseCount - S - d);
        const njHi = Math.min(c0hi, pj.topCount - S - d, countUpper - slotsAfter - S - d);
        const ntLo = Math.max(pj.lo, tLo + d * minInterval);
        const ntHi = Math.min(pj.hi, tHi + d * maxInterval);
        if (njLo > njHi || ntLo > ntHi) return null;
        if (jc !== undefined) {
          const forced = edgeJitterLB(d, tLo, tHi, ntLo, ntHi);
          if (jlb + forced > jitterBudget) return null;
        }
        return { j, d, c0lo: njLo, c0hi: njHi, tLo: ntLo, tHi: ntHi };
      };

      for (let d = dMin; d <= dHi0; d += modulus) {
        // The rising time lower bound is monotone in d; once it passes j's
        // interval no larger gap can work.
        if (d * minInterval > pj.hi - tLo) break;
        const mv = consider(d);
        if (mv) moves.push(mv);
      }
    }
    moves.sort((a, b) => a.d - b.d || compareId(packets[a.j].id, packets[b.j].id));
    return moves;
  };

  const recordDead = (
    depth: number,
    last: number,
    S: number,
    c0lo: number,
    c0hi: number,
    tLo: number,
    tHi: number,
    jlb = Infinity,
  ): void => {
    if (bestDead === null || depth > bestDead.depth) {
      bestDead = {
        depth,
        placed: orderArr.slice(0, depth),
        gaps: gapsArr.slice(0, Math.max(0, depth - 1)),
        last,
        S,
        c0lo,
        c0hi,
        tLo,
        tHi,
        jitterLB: jc !== undefined ? jlb : Infinity,
      };
    }
  };

  /** Exact fixed-chain budget test for the complete order in the shared
   * arrays (orderArr[0..depth-1], gapsArr[0..depth-2]). Returns the minimum
   * total jitter, or Infinity when the chain is infeasible / over budget. */
  const chainMinJitter = (depth: number): number => {
    if (jc === undefined) return 0;
    const order = orderArr.slice(0, depth);
    const gaps = gapsArr.slice(0, depth - 1);
    const windows = tightenWindows(packets, order, gaps, minInterval, maxInterval);
    if (windows === null) return Infinity;
    const cands = jitterCandidates(
      packets,
      order,
      windows,
      gaps,
      minInterval,
      maxInterval,
      jc.nominalInterval,
    );
    const v = minChainJitter(cands, gaps, minInterval, maxInterval, jc.nominalInterval);
    return v <= jc.totalJitterBudget ? v : Infinity;
  };

  // ------------------------------------------------------------------ Phase A
  // Minimum total counter gap. A state memo caches the best completion gap
  // sum (Infinity = dead); state = (used set, last packet, fixed prefix gap
  // sum, tightened c0 and last-timestamp windows).
  const memoA = new Map<string, number>();
  let bestA = Infinity;
  let stopA = false;

  const dfsA = (depth: number, last: number, S: number, c0lo: number, c0hi: number, tLo: number, tHi: number, jlb = 0): number => {
    if (stopA) return Infinity;
    if (depth === n) {
      const jm = chainMinJitter(depth);
      if (!Number.isFinite(jm)) {
        recordDead(depth, last, S, c0lo, c0hi, tLo, tHi, jlb);
        return Infinity;
      }
      if (S < bestA) bestA = S;
      if (bestA === globalPrimaryLB) stopA = true;
      return S;
    }
    const mask = usedMask();
    const remaining = full ^ mask;
    // Bound prune only once a feasible solution exists: before that, an
    // infinite intrinsic completion must still be explored to record the
    // deepest non-extendable state for failure evidence.
    if (Number.isFinite(bestA) && S + cont[remaining][last] >= bestA) return Infinity;

    // With the jitter budget enabled, prefixes with different accumulated
    // jitter are not interchangeable; key on the complete prefix.
    const key =
      jc === undefined
        ? `A|${mask}|${last}|${S}|${c0lo}|${c0hi}|${tLo}|${tHi}`
        : prefixKey('A', depth, last, S, c0lo, c0hi, tLo, tHi);
    const cached = memoA.get(key);
    if (cached !== undefined) return cached;

    const moves = enumerateMoves(depth, last, S, c0lo, c0hi, tLo, tHi, mask, jlb);
    if (moves.length === 0) {
      recordDead(depth, last, S, c0lo, c0hi, tLo, tHi, jlb);
      memoA.set(key, Infinity);
      return Infinity;
    }

    let best = Infinity;
    for (const mv of moves) {
      if (Number.isFinite(bestA) && S + mv.d + cont[remaining ^ (1 << mv.j)][mv.j] >= bestA) break;
      used[mv.j] = 1;
      orderArr[depth] = mv.j;
      gapsArr[depth - 1] = mv.d;
      tLoArr[depth] = mv.tLo;
      tHiArr[depth] = mv.tHi;
      const childJlb =
        jc === undefined
          ? 0
          : jlb + edgeJitterLB(mv.d, tLo, tHi, mv.tLo, mv.tHi);
      const v = dfsA(depth + 1, mv.j, S + mv.d, mv.c0lo, mv.c0hi, mv.tLo, mv.tHi, childJlb);
      used[mv.j] = 0;
      if (v < best) best = v;
      if (stopA) break;
    }
    if (best === Infinity) {
      recordDead(depth, last, S, c0lo, c0hi, tLo, tHi, jlb);
    }
    memoA.set(key, best);
    return best;
  };

  for (const seed of seedOrder) {
    if (stopA) break;
    const c0hi0 = Math.min(seed.topCount, countUpper - n + 1);
    used.fill(0);
    used[seed.index] = 1;
    orderArr[0] = seed.index;
    tLoArr[0] = seed.lo;
    tHiArr[0] = seed.hi;
    dfsA(1, seed.index, 0, seed.baseCount, c0hi0, seed.lo, seed.hi);
  }
  if (bestA === Infinity) {
    throw buildFailureEvidence(packets, pair, bestDead, modulus, countUpper, minInterval, maxInterval, jc);
  }
  const Pstar = bestA;

  // ------------------------------------------------------- Phase B/C: prefix
  // The deviation and lexicographic phases (and the primary-optimal oracle
  // they share) carry the fixed partial order as an IMMUTABLE prefix of
  // nodes instead of mutating the shared search arrays. This keeps memoized
  // states self-contained — essential when the jitter budget is enabled,
  // because two frontiers reached through different prefixes consume
  // different amounts of jitter and are never interchangeable.
  interface PNode {
    j: number;
    /** Counter gap into this node (0 for the seed). */
    d: number;
    c0lo: number;
    c0hi: number;
    tLo: number;
    tHi: number;
  }

  const seedNode = (seedIndex: number): PNode => {
    const seed = packets[seedIndex];
    return {
      j: seedIndex,
      d: 0,
      c0lo: seed.baseCount,
      c0hi: Math.min(seed.topCount, countUpper - n + 1),
      tLo: seed.lo,
      tHi: seed.hi,
    };
  };

  const nodeMask = (nodes: PNode[]): number => {
    let mask = 0;
    for (const nd of nodes) mask |= 1 << nd.j;
    return mask;
  };

  const nodeKey = (tag: string, nodes: PNode[], S: number): string => {
    let s = `${tag}|${S}`;
    for (const nd of nodes) s += `>${nd.j}:${nd.d}:${nd.tLo},${nd.tHi}:${nd.c0lo},${nd.c0hi}`;
    return s;
  };

  /** Forced lower bound on jitter already consumed by the fixed prefix. */
  const nodesJitterLB = (nodes: PNode[]): number => {
    let sum = 0;
    for (let k = 1; k < nodes.length; k++) {
      const a = nodes[k - 1];
      const b = nodes[k];
      const v = edgeJitterLB(b.d, a.tLo, a.tHi, b.tLo, b.tHi);
      if (!Number.isFinite(v)) return Infinity;
      sum += v;
    }
    return sum;
  };

  /** Exact minimum total jitter of a complete prefix; Infinity if over budget
   * or (defensively) structurally infeasible. */
  const chainJitterOf = (nodes: PNode[]): number => {
    if (jc === undefined) return 0;
    const order = nodes.map((nd) => nd.j);
    const gaps = nodes.slice(1).map((nd) => nd.d);
    const windows = tightenWindows(packets, order, gaps, minInterval, maxInterval);
    if (windows === null) return Infinity;
    const cands = jitterCandidates(
      packets, order, windows, gaps, minInterval, maxInterval, jc.nominalInterval,
    );
    const v = minChainJitter(cands, gaps, minInterval, maxInterval, jc.nominalInterval);
    return v <= jc.totalJitterBudget ? v : Infinity;
  };

  /** Optimal midpoint deviation2 of a complete prefix under the jitter budget. */
  const deviationOf = (nodes: PNode[]): number => {
    const order = nodes.map((nd) => nd.j);
    const gaps = nodes.slice(1).map((nd) => nd.d);
    const windows = tightenWindows(packets, order, gaps, minInterval, maxInterval);
    if (windows === null) return Infinity;
    if (jc === undefined) {
      return optimalTimes(packets, order, windows, gaps, minInterval, maxInterval).deviation2;
    }
    const sol = optimalTimesBudgeted(
      packets, order, windows, gaps, minInterval, maxInterval,
      jc.nominalInterval, jc.totalJitterBudget,
    );
    return sol === null ? Infinity : sol.deviation2;
  };

  /**
   * Primary-optimal-chain oracle: true exactly when the given prefix admits a
   * completion reaching total gap Pstar (and, when enabled, the jitter
   * budget). The Held-Karp table provides a necessary exact gap bound;
   * time/count feasibility is checked by enumeration and the cumulative
   * jitter by the fixed-chain solver at the leaf.
   */
  const memoOpt = new Map<string, boolean>();
  const oracle = (nodes: PNode[], S: number): boolean => {
    const depth = nodes.length;
    if (depth === n) return S === Pstar && Number.isFinite(chainJitterOf(nodes));
    const key = jc === undefined ? nodeKey('O', nodes, S) : nodeKey('O', nodes, S);
    const cached = memoOpt.get(key);
    if (cached !== undefined) return cached;

    const last = nodes[depth - 1];
    const mask = nodeMask(nodes);
    const remaining = full ^ mask;
    const jlb = jc === undefined ? 0 : nodesJitterLB(nodes);
    const moves = enumerateMoves(
      depth, last.j, S, last.c0lo, last.c0hi, last.tLo, last.tHi, mask, jlb,
    );
    let ok = false;
    for (const mv of moves) {
      if (S + mv.d + cont[remaining ^ (1 << mv.j)][mv.j] > Pstar) continue;
      const child: PNode = {
        j: mv.j, d: mv.d, c0lo: mv.c0lo, c0hi: mv.c0hi, tLo: mv.tLo, tHi: mv.tHi,
      };
      if (oracle([...nodes, child], S + mv.d)) {
        ok = true;
        break;
      }
    }
    memoOpt.set(key, ok);
    return ok;
  };

  /** Moves from a prefix that lie on at least one primary-optimal completion. */
  const optimalMovesFrom = (nodes: PNode[], S: number): Move[] => {
    const depth = nodes.length;
    const last = nodes[depth - 1];
    const mask = nodeMask(nodes);
    const remaining = full ^ mask;
    const jlb = jc === undefined ? 0 : nodesJitterLB(nodes);
    const all = enumerateMoves(
      depth, last.j, S, last.c0lo, last.c0hi, last.tLo, last.tHi, mask, jlb,
    );
    return all.filter((mv) => {
      if (S + mv.d + cont[remaining ^ (1 << mv.j)][mv.j] > Pstar) return false;
      const child: PNode = {
        j: mv.j, d: mv.d, c0lo: mv.c0lo, c0hi: mv.c0hi, tLo: mv.tLo, tHi: mv.tHi,
      };
      return oracle([...nodes, child], S + mv.d);
    });
  };

  const seedIsOptimal = (seedIndex: number): boolean => {
    const node = seedNode(seedIndex);
    if (node.c0lo > node.c0hi) return false;
    return oracle([node], 0);
  };

  // ------------------------------------------------------------------ Phase B
  // Minimum deviation2 over primary-optimal, budget-feasible chains.
  const memoB = new Map<string, number>();
  let bestB = Infinity;

  const independentDevLB = (mask: number): number => {
    let sum = 0;
    const remaining = full ^ mask;
    for (let j = 0; j < n; j++) {
      if (remaining & (1 << j)) sum += minDeviation2(packets[j].lo, packets[j].hi, packets[j].mid2);
    }
    return sum;
  };

  const dfsB = (nodes: PNode[], S: number): number => {
    const depth = nodes.length;
    if (depth === n) {
      const v = deviationOf(nodes);
      if (v < bestB) bestB = v;
      return v;
    }
    const mask = nodeMask(nodes);

    let placedLB = 0;
    for (const nd of nodes) {
      placedLB += minDeviation2(nd.tLo, nd.tHi, packets[nd.j].mid2);
    }
    if (placedLB + independentDevLB(mask) >= bestB) return Infinity;

    const key = nodeKey('B', nodes, S);
    const cached = memoB.get(key);
    if (cached !== undefined) return cached;

    let best = Infinity;
    for (const mv of optimalMovesFrom(nodes, S)) {
      const child: PNode = {
        j: mv.j, d: mv.d, c0lo: mv.c0lo, c0hi: mv.c0hi, tLo: mv.tLo, tHi: mv.tHi,
      };
      const v = dfsB([...nodes, child], S + mv.d);
      if (v < best) best = v;
    }
    memoB.set(key, best);
    return best;
  };

  for (const seed of seedOrder) {
    if (!seedIsOptimal(seed.index)) continue;
    dfsB([seedNode(seed.index)], 0);
  }
  if (bestB === Infinity) {
    // Defensive: phase A guarantees a primary-optimal feasible leaf.
    throw new SolveError(
      'NO_CONSISTENT_INTERPRETATION',
      'internal search failure during deviation optimization',
    );
  }
  const Dstar = bestB;

  // ------------------------------------------------------------------ Phase C
  // Greedily fix the lexicographically smallest id sequence using a memoized
  // boolean oracle over immutable prefixes.
  const memoC = new Map<string, boolean>();
  const feasibleWithDstar = (nodes: PNode[], S: number): boolean => {
    if (nodes.length === n) return deviationOf(nodes) === Dstar;
    const key = nodeKey('C', nodes, S);
    const cached = memoC.get(key);
    if (cached !== undefined) return cached;
    let ok = false;
    for (const mv of optimalMovesFrom(nodes, S)) {
      const child: PNode = {
        j: mv.j, d: mv.d, c0lo: mv.c0lo, c0hi: mv.c0hi, tLo: mv.tLo, tHi: mv.tHi,
      };
      if (feasibleWithDstar([...nodes, child], S + mv.d)) {
        ok = true;
        break;
      }
    }
    memoC.set(key, ok);
    return ok;
  };

  const fixedNodes: PNode[] = [];
  let fixedS = 0;
  for (let depth = 0; depth < n; depth++) {
    let candidates: PNode[];
    if (depth === 0) {
      candidates = seedOrder.filter((p) => seedIsOptimal(p.index)).map((p) => seedNode(p.index));
    } else {
      candidates = optimalMovesFrom(fixedNodes, fixedS).map((mv) => ({
        j: mv.j, d: mv.d, c0lo: mv.c0lo, c0hi: mv.c0hi, tLo: mv.tLo, tHi: mv.tHi,
      }));
    }
    candidates.sort((a, b) => compareId(packets[a.j].id, packets[b.j].id));

    let picked: PNode | null = null;
    for (const cand of candidates) {
      const nextS = depth === 0 ? 0 : fixedS + cand.d;
      if (feasibleWithDstar([...fixedNodes, cand], nextS)) {
        picked = cand;
        break;
      }
    }
    if (picked === null) {
      // Defensive: phases A/B certify a feasible choice at every position.
      throw new SolveError('NO_CONSISTENT_INTERPRETATION', 'internal failure reconstructing lex-min order');
    }
    fixedNodes.push(picked);
    fixedS = depth === 0 ? 0 : fixedS + picked.d;
  }

  // Assemble the certified solution: smallest admissible c0, optimal times.
  const finalOrder = fixedNodes.map((nd) => nd.j);
  const finalGaps = fixedNodes.slice(1).map((nd) => nd.d);
  const finalC0 = fixedNodes[0].c0lo;
  const windows = tightenWindows(packets, finalOrder, finalGaps, minInterval, maxInterval);
  if (!windows) {
    throw new SolveError('NO_CONSISTENT_INTERPRETATION', 'internal failure tightening final windows');
  }
  let finalTimes: number[];
  let dev2: number;
  let finalJitter = 0;
  if (jc === undefined) {
    const sol = optimalTimes(packets, finalOrder, windows, finalGaps, minInterval, maxInterval);
    finalTimes = sol.times;
    dev2 = sol.deviation2;
  } else {
    const sol = optimalTimesBudgeted(
      packets,
      finalOrder,
      windows,
      finalGaps,
      minInterval,
      maxInterval,
      jc.nominalInterval,
      jc.totalJitterBudget,
    );
    if (sol === null) {
      throw new SolveError(
        'NO_CONSISTENT_INTERPRETATION',
        'internal failure: certified chain does not admit a budgeted timestamp vector',
      );
    }
    finalTimes = sol.times;
    dev2 = sol.deviation2;
    finalJitter = sol.jitter;
  }
  let gapSum = 0;
  for (const d of finalGaps) gapSum += d;

  return buildResult(
    packets,
    {
      gapSum,
      deviation2: dev2,
      times: finalTimes,
      order: finalOrder,
      c0: finalC0,
      gaps: finalGaps,
    },
    modulus,
    minInterval,
    maxInterval,
    jc,
    finalJitter,
  );
}

function buildResult(
  packets: Packet[],
  cand: { gapSum: number; deviation2: number; times: number[]; order: number[]; c0: number; gaps: number[] },
  modulus: number,
  minInterval: number,
  maxInterval: number,
  jc: JitterConfig | undefined,
  totalJitter: number,
): SolveResult {
  const n = cand.order.length;
  const order = cand.order.map((ix) => packets[ix].id);
  const assignments: AssignedPacket[] = [];
  const adjacency: AdjacencyEvidence[] = [];
  const missingSegments: MissingSegment[] = [];

  let count = cand.c0;
  let cumulative = 0;
  for (let k = 0; k < n; k++) {
    const p = packets[cand.order[k]];
    assignments.push({
      position: k,
      id: p.id,
      absoluteCount: count,
      time: cand.times[k],
      remainder: p.remainder,
      timeInterval: { lower: p.lo, upper: p.hi },
    });
    if (k > 0) {
      const d = cand.gaps[k - 1];
      const prevCount = count - d;
      if (d > 1) {
        missingSegments.push({ fromCount: prevCount + 1, toCount: count - 1, length: d - 1 });
      }
      const tGap = cand.times[k] - cand.times[k - 1];
      const prevP = packets[cand.order[k - 1]];
      let nominalGap: number | undefined;
      let jitter: number | undefined;
      if (jc !== undefined) {
        nominalGap = d * jc.nominalInterval;
        jitter = Math.abs(tGap - nominalGap);
        cumulative += jitter;
      }
      adjacency.push({
        index: k - 1,
        fromId: prevP.id,
        toId: p.id,
        fromCount: prevCount,
        toCount: count,
        countGap: d,
        fromTime: cand.times[k - 1],
        toTime: cand.times[k],
        timeGap: tGap,
        allowedTimeGap: { min: d * minInterval, max: d * maxInterval },
        ...(jc !== undefined
          ? {
              nominalTimeGap: nominalGap!,
              jitter: jitter!,
              cumulativeJitter: cumulative,
            }
          : {}),
        missingBetween: d - 1,
        congruence: { remainder: p.remainder, modulus },
        absoluteCountCongruent: modNonNeg(count, modulus) === p.remainder,
        timeWithinInterval: {
          from: { lower: prevP.lo, upper: prevP.hi },
          to: { lower: p.lo, upper: p.hi },
        },
        satisfied:
          tGap >= d * minInterval &&
          tGap <= d * maxInterval &&
          cand.times[k - 1] >= prevP.lo &&
          cand.times[k - 1] <= prevP.hi &&
          cand.times[k] >= p.lo &&
          cand.times[k] <= p.hi &&
          modNonNeg(prevCount, modulus) === prevP.remainder &&
          modNonNeg(count, modulus) === p.remainder,
      });
    }
    if (k < n - 1) count += cand.gaps[k];
  }

  return {
    order,
    assignments,
    missingSegments,
    missingCountTotal: cand.gapSum - (n - 1),
    adjacency,
    observedCountRange: { first: cand.c0, last: cand.c0 + cand.gapSum },
    ...(jc !== undefined
      ? {
          jitterBudget: {
            nominalInterval: jc.nominalInterval,
            budget: jc.totalJitterBudget,
            used: totalJitter,
            remaining: jc.totalJitterBudget - totalJitter,
            exhausted: totalJitter === jc.totalJitterBudget,
          },
        }
      : {}),
  };
}

function buildFailureEvidence(
  packets: Packet[],
  pair: PairFeas[][],
  bestDead: DeadState | null,
  modulus: number,
  countUpper: number,
  minInterval: number,
  maxInterval: number,
  jc?: JitterConfig,
): SolveError {
  const make = (evidence: ConstraintFailureEvidence): SolveError =>
    new SolveError(
      'NO_CONSISTENT_INTERPRETATION',
      'no globally consistent interpretation exists within the search window',
      evidence,
    );

  if (bestDead === null) {
    return make({
      stage: 'seed',
      partialLength: 0,
      partialOrder: [],
      candidateId: packets[0].id,
      reason: 'no packet can be seeded inside the absolute count search window',
    });
  }

  const n = packets.length;
  const { depth, placed, last, S, c0lo, c0hi, tLo, tHi, gaps: deadGaps, jitterLB } = bestDead;
  const partialOrder = placed.map((ix) => packets[ix].id);
  const usedNow = new Set(placed);
  const slotsAfter = n - 1 - depth;

  // Complete chain whose every structural constraint holds but whose exact
  // minimum cumulative jitter exceeds the budget: report the shortfall.
  if (jc !== undefined && depth === n) {
    const windows = tightenWindows(packets, placed, deadGaps, minInterval, maxInterval);
    let jMin = Number.isFinite(jitterLB) ? jitterLB : Infinity;
    if (windows !== null) {
      const cands = jitterCandidates(
        packets, placed, windows, deadGaps, minInterval, maxInterval, jc.nominalInterval,
      );
      const v = minChainJitter(cands, deadGaps, minInterval, maxInterval, jc.nominalInterval);
      if (Number.isFinite(v)) jMin = v;
    }
    if (Number.isFinite(jMin) && jMin > jc.totalJitterBudget) {
      const lastP = packets[last];
      return make({
        stage: 'extension',
        partialLength: depth,
        partialOrder,
        candidateId: lastP.id,
        reason:
          `cannot interpret the complete order ${partialOrder.map(String).join(' -> ')} within the ` +
          `cumulative jitter budget: the fixed order and absolute counts force a minimum total ` +
          `jitter of ${jMin}, which exceeds the budget ${jc.totalJitterBudget} by ` +
          `${jMin - jc.totalJitterBudget} (nominal interval ${jc.nominalInterval})`,
        detail: {
          cause: 'JITTER_BUDGET',
          jitter: {
            nominalInterval: jc.nominalInterval,
            budget: jc.totalJitterBudget,
            used: jMin,
            minimumAdditional: jMin - jc.totalJitterBudget,
          },
        },
      });
    }
  }

  // Forced jitter of the fixed partial order, needed to report cumulative
  // budget usage at the dead end. Reconstruct the FORWARD-tightened per-node
  // windows exactly as the search carried them (the dead frontier tLo/tHi is
  // such a forward window), then sum each edge's forced-jitter lower bound.
  let prefixJitterUsed = 0;
  if (jc !== undefined && placed.length > 0) {
    const fw: { lo: number; hi: number }[] = [];
    for (let k = 0; k < placed.length; k++) {
      const pp = packets[placed[k]];
      if (k === 0) fw.push({ lo: pp.lo, hi: pp.hi });
      else {
        const dd = deadGaps[k - 1];
        fw.push({
          lo: Math.max(pp.lo, fw[k - 1].lo + dd * minInterval),
          hi: Math.min(pp.hi, fw[k - 1].hi + dd * maxInterval),
        });
      }
    }
    for (let k = 0; k < deadGaps.length; k++) {
      prefixJitterUsed += edgeJitterLowerBound(
        deadGaps[k],
        fw[k].lo,
        fw[k].hi,
        fw[k + 1].lo,
        fw[k + 1].hi,
        minInterval,
        maxInterval,
        jc.nominalInterval,
      );
    }
    if (!Number.isFinite(prefixJitterUsed)) prefixJitterUsed = Number.isFinite(jitterLB) ? jitterLB : 0;
  }

  // Reproduce the canonical successor scan at the deepest dead end. For each
  // unused successor derive the feasible counter-gap range implied by each
  // constraint class independently:
  //   time:  [Tlo, Thi] from the tightened timestamp windows
  //   count: [Clo, Chi] from the c0 window and remaining absolute slots
  // plus the intrinsic ceiling pair.dHi (raw pair intervals + search window)
  // and the congruence residue. Their intersection is empty at a dead end;
  // the first blocker in canonical order (cause, gap, id) is reported.
  type Blocker = {
    j: number;
    cause: 'TIME_GAP' | 'COUNT_WINDOW' | 'CONGRUENCE' | 'JITTER_BUDGET';
    dStar: number;
    delta: number;
    timeRange: { min: number; max: number };
    countRange: { min: number; max: number };
    intrinsicCeiling: number;
    achievable: { min: number; max: number };
    /** Smallest forced edge jitter across this successor's admissible gaps. */
    minEdgeJitter: number;
    /** Congruent gap attaining minEdgeJitter. */
    minEdgeGap: number;
  };
  const blockers: Blocker[] = [];

  /** Smallest value >= lo congruent to `delta` and <= hi, else Infinity. */
  const snap = (lo: number, hi: number, delta: number): number => {
    if (lo > hi) return Infinity;
    const v = ceilResidue(lo, delta, modulus);
    return v <= hi ? v : Infinity;
  };

  for (let j = 0; j < n; j++) {
    if (usedNow.has(j)) continue;
    const pj = packets[j];
    const pf = pair[last][j];
    const d0 = pf.delta === 0 ? modulus : pf.delta;

    const Tlo = Math.ceil((pj.lo - tHi) / maxInterval);
    const Thi = Math.floor((pj.hi - tLo) / minInterval);
    const Clo = pj.baseCount - S - c0hi;
    const Chi = Math.min(pj.topCount - S - c0lo, countUpper - slotsAfter - S - c0lo);
    const loAll = Math.max(d0, Tlo, Clo);
    const hiAll = Math.min(pf.dHi, Thi, Chi);

    const snapOrInf = (lo: number, hi: number): number => {
      if (lo > hi) return Infinity;
      const d = ceilResidue(lo, pf.delta, modulus);
      return d <= hi ? d : Infinity;
    };
    const dTime = snapOrInf(Math.max(d0, Tlo), Thi);
    const dCount = snapOrInf(Math.max(d0, Clo), Chi);
    const dBoth = snapOrInf(loAll, hiAll);

    // Enumerate every congruent gap admissible by time+count windows and the
    // raw feasibility test, tracking the smallest forced next-edge jitter.
    let minEdgeJitter = Infinity;
    let minEdgeGap = Infinity;
    if (jc !== undefined && dBoth !== Infinity) {
      for (let d = dBoth; d <= hiAll; d += modulus) {
        if (d * minInterval > pj.hi - tLo) break;
        const njLo = Math.max(c0lo, pj.baseCount - S - d);
        const njHi = Math.min(c0hi, pj.topCount - S - d, countUpper - slotsAfter - S - d);
        const ntLo = Math.max(pj.lo, tLo + d * minInterval);
        const ntHi = Math.min(pj.hi, tHi + d * maxInterval);
        if (njLo > njHi || ntLo > ntHi) continue;
        const forced = edgeJitterLowerBound(
          d, tLo, tHi, ntLo, ntHi, minInterval, maxInterval, jc.nominalInterval,
        );
        if (forced < minEdgeJitter) {
          minEdgeJitter = forced;
          minEdgeGap = d;
        }
      }
    }

    const budgetBlocks =
      jc !== undefined &&
      dBoth !== Infinity &&
      Number.isFinite(minEdgeJitter) &&
      prefixJitterUsed + minEdgeJitter > jc.totalJitterBudget;

    let cause: Blocker['cause'];
    let dStar: number;
    if (budgetBlocks) {
      // Time/count/congruence all admit a gap; only the jitter budget fails.
      cause = 'JITTER_BUDGET';
      dStar = minEdgeGap;
    } else if (dBoth !== Infinity) {
      // This successor itself looks extendable; it cannot be the reported
      // blocker of the dead end (a later extension or the leaf budget fails).
      continue;
    } else if (dCount !== Infinity) {
      // The smallest gap satisfying congruence + the count window exists;
      // the extension attempt at it fails on the time-difference range.
      cause = 'TIME_GAP';
      dStar = dCount;
    } else if (dTime !== Infinity) {
      // Timing admits a congruent gap but the absolute-count window does not.
      cause = 'COUNT_WINDOW';
      dStar = dTime;
    } else {
      // Neither range alone contains a congruent value.
      cause = Tlo > Thi ? 'TIME_GAP' : 'CONGRUENCE';
      dStar = Infinity;
    }

    blockers.push({
      j,
      cause,
      dStar,
      delta: pf.delta,
      timeRange: { min: Tlo, max: Thi },
      countRange: { min: Clo, max: Chi },
      intrinsicCeiling: pf.dHi,
      achievable: { min: pj.lo - tHi, max: pj.hi - tLo },
      minEdgeJitter: Number.isFinite(minEdgeJitter) ? minEdgeJitter : Infinity,
      minEdgeGap: Number.isFinite(minEdgeGap) ? minEdgeGap : Infinity,
    });
  }

  const causeRank = { TIME_GAP: 0, COUNT_WINDOW: 1, CONGRUENCE: 2, JITTER_BUDGET: 3 } as const;
  blockers.sort(
    (a, b) =>
      causeRank[a.cause] - causeRank[b.cause] ||
      a.dStar - b.dStar ||
      compareId(packets[a.j].id, packets[b.j].id),
  );

  if (blockers.length > 0) {
    const b = blockers[0];
    const pj = packets[b.j];
    const prevId = String(packets[last].id);
    const finite = (x: number): number => (Number.isFinite(x) ? x : -1);
    const d = b.dStar;

    if (b.cause === 'JITTER_BUDGET') {
      const minAdd = b.minEdgeJitter;
      return make({
        stage: 'extension',
        partialLength: depth,
        partialOrder,
        candidateId: pj.id,
        reason:
          `cannot append packet ${String(pj.id)} after packet ${prevId} within the cumulative ` +
          `jitter budget: the fixed partial order has already forced ${prefixJitterUsed} of the ` +
          `${jc!.totalJitterBudget} budget, and every otherwise admissible congruent gap for this ` +
          `packet adds at least ${minAdd} more jitter (first reached at gap ${finite(b.minEdgeGap)}), ` +
          `so ${prefixJitterUsed + minAdd} > ${jc!.totalJitterBudget}; nominal interval ` +
          `${jc!.nominalInterval} (time/count/congruence constraints all admit the extension)`,
        detail: {
          cause: 'JITTER_BUDGET',
          minimalCongruentGap: Number.isFinite(b.minEdgeGap) ? b.minEdgeGap : undefined,
          countGap: Number.isFinite(b.minEdgeGap) ? b.minEdgeGap : undefined,
          countGapWindow: { min: b.countRange.min, max: b.countRange.max },
          actualTimeGapRange: b.achievable,
          jitter: {
            nominalInterval: jc!.nominalInterval,
            budget: jc!.totalJitterBudget,
            used: prefixJitterUsed,
            minimumAdditional: minAdd,
          },
        },
      });
    }

    if (b.cause === 'TIME_GAP') {
      // Smallest congruent gap that would satisfy the time-difference range.
      const firstPositive = b.delta === 0 ? modulus : b.delta;
      const dTiming = snap(Math.max(firstPositive, b.timeRange.min), b.timeRange.max, b.delta);
      const dCountVal = b.dStar;
      return make({
        stage: 'extension',
        partialLength: depth,
        partialOrder,
        candidateId: pj.id,
        reason:
          `cannot append packet ${String(pj.id)} after packet ${prevId}: the time-difference ` +
          `constraint needs a counter gap in [${b.timeRange.min}, ${b.timeRange.max}] but the ` +
          `absolute-count window only permits [${b.countRange.min}, ${b.countRange.max}] ` +
          `(smallest congruent gap satisfying the count window: ${finite(dCountVal)}; satisfying ` +
          `the time range: ${finite(dTiming)}). The tightened closed intervals only admit time ` +
          `differences in [${b.achievable.min}, ${b.achievable.max}], so no single gap satisfies ` +
          `both constraints`,
        detail: {
          cause: 'TIME_GAP',
          minimalCongruentGap: Number.isFinite(dCountVal) ? dCountVal : undefined,
          countGap: Number.isFinite(dCountVal) ? dCountVal : undefined,
          requiredTimeGap: Number.isFinite(dTiming)
            ? { min: dTiming * minInterval, max: dTiming * maxInterval }
            : undefined,
          actualTimeGapRange: b.achievable,
          countGapWindow: { min: b.countRange.min, max: b.countRange.max },
        },
      });
    }

    if (b.cause === 'COUNT_WINDOW') {
      return make({
        stage: 'extension',
        partialLength: depth,
        partialOrder,
        candidateId: pj.id,
        reason:
          `cannot append packet ${String(pj.id)} after packet ${prevId}: the absolute-count ` +
          `window only admits a counter gap in [${b.countRange.min}, ${b.countRange.max}] (intrinsic ` +
          `ceiling ${b.intrinsicCeiling}), but the time-difference constraint needs a gap in ` +
          `[${b.timeRange.min}, ${b.timeRange.max}]; the two ranges have no congruent value in common`,
        detail: {
          cause: 'COUNT_WINDOW',
          minimalCongruentGap: Number.isFinite(d) ? d : undefined,
          countGapWindow: { min: b.countRange.min, max: b.countRange.max },
          requiredTimeGap: Number.isFinite(d)
            ? { min: d * minInterval, max: d * maxInterval }
            : undefined,
          actualTimeGapRange: b.achievable,
        },
      });
    }

    return make({
      stage: 'extension',
      partialLength: depth,
      partialOrder,
      candidateId: pj.id,
      reason:
        `cannot append packet ${String(pj.id)} after packet ${prevId}: the time-feasible gap range ` +
        `[${b.timeRange.min}, ${b.timeRange.max}] and count-feasible gap range ` +
        `[${b.countRange.min}, ${b.countRange.max}] overlap but contain no positive counter gap ` +
        `congruent to ${pair[last][b.j].delta} modulo ${modulus}`,
      detail: {
        cause: 'CONGRUENCE',
        minimalCongruentGap: Number.isFinite(d) ? d : undefined,
        countGapWindow: { min: b.countRange.min, max: b.countRange.max },
        actualTimeGapRange: b.achievable,
      },
    });
  }

  const candidate = packets[last];
  return make({
    stage: 'extension',
    partialLength: depth,
    partialOrder,
    candidateId: candidate.id,
    reason: `cannot extend from packet ${String(candidate.id)}: no unused packet remains`,
  });
}
