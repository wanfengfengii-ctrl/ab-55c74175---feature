import type {
  AdjacencyEvidence,
  AssignedPacket,
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
  /** Gaps already fixed between the placed packets (length depth - 1). */
  gaps: number[];
  last: number;
  S: number;
  c0lo: number;
  c0hi: number;
  tLo: number;
  tHi: number;
}

/** Active jitter configuration. */
interface JitterSpec {
  nominal: number;
  total: number;
}

/** Outcome of the exact fixed-chain integer program. */
interface ChainSolution {
  /** Infinity when no timestamp assignment satisfies the chain (+ budget). */
  deviation2: number;
  /** Total jitter of the returned assignment (Infinity when infeasible). */
  jitter: number;
  /** Lexicographically smallest optimal integer timestamp vector. */
  times: number[];
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
 * [t_next - U, t_next - L]. For an integer difference-constraint CHAIN these
 * two passes yield the exact per-variable feasible domains. Null = infeasible.
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
 * Exact integer optimization on a FIXED order with fixed gaps.
 *
 * Without a jitter spec: minimize Σ |2 t_k - mid2_k| subject to
 * t_k ∈ [lo_k, hi_k] and L_e ≤ t_{e+1} - t_e ≤ U_e.
 *
 * With a jitter spec: the cumulative jitter constraint
 * Σ_e |Δt_e - d_e·nominal| ≤ total is a HARD constraint; mode selects
 * whether to minimize midpoint deviation ('deviation') or only determine the
 * minimum achievable jitter ('feasibility').
 *
 * At an integral optimum every variable is pinned — directly or through a
 * chain of tight constraints — to a pivot: an interval bound, one of the two
 * integers adjacent to its midpoint, a tight L/U edge, or (with jitter) a
 * nominal edge Δt = d·nominal. Closure under these tight-edge chains gives a
 * small exact candidate set per position. A suffix DP over candidates and
 * jitter consumed computes the optimum; reconstruction returns the
 * lexicographically smallest optimal timestamp vector.
 *
 * Exported for direct differential testing against a full-domain DP.
 */
export function optimalTimes(
  packets: Packet[],
  order: number[],
  windows: { lo: number; hi: number }[],
  gaps: number[],
  minInterval: number,
  maxInterval: number,
  jb: JitterSpec | null = null,
  mode: 'deviation' | 'feasibility' = 'deviation',
): ChainSolution {
  const infeasible: ChainSolution = { deviation2: Infinity, jitter: Infinity, times: [] };
  const n = order.length;
  const L = gaps.map((d) => d * minInterval);
  const U = gaps.map((d) => d * maxInterval);
  const N = gaps.map((d) => d * (jb ? jb.nominal : 0));

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
  const steps = (e: number): number[] => (jb ? [L[e], U[e], N[e]] : [L[e], U[e]]);

  // Closure under tight-edge chains. An integral optimum pins every
  // variable via a chain of tight constraints (L/U edges, plus nominal edges
  // when jitter is tracked) back to a pivot.
  const reached: Set<number>[] = candSets.map(() => new Set<number>());
  for (let j = 0; j < n; j++) {
    for (const s of pivotsOf(j)) {
      add(j, s);
      reached[j].add(s);
    }
  }

  /**
   * With a jitter budget the linearized problem adds ONE global constraint
   * Σ s_e ≤ B (s_e = |Δt_e - N_e|). At an integral optimum every edge is
   * tight (Δt ∈ {L_e, U_e, N_e}, or pinned through endpoint pivots) except
   * at most ONE residual edge e0 whose s carries the budget remainder:
   * s_e0 = B − Σ_{e≠e0} s_e, Δt_e0 = N_e0 ± s_e0. The possible per-edge
   * jitter contributions of tight edges therefore form a finite option set;
   * their subset sums (excluding e0) yield every possible residual step.
   */
  const jitterSteps: number[][] = new Array(n - 1);
  if (jb) {
    for (let e = 0; e < n - 1; e++) {
      const opts = new Set<number>([0, Math.abs(L[e] - N[e]), Math.abs(U[e] - N[e])]);
      // Edges whose endpoints are both pinned by tight chains contribute the
      // jitter of their difference; added after the first closure pass below.
      jitterSteps[e] = [...opts].filter((x) => Number.isSafeInteger(x));
    }
  }

  const runClosure = (): void => {
    let grew = true;
    while (grew) {
      grew = false;
      for (let pos = 0; pos < n; pos++) {
        for (const v of [...reached[pos]]) {
          const expand = (np: number, nv: number): void => {
            if (np < 0 || np >= n) return;
            if (nv < windows[np].lo || nv > windows[np].hi) return;
            if (reached[np].has(nv)) return;
            reached[np].add(nv);
            add(np, nv);
            grew = true;
          };
          if (pos < n - 1) for (const st of steps(pos)) expand(pos + 1, v + st);
          if (pos > 0) for (const st of steps(pos - 1)) expand(pos - 1, v - st);
        }
      }
    }
  };
  runClosure();

  if (jb && Number.isFinite(jb.total)) {
    // Alternate: tight-edge jitter options of reached endpoint pairs,
    // residual-edge expansion from subset sums, closure — to fixed point.
    let grewOuter = true;
    let guard = 0;
    while (grewOuter) {
      grewOuter = false;
      if (++guard > 512) {
        throw new Error('jitter candidate closure did not converge');
      }
      for (let e = 0; e < n - 1; e++) {
        const opts = new Set<number>(jitterSteps[e]);
        for (const a of reached[e]) {
          for (const b of reached[e + 1]) {
            const dt = b - a;
            if (dt < L[e] || dt > U[e]) continue;
            if (!opts.has(Math.abs(dt - N[e]))) {
              opts.add(Math.abs(dt - N[e]));
              grewOuter = true;
            }
          }
        }
        jitterSteps[e] = [...opts];
      }

      // Subset sums of per-edge tight jitter options, with prefix/suffix
      // sets so the sums EXCLUDING a chosen residual edge e0 are available.
      const prefixSums: Set<number>[] = new Array(n);
      const suffixSums: Set<number>[] = new Array(n);
      prefixSums[0] = new Set<number>([0]);
      for (let e = 0; e < n - 1; e++) {
        const next = new Set<number>();
        for (const s of prefixSums[e]) {
          for (const o of jitterSteps[e]) {
            const v = s + o;
            if (v <= jb.total) next.add(v);
          }
        }
        prefixSums[e + 1] = next;
      }
      suffixSums[n - 1] = new Set<number>([0]);
      for (let e = n - 2; e >= 0; e--) {
        const next = new Set<number>();
        for (const s of suffixSums[e + 1]) {
          for (const o of jitterSteps[e]) {
            const v = s + o;
            if (v <= jb.total) next.add(v);
          }
        }
        suffixSums[e] = next;
      }

      for (let e0 = 0; e0 < n - 1; e0++) {
        const sumsExcept = new Set<number>();
        for (const a of prefixSums[e0]) {
          for (const b of suffixSums[e0 + 1]) {
            const v = a + b;
            if (v <= jb.total) sumsExcept.add(v);
          }
        }
        for (const sOthers of sumsExcept) {
          const residual = jb.total - sOthers; // s_e0 >= 0
          if (residual < 0) continue;
          for (const sign of [1, -1]) {
            const step = N[e0] + sign * residual;
            for (const v of [...reached[e0]]) {
              const nv = v + step;
              if (nv >= windows[e0 + 1].lo && nv <= windows[e0 + 1].hi && !reached[e0 + 1].has(nv)) {
                reached[e0 + 1].add(nv);
                add(e0 + 1, nv);
                grewOuter = true;
              }
            }
            for (const v of [...reached[e0 + 1]]) {
              const nv = v - step;
              if (nv >= windows[e0].lo && nv <= windows[e0].hi && !reached[e0].has(nv)) {
                reached[e0].add(nv);
                add(e0, nv);
                grewOuter = true;
              }
            }
          }
        }
      }
      runClosure();
    }
  }

  const candidates = candSets.map((set) => [...set].sort((a, b) => a - b));
  if (candidates.some((c) => c.length === 0)) return infeasible;
  const dev = (k: number, t: number): number => Math.abs(2 * t - packets[order[k]].mid2);

  if (jb === null) {
    // suffix[k][c] = minimal deviation on positions k..n-1 with t_k = c[k][c].
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
    if (!Number.isFinite(globalBest)) return infeasible;
    // Lexicographically smallest optimal vector.
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
      if (chosen < 0) return infeasible;
      times[k] = candidates[k][chosen];
      target -= dev(k, times[k]);
      prevTime = times[k];
    }
    return { times, deviation2: globalBest, jitter: 0 };
  }

  // Budgeted: table[k][c] maps jitter spent on suffix edges k..n-2 to the
  // minimum deviation attainable on positions k..n-1 (0 in feasibility mode).
  const table: Map<number, number>[][] = new Array(n);
  table[n - 1] = candidates[n - 1].map(
    (t) => new Map<number, number>([[0, mode === 'feasibility' ? 0 : dev(n - 1, t)]]),
  );
  for (let k = n - 2; k >= 0; k--) {
    table[k] = candidates[k].map(() => new Map<number, number>());
    for (let c = 0; c < candidates[k].length; c++) {
      const t = candidates[k][c];
      const target = table[k][c];
      const ownCost = mode === 'feasibility' ? 0 : dev(k, t);
      for (let nc = 0; nc < candidates[k + 1].length; nc++) {
        const nt = candidates[k + 1][nc];
        const dt = nt - t;
        if (dt < L[k] || dt > U[k]) continue;
        const edgeJitter = Math.abs(dt - N[k]);
        for (const [tailUsed, tailCost] of table[k + 1][nc]) {
          const used = edgeJitter + tailUsed;
          if (used > jb.total) continue;
          const v = ownCost + tailCost;
          const old = target.get(used);
          if (old === undefined || v < old) target.set(used, v);
        }
      }
    }
  }

  let globalBest = Infinity;
  let bestJitter = Infinity;
  for (const m of table[0]) {
    for (const [usedJ, v] of m) {
      const better =
        mode === 'feasibility' ? usedJ < bestJitter : v < globalBest || (v === globalBest && usedJ < bestJitter);
      if (better) {
        globalBest = v;
        bestJitter = usedJ;
      }
    }
  }
  if (bestJitter === Infinity) return infeasible;

  // Forward reconstruction of the lexicographically smallest optimal vector.
  const times = new Array<number>(n);
  let needJitter = bestJitter;
  let needCost = globalBest;
  let prevTime = Number.NaN;
  for (let k = 0; k < n; k++) {
    let chosen = -1;
    for (let c = 0; c < candidates[k].length; c++) {
      const t = candidates[k][c];
      if (k === 0) {
        if (table[0][c].get(needJitter) !== needCost) continue;
      } else {
        const dt = t - prevTime;
        if (dt < L[k - 1] || dt > U[k - 1]) continue;
        const edgeJitter = Math.abs(dt - N[k - 1]);
        if (edgeJitter > needJitter) continue;
        if (table[k][c].get(needJitter - edgeJitter) !== needCost) continue;
      }
      chosen = c;
      break;
    }
    if (chosen < 0) return infeasible;
    times[k] = candidates[k][chosen];
    if (k > 0) {
      const edgeJitter = Math.abs(times[k] - prevTime - N[k - 1]);
      needJitter -= edgeJitter;
    }
    // Position k's own deviation is now fixed: the next suffix table covers
    // positions k+1..n-1, so account for dev(k) before the next lookup.
    needCost -= mode === 'feasibility' ? 0 : dev(k, times[k]);
    prevTime = times[k];
  }
  return {
    times,
    deviation2: mode === 'feasibility' ? 0 : globalBest,
    jitter: bestJitter,
  };
}

interface Move {
  j: number;
  d: number;
  c0lo: number;
  c0hi: number;
  tLo: number;
  tHi: number;
}

export interface SolveOptions {
  /** Active only when both are provided. */
  nominalInterval?: number;
  totalJitterBudget?: number;
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
 * When nominalInterval + totalJitterBudget are supplied, the cumulative
 * jitter Σ|Δt - d·nominalInterval| ≤ budget is a HARD constraint solved
 * JOINTLY with order, absolute counts and timestamps in every phase — the
 * unconstrained optimum is never computed and filtered afterwards.
 *
 * Three exhaustive branch-and-bound phases share the state space:
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
  options: SolveOptions = {},
): SolveResult {
  const n = inputs.length;
  const W = countUpper - countLower;
  const nominal = options.nominalInterval;
  const budget = options.totalJitterBudget;
  const jitterOn = nominal !== undefined && budget !== undefined;
  const jb: JitterSpec | null = jitterOn ? { nominal: nominal!, total: budget! } : null;

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
   * sorted by smallest gap then smallest target id. */
  const enumerateMoves = (
    depth: number,
    last: number,
    S: number,
    c0lo: number,
    c0hi: number,
    tLo: number,
    tHi: number,
    mask: number,
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

  /**
   * Exact minimum jitter of the fixed prefix currently in orderArr/gapsArr
   * (positions 0..depth-1), over integer timestamps consistent with the
   * prefix's tightened windows. The suffix is ignored, so this is a lower
   * bound on the jitter of ANY completion: Infinity / a value > budget
   * certifies that no completion can satisfy the budget. Memoized by the
   * full prefix signature because the frontier windows alone do not
   * determine the prefix jitter already spent.
   */
  const memoPrefixJitter = new Map<string, number>();
  const minPrefixJitter = (depth: number): number => {
    if (!jitterOn) return 0;
    if (depth < 2) return 0;
    let key = `J|${depth}`;
    for (let k = 0; k < depth; k++) {
      key += `>${orderArr[k]}:${k > 0 ? gapsArr[k - 1] : 0}:${tLoArr[k]},${tHiArr[k]}`;
    }
    const cached = memoPrefixJitter.get(key);
    if (cached !== undefined) return cached;
    const order = orderArr.slice(0, depth);
    const gaps = gapsArr.slice(0, depth - 1);
    const windows = tightenWindows(packets, order, gaps, minInterval, maxInterval);
    const value =
      windows === null
        ? Infinity
        : optimalTimes(packets, order, windows, gaps, minInterval, maxInterval, jb, 'feasibility')
            .jitter;
    memoPrefixJitter.set(key, value);
    return value;
  };

  const recordDead = (
    depth: number,
    last: number,
    S: number,
    c0lo: number,
    c0hi: number,
    tLo: number,
    tHi: number,
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
      };
    }
  };

  // ------------------------------------------------------------------ Phase A
  // Minimum total counter gap. With an active jitter budget only chains whose
  // full timestamp assignment fits the budget are eligible; the budget is
  // enforced DURING the search (per-prefix exact jitter pruning + exact leaf
  // checks), never by post-filtering the unconstrained optimum.
  const memoA = new Map<string, number>();
  let bestA = Infinity;
  let stopA = false;

  /** Drop moves whose fixed prefix already forces jitter above the budget. */
  const jitterFeasibleMoves = (depth: number, moves: Move[]): Move[] => {
    if (!jitterOn) return moves;
    return moves.filter((mv) => {
      orderArr[depth] = mv.j;
      gapsArr[depth - 1] = mv.d;
      tLoArr[depth] = mv.tLo;
      tHiArr[depth] = mv.tHi;
      const usedJ = minPrefixJitter(depth + 1);
      return Number.isFinite(usedJ) && usedJ <= jb!.total;
    });
  };

  const dfsA = (depth: number, last: number, S: number, c0lo: number, c0hi: number, tLo: number, tHi: number): number => {
    if (stopA) return Infinity;
    if (depth === n) {
      if (jitterOn) {
        const order = orderArr.slice();
        const gaps = gapsArr.slice();
        const windows = tightenWindows(packets, order, gaps, minInterval, maxInterval);
        const ok =
          windows !== null &&
          Number.isFinite(
            optimalTimes(packets, order, windows, gaps, minInterval, maxInterval, jb, 'feasibility')
              .jitter,
          );
        if (!ok) return Infinity;
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

    // Prefix history is part of the state whenever jitter is tracked: two
    // chains with the same frontier can carry different committed jitter.
    const key = jitterOn
      ? `A|${mask}|${last}|${S}|${c0lo}|${c0hi}|${tLo}|${tHi}|${orderArr
          .slice(0, depth)
          .map((ix, k) => `${ix}:${k > 0 ? gapsArr[k - 1] : 0}`)
          .join('#')}`
      : `A|${mask}|${last}|${S}|${c0lo}|${c0hi}|${tLo}|${tHi}`;
    const cached = memoA.get(key);
    if (cached !== undefined) return cached;

    const moves = jitterFeasibleMoves(
      depth,
      enumerateMoves(depth, last, S, c0lo, c0hi, tLo, tHi, mask),
    );
    if (moves.length === 0) {
      recordDead(depth, last, S, c0lo, c0hi, tLo, tHi);
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
      const v = dfsA(depth + 1, mv.j, S + mv.d, mv.c0lo, mv.c0hi, mv.tLo, mv.tHi);
      used[mv.j] = 0;
      if (v < best) best = v;
      if (stopA) break;
    }
    if (best === Infinity) {
      recordDead(depth, last, S, c0lo, c0hi, tLo, tHi);
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
    // When a budget is active, distinguish "no interpretation exists at all"
    // from "interpretations exist but every one overspends the jitter
    // budget". The latter is certified against the unconstrained optimum
    // (which the joint search never computed as a candidate): its first
    // prefix edge whose exact minimum cumulative jitter crosses the budget
    // is the canonical first blocking evidence. If even the unconstrained
    // problem is infeasible, its structural evidence is the correct answer
    // (not a JITTER_BUDGET blocker).
    if (jb) {
      const unconstrained = solve(
        inputs,
        modulus,
        countLower,
        countUpper,
        minInterval,
        maxInterval,
      );
      throw buildJitterFailureEvidence(
        packets,
        pair,
        modulus,
        countUpper,
        minInterval,
        maxInterval,
        jb,
        unconstrained,
      );
    }
    throw buildFailureEvidence(
      packets,
      pair,
      bestDead,
      modulus,
      countUpper,
      minInterval,
      maxInterval,
    );
  }
  const Pstar = bestA;

  // ------------------------------------------------------------- Phase B/C key
  /** Exact state signature for the deviation/lex phases: full prefix packet
   * sequence, its gaps and every position's tightened window. Paths sharing
   * this signature have identical prefix deviation/jitter and an identical
   * frontier, so memoized results are interchangeable. */
  const stateKeyBC = (depth: number, last: number, S: number, c0lo: number, c0hi: number, tLo: number, tHi: number): string => {
    let s = `${depth}|${last}|${S}|${c0lo}|${c0hi}|${tLo}|${tHi}`;
    for (let k = 0; k < depth; k++) {
      s += `>${orderArr[k]}:${k > 0 ? gapsArr[k - 1] : 0}:${tLoArr[k]},${tHiArr[k]}`;
    }
    return s;
  };

  /**
   * Exact primary-optimal-chain oracle. Returns true exactly when a
   * completion of the CURRENT state reaches total gap Pstar AND, when a
   * jitter budget is active, admits timestamps whose cumulative jitter fits.
   * Unlike the intrinsic Held-Karp bound, this accounts for time/count/
   * jitter feasibility, so it is the correct filter for phases B and C.
   */
  const memoOpt = new Map<string, boolean>();
  const optimalFromState = (
    depth: number,
    last: number,
    S: number,
    c0lo: number,
    c0hi: number,
    tLo: number,
    tHi: number,
    mask: number,
  ): boolean => {
    if (depth === n) {
      if (S !== Pstar) return false;
      if (jitterOn) {
        const order = orderArr.slice();
        const gaps = gapsArr.slice();
        const windows = tightenWindows(packets, order, gaps, minInterval, maxInterval);
        return (
          windows !== null &&
          Number.isFinite(
            optimalTimes(packets, order, windows, gaps, minInterval, maxInterval, jb, 'feasibility')
              .jitter,
          )
        );
      }
      return true;
    }
    // Frontier-only state is Markovian without jitter; with jitter the
    // committed prefix is part of the state.
    const key = jitterOn
      ? `O|${stateKeyBC(depth, last, S, c0lo, c0hi, tLo, tHi)}`
      : `O|${mask}|${last}|${S}|${c0lo}|${c0hi}|${tLo}|${tHi}`;
    const cached = memoOpt.get(key);
    if (cached !== undefined) return cached;

    const remaining = full ^ mask;
    let moves = enumerateMoves(depth, last, S, c0lo, c0hi, tLo, tHi, mask);
    if (jitterOn) moves = jitterFeasibleMoves(depth, moves);
    let ok = false;
    for (const mv of moves) {
      // Necessary bound for reaching Pstar; exact feasibility checked below.
      if (S + mv.d + cont[remaining ^ (1 << mv.j)][mv.j] > Pstar) continue;
      used[mv.j] = 1;
      orderArr[depth] = mv.j;
      gapsArr[depth - 1] = mv.d;
      tLoArr[depth] = mv.tLo;
      tHiArr[depth] = mv.tHi;
      const v = optimalFromState(
        depth + 1,
        mv.j,
        S + mv.d,
        mv.c0lo,
        mv.c0hi,
        mv.tLo,
        mv.tHi,
        mask | (1 << mv.j),
      );
      used[mv.j] = 0;
      if (v) {
        ok = true;
        break;
      }
    }
    memoOpt.set(key, ok);
    return ok;
  };

  /** Successor moves that lie on at least one primary-optimal completion. */
  const optimalMoves = (
    depth: number,
    last: number,
    S: number,
    c0lo: number,
    c0hi: number,
    tLo: number,
    tHi: number,
    mask: number,
  ): Move[] => {
    const remaining = full ^ mask;
    const all = enumerateMoves(depth, last, S, c0lo, c0hi, tLo, tHi, mask);
    return all.filter((mv) => {
      if (S + mv.d + cont[remaining ^ (1 << mv.j)][mv.j] > Pstar) return false;
      used[mv.j] = 1;
      orderArr[depth] = mv.j;
      gapsArr[depth - 1] = mv.d;
      tLoArr[depth] = mv.tLo;
      tHiArr[depth] = mv.tHi;
      let feasible = true;
      if (jitterOn) {
        const usedJ = minPrefixJitter(depth + 1);
        feasible = Number.isFinite(usedJ) && usedJ <= jb!.total;
      }
      const v =
        feasible &&
        optimalFromState(
          depth + 1,
          mv.j,
          S + mv.d,
          mv.c0lo,
          mv.c0hi,
          mv.tLo,
          mv.tHi,
          mask | (1 << mv.j),
        );
      used[mv.j] = 0;
      return v;
    });
  };

  /** Whether a seed packet can begin any primary-optimal completion. */
  const seedIsOptimal = (seedIndex: number): boolean => {
    const seed = packets[seedIndex];
    const c0hi0 = Math.min(seed.topCount, countUpper - n + 1);
    if (seed.baseCount > c0hi0) return false;
    return optimalFromState(1, seedIndex, 0, seed.baseCount, c0hi0, seed.lo, seed.hi, 1 << seedIndex);
  };

  const leafObjective = (): { dev: number; jit: number } => {
    const order = orderArr.slice();
    const gaps = gapsArr.slice();
    const windows = tightenWindows(packets, order, gaps, minInterval, maxInterval);
    if (windows === null) return { dev: Infinity, jit: Infinity };
    const r = optimalTimes(packets, order, windows, gaps, minInterval, maxInterval, jb, 'deviation');
    return { dev: r.deviation2, jit: r.jitter };
  };

  const leafDeviation = (): number => leafObjective().dev;

  // ------------------------------------------------------------------ Phase B
  // Minimum total deviation2 over primary-optimal (and budget-feasible)
  // chains. The returned chain is fixed by phase C afterwards; on that chain
  // optimalTimes already minimizes the achieved jitter among dev-optimal
  // timestamp vectors, so jitter needs no separate chain-level objective.
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

  const dfsB = (depth: number, last: number, S: number, c0lo: number, c0hi: number, tLo: number, tHi: number): number => {
    if (depth === n) {
      const v = leafDeviation();
      if (Number.isFinite(v) && v < bestB) bestB = v;
      return v;
    }
    const mask = usedMask();

    let placedLB = 0;
    for (let k = 0; k < depth; k++) {
      placedLB += minDeviation2(tLoArr[k], tHiArr[k], packets[orderArr[k]].mid2);
    }
    if (placedLB + independentDevLB(mask) >= bestB) return Infinity;

    const key = stateKeyBC(depth, last, S, c0lo, c0hi, tLo, tHi);
    const cached = memoB.get(key);
    if (cached !== undefined) return cached;

    const moves = optimalMoves(depth, last, S, c0lo, c0hi, tLo, tHi, mask);
    let best = Infinity;
    for (const mv of moves) {
      used[mv.j] = 1;
      orderArr[depth] = mv.j;
      gapsArr[depth - 1] = mv.d;
      tLoArr[depth] = mv.tLo;
      tHiArr[depth] = mv.tHi;
      const v = dfsB(depth + 1, mv.j, S + mv.d, mv.c0lo, mv.c0hi, mv.tLo, mv.tHi);
      used[mv.j] = 0;
      if (Number.isFinite(v) && v < best) best = v;
    }
    memoB.set(key, best);
    return best;
  };

  for (const seed of seedOrder) {
    used.fill(0);
    used[seed.index] = 1;
    orderArr[0] = seed.index;
    tLoArr[0] = seed.lo;
    tHiArr[0] = seed.hi;
    if (!seedIsOptimal(seed.index)) continue;
    dfsB(1, seed.index, 0, seed.baseCount, Math.min(seed.topCount, countUpper - n + 1), seed.lo, seed.hi);
  }
  if (!Number.isFinite(bestB)) {
    // Defensive: phase A guarantees a primary-optimal feasible leaf.
    throw new SolveError(
      'NO_CONSISTENT_INTERPRETATION',
      'internal search failure during deviation optimization',
    );
  }
  const Dstar = bestB;

  // ------------------------------------------------------------------ Phase C
  // Greedily construct the lexicographically smallest id sequence. At every
  // position candidate packets are tried in ascending id order; a memoized
  // boolean oracle decides whether a primary-optimal, deviation-optimal
  // completion exists with the candidate fixed at the current position.
  // Jitter stays a hard constraint (via optimalFromState); once the unique
  // lex-min chain is fixed, optimalTimes returns its minimum-jitter vector.
  const memoC = new Map<string, boolean>();

  const dfsCfeasible = (depth: number, last: number, S: number, c0lo: number, c0hi: number, tLo: number, tHi: number): boolean => {
    if (depth === n) {
      return leafDeviation() === Dstar;
    }
    const mask = usedMask();
    const key = stateKeyBC(depth, last, S, c0lo, c0hi, tLo, tHi);
    const cached = memoC.get(key);
    if (cached !== undefined) return cached;

    const moves = optimalMoves(depth, last, S, c0lo, c0hi, tLo, tHi, mask);
    let ok = false;
    for (const mv of moves) {
      used[mv.j] = 1;
      orderArr[depth] = mv.j;
      gapsArr[depth - 1] = mv.d;
      tLoArr[depth] = mv.tLo;
      tHiArr[depth] = mv.tHi;
      ok = dfsCfeasible(depth + 1, mv.j, S + mv.d, mv.c0lo, mv.c0hi, mv.tLo, mv.tHi);
      used[mv.j] = 0;
      if (ok) break;
    }
    memoC.set(key, ok);
    return ok;
  };

  const chosen: number[] = [];
  const fixedGaps: number[] = [];
  let curLast = -1;
  let curS = 0;
  let curC0lo = 0;
  let curC0hi = 0;
  let curTLo = 0;
  let curTHi = 0;
  let curMask = 0;

  /** Reproduce the forward-tightened windows of the fixed prefix so the
   * oracle's memoization keys and leaf tightening see a consistent state. */
  const replayPrefixWindows = (): void => {
    for (let k = 0; k < chosen.length; k++) {
      const p = packets[chosen[k]];
      if (k === 0) {
        tLoArr[0] = p.lo;
        tHiArr[0] = p.hi;
      } else {
        const d = fixedGaps[k - 1];
        tLoArr[k] = Math.max(p.lo, tLoArr[k - 1] + d * minInterval);
        tHiArr[k] = Math.min(p.hi, tHiArr[k - 1] + d * maxInterval);
      }
      orderArr[k] = chosen[k];
      if (k > 0) gapsArr[k - 1] = fixedGaps[k - 1];
    }
  };

  for (let depth = 0; depth < n; depth++) {
    let candidates: { j: number; mv: Move | null }[];
    if (depth === 0) {
      // Every seedable packet is tried in id order; the in-loop oracle
      // certifies primary-optimality + budget feasibility (it needs the
      // shared prefix arrays installed first, so do not pre-filter here).
      candidates = seedOrder.map((p) => ({ j: p.index, mv: null }));
    } else {
      candidates = optimalMoves(depth, curLast, curS, curC0lo, curC0hi, curTLo, curTHi, curMask).map(
        (mv) => ({ j: mv.j, mv }),
      );
    }
    candidates.sort((a, b) => compareId(packets[a.j].id, packets[b.j].id));

    let picked: { j: number; mv: Move | null } | null = null;
    for (const cand of candidates) {
      const j = cand.j;
      used.fill(0);
      for (const ix of chosen) used[ix] = 1;
      used[j] = 1;
      replayPrefixWindows();
      orderArr[depth] = j;

      let ok: boolean;
      if (depth === 0) {
        const seed = packets[j];
        const c0hi0 = Math.min(seed.topCount, countUpper - n + 1);
        tLoArr[0] = seed.lo;
        tHiArr[0] = seed.hi;
        ok = dfsCfeasible(1, j, 0, seed.baseCount, c0hi0, seed.lo, seed.hi);
      } else {
        const mv = cand.mv!;
        gapsArr[depth - 1] = mv.d;
        tLoArr[depth] = mv.tLo;
        tHiArr[depth] = mv.tHi;
        ok = dfsCfeasible(depth + 1, j, curS + mv.d, mv.c0lo, mv.c0hi, mv.tLo, mv.tHi);
      }
      used[j] = 0;
      if (ok) {
        picked = cand;
        break;
      }
    }

    if (!picked) {
      // Defensive: phases A/B certify a feasible choice at every position.
      throw new SolveError('NO_CONSISTENT_INTERPRETATION', 'internal failure reconstructing lex-min order');
    }

    chosen.push(picked.j);
    if (depth === 0) {
      const seed = packets[picked.j];
      curC0lo = seed.baseCount;
      curC0hi = Math.min(seed.topCount, countUpper - n + 1);
      curTLo = seed.lo;
      curTHi = seed.hi;
    } else {
      const mv = picked.mv!;
      fixedGaps.push(mv.d);
      curS += mv.d;
      curC0lo = mv.c0lo;
      curC0hi = mv.c0hi;
      curTLo = mv.tLo;
      curTHi = mv.tHi;
    }
    curLast = picked.j;
    curMask |= 1 << picked.j;
  }

  // Assemble the certified solution: smallest admissible c0, optimal times.
  const finalOrder = chosen.slice();
  const finalGaps = fixedGaps.slice();
  const windows = tightenWindows(packets, finalOrder, finalGaps, minInterval, maxInterval);
  if (!windows) {
    throw new SolveError('NO_CONSISTENT_INTERPRETATION', 'internal failure tightening final windows');
  }
  const sol = optimalTimes(
    packets,
    finalOrder,
    windows,
    finalGaps,
    minInterval,
    maxInterval,
    jb,
    'deviation',
  );
  if (!Number.isFinite(sol.deviation2)) {
    throw new SolveError(
      'NO_CONSISTENT_INTERPRETATION',
      'internal failure: reconstructed chain violates the jitter budget',
    );
  }
  let gapSum = 0;
  for (const d of finalGaps) gapSum += d;

  return buildResult(
    packets,
    {
      gapSum,
      times: sol.times,
      order: finalOrder,
      c0: curC0lo,
      gaps: finalGaps,
    },
    modulus,
    minInterval,
    maxInterval,
    jb,
  );
}

function buildResult(
  packets: Packet[],
  cand: { gapSum: number; times: number[]; order: number[]; c0: number; gaps: number[] },
  modulus: number,
  minInterval: number,
  maxInterval: number,
  jb: JitterSpec | null,
): SolveResult {
  const n = cand.order.length;
  const order = cand.order.map((ix) => packets[ix].id);
  const assignments: AssignedPacket[] = [];
  const adjacency: AdjacencyEvidence[] = [];
  const missingSegments: MissingSegment[] = [];

  let count = cand.c0;
  let cumulativeJitter = 0;
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
      let jitterFields = {};
      if (jb) {
        const jitter = Math.abs(tGap - d * jb.nominal);
        cumulativeJitter += jitter;
        jitterFields = {
          nominalTimeGap: d * jb.nominal,
          jitter,
          cumulativeJitter,
        };
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
        ...jitterFields,
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
          modNonNeg(count, modulus) === p.remainder &&
          (jb ? cumulativeJitter <= jb.total : true),
      });
    }
    if (k < n - 1) count += cand.gaps[k];
  }

  const result: SolveResult = {
    order,
    assignments,
    missingSegments,
    missingCountTotal: cand.gapSum - (n - 1),
    adjacency,
    observedCountRange: { first: cand.c0, last: cand.c0 + cand.gapSum },
  };
  if (jb) {
    result.jitter = {
      nominalInterval: jb.nominal,
      budget: jb.total,
      used: cumulativeJitter,
      remaining: jb.total - cumulativeJitter,
      exhausted: cumulativeJitter === jb.total,
    };
  }
  return result;
}

/**
 * Exact minimum cumulative jitter of a fixed chain WITHOUT the budget cap
 * (evidence must report the forced overshoot, which a capped computation
 * would only reveal as "infeasible"). Infinity if the chain admits no
 * integer timestamps at all.
 */
function minChainJitterUncapped(
  packets: Packet[],
  order: number[],
  gaps: number[],
  minInterval: number,
  maxInterval: number,
  jb: JitterSpec,
): number {
  const wins = tightenWindows(packets, order, gaps, minInterval, maxInterval);
  if (wins === null) return Infinity;
  return optimalTimes(
    packets,
    order,
    wins,
    gaps,
    minInterval,
    maxInterval,
    { nominal: jb.nominal, total: Infinity },
    'feasibility',
  ).jitter;
}

/**
 * Build the first jitter-budget blocking evidence along an otherwise fully
 * consistent chain. `reference` is a chain feasible under every original
 * constraint (the unconstrained lexicographic optimum), which is guaranteed
 * to overspend the budget because the joint search found nothing feasible.
 *
 * The "first blocking edge" is the first prefix whose EXACT minimum
 * cumulative jitter exceeds the budget. The evidence reports the exact
 * jitter already committed before that edge, the budget cap, and — across
 * every congruent counter gap still admissible for the successor — the
 * minimum fresh jitter forced by the extension.
 */
function buildJitterFailureEvidence(
  packets: Packet[],
  pair: PairFeas[][],
  modulus: number,
  countUpper: number,
  minInterval: number,
  maxInterval: number,
  jb: JitterSpec,
  reference: SolveResult,
): SolveError {
  const make = (evidence: ConstraintFailureEvidence): SolveError =>
    new SolveError(
      'NO_CONSISTENT_INTERPRETATION',
      'no interpretation fits the cumulative jitter budget',
      evidence,
    );

  const n = packets.length;
  const byId = new Map<string, number>();
  packets.forEach((p, ix) => byId.set(String(p.id), ix));
  const order = reference.order.map((id) => byId.get(String(id))!);
  const gaps = reference.adjacency.map((e) => e.countGap);

  // Exact minimum cumulative jitter for every prefix, independently of the
  // full chain's later edges and WITHOUT the budget cap.
  const prefixJitter = new Array<number>(n).fill(0);
  for (let k = 2; k <= n; k++) {
    prefixJitter[k - 1] = minChainJitterUncapped(
      packets,
      order.slice(0, k),
      gaps.slice(0, k - 1),
      minInterval,
      maxInterval,
      jb,
    );
  }

  // For each edge k evaluate EVERY congruent gap still admissible for the
  // reference successor (time/count/congruence), and compute the exact
  // uncapped minimum jitter of prefix + that gap. The "first blocking edge"
  // is the first k whose minimum attainable fresh jitter across all gaps
  // already forces total jitter above the budget — i.e. no single extension
  // of this prefix can keep the chain within budget.
  const edgeData: {
    k: number;
    used: number;
    tLo: number;
    tHi: number;
    clo: number;
    chi: number;
    minimalByGap: { countGap: number; minEdgeJitter: number }[];
    minAdditional: number;
  }[] = [];
  for (let k = 1; k < n; k++) {
    const lastIx = order[k - 1];
    const succIx = order[k];
    const pj = packets[succIx];
    const prefixWins = tightenWindows(
      packets,
      order.slice(0, k),
      gaps.slice(0, k - 1),
      minInterval,
      maxInterval,
    );
    const used = k >= 2 ? prefixJitter[k - 1] : 0;
    const tLo = prefixWins ? prefixWins[k - 1].lo : packets[lastIx].lo;
    const tHi = prefixWins ? prefixWins[k - 1].hi : packets[lastIx].hi;
    const pf = pair[lastIx][succIx];
    const slotsAfter = n - 1 - k;
    let S = 0;
    for (let e = 0; e < k - 1; e++) S += gaps[e];
    const c0 = reference.observedCountRange.first;
    const clo = Math.max(pf.dLo, Math.ceil((pj.lo - tHi) / maxInterval), pj.baseCount - S - c0);
    const chi = Math.min(
      pf.dHi,
      Math.floor((pj.hi - tLo) / minInterval),
      pj.topCount - S - c0,
      countUpper - slotsAfter - S - c0,
    );
    const minimalByGap: { countGap: number; minEdgeJitter: number }[] = [];
    let minAdditional = Infinity;
    if (clo <= chi) {
      for (let d = ceilResidue(clo, pf.delta, modulus); d <= chi; d += modulus) {
        if (d * minInterval > pj.hi - tLo) break;
        const joint = minChainJitterUncapped(
          packets,
          order.slice(0, k + 1),
          [...gaps.slice(0, k - 1), d],
          minInterval,
          maxInterval,
          jb,
        );
        if (!Number.isFinite(joint)) continue;
        const marginal = joint - used;
        minimalByGap.push({ countGap: d, minEdgeJitter: marginal });
        if (marginal < minAdditional) minAdditional = marginal;
      }
    }
    edgeData.push({ k, used, tLo, tHi, clo, chi, minimalByGap, minAdditional });
  }

  const blocker = edgeData.find((e) => e.minAdditional > jb.total - e.used);
  const chosenEdge = blocker ?? edgeData[edgeData.length - 1];
  const k = chosenEdge.k;
  const lastIx = order[k - 1];
  const succIx = order[k];
  const pj = packets[succIx];
  const prevId = packets[lastIx].id;
  const { used, tLo, tHi } = chosenEdge;
  const minimalByGap = chosenEdge.minimalByGap;
  const minAdditional = chosenEdge.minAdditional;

  const partialOrder = order.slice(0, k).map((ix) => packets[ix].id);
  const perGap = minimalByGap
    .map((g) => `gap ${g.countGap} adds >= ${g.minEdgeJitter}`)
    .join('; ');
  return make({
    stage: 'extension',
    partialLength: k,
    partialOrder,
    candidateId: pj.id,
    reason:
      `cannot extend to packet ${String(pj.id)} after packet ${String(prevId)}: order, absolute ` +
      `counts and integer timestamps are otherwise jointly consistent, but the cumulative jitter ` +
      `budget caps total deviation from nominal at ${jb.total} ticks (nominal interval ` +
      `${jb.nominal}); the prefix already commits ${used} ticks and every admissible counter gap ` +
      `to packet ${String(pj.id)} forces at least ${minAdditional} fresh ticks ` +
      `(${perGap || 'no admissible gap'}), so the budget cannot be honored`,
    detail: {
      cause: 'JITTER_BUDGET',
      actualTimeGapRange: { min: pj.lo - tHi, max: pj.hi - tLo },
      jitterBudget: {
        nominalInterval: jb.nominal,
        budget: jb.total,
        used,
        remaining: Math.max(0, jb.total - used),
        minAdditionalJitter: minAdditional,
        minimalByGap,
      },
    },
  });
}

function buildFailureEvidence(
  packets: Packet[],
  pair: PairFeas[][],
  bestDead: DeadState | null,
  modulus: number,
  countUpper: number,
  minInterval: number,
  maxInterval: number,
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
  const { depth, placed, last, S, c0lo, c0hi, tLo, tHi } = bestDead;
  const partialOrder = placed.map((ix) => packets[ix].id);
  const usedNow = new Set(placed);
  const slotsAfter = n - 1 - depth;

  type Blocker = {
    j: number;
    cause: 'TIME_GAP' | 'COUNT_WINDOW' | 'CONGRUENCE';
    dStar: number;
    delta: number;
    timeRange: { min: number; max: number };
    countRange: { min: number; max: number };
    intrinsicCeiling: number;
    achievable: { min: number; max: number };
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

    if (dBoth !== Infinity) continue; // extendable; cannot occur at a dead end
    let cause: Blocker['cause'];
    let dStar: number;
    if (dCount !== Infinity) {
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
    });
  }

  const causeRank = { TIME_GAP: 0, COUNT_WINDOW: 1, CONGRUENCE: 2 } as const;
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
