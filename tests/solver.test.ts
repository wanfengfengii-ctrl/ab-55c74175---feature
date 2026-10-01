import { describe, expect, it } from 'vitest';
import { solve } from '../src/core/solver.js';
import { SolveError } from '../src/core/types.js';
import type { PacketInput } from '../src/core/types.js';
import { sampleRequest, sampleExpected } from './fixtures/sample.js';
import { bruteSolve, lexIds, makeRng, type RefSolution } from './helpers/brute.js';

describe('solver: cross-week sample with missing packets', () => {
  const result = solve(
    sampleRequest.packets,
    sampleRequest.modulus,
    sampleRequest.countLower,
    sampleRequest.countUpper,
    sampleRequest.minInterval,
    sampleRequest.maxInterval,
  );

  it('recovers the true order despite scrambled download order', () => {
    expect(result.order).toEqual(sampleExpected.order);
  });

  it('assigns wrap-crossing absolute counts congruent to remainders', () => {
    expect(result.assignments.map((a) => a.absoluteCount)).toEqual(sampleExpected.counts);
    for (const a of result.assignments) {
      expect(((a.absoluteCount % 10) + 10) % 10).toBe(a.remainder);
    }
    expect(result.observedCountRange).toEqual({ first: 8, last: 31 });
  });

  it('selects midpoint-exact transmit timestamps', () => {
    expect(result.assignments.map((a) => a.time)).toEqual(sampleExpected.times);
  });

  it('reports the true missing count segments', () => {
    expect(result.missingCountTotal).toBe(sampleExpected.missingTotal);
    expect(result.missingSegments).toEqual(sampleExpected.missingSegments);
  });

  it('provides per-adjacency constraint evidence', () => {
    expect(result.adjacency).toHaveLength(6);
    for (const ev of result.adjacency) {
      expect(ev.satisfied).toBe(true);
      expect(ev.absoluteCountCongruent).toBe(true);
      expect(ev.timeGap).toBeGreaterThanOrEqual(ev.allowedTimeGap.min);
      expect(ev.timeGap).toBeLessThanOrEqual(ev.allowedTimeGap.max);
      expect(ev.toCount - ev.fromCount).toBe(ev.countGap);
      expect(ev.missingBetween).toBe(Math.max(0, ev.countGap - 1));
    }
    expect(result.adjacency.map((e) => [e.fromId, e.toId])).toEqual([
      ['A', 'B'],
      ['B', 'C'],
      ['C', 'D'],
      ['D', 'E'],
      ['E', 'F'],
      ['F', 'G'],
    ]);
  });
});

describe('solver: lexicographic optimization', () => {
  it('prefers fewer missing counts when larger congruent gaps are also feasible', () => {
    // All six packets have remainder 0 (mod 10): every adjacent gap must be a
    // positive multiple of 10. Wide timing intervals would permit gaps of 20+,
    // but the primary objective forces the minimum gap of 10 each.
    const packets: PacketInput[] = Array.from({ length: 6 }, (_, i) => ({
      id: i,
      remainder: 0,
      timeLower: 0,
      timeUpper: 1000,
    }));
    const r = solve(packets, 10, 0, 100, 1, 100);
    expect(r.assignments.map((x) => x.absoluteCount)).toEqual([0, 10, 20, 30, 40, 50]);
    expect(r.missingCountTotal).toBe(45);
  });

  it('minimizes id sequence lexicographically among objective-equal solutions', () => {
    // Six packets with remainders 0..5 and wide timing: the 6! chains all
    // have the same missing/deviation; lex-min must be ascending by id order.
    const packets: PacketInput[] = [
      { id: 5, remainder: 5, timeLower: 0, timeUpper: 1000 },
      { id: 2, remainder: 2, timeLower: 0, timeUpper: 1000 },
      { id: 0, remainder: 0, timeLower: 0, timeUpper: 1000 },
      { id: 4, remainder: 4, timeLower: 0, timeUpper: 1000 },
      { id: 1, remainder: 1, timeLower: 0, timeUpper: 1000 },
      { id: 3, remainder: 3, timeLower: 0, timeUpper: 1000 },
    ];
    const r = solve(packets, 6, 0, 200, 1, 200);
    expect(r.order).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('handles identical twins with symmetry-preserving lex order', () => {
    // a and b are identical twins (remainder 0, same interval); their minimum
    // gap is a full modulus (10), so times are spaced 60..100 apart. The
    // following rem 1,2,3 chain continues with gap 1 each. The lex-min order
    // must place twin 'a' before twin 'b'.
    const packets: PacketInput[] = [
      { id: 'b', remainder: 0, timeLower: 0, timeUpper: 100 },
      { id: 'a', remainder: 0, timeLower: 0, timeUpper: 100 },
      { id: 'c', remainder: 1, timeLower: 60, timeUpper: 80 },
      { id: 'd', remainder: 2, timeLower: 70, timeUpper: 90 },
      { id: 'e', remainder: 3, timeLower: 75, timeUpper: 100 },
      { id: 'f', remainder: 4, timeLower: 85, timeUpper: 110 },
    ];
    const r = solve(packets, 10, 0, 100, 6, 10);
    expect(r.order.slice(0, 2)).toEqual(['a', 'b']);
    expect(r.order).toHaveLength(6);
    for (const ev of r.adjacency) expect(ev.satisfied).toBe(true);
  });
});

describe('solver: infeasibility evidence', () => {
  it('reports seed-stage failure when no packet starts a long enough chain', () => {
    const packets: PacketInput[] = Array.from({ length: 10 }, (_, i) => ({
      id: i,
      remainder: i % 7,
      timeLower: i,
      timeUpper: i + 100,
    }));
    let caught: SolveError | null = null;
    try {
      solve(packets, 7, 0, 5, 1, 5);
    } catch (e) {
      caught = e as SolveError;
    }
    expect(caught).not.toBeNull();
    expect(caught!.code).toBe('NO_CONSISTENT_INTERPRETATION');
    expect(caught!.evidence?.stage).toBe('seed');
  });

  it('reports the first non-extendable adjacency with disjoint time ranges', () => {
    const packets: PacketInput[] = [
      { id: 0, remainder: 0, timeLower: 0, timeUpper: 100 },
      { id: 1, remainder: 1, timeLower: 0, timeUpper: 100 },
      { id: 2, remainder: 2, timeLower: 0, timeUpper: 100 },
      { id: 3, remainder: 3, timeLower: 9000, timeUpper: 9001 },
      { id: 4, remainder: 4, timeLower: 0, timeUpper: 100 },
      { id: 5, remainder: 0, timeLower: 0, timeUpper: 100 },
      { id: 6, remainder: 1, timeLower: 0, timeUpper: 100 },
      { id: 7, remainder: 2, timeLower: 0, timeUpper: 100 },
    ];
    let caught: SolveError | null = null;
    try {
      solve(packets, 5, 0, 200, 1, 20);
    } catch (e) {
      caught = e as SolveError;
    }
    expect(caught).not.toBeNull();
    expect(caught!.code).toBe('NO_CONSISTENT_INTERPRETATION');
    expect(caught!.evidence?.stage).toBe('extension');
    expect(caught!.evidence?.candidateId).toBe(3);
    expect(caught!.evidence?.detail?.cause).toBe('TIME_GAP');
  });
});

describe('solver: cumulative jitter budget', () => {
  it('returns zero jitter and an exhausted budget when nominal matches truth exactly', () => {
    const r = solve(
      sampleRequest.packets,
      sampleRequest.modulus,
      sampleRequest.countLower,
      sampleRequest.countUpper,
      sampleRequest.minInterval,
      sampleRequest.maxInterval,
      { nominalInterval: 10, totalJitterBudget: 0 },
    );
    expect(r.order).toEqual(sampleExpected.order);
    expect(r.jitter).toEqual({
      nominalInterval: 10,
      budget: 0,
      used: 0,
      remaining: 0,
      exhausted: true,
    });
    for (const ev of r.adjacency) {
      expect(ev.nominalTimeGap).toBe(ev.countGap * 10);
      expect(ev.jitter).toBe(0);
      expect(ev.cumulativeJitter).toBe(0);
      expect(ev.satisfied).toBe(true);
    }
  });

  it('reproduces the unconstrained result with a loose budget (only additive fields)', () => {
    const plain = solve(
      sampleRequest.packets,
      sampleRequest.modulus,
      sampleRequest.countLower,
      sampleRequest.countUpper,
      sampleRequest.minInterval,
      sampleRequest.maxInterval,
    );
    const r = solve(
      sampleRequest.packets,
      sampleRequest.modulus,
      sampleRequest.countLower,
      sampleRequest.countUpper,
      sampleRequest.minInterval,
      sampleRequest.maxInterval,
      { nominalInterval: 10, totalJitterBudget: 1_000 },
    );
    expect(r.order).toEqual(plain.order);
    expect(r.assignments).toEqual(plain.assignments);
    expect(r.jitter?.used).toBe(0);
    expect(r.jitter?.exhausted).toBe(false);
  });

  it('solves the budget jointly: a tight budget changes the absolute counts, not just filters', () => {
    // Unconstrained chain uses gaps summing to 7 (2 missing packets) and
    // spends 8 jitter. A budget of 4 forces the search to choose larger
    // congruent gaps (sum 9, 4 missing) — order is unchanged, so this proves
    // the budget is solved jointly with the ABSOLUTE COUNTS rather than by
    // taking the unconstrained solution and filtering it.
    const packets: PacketInput[] = [
      { id: 423, remainder: 0, timeLower: 23, timeUpper: 26 },
      { id: 307, remainder: 0, timeLower: 6, timeUpper: 8 },
      { id: 243, remainder: 0, timeLower: 16, timeUpper: 19 },
      { id: 642, remainder: 0, timeLower: 10, timeUpper: 13 },
      { id: 578, remainder: 1, timeLower: 22, timeUpper: 25 },
      { id: 146, remainder: 1, timeLower: 1, timeUpper: 2 },
    ];
    const args = [2, 7, 16, 2, 4] as const;
    const free = solve(packets, ...args);
    expect(free.missingCountTotal).toBe(2);
    expect(free.adjacency.map((e) => e.countGap)).toEqual([1, 2, 2, 1, 1]);

    const bound = solve(packets, ...args, { nominalInterval: 2, totalJitterBudget: 4 });
    expect(bound.order).toEqual(free.order);
    expect(bound.missingCountTotal).toBe(4);
    expect(bound.adjacency.map((e) => e.countGap)).toEqual([3, 2, 2, 1, 1]);
    expect(bound.jitter?.used).toBe(4);
    expect(bound.jitter?.exhausted).toBe(true);
    for (const ev of bound.adjacency) expect(ev.satisfied).toBe(true);
  });

  it('solves the budget jointly: a tight budget changes timestamps, not just filters', () => {
    // Unconstrained optimum deviation is 6 at jitter 3; budget 2 forces
    // deviation 8 with exactly 2 jitter (budget exhausted, still accepted).
    const packets: PacketInput[] = [
      { id: 55, remainder: 0, timeLower: 5, timeUpper: 7 },
      { id: 509, remainder: 1, timeLower: 14, timeUpper: 14 },
      { id: 640, remainder: 0, timeLower: 15, timeUpper: 18 },
      { id: 598, remainder: 1, timeLower: 14, timeUpper: 17 },
      { id: 412, remainder: 0, timeLower: 7, timeUpper: 10 },
      { id: 145, remainder: 0, timeLower: 13, timeUpper: 14 },
    ];
    const args = [2, 5, 13, 1, 2] as const;
    const plain = solve(packets, ...args);
    const plainDev = plain.assignments.reduce(
      (sum, a) => sum + Math.abs(2 * a.time - (a.timeInterval.lower + a.timeInterval.upper)),
      0,
    );
    expect(plainDev).toBe(6);
    expect(plain.jitter).toBeUndefined();

    const tight = solve(packets, ...args, { nominalInterval: 2, totalJitterBudget: 2 });
    expect(tight.jitter).toMatchObject({ used: 2, remaining: 0, exhausted: true });
    const tightDev = tight.assignments.reduce(
      (sum, a) => sum + Math.abs(2 * a.time - (a.timeInterval.lower + a.timeInterval.upper)),
      0,
    );
    expect(tightDev).toBe(8);
    let cum = 0;
    for (const ev of tight.adjacency) {
      cum += ev.jitter!;
      expect(ev.cumulativeJitter).toBe(cum);
      expect(ev.nominalTimeGap).toBe(ev.countGap * 2);
    }
    expect(cum).toBe(2);
  });

  it('reports first jitter-budget blocker with used, minimum additional and cap', () => {
    const packets: PacketInput[] = [
      { id: 0, remainder: 0, timeLower: 0, timeUpper: 0 },
      { id: 1, remainder: 1, timeLower: 3, timeUpper: 3 },
      { id: 2, remainder: 2, timeLower: 6, timeUpper: 6 },
      { id: 3, remainder: 3, timeLower: 10, timeUpper: 10 },
      { id: 4, remainder: 4, timeLower: 12, timeUpper: 12 },
      { id: 5, remainder: 5, timeLower: 15, timeUpper: 15 },
    ];
    // Fixed counts 0..5; nominal 3 forces jitters 0,0,1,1,0 (total 2).
    expect(() =>
      solve(packets, 10, 0, 100, 1, 100, { nominalInterval: 3, totalJitterBudget: 2 }),
    ).not.toThrow();
    let caught: SolveError | null = null;
    try {
      solve(packets, 10, 0, 100, 1, 100, { nominalInterval: 3, totalJitterBudget: 1 });
    } catch (e) {
      caught = e as SolveError;
    }
    expect(caught).not.toBeNull();
    expect(caught!.code).toBe('NO_CONSISTENT_INTERPRETATION');
    expect(caught!.evidence?.stage).toBe('extension');
    expect(caught!.evidence?.detail?.cause).toBe('JITTER_BUDGET');
    const jb = caught!.evidence?.detail?.jitterBudget!;
    expect(jb).toBeTruthy();
    expect(jb.budget).toBe(1);
    expect(jb.used).toBe(1);
    expect(jb.remaining).toBe(0);
    expect(jb.minAdditionalJitter).toBe(1);
    expect(jb.minimalByGap.length).toBeGreaterThan(0);
    expect(jb.minimalByGap.every((g) => g.minEdgeJitter >= 1)).toBe(true);
  });

  it('keeps original NO_CONSISTENT_INTERPRETATION semantics when budget is irrelevant', () => {
    const packets: PacketInput[] = Array.from({ length: 10 }, (_, i) => ({
      id: i,
      remainder: i % 7,
      timeLower: i,
      timeUpper: i + 100,
    }));
    let caught: SolveError | null = null;
    try {
      solve(packets, 7, 0, 5, 1, 5, { nominalInterval: 3, totalJitterBudget: 1_000 });
    } catch (e) {
      caught = e as SolveError;
    }
    expect(caught!.code).toBe('NO_CONSISTENT_INTERPRETATION');
    expect(caught!.evidence?.stage).toBe('seed');
    expect(caught!.evidence?.detail?.cause).not.toBe('JITTER_BUDGET');
  });
});

describe('solver: differential fuzzing against exhaustive reference', () => {
  const batches: [number, number][] = [
    [1, 120],
    [7, 120],
    [42, 120],
  ];

  for (const [seedStart, count] of batches) {
    it(`agrees with brute force on ${count} random instances (seed ${seedStart})`, () => {
      const rnd = makeRng(seedStart);
      for (let iter = 0; iter < count; iter++) {
        const n = 6;
        const modulus = 2 + Math.floor(rnd() * 6);
        const ids = new Set<number>();
        const packets: PacketInput[] = [];
        for (let i = 0; i < n; i++) {
          let id = Math.floor(rnd() * 1000);
          while (ids.has(id)) id = Math.floor(rnd() * 1000);
          ids.add(id);
          const lo = Math.floor(rnd() * 24);
          packets.push({
            id,
            remainder: Math.floor(rnd() * modulus),
            timeLower: lo,
            timeUpper: lo + Math.floor(rnd() * 3),
          });
        }
        const minInterval = 1 + Math.floor(rnd() * 3);
        const maxInterval = minInterval + Math.floor(rnd() * 3);
        const countLower = Math.floor(rnd() * 8);
        const countUpper = countLower + (n - 1) + Math.floor(rnd() * 3);

        // Inject identical twins in some cases.
        if (iter % 3 === 0) {
          packets[1].remainder = packets[0].remainder;
          packets[1].timeLower = packets[0].timeLower;
          packets[1].timeUpper = packets[0].timeUpper;
        }

        let got: RefSolution | null = null;
        let gotError = false;
        try {
          const r = solve(packets, modulus, countLower, countUpper, minInterval, maxInterval);
          for (const a of r.assignments) {
            const p = packets.find((pp) => pp.id === a.id)!;
            expect(a.time).toBeGreaterThanOrEqual(p.timeLower);
            expect(a.time).toBeLessThanOrEqual(p.timeUpper);
            expect(a.absoluteCount).toBeGreaterThanOrEqual(countLower);
            expect(a.absoluteCount).toBeLessThanOrEqual(countUpper);
          }
          for (const ev of r.adjacency) expect(ev.satisfied).toBe(true);
          let deviation = 0;
          for (const a of r.assignments) {
            const p = packets.find((pp) => pp.id === a.id)!;
            deviation += Math.abs(2 * a.time - (p.timeLower + p.timeUpper));
          }
          got = { missing: r.missingCountTotal, deviation, jitter: r.jitter?.used ?? 0, idSeq: r.order };
        } catch (e) {
          if (!(e instanceof SolveError)) throw e;
          gotError = true;
        }

        const ref = bruteSolve(packets, modulus, countLower, countUpper, minInterval, maxInterval);
        expect(gotError).toBe(ref === null);
        if (got && ref) {
          expect(got.missing).toBe(ref.missing);
          expect(got.deviation).toBe(ref.deviation);
          expect(lexIds(got.idSeq, ref.idSeq)).toBe(0);
        }
      }
    });
  }

  // Same differential contract with the cumulative jitter budget active:
  // the reference enforces the budget DURING enumeration, so agreement proves
  // the solver solves order/counts/times/jitter jointly rather than
  // post-filtering its unconstrained optimum.
  const budgetBatches: [number, number][] = [
    [101, 90],
    [202, 90],
  ];
  for (const [seedStart, count] of budgetBatches) {
    it(`agrees with brute force under a jitter budget on ${count} random instances (seed ${seedStart})`, () => {
      const rnd = makeRng(seedStart);
      for (let iter = 0; iter < count; iter++) {
        const n = 6;
        const modulus = 2 + Math.floor(rnd() * 6);
        const ids = new Set<number>();
        const packets: PacketInput[] = [];
        for (let i = 0; i < n; i++) {
          let id = Math.floor(rnd() * 1000);
          while (ids.has(id)) id = Math.floor(rnd() * 1000);
          ids.add(id);
          const lo = Math.floor(rnd() * 24);
          packets.push({
            id,
            remainder: Math.floor(rnd() * modulus),
            timeLower: lo,
            timeUpper: lo + Math.floor(rnd() * 3),
          });
        }
        const minInterval = 1 + Math.floor(rnd() * 3);
        const maxInterval = minInterval + Math.floor(rnd() * 3);
        const nominalInterval = minInterval + Math.floor(rnd() * (maxInterval - minInterval + 1));
        const countLower = Math.floor(rnd() * 8);
        const countUpper = countLower + (n - 1) + Math.floor(rnd() * 3);

        if (iter % 3 === 0) {
          packets[1].remainder = packets[0].remainder;
          packets[1].timeLower = packets[0].timeLower;
          packets[1].timeUpper = packets[0].timeUpper;
        }

        const baseArgs = [packets, modulus, countLower, countUpper, minInterval, maxInterval] as const;
        const freeRef = bruteSolve(...baseArgs, { nominalInterval, totalJitterBudget: 1_000_000 });
        // Budgets: zero, the unconstrained optimum's jitter, and a midpoint.
        const budgets = [0];
        if (freeRef) budgets.push(freeRef.jitter, Math.floor(freeRef.jitter / 2));
        for (const totalJitterBudget of new Set(budgets)) {
          let got: RefSolution | null = null;
          let gotError = false;
          try {
            const r = solve(...baseArgs, { nominalInterval, totalJitterBudget });
            for (const a of r.assignments) {
              const p = packets.find((pp) => pp.id === a.id)!;
              expect(a.time).toBeGreaterThanOrEqual(p.timeLower);
              expect(a.time).toBeLessThanOrEqual(p.timeUpper);
            }
            for (const ev of r.adjacency) {
              expect(ev.satisfied).toBe(true);
              expect(ev.cumulativeJitter).toBeLessThanOrEqual(totalJitterBudget);
            }
            expect(r.jitter!.used).toBeLessThanOrEqual(totalJitterBudget);
            const deviation = r.assignments.reduce(
              (sum, a) =>
                sum + Math.abs(2 * a.time - (a.timeInterval.lower + a.timeInterval.upper)),
              0,
            );
            got = {
              missing: r.missingCountTotal,
              deviation,
              jitter: r.jitter!.used,
              idSeq: r.order,
            };
          } catch (e) {
            if (!(e instanceof SolveError)) throw e;
            gotError = true;
          }

          const ref = bruteSolve(...baseArgs, { nominalInterval, totalJitterBudget });
          expect(gotError).toBe(ref === null);
          if (got && ref) {
            expect(got.missing).toBe(ref.missing);
            expect(got.deviation).toBe(ref.deviation);
            expect(lexIds(got.idSeq, ref.idSeq)).toBe(0);
            expect(got.jitter).toBe(ref.jitter);
          }
        }
      }
    });
  }
});
