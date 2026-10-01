import { describe, expect, it } from 'vitest';
import { handleSolve } from '../src/api/handler.js';
import { sampleRequest, sampleExpected } from './fixtures/sample.js';

describe('handleSolve', () => {
  it('returns the recovered order for the canonical sample', () => {
    const res = handleSolve(sampleRequest);
    expect(res.status).toBe('ok');
    if (res.status !== 'ok') throw new Error('expected ok');
    expect(res.data.order).toEqual(sampleExpected.order);
    expect(res.data.missingCountTotal).toBe(sampleExpected.missingTotal);
    for (const ev of res.data.adjacency) expect(ev.satisfied).toBe(true);
  });

  it('maps validation failures to INVALID_REQUEST errors', () => {
    const res = handleSolve({ ...sampleRequest, modulus: 1 });
    expect(res.status).toBe('error');
    if (res.status !== 'error') throw new Error('expected error');
    expect(res.error.code).toBe('INVALID_REQUEST');
  });

  it('maps infeasible instances to the stable business error code with evidence', () => {
    const packets = Array.from({ length: 8 }, (_, i) => ({
      id: i,
      remainder: i % 5,
      timeLower: i === 3 ? 9000 : 0,
      timeUpper: i === 3 ? 9001 : 100,
    }));
    const res = handleSolve({ packets, modulus: 5, countLower: 0, countUpper: 200, minInterval: 1, maxInterval: 20 });
    expect(res.status).toBe('error');
    if (res.status !== 'error') throw new Error('expected error');
    expect(res.error.code).toBe('NO_CONSISTENT_INTERPRETATION');
    expect(res.error.evidence).toBeTruthy();
    expect(['seed', 'extension']).toContain((res.error.evidence as { stage: string }).stage);
  });

  it('ignores download order: shuffling input packets yields the same solution', () => {
    const shuffled = {
      ...sampleRequest,
      packets: [...sampleRequest.packets].reverse(),
    };
    const a = handleSolve(sampleRequest);
    const b = handleSolve(shuffled);
    expect(a).toEqual(b);
  });

  it('runs the cumulative-jitter variant and echoes budget accounting', () => {
    const res = handleSolve({ ...sampleRequest, nominalInterval: 10, totalJitterBudget: 100 });
    expect(res.status).toBe('ok');
    if (res.status !== 'ok') throw new Error('expected ok');
    expect(res.data.jitterBudget).toBeTruthy();
    const { nominalInterval, budget, used, remaining, exhausted } = res.data.jitterBudget!;
    expect(nominalInterval).toBe(10);
    expect(budget).toBe(100);
    let running = 0;
    for (const ev of res.data.adjacency) {
      expect(ev.nominalTimeGap).toBe(ev.countGap * 10);
      expect(ev.jitter).toBe(Math.abs(ev.timeGap - ev.nominalTimeGap!));
      running += ev.jitter!;
      expect(ev.cumulativeJitter).toBe(running);
    }
    expect(used).toBe(running);
    expect(remaining).toBe(100 - used);
    expect(exhausted).toBe(used === 100);
  });

  it('rejects an out-of-range nominalInterval as INVALID_REQUEST', () => {
    const res = handleSolve({ ...sampleRequest, nominalInterval: 100, totalJitterBudget: 0 });
    expect(res.status).toBe('error');
    if (res.status !== 'error') throw new Error('expected error');
    expect(res.error.code).toBe('INVALID_REQUEST');
  });

  it('maps a budget-only infeasible instance to NO_CONSISTENT_INTERPRETATION with jitter detail', () => {
    // Canonical sample has zero per-edge jitter at nominal=10 (times are exact
    // multiples of 10 * counts), so a budget of 0 succeeds; force a nominal
    // that the recovered chain cannot meet.
    const res = handleSolve({ ...sampleRequest, nominalInterval: 9, totalJitterBudget: 0 });
    expect(res.status).toBe('error');
    if (res.status !== 'error') throw new Error('expected error');
    expect(res.error.code).toBe('NO_CONSISTENT_INTERPRETATION');
    const evidence = res.error.evidence as {
      detail?: { cause?: string; jitter?: { budget: number; used: number; minimumAdditional: number } };
    };
    expect(evidence.detail?.jitter).toBeTruthy();
    expect(evidence.detail!.jitter!.budget).toBe(0);
  });
});
