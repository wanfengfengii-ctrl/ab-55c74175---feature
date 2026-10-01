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

  it('is fully backward compatible when no budget parameters are sent', () => {
    const res = handleSolve(sampleRequest);
    if (res.status !== 'ok') throw new Error('expected ok');
    expect(res.data.jitter).toBeUndefined();
    for (const ev of res.data.adjacency) {
      expect(ev.jitter).toBeUndefined();
      expect(ev.nominalTimeGap).toBeUndefined();
      expect(ev.cumulativeJitter).toBeUndefined();
    }
  });

  it('enforces the jitter budget and reports per-edge jitter accounting', () => {
    const res = handleSolve({ ...sampleRequest, nominalInterval: 10, totalJitterBudget: 0 });
    expect(res.status).toBe('ok');
    if (res.status !== 'ok') throw new Error('expected ok');
    expect(res.data.jitter).toEqual({
      nominalInterval: 10,
      budget: 0,
      used: 0,
      remaining: 0,
      exhausted: true,
    });
    let cumulative = 0;
    for (const ev of res.data.adjacency) {
      expect(ev.nominalTimeGap).toBe(ev.countGap * 10);
      expect(typeof ev.jitter).toBe('number');
      cumulative += ev.jitter!;
      expect(ev.cumulativeJitter).toBe(cumulative);
    }
    expect(cumulative).toBe(0);
  });

  it('returns INVALID_REQUEST for half-specified or out-of-range budget parameters', () => {
    const half = handleSolve({ ...sampleRequest, nominalInterval: 10 });
    expect(half.status).toBe('error');
    if (half.status !== 'error') throw new Error('expected error');
    expect(half.error.code).toBe('INVALID_REQUEST');

    const outOfRange = handleSolve({ ...sampleRequest, nominalInterval: 12, totalJitterBudget: 3 });
    expect(outOfRange.status).toBe('error');
    if (outOfRange.status !== 'error') throw new Error('expected error');
    expect(outOfRange.error.code).toBe('INVALID_REQUEST');
  });

  it('returns NO_CONSISTENT_INTERPRETATION with jitter blocker evidence when only the budget fails', () => {
    const packets = [
      { id: 0, remainder: 0, timeLower: 0, timeUpper: 0 },
      { id: 1, remainder: 1, timeLower: 3, timeUpper: 3 },
      { id: 2, remainder: 2, timeLower: 6, timeUpper: 6 },
      { id: 3, remainder: 3, timeLower: 10, timeUpper: 10 },
      { id: 4, remainder: 4, timeLower: 12, timeUpper: 12 },
      { id: 5, remainder: 5, timeLower: 15, timeUpper: 15 },
    ];
    const res = handleSolve({
      packets,
      modulus: 10,
      countLower: 0,
      countUpper: 100,
      minInterval: 1,
      maxInterval: 100,
      nominalInterval: 3,
      totalJitterBudget: 1,
    });
    expect(res.status).toBe('error');
    if (res.status !== 'error') throw new Error('expected error');
    expect(res.error.code).toBe('NO_CONSISTENT_INTERPRETATION');
    const evidence = res.error.evidence as {
      detail: { cause: string; jitterBudget: Record<string, unknown> };
    };
    expect(evidence.detail.cause).toBe('JITTER_BUDGET');
    expect(evidence.detail.jitterBudget.budget).toBe(1);
    expect(evidence.detail.jitterBudget.used).toBe(1);
    expect(evidence.detail.jitterBudget.minAdditionalJitter).toBe(1);
  });
});
