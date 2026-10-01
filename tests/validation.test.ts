import { describe, expect, it } from 'vitest';
import { validateRequest } from '../src/core/validation.js';
import { SolveError } from '../src/core/types.js';
import { sampleRequest } from './fixtures/sample.js';

const valid = (): typeof sampleRequest => JSON.parse(JSON.stringify(sampleRequest));

describe('validateRequest', () => {
  it('accepts the canonical sample', () => {
    expect(() => validateRequest(valid())).not.toThrow();
  });

  it.each([
    ['not an object', null],
    ['an array', []],
    ['missing modulus', { ...valid(), modulus: undefined }],
    ['non-integer interval', { ...valid(), minInterval: 1.5 }],
  ])('rejects %s', (_label, body) => {
    expect(() => validateRequest(body)).toThrow(SolveError);
  });

  it('rejects modulus below 2', () => {
    const b = valid();
    b.modulus = 1;
    try {
      validateRequest(b);
      throw new Error('should have thrown');
    } catch (e) {
      expect((e as SolveError).code).toBe('INVALID_REQUEST');
    }
  });

  it('rejects inverted count window', () => {
    const b = valid();
    b.countLower = 100;
    b.countUpper = 0;
    expect(() => validateRequest(b)).toThrow(SolveError);
  });

  it('rejects negative or inverted sample interval', () => {
    const b1 = valid();
    b1.minInterval = 0;
    expect(() => validateRequest(b1)).toThrow(SolveError);
    const b2 = valid();
    b2.maxInterval = 1;
    b2.minInterval = 5;
    expect(() => validateRequest(b2)).toThrow(SolveError);
  });

  it('rejects packet counts outside 6..14', () => {
    const b = valid();
    b.packets = b.packets.slice(0, 5);
    expect(() => validateRequest(b)).toThrow(SolveError);
  });

  it('rejects duplicate packet ids', () => {
    const b = valid();
    b.packets[1] = { ...b.packets[0] };
    expect(() => validateRequest(b)).toThrow(SolveError);
  });

  it('rejects remainder outside [0, modulus)', () => {
    const b = valid();
    b.packets[0].remainder = 10;
    expect(() => validateRequest(b)).toThrow(SolveError);
  });

  it('rejects inverted time intervals', () => {
    const b = valid();
    b.packets[0].timeLower = 100;
    b.packets[0].timeUpper = 0;
    expect(() => validateRequest(b)).toThrow(SolveError);
  });

  it('rejects excessively large count search windows', () => {
    const b = valid();
    b.countUpper = b.countLower + 2_000_000;
    expect(() => validateRequest(b)).toThrow(SolveError);
  });

  describe('jitter budget parameters', () => {
    it('accepts both parameters with nominalInterval inside the sampling range', () => {
      const b = valid();
      b.nominalInterval = 10;
      b.totalJitterBudget = 0;
      const req = validateRequest(b);
      expect(req.nominalInterval).toBe(10);
      expect(req.totalJitterBudget).toBe(0);
    });

    it('leaves both parameters undefined when absent (full backward compatibility)', () => {
      const req = validateRequest(valid());
      expect(req.nominalInterval).toBeUndefined();
      expect(req.totalJitterBudget).toBeUndefined();
    });

    it.each([
      ['nominalInterval only', { nominalInterval: 10 }],
      ['totalJitterBudget only', { totalJitterBudget: 5 }],
    ])('rejects %s', (_label, extra) => {
      const b = { ...valid(), ...extra };
      try {
        validateRequest(b);
        throw new Error('should have thrown');
      } catch (e) {
        expect((e as SolveError).code).toBe('INVALID_REQUEST');
      }
    });

    it('rejects non-integer nominalInterval or totalJitterBudget', () => {
      expect(() => validateRequest({ ...valid(), nominalInterval: 9.5, totalJitterBudget: 5 })).toThrow(SolveError);
      expect(() => validateRequest({ ...valid(), nominalInterval: 10, totalJitterBudget: -1 })).toThrow(SolveError);
    });

    it('rejects nominalInterval outside [minInterval, maxInterval]', () => {
      const below = valid();
      below.nominalInterval = 8;
      below.totalJitterBudget = 5;
      expect(() => validateRequest(below)).toThrow(SolveError);
      const above = valid();
      above.nominalInterval = 12;
      above.totalJitterBudget = 5;
      expect(() => validateRequest(above)).toThrow(SolveError);
    });

    it('accepts nominalInterval exactly at either sampling bound', () => {
      const lo = valid();
      lo.nominalInterval = 9;
      lo.totalJitterBudget = 0;
      expect(() => validateRequest(lo)).not.toThrow();
      const hi = valid();
      hi.nominalInterval = 11;
      hi.totalJitterBudget = 0;
      expect(() => validateRequest(hi)).not.toThrow();
    });
  });
});
