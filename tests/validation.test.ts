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
});

describe('validateRequest: nominalInterval / totalJitterBudget', () => {
  it('accepts the pair together and parses both fields', () => {
    const b = valid();
    b.nominalInterval = 10;
    b.totalJitterBudget = 5;
    const req = validateRequest(b);
    expect(req.nominalInterval).toBe(10);
    expect(req.totalJitterBudget).toBe(5);
  });

  it('leaves both fields undefined when neither is supplied', () => {
    const req = validateRequest(valid());
    expect(req.nominalInterval).toBeUndefined();
    expect(req.totalJitterBudget).toBeUndefined();
  });

  it('rejects nominalInterval without totalJitterBudget (and vice versa)', () => {
    const onlyNominal = valid();
    onlyNominal.nominalInterval = 10;
    const onlyBudget = valid();
    onlyBudget.totalJitterBudget = 5;
    for (const b of [onlyNominal, onlyBudget]) {
      try {
        validateRequest(b);
        throw new Error('should have thrown');
      } catch (e) {
        expect((e as SolveError).code).toBe('INVALID_REQUEST');
      }
    }
  });

  it.each([
    ['below minInterval', 8, 5],
    ['above maxInterval', 12, 5],
  ])('rejects nominalInterval %s', (_label, nominal, budget) => {
    const b = valid();
    b.nominalInterval = nominal;
    b.totalJitterBudget = budget;
    expect(() => validateRequest(b)).toThrow(SolveError);
  });

  it('accepts nominalInterval exactly on either interval bound', () => {
    const low = valid();
    low.nominalInterval = 9;
    low.totalJitterBudget = 0;
    expect(validateRequest(low).nominalInterval).toBe(9);
    const high = valid();
    high.nominalInterval = 11;
    high.totalJitterBudget = 0;
    expect(validateRequest(high).nominalInterval).toBe(11);
  });

  it('rejects a negative budget and non-integer values', () => {
    const neg = valid();
    neg.nominalInterval = 10;
    neg.totalJitterBudget = -1;
    expect(() => validateRequest(neg)).toThrow(SolveError);
    const nonInt = valid();
    nonInt.nominalInterval = 10;
    nonInt.totalJitterBudget = 1.5;
    expect(() => validateRequest(nonInt)).toThrow(SolveError);
  });

  it('accepts a zero (fully tight) budget', () => {
    const b = valid();
    b.nominalInterval = 10;
    b.totalJitterBudget = 0;
    expect(validateRequest(b).totalJitterBudget).toBe(0);
  });
});
