/**
 * Unit tests for pooler + normalizer.
 */
import { describe, it, expect } from 'vitest';
import { meanPool, clsPool, lastTokenPool } from '../../src/pooler';
import { l2Normalize, cosineSimilarity } from '../../src/normalizer';

describe('meanPool', () => {
  it('averages over non-padding tokens', () => {
    const h = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const mask = new Int32Array([1, 1, 0]);
    const r = meanPool(h, mask, 1, 3, 2);
    expect(r[0]).toBeCloseTo(2.0);
    expect(r[1]).toBeCloseTo(3.0);
  });

  it('handles single token', () => {
    const h = new Float32Array([7.0, 8.0]);
    const mask = new Int32Array([1]);
    const r = meanPool(h, mask, 1, 1, 2);
    expect(r[0]).toBeCloseTo(7.0);
    expect(r[1]).toBeCloseTo(8.0);
  });

  it('all-padding returns zeros', () => {
    const h = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const mask = new Int32Array([0, 0]);
    const r = meanPool(h, mask, 1, 2, 2);
    expect(r[0]).toBe(0);
    expect(r[1]).toBe(0);
  });

  it('handles batch of 2', () => {
    const h = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const mask = new Int32Array([1, 1, 1, 0]);
    const r = meanPool(h, mask, 2, 2, 2);
    expect(r[0]).toBeCloseTo(2);
    expect(r[1]).toBeCloseTo(3);
    expect(r[2]).toBeCloseTo(5);
    expect(r[3]).toBeCloseTo(6);
  });
});

describe('clsPool', () => {
  it('extracts first token (CLS)', () => {
    const h = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const r = clsPool(h, 1, 3, 2);
    expect(r[0]).toBe(1.0);
    expect(r[1]).toBe(2.0);
  });

  it('handles batch', () => {
    const h = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const r = clsPool(h, 2, 2, 2);
    expect(r[0]).toBe(1);
    expect(r[1]).toBe(2);
    expect(r[2]).toBe(5);
    expect(r[3]).toBe(6);
  });
});

describe('lastTokenPool', () => {
  it('extracts last non-padding token', () => {
    const h = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const mask = new Int32Array([1, 1, 0]);
    const r = lastTokenPool(h, mask, 1, 3, 2);
    expect(r[0]).toBe(3.0);
    expect(r[1]).toBe(4.0);
  });

  it('falls back to position 0 when all padding', () => {
    const h = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const mask = new Int32Array([0, 0]);
    const r = lastTokenPool(h, mask, 1, 2, 2);
    expect(r[0]).toBe(1.0);
    expect(r[1]).toBe(2.0);
  });

  it('handles batch', () => {
    const h = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const mask = new Int32Array([1, 1, 1, 0]);
    const r = lastTokenPool(h, mask, 2, 2, 2);
    expect(r[0]).toBe(3);
    expect(r[1]).toBe(4);
    expect(r[2]).toBe(5);
    expect(r[3]).toBe(6);
  });
});

describe('l2Normalize', () => {
  it('normalizes to unit norm', () => {
    const e = new Float32Array([3.0, 4.0]);
    l2Normalize(e, 1, 2);
    expect(e[0]).toBeCloseTo(0.6);
    expect(e[1]).toBeCloseTo(0.8);
  });

  it('handles zero norm', () => {
    const e = new Float32Array([0.0, 0.0]);
    l2Normalize(e, 1, 2);
    expect(e[0]).toBe(0);
    expect(e[1]).toBe(0);
  });

  it('handles batch', () => {
    const e = new Float32Array([3, 4, 0, 5]);
    l2Normalize(e, 2, 2);
    expect(e[0]).toBeCloseTo(0.6);
    expect(e[1]).toBeCloseTo(0.8);
    expect(e[2]).toBe(0);
    expect(e[3]).toBe(1);
  });
});

describe('cosineSimilarity', () => {
  it('returns 1 for identical', () => {
    expect(cosineSimilarity([1, 2, 3], [1, 2, 3])).toBeCloseTo(1.0);
  });

  it('returns 0 for orthogonal', () => {
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0.0);
  });

  it('throws on dimension mismatch', () => {
    expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow(/Dimension mismatch/i);
  });

  it('zero vectors produce 0', () => {
    expect(cosineSimilarity([0, 0], [0, 0])).toBe(0);
  });

  it('handles Float32Array inputs', () => {
    const a = new Float32Array([1, 0]);
    const b = new Float32Array([0, 1]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0.0);
  });
});
