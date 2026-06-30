/**
 * Unit tests for the pooling module.
 */
import { describe, it, expect } from 'vitest';
import { poolEmbeddings, normalizeEmbeddings, cosineSimilarity } from '../../src/pooling';

describe('poolEmbeddings', () => {
  it('last-token pooling picks the last non-padding token', () => {
    const h = new Float32Array([
      0.1,
      0.2, // token 0
      0.3,
      0.4, // token 1
      0.5,
      0.6, // token 2 (last non-padding)
    ]);
    const mask = new Int32Array([1, 1, 1]);

    const result = poolEmbeddings(h, mask, 1, 3, 2, 'last_token');
    expect(result[0]).toBeCloseTo(0.5);
    expect(result[1]).toBeCloseTo(0.6);
  });

  it('last-token pooling skips padding tokens', () => {
    const h = new Float32Array([
      0.1,
      0.2, // token 0
      0.3,
      0.4, // token 1 (last real)
      0.5,
      0.6, // padding
    ]);
    const mask = new Int32Array([1, 1, 0]);

    const result = poolEmbeddings(h, mask, 1, 3, 2, 'last_token');
    expect(result[0]).toBeCloseTo(0.3);
    expect(result[1]).toBeCloseTo(0.4);
  });

  it('mean pooling computes average over non-padding tokens', () => {
    const h = new Float32Array([
      1.0,
      2.0, // token 0
      3.0,
      4.0, // token 1
      5.0,
      6.0, // padding
    ]);
    const mask = new Int32Array([1, 1, 0]);

    const result = poolEmbeddings(h, mask, 1, 3, 2, 'mean');
    expect(result[0]).toBeCloseTo(2.0);
    expect(result[1]).toBeCloseTo(3.0);
  });

  it('mean pooling handles single token', () => {
    const h = new Float32Array([7.0, 8.0]);
    const mask = new Int32Array([1]);

    const result = poolEmbeddings(h, mask, 1, 1, 2, 'mean');
    expect(result[0]).toBeCloseTo(7.0);
    expect(result[1]).toBeCloseTo(8.0);
  });

  it('CLS pooling takes first token', () => {
    const h = new Float32Array([
      0.1,
      0.2, // token 0 (CLS)
      0.3,
      0.4, // token 1
      0.5,
      0.6, // token 2
    ]);
    const mask = new Int32Array([1, 1, 1]);

    const result = poolEmbeddings(h, mask, 1, 3, 2, 'cls');
    expect(result[0]).toBeCloseTo(0.1);
    expect(result[1]).toBeCloseTo(0.2);
  });

  it('handles batch of 2 with last-token pooling', () => {
    const h = new Float32Array([
      // batch 0: 2 tokens dim=3
      1, 2, 3, 4, 5, 6,
      // batch 1: 2 tokens dim=3
      7, 8, 9, 10, 11, 12,
    ]);
    const mask = new Int32Array([1, 1, 1, 1]);

    const result = poolEmbeddings(h, mask, 2, 2, 3, 'last_token');
    // batch 0 last token: [4, 5, 6]
    expect(result[0]).toBeCloseTo(4);
    expect(result[1]).toBeCloseTo(5);
    expect(result[2]).toBeCloseTo(6);
    // batch 1 last token: [10, 11, 12]
    expect(result[3]).toBeCloseTo(10);
    expect(result[4]).toBeCloseTo(11);
    expect(result[5]).toBeCloseTo(12);
  });

  it('last-token pooling with all-padding tokens falls back to last position', () => {
    const h = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    const mask = new Int32Array([0, 0, 0]); // all padding
    const result = poolEmbeddings(h, mask, 1, 3, 2, 'last_token');
    // Falls back to index S-1 = 2
    expect(result[0]).toBeCloseTo(0.5);
    expect(result[1]).toBeCloseTo(0.6);
  });
});

describe('normalizeEmbeddings', () => {
  it('normalizes to unit L2 norm', () => {
    const e = new Float32Array([3.0, 4.0]);
    normalizeEmbeddings(e, 1, 2);
    expect(e[0]).toBeCloseTo(0.6);
    expect(e[1]).toBeCloseTo(0.8);
  });

  it('handles zero norm gracefully', () => {
    const e = new Float32Array([0.0, 0.0]);
    normalizeEmbeddings(e, 1, 2);
    expect(e[0]).toBe(0);
    expect(e[1]).toBe(0);
  });

  it('handles batch normalization', () => {
    const e = new Float32Array([3, 4, 0, 5]);
    normalizeEmbeddings(e, 2, 2);
    expect(e[0]).toBeCloseTo(0.6);
    expect(e[1]).toBeCloseTo(0.8);
    expect(e[2]).toBe(0);
    expect(e[3]).toBe(1);
  });
});

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    expect(cosineSimilarity(new Float32Array([1, 2, 3]), new Float32Array([1, 2, 3]))).toBeCloseTo(
      1.0,
    );
  });

  it('returns 0 for orthogonal vectors', () => {
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([0, 1]))).toBeCloseTo(0.0);
  });

  it('returns -1 for opposite vectors', () => {
    expect(cosineSimilarity(new Float32Array([1, 0]), new Float32Array([-1, 0]))).toBeCloseTo(-1.0);
  });

  it('throws on dimension mismatch', () => {
    expect(() => cosineSimilarity(new Float32Array([1, 2]), new Float32Array([1, 2, 3]))).toThrow();
  });

  it('returns 0 for zero vectors', () => {
    expect(cosineSimilarity(new Float32Array([0, 0]), new Float32Array([1, 2]))).toBeCloseTo(0.0);
    expect(cosineSimilarity(new Float32Array([1, 2]), new Float32Array([0, 0]))).toBeCloseTo(0.0);
    expect(cosineSimilarity(new Float32Array([0, 0]), new Float32Array([0, 0]))).toBeCloseTo(0.0);
  });
});
