/**
 * Unit tests for matmul.ts — matrix multiplication operations.
 */
import { describe, it, expect } from 'vitest';
import { matmul, matmulBiased, matvec, hadamardAccumulate, transposeSquare } from '../../src/inference/matmul';

describe('matmul', () => {
  it('multiplies 2×3 by 3×2 = 2×2', () => {
    // A = [[1, 2, 3], [4, 5, 6]]
    const a = new Float32Array([1, 2, 3, 4, 5, 6]);
    // B = [[7, 8], [9, 10], [11, 12]]
    const b = new Float32Array([7, 8, 9, 10, 11, 12]);
    const c = new Float32Array(4).fill(0);

    matmul(a, b, c, 2, 3, 2);

    // C[0,0] = 1*7 + 2*9 + 3*11 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 154
    expect(c[0]).toBeCloseTo(58);
    expect(c[1]).toBeCloseTo(64);
    expect(c[2]).toBeCloseTo(139);
    expect(c[3]).toBeCloseTo(154);
  });

  it('handles identity matrix multiply', () => {
    // A = I_3
    const a = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
    // B = [[1, 2], [3, 4], [5, 6]]
    const b = new Float32Array([1, 2, 3, 4, 5, 6]);
    const c = new Float32Array(6).fill(0);

    matmul(a, b, c, 3, 3, 2);

    // C = B
    expect(c[0]).toBeCloseTo(1);
    expect(c[1]).toBeCloseTo(2);
    expect(c[2]).toBeCloseTo(3);
    expect(c[3]).toBeCloseTo(4);
    expect(c[4]).toBeCloseTo(5);
    expect(c[5]).toBeCloseTo(6);
  });

  it('accumulates into pre-filled output (C += A @ B)', () => {
    const a = new Float32Array([1, 2, 3, 4]);
    const b = new Float32Array([5, 6, 7, 8]);
    const c = new Float32Array([10, 10, 10, 10]); // pre-filled

    matmul(a, b, c, 2, 2, 2);

    // A @ B = [[19, 22], [43, 50]]
    // C = [[29, 32], [53, 60]]
    expect(c[0]).toBeCloseTo(29);
    expect(c[1]).toBeCloseTo(32);
    expect(c[2]).toBeCloseTo(53);
    expect(c[3]).toBeCloseTo(60);
  });

  it('multiplies larger matrices (BERT hidden size × intermediate)', () => {
    // Simulate a slice of BERT FFN: [1, 768] @ [768, 3072]
    const m = 1,
      k = 96,
      n = 128; // smaller for test speed
    const a = new Float32Array(m * k);
    const b = new Float32Array(k * n);
    const c = new Float32Array(m * n);

    for (let i = 0; i < a.length; i++) a[i] = Math.sin(i * 0.01) * 0.1;
    for (let i = 0; i < b.length; i++) b[i] = Math.cos(i * 0.01) * 0.1;

    matmul(a, b, c, m, k, n);

    // Output should be finite
    for (let i = 0; i < c.length; i++) {
      expect(Number.isFinite(c[i])).toBe(true);
    }
  });

  it('handles zero matrix multiply', () => {
    const a = new Float32Array([0, 0, 0, 0]);
    const b = new Float32Array([1, 2, 3, 4]);
    const c = new Float32Array(4).fill(0); // zero-init

    matmul(a, b, c, 2, 2, 2);

    expect(c[0]).toBeCloseTo(0);
    expect(c[1]).toBeCloseTo(0);
    expect(c[2]).toBeCloseTo(0);
    expect(c[3]).toBeCloseTo(0);
  });
});

describe('matmulBiased', () => {
  it('adds bias after matrix multiply', () => {
    const a = new Float32Array([1, 0, 0, 1]); // I_2
    const b = new Float32Array([2, 3, 4, 5]);
    const bias = new Float32Array([10, 20]);
    const c = new Float32Array(4);

    matmulBiased(a, b, bias, c, 2, 2, 2);

    // A@B = B = [[2,3],[4,5]]
    // + bias [[10,20],[10,20]] = [[12,23],[14,25]]
    expect(c[0]).toBeCloseTo(12);
    expect(c[1]).toBeCloseTo(23);
    expect(c[2]).toBeCloseTo(14);
    expect(c[3]).toBeCloseTo(25);
  });
});

describe('matvec', () => {
  it('computes matrix-vector product', () => {
    const a = new Float32Array([1, 2, 3, 4, 5, 6]); // 2×3
    const x = new Float32Array([7, 8, 9]);
    const y = new Float32Array(2);

    matvec(a, x, y, 2, 3);

    // y[0] = 1*7 + 2*8 + 3*9 = 50
    // y[1] = 4*7 + 5*8 + 6*9 = 122
    expect(y[0]).toBeCloseTo(50);
    expect(y[1]).toBeCloseTo(122);
  });
});

describe('hadamardAccumulate', () => {
  it('accumulates element-wise products', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([4, 5, 6]);
    const c = new Float32Array([10, 20, 30]);

    hadamardAccumulate(a, b, c, 3);

    expect(c[0]).toBeCloseTo(14); // 10 + 1*4
    expect(c[1]).toBeCloseTo(30); // 20 + 2*5
    expect(c[2]).toBeCloseTo(48); // 30 + 3*6
  });
});

describe('transposeSquare', () => {
  it('transposes a 3x3 matrix', () => {
    // [1, 2, 3]
    // [4, 5, 6]
    // [7, 8, 9]
    const m = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const out = new Float32Array(9);
    transposeSquare(m, 3, out);

    // Expected transpose
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(4);
    expect(out[2]).toBe(7);
    expect(out[3]).toBe(2);
    expect(out[4]).toBe(5);
    expect(out[5]).toBe(8);
    expect(out[6]).toBe(3);
    expect(out[7]).toBe(6);
    expect(out[8]).toBe(9);
  });
});
