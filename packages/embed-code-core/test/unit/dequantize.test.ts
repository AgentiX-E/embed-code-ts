/**
 * Unit tests for dequantize.ts — Int8 quantization and online GEMM.
 */
import { describe, it, expect } from 'vitest';
import {
  int8Gemm,
  int8GemmBiased,
  int8Linear,
  dequantize,
  quantizePerChannel,
} from '../../src/inference/dequantize';

describe('quantizePerChannel', () => {
  it('quantizes a simple vector and scales back correctly', () => {
    // 2 channels, 2 elements per channel = 4 elements
    // Channel 0: values at indices 0, 2
    // Channel 1: values at indices 1, 3
    const f = new Float32Array([1.0, -0.5, 0.3, 0.8]);
    const n = 2;

    const w = new Int8Array(f.length);
    const scales = quantizePerChannel(f, w, n);

    // Verify scales are positive
    expect(scales[0]).toBeGreaterThan(0);
    expect(scales[1]).toBeGreaterThan(0);

    // Verify round-trip: w_float ≈ f within quantization error
    const recovered = new Float32Array(f.length);
    dequantize(w, scales, recovered, n);

    for (let i = 0; i < f.length; i++) {
      const absErr = Math.abs(f[i]! - recovered[i]!);
      const channel = i % n;
      expect(absErr).toBeLessThanOrEqual(scales[channel]! / 2 + 1e-6);
    }
  });

  it('handles uniform values', () => {
    const f = new Float32Array(8).fill(3.14);
    const n = 2;
    const w = new Int8Array(f.length);
    const scales = quantizePerChannel(f, w, n);

    // Recovered values should be close to 3.14
    const recovered = new Float32Array(f.length);
    dequantize(w, scales, recovered, n);

    for (let i = 0; i < f.length; i++) {
      expect(recovered[i]).toBeCloseTo(3.14, 1);
    }
  });

  it('handles zero values', () => {
    const f = new Float32Array([0, 0, 0, 0]);
    const n = 1;
    const w = new Int8Array(f.length);
    const scales = quantizePerChannel(f, w, n);

    expect(scales[0]).toBeGreaterThan(0); // fallback to 1.0

    const recovered = new Float32Array(f.length);
    dequantize(w, scales, recovered, n);

    for (let i = 0; i < f.length; i++) {
      expect(recovered[i]).toBeCloseTo(0);
    }
  });

  it('clamps values to [-128, 127]', () => {
    const f = new Float32Array([200, -200]);
    const n = 1;
    const w = new Int8Array(f.length);
    quantizePerChannel(f, w, n);

    // Int8 range check
    expect(w[0]).toBeGreaterThanOrEqual(-128);
    expect(w[0]).toBeLessThanOrEqual(127);
    expect(w[1]).toBeGreaterThanOrEqual(-128);
    expect(w[1]).toBeLessThanOrEqual(127);
  });
});

describe('int8Gemm', () => {
  it('multiplies with identity-like int8 weights', () => {
    // Identity matrix in int8: W = Int8Array([127, 0, 0, 127]) reshaped 2×2
    // Row-major: W[0,0]=127, W[0,1]=0, W[1,0]=0, W[1,1]=127
    const w = new Int8Array([64, 0, 0, 64]);
    const scale = new Float32Array([1 / 64, 1 / 64]); // scale so 64 * 1/64 = 1.0
    // A = [[1, 2], [3, 4]]
    const a = new Float32Array([1, 2, 3, 4]);
    const c = new Float32Array(4);

    int8Gemm(a, w, scale, c, 2, 2, 2);

    // Dequantized W = [[1, 0], [0, 1]]
    // A @ I = A
    expect(c[0]).toBeCloseTo(1);
    expect(c[1]).toBeCloseTo(2);
    expect(c[2]).toBeCloseTo(3);
    expect(c[3]).toBeCloseTo(4);
  });

  it('skips zero activations (sparsity optimization)', () => {
    const w = new Int8Array([64, 64, 64, 64]);
    const scale = new Float32Array([1 / 64, 1 / 64]);
    // A with zeros in some positions
    const a = new Float32Array([0, 2, 0, 4]);
    const c = new Float32Array(4);

    int8Gemm(a, w, scale, c, 2, 2, 2);

    // Row 0: 0*1 + 2*1 = 2 in both cols
    expect(c[0]).toBeCloseTo(2);
    expect(c[1]).toBeCloseTo(2);
    // Row 1: 0*1 + 4*1 = 4 in both cols
    expect(c[2]).toBeCloseTo(4);
    expect(c[3]).toBeCloseTo(4);
  });
});

describe('int8GemmBiased', () => {
  it('adds bias to int8 GEMM output', () => {
    const w = new Int8Array([64, 0, 0, 64]);
    const wScale = new Float32Array([1 / 64, 1 / 64]);
    // bias must fit in int8 max 127: 10 * 10 = 100, 20 * 10 = 200 → overflow!
    // Use scale=10 and bias=10,20 → dequantized bias = 10*10, 20*10 = 100, 200 overflows
    // Instead: bias values 10, 20 with scale 1.0
    const bias = new Int8Array([10, 20]);
    const bScale = new Float32Array([1.0, 1.0]);
    const a = new Float32Array([1, 0, 0, 1]);
    const c = new Float32Array(4);

    int8GemmBiased(a, w, wScale, bias, bScale, c, 2, 2, 2);

    // C = I @ I + [[10, 20], [10, 20]] = [[11, 20], [10, 21]]
    expect(c[0]).toBeCloseTo(11);
    expect(c[1]).toBeCloseTo(20);
    expect(c[2]).toBeCloseTo(10);
    expect(c[3]).toBeCloseTo(21);
  });
});

describe('int8Linear', () => {
  it('computes linear layer with transposed weight layout', () => {
    // Simulates nn.Linear(2, 3) with identity weights
    // W stored as [3 × 2] (transposed): [[64, 0], [0, 64], [0, 0]]
    // That's [64, 0, 0, 64, 0, 0] row-major
    const w = new Int8Array([64, 0, 0, 64, 0, 0]);
    const wScale = new Float32Array([1 / 64, 1 / 64, 1.0]);
    const bias = new Int8Array([10, 20, 0]);
    const bScale = new Float32Array([1.0, 1.0, 1.0]);
    const a = new Float32Array([1, 2]); // single input
    const c = new Float32Array(3);

    int8Linear(a, w, wScale, bias, bScale, c, 1, 2, 3);

    // C[0] = 1*1 + 2*0 + 10 = 11
    // C[1] = 1*0 + 2*1 + 20 = 22
    // C[2] = 1*0 + 2*0 + 0 = 0
    expect(c[0]).toBeCloseTo(11);
    expect(c[1]).toBeCloseTo(22);
    expect(c[2]).toBeCloseTo(0);
  });
});

describe('dequantize', () => {
  it('dequantizes int8 tensor correctly', () => {
    const w = new Int8Array([64, -64, 32, -32]);
    const scale = new Float32Array([1 / 64, 1 / 32]); // channel 0: 1/64, channel 1: 1/32
    const out = new Float32Array(4);

    dequantize(w, scale, out, 2);

    // Channel 0 (indices 0, 2): 64 * 1/64 = 1, 32 * 1/64 = 0.5
    // Channel 1 (indices 1, 3): -64 * 1/32 = -2, -32 * 1/32 = -1
    expect(out[0]).toBeCloseTo(1);
    expect(out[1]).toBeCloseTo(-2);
    expect(out[2]).toBeCloseTo(0.5);
    expect(out[3]).toBeCloseTo(-1);
  });
});
