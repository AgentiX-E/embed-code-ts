/**
 * Unit tests for activations.ts — GELU, Softmax, and batch variants.
 */
import { describe, it, expect } from 'vitest';
import { gelu, softmax, softmaxBatch, attentionSoftmax } from '../../src/inference/activations';

describe('gelu', () => {
  it('gelu(0) = 0', () => {
    const x = new Float32Array([0]);
    gelu(x);
    expect(x[0]).toBeCloseTo(0);
  });

  it('gelu approaches identity for large positive inputs', () => {
    const x = new Float32Array([10]);
    gelu(x);
    // GELU(10) ≈ 10 for large x
    expect(x[0]).toBeCloseTo(10, 1); // within 10%
  });

  it('gelu approaches 0 for large negative inputs', () => {
    const x = new Float32Array([-10]);
    gelu(x);
    expect(x[0]).toBeCloseTo(0, 1);
  });

  it('gelu is monotonic for non-negative inputs', () => {
    const x = new Float32Array([0, 0.5, 1, 2, 3, 5]);
    gelu(x);
    for (let i = 1; i < x.length; i++) {
      expect(x[i]).toBeGreaterThanOrEqual(x[i - 1]!);
    }
  });

  it('gelu negative inputs produce values ≤ 0', () => {
    const x = new Float32Array([-5, -3, -1, -0.1]);
    gelu(x);
    for (let i = 0; i < x.length; i++) {
      expect(x[i]).toBeLessThanOrEqual(0);
    }
  });

  it('matches PyTorch approximate tanh GELU reference values', () => {
    // Reference values computed with:
    //   torch.nn.functional.gelu(x, approximate='tanh')
    const inputs = new Float32Array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]);
    const expected = [
      -0.0036372468, -0.04540229, -0.158808, -0.1542857, 0.0, 0.3457143, 0.841192, 1.9545977,
      2.9963627,
    ];

    gelu(inputs);
    for (let i = 0; i < inputs.length; i++) {
      expect(inputs[i]).toBeCloseTo(expected[i]!, 4);
    }
  });

  it('handles array of all zeros', () => {
    const x = new Float32Array(100).fill(0);
    gelu(x);
    for (let i = 0; i < x.length; i++) {
      expect(x[i]).toBeCloseTo(0);
    }
  });
});

describe('softmax', () => {
  it('sums to 1', () => {
    const x = new Float32Array([1, 2, 3, 4]);
    softmax(x);
    expect(x[0] + x[1] + x[2] + x[3]).toBeCloseTo(1.0);
  });

  it('largest input gets largest probability', () => {
    const x = new Float32Array([1, 5, 2, 3]);
    softmax(x);
    expect(x[1]).toBeGreaterThan(x[0]!);
    expect(x[1]).toBeGreaterThan(x[2]!);
    expect(x[1]).toBeGreaterThan(x[3]!);
  });

  it('equal inputs produce equal outputs', () => {
    const x = new Float32Array([2, 2, 2, 2]);
    softmax(x);
    for (let i = 0; i < x.length; i++) {
      expect(x[i]).toBeCloseTo(0.25);
    }
  });

  it('handles very large values (numerical stability)', () => {
    const x = new Float32Array([1000, 1000, 1000]);
    softmax(x);
    for (let i = 0; i < x.length; i++) {
      expect(x[i]).toBeCloseTo(1 / 3);
    }
  });

  it('handles very small values (all negative large)', () => {
    const x = new Float32Array([-1000, -1000, -1000]);
    softmax(x);
    for (let i = 0; i < x.length; i++) {
      expect(x[i]).toBeCloseTo(1 / 3);
    }
  });
});

describe('softmaxBatch', () => {
  it('applies softmax independently to each row', () => {
    const x = new Float32Array([
      1,
      2,
      3,
      0,
      0,
      0, // row 0
    ]);
    softmaxBatch(x, 1, 3);

    const sum0 = x[0]! + x[1]! + x[2]!;
    expect(sum0).toBeCloseTo(1.0);
  });

  it('handles batch of 2 correctly', () => {
    const x = new Float32Array([
      1,
      2, // row 0
      3,
      4, // row 1
    ]);
    softmaxBatch(x, 2, 2);

    // Row 0 sums to 1
    expect(x[0]! + x[1]!).toBeCloseTo(1.0);
    // Row 1 sums to 1
    expect(x[2]! + x[3]!).toBeCloseTo(1.0);
    // Row 1's second element should be larger
    expect(x[3]).toBeGreaterThan(x[2]!);
  });

  it('handles all-zeros row without NaN', () => {
    const x = new Float32Array([0, 0, 7, 8]);
    softmaxBatch(x, 2, 2);
    // Row 0: all zeros → each should be 0.5
    expect(x[0]! + x[1]!).toBeCloseTo(1.0);
    expect(Number.isFinite(x[0])).toBe(true);
    // Row 1: [7, 8] → 8 gets larger probability
    expect(x[2]! + x[3]!).toBeCloseTo(1.0);
    expect(x[3]).toBeGreaterThan(x[2]!);
  });
});

describe('attentionSoftmax', () => {
  it('applies softmax correctly for 1 head, 1 batch, L=4', () => {
    // [B=1, heads=1, L=4, L=4]
    const scores = new Float32Array(16);
    for (let i = 0; i < 16; i++) scores[i] = (i % 4) * 0.5;

    attentionSoftmax(scores, 1, 1, 4);

    // Each of the 4 rows should sum to 1
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let c = 0; c < 4; c++) {
        sum += scores[r * 4 + c]!;
      }
      expect(sum).toBeCloseTo(1.0);
    }
  });

  it('handles 2 heads, batch=1, L=3', () => {
    const scores = new Float32Array(1 * 2 * 3 * 3);
    for (let i = 0; i < scores.length; i++) scores[i] = Math.sin(i * 0.5);

    attentionSoftmax(scores, 1, 2, 3);

    for (let r = 0; r < 6; r++) {
      let sum = 0;
      for (let c = 0; c < 3; c++) sum += scores[r * 3 + c]!;
      expect(sum).toBeCloseTo(1.0);
    }
  });

  it('handles extreme values without NaN', () => {
    const scores = new Float32Array(4).fill(-1e6);
    attentionSoftmax(scores, 1, 1, 2);
    for (let i = 0; i < 4; i++) {
      expect(Number.isFinite(scores[i])).toBe(true);
    }
  });
});
