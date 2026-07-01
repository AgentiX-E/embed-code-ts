/**
 * Activation functions used in BERT-base inference.
 *
 *   • GELU  — Gaussian Error Linear Unit (approximate tanh form)
 *   • Softmax — numerically stable softmax with max-subtraction
 *
 * All functions operate in-place on Float32Array for zero-allocation.
 */

/**
 * GELU activation (approximate, matches PyTorch `approximate='tanh'`).
 *
 *   GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
 *
 * Operates in-place on the provided Float32Array.
 */
export function gelu(x: Float32Array): void {
  // Pre-computed constants
  const SQRT_2_OVER_PI = 0.7978845608028654; // sqrt(2 / π)
  const COEFF = 0.044715;

  for (let i = 0; i < x.length; i++) {
    const v = x[i]!;
    const inner = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
    // tanh approximation via Math.tanh (native, fast)
    x[i] = 0.5 * v * (1.0 + Math.tanh(inner));
  }
}

/**
 * Numerically stable Softmax.
 *
 *   softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
 *
 * The max-subtraction prevents overflow for large inputs.
 * Operates in-place.
 */
export function softmax(x: Float32Array): void {
  // 1. Find max
  let maxVal = -Infinity;
  for (let i = 0; i < x.length; i++) {
    if (x[i]! > maxVal) maxVal = x[i]!;
  }

  // 2. Compute exp(x - max) and sum
  let sum = 0.0;
  for (let i = 0; i < x.length; i++) {
    const expVal = Math.exp(x[i]! - maxVal);
    x[i] = expVal;
    sum += expVal;
  }

  // 3. Normalize
  const invSum = sum > 0 ? 1.0 / sum : 1.0;
  for (let i = 0; i < x.length; i++) {
    x[i] = x[i]! * invSum;
  }
}

/**
 * Batch Softmax — applies softmax along the last dimension for each item in the batch.
 *
 *   x: Float32Array [batch × dim]  (row-major)
 *   softmax is applied independently to each row.
 *
 * Operates in-place.
 */
export function softmaxBatch(x: Float32Array, batch: number, dim: number): void {
  for (let b = 0; b < batch; b++) {
    const start = b * dim;
    const slice = x.subarray(start, start + dim);

    // Find max
    let maxVal = -Infinity;
    for (let i = 0; i < dim; i++) {
      if (slice[i]! > maxVal) maxVal = slice[i]!;
    }

    // Compute exp and sum
    let sum = 0.0;
    for (let i = 0; i < dim; i++) {
      const expVal = Math.exp(slice[i]! - maxVal);
      slice[i] = expVal;
      sum += expVal;
    }

    // Normalize
    const invSum = sum > 0 ? 1.0 / sum : 1.0;
    for (let i = 0; i < dim; i++) {
      slice[i] = slice[i]! * invSum;
    }
  }
}

/**
 * Multi-head attention softmax: applies softmax across the last dimension
 * for each head × batch × position independently.
 *
 *   x: Float32Array [B × heads × L × L]  (row-major, last dim is L)
 */
export function attentionSoftmax(
  scores: Float32Array,
  batch: number,
  heads: number,
  seqLen: number,
): void {
  const dim = seqLen; // softmax over the key dimension (columns)
  const totalRows = batch * heads * seqLen;

  for (let r = 0; r < totalRows; r++) {
    const start = r * dim;
    const slice = scores.subarray(start, start + dim);

    let maxVal = -Infinity;
    for (let i = 0; i < dim; i++) {
      if (slice[i]! > maxVal) maxVal = slice[i]!;
    }

    let sum = 0.0;
    for (let i = 0; i < dim; i++) {
      const expVal = Math.exp(slice[i]! - maxVal);
      slice[i] = expVal;
      sum += expVal;
    }

    const invSum = sum > 0 ? 1.0 / sum : 1.0;
    for (let i = 0; i < dim; i++) {
      slice[i] = slice[i]! * invSum;
    }
  }
}
