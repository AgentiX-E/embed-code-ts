/**
 * Layer Normalization — BERT-style (ε in denominator, learnable affine params).
 *
 *   LayerNorm(x) = (x - μ) / sqrt(σ² + ε) * γ + β
 *
 * All weights stored as int8 with per-channel float32 scales.
 */

/**
 * Apply LayerNorm to a 2D tensor [batch, dim].
 *
 *   x:      Float32Array [B × D]  (row-major, in-place)
 *   weight: Int8Array    [D]       (gamma, learnable scale)
 *   wScale: Float32Array [1]       (gamma dequant scale, single channel)
 *   bias:   Int8Array    [D]       (beta, learnable shift)
 *   bScale: Float32Array [1]       (bias dequant scale, single channel)
 *   batch:  number                 (number of samples)
 *   dim:    number                 (hidden dimension)
 */
export function layerNorm(
  x: Float32Array,
  weight: Int8Array,
  wScale: Float32Array,
  bias: Int8Array,
  bScale: Float32Array,
  batch: number,
  dim: number,
): void {
  const eps = 1e-12;

  for (let b = 0; b < batch; b++) {
    const off = b * dim;

    // 1. Compute mean
    let mean = 0;
    for (let d = 0; d < dim; d++) {
      mean += x[off + d]!;
    }
    mean /= dim;

    // 2. Compute variance
    let variance = 0;
    for (let d = 0; d < dim; d++) {
      const diff = x[off + d]! - mean;
      variance += diff * diff;
    }
    variance /= dim;

    // 3. Normalize and apply affine
    const invStd = 1.0 / Math.sqrt(variance + eps);
    for (let d = 0; d < dim; d++) {
      const normalized = (x[off + d]! - mean) * invStd;
      const gamma = weight[d]! * wScale[0]!;
      const beta = bias[d]! * bScale[0]!;
      x[off + d] = normalized * gamma + beta;
    }
  }
}

/**
 * Apply LayerNorm to a 3D tensor [batch, seqLen, dim].
 *
 * The last dimension (dim) is normalized independently for each
 * position in each batch element.
 */
export function layerNorm3D(
  x: Float32Array,
  weight: Int8Array,
  wScale: Float32Array,
  bias: Int8Array,
  bScale: Float32Array,
  batch: number,
  seqLen: number,
  dim: number,
): void {
  const eps = 1e-12;
  const totalRows = batch * seqLen;

  for (let r = 0; r < totalRows; r++) {
    const off = r * dim;

    let mean = 0;
    for (let d = 0; d < dim; d++) {
      mean += x[off + d]!;
    }
    mean /= dim;

    let variance = 0;
    for (let d = 0; d < dim; d++) {
      const diff = x[off + d]! - mean;
      variance += diff * diff;
    }
    variance /= dim;

    const invStd = 1.0 / Math.sqrt(variance + eps);
    for (let d = 0; d < dim; d++) {
      const normalized = (x[off + d]! - mean) * invStd;
      const gamma = weight[d]! * wScale[0]!;
      const beta = bias[d]! * bScale[0]!;
      x[off + d] = normalized * gamma + beta;
    }
  }
}
