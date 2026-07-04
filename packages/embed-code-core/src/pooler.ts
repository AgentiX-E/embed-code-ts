/**
 * Mean pooling — mask-aware aggregation of hidden states.
 *
 * Computes: pooled[b][d] = sum_s hidden[b][s][d] * mask[b][s] / sum_s mask[b][s]
 */
export function meanPool(
  hidden: Float32Array,
  attentionMask: Int32Array,
  batch: number,
  seqLen: number,
  dim: number,
): Float32Array {
  const out = new Float32Array(batch * dim);
  for (let b = 0; b < batch; b++) {
    let count = 0;
    for (let s = 0; s < seqLen; s++) {
      if (attentionMask[b * seqLen + s]! === 1) {
        count++;
        const src = (b * seqLen + s) * dim;
        const dst = b * dim;
        for (let d = 0; d < dim; d++) out[dst + d] += hidden[src + d]!;
      }
    }
    if (count > 0) {
      for (let d = 0; d < dim; d++) out[b * dim + d] /= count;
    }
  }
  return out;
}

/**
 * CLS pooling — returns the [CLS] token embedding (position 0).
 */
export function clsPool(
  hidden: Float32Array,
  batch: number,
  seqLen: number,
  dim: number,
): Float32Array {
  const out = new Float32Array(batch * dim);
  for (let b = 0; b < batch; b++) {
    const src = b * seqLen * dim; // position 0 of sequence b
    const dst = b * dim;
    for (let d = 0; d < dim; d++) out[dst + d] = hidden[src + d]!;
  }
  return out;
}

/**
 * Last-token pooling — returns the last non-padding token embedding.
 * Uses attentionMask to find the last real token in each sequence.
 */
export function lastTokenPool(
  hidden: Float32Array,
  attentionMask: Int32Array,
  batch: number,
  seqLen: number,
  dim: number,
): Float32Array {
  const out = new Float32Array(batch * dim);
  for (let b = 0; b < batch; b++) {
    let lastIdx = 0;
    for (let s = seqLen - 1; s >= 0; s--) {
      if (attentionMask[b * seqLen + s]! === 1) {
        lastIdx = s;
        break;
      }
    }
    const src = (b * seqLen + lastIdx) * dim;
    const dst = b * dim;
    for (let d = 0; d < dim; d++) out[dst + d] = hidden[src + d]!;
  }
  return out;
}
