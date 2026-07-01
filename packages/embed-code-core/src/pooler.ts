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
