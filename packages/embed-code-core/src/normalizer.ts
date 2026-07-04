/**
 * L2 normalization — in-place unit-vector normalization.
 *
 * After normalization: ||v||₂ ≈ 1.0 for each batch element.
 * Zero vectors are left unchanged.
 */
export function l2Normalize(embeddings: Float32Array, batch: number, dim: number): void {
  for (let b = 0; b < batch; b++) {
    const offset = b * dim;
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += embeddings[offset + d]! ** 2;
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let d = 0; d < dim; d++) embeddings[offset + d] /= norm;
    }
  }
}

/**
 * Cosine similarity between two embedding vectors.
 */
export function cosineSimilarity(a: Float32Array | number[], b: Float32Array | number[]): number {
  if (a.length !== b.length)
    throw new Error(`Dimension mismatch in cosineSimilarity: ${a.length} vs ${b.length}`);
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! ** 2;
    normB += b[i]! ** 2;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
