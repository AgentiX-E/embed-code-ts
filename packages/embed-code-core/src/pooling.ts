/**
 * Pooling strategies for converting hidden states to embeddings.
 *
 * nomic-embed-code: last_token pooling
 * nomic-embed-text: mean pooling
 */

export type PoolingStrategy = 'mean' | 'cls' | 'last_token';

export function poolEmbeddings(
  hiddenStates: Float32Array,
  attentionMask: Int32Array,
  batchSize: number,
  seqLen: number,
  hiddenDim: number,
  strategy: PoolingStrategy,
): Float32Array {
  switch (strategy) {
    case 'last_token':
      return lastTokenPooling(hiddenStates, attentionMask, batchSize, seqLen, hiddenDim);
    case 'mean':
      return meanPooling(hiddenStates, attentionMask, batchSize, seqLen, hiddenDim);
    case 'cls':
      return clsPooling(hiddenStates, batchSize, seqLen, hiddenDim);
    default:
      return lastTokenPooling(hiddenStates, attentionMask, batchSize, seqLen, hiddenDim);
  }
}

function lastTokenPooling(
  h: Float32Array,
  mask: Int32Array,
  B: number,
  S: number,
  D: number,
): Float32Array {
  const out = new Float32Array(B * D);
  for (let b = 0; b < B; b++) {
    let lastIdx = S - 1;
    for (let s = S - 1; s >= 0; s--) {
      if (mask[b * S + s] === 1) {
        lastIdx = s;
        break;
      }
    }
    const src = (b * S + lastIdx) * D;
    const dst = b * D;
    for (let d = 0; d < D; d++) out[dst + d] = h[src + d]!;
  }
  return out;
}

function meanPooling(
  h: Float32Array,
  mask: Int32Array,
  B: number,
  S: number,
  D: number,
): Float32Array {
  const out = new Float32Array(B * D);
  for (let b = 0; b < B; b++) {
    let count = 0;
    const dst = b * D;
    for (let s = 0; s < S; s++) {
      if (mask[b * S + s] === 1) {
        count++;
        const src = (b * S + s) * D;
        for (let d = 0; d < D; d++) out[dst + d] += h[src + d];
      }
    }
    if (count > 0) {
      for (let d = 0; d < D; d++) out[dst + d] /= count;
    }
  }
  return out;
}

function clsPooling(h: Float32Array, B: number, S: number, D: number): Float32Array {
  const out = new Float32Array(B * D);
  for (let b = 0; b < B; b++) {
    const src = b * S * D;
    const dst = b * D;
    for (let d = 0; d < D; d++) out[dst + d] = h[src + d]!;
  }
  return out;
}

export function normalizeEmbeddings(
  embeddings: Float32Array,
  batchSize: number,
  dim: number,
): void {
  for (let b = 0; b < batchSize; b++) {
    const offset = b * dim;
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += embeddings[offset + d] ** 2;
    norm = Math.sqrt(norm);
    if (norm > 0) {
      for (let d = 0; d < dim; d++) embeddings[offset + d] /= norm;
    }
  }
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error(`Dimension mismatch: ${a.length} vs ${b.length}`);
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] ** 2;
    normB += b[i] ** 2;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}
