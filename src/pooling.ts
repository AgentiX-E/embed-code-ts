/**
 * Pooling strategies for extracting embeddings from model outputs.
 *
 * nomic-embed-code uses last-token pooling (matching its training methodology).
 * nomic-embed-text uses mean pooling.
 */

import type { PoolingStrategy } from './types';

/**
 * Apply the specified pooling strategy to model hidden states.
 *
 * @param hiddenStates - Model output tensor [batchSize, seqLen, hiddenDim]
 * @param attentionMask - Attention mask [batchSize, seqLen]
 * @param strategy - Pooling strategy
 * @returns Pooled embeddings [batchSize, hiddenDim]
 */
export function poolEmbeddings(
  hiddenStates: Float32Array,
  attentionMask: Int32Array,
  shape: { batchSize: number; seqLen: number; hiddenDim: number },
  strategy: PoolingStrategy,
): Float32Array {
  const { batchSize, seqLen, hiddenDim } = shape;

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

/**
 * Last-token pooling: take the hidden state of the last non-padding token.
 * This is what nomic-embed-code uses during training.
 */
function lastTokenPooling(
  hiddenStates: Float32Array,
  attentionMask: Int32Array,
  batchSize: number,
  seqLen: number,
  hiddenDim: number,
): Float32Array {
  const result = new Float32Array(batchSize * hiddenDim);

  for (let b = 0; b < batchSize; b++) {
    // Find the last non-padding token
    let lastIdx = seqLen - 1;
    for (let s = seqLen - 1; s >= 0; s--) {
      if (attentionMask[b * seqLen + s] === 1) {
        lastIdx = s;
        break;
      }
    }

    // Extract hidden state at lastIdx
    const srcOffset = (b * seqLen + lastIdx) * hiddenDim;
    const dstOffset = b * hiddenDim;
    for (let d = 0; d < hiddenDim; d++) {
      result[dstOffset + d] = hiddenStates[srcOffset + d]!;
    }
  }

  return result;
}

/**
 * Mean pooling: average all non-padding token hidden states.
 */
function meanPooling(
  hiddenStates: Float32Array,
  attentionMask: Int32Array,
  batchSize: number,
  seqLen: number,
  hiddenDim: number,
): Float32Array {
  const result = new Float32Array(batchSize * hiddenDim);

  for (let b = 0; b < batchSize; b++) {
    let count = 0;

    // Sum over all non-padding tokens
    for (let s = 0; s < seqLen; s++) {
      if (attentionMask[b * seqLen + s] === 1) {
        count++;
        const srcOffset = (b * seqLen + s) * hiddenDim;
        const dstOffset = b * hiddenDim;
        for (let d = 0; d < hiddenDim; d++) {
          result[dstOffset + d]! += hiddenStates[srcOffset + d]!;
        }
      }
    }

    // Divide by count
    if (count > 0) {
      const dstOffset = b * hiddenDim;
      for (let d = 0; d < hiddenDim; d++) {
        result[dstOffset + d]! /= count;
      }
    }
  }

  return result;
}

/**
 * CLS pooling: take the first token's hidden state.
 */
function clsPooling(
  hiddenStates: Float32Array,
  batchSize: number,
  seqLen: number,
  hiddenDim: number,
): Float32Array {
  const result = new Float32Array(batchSize * hiddenDim);

  for (let b = 0; b < batchSize; b++) {
    const srcOffset = b * seqLen * hiddenDim;
    const dstOffset = b * hiddenDim;
    for (let d = 0; d < hiddenDim; d++) {
      result[dstOffset + d] = hiddenStates[srcOffset + d]!;
    }
  }

  return result;
}

/**
 * L2 normalize embeddings in-place.
 */
export function normalizeEmbeddings(
  embeddings: Float32Array,
  batchSize: number,
  hiddenDim: number,
): Float32Array {
  for (let b = 0; b < batchSize; b++) {
    const offset = b * hiddenDim;

    // Compute L2 norm
    let norm = 0;
    for (let d = 0; d < hiddenDim; d++) {
      norm += embeddings[offset + d]! ** 2;
    }
    norm = Math.sqrt(norm);

    // Normalize
    if (norm > 0) {
      for (let d = 0; d < hiddenDim; d++) {
        embeddings[offset + d]! /= norm;
      }
    }
  }

  return embeddings;
}

/**
 * Compute cosine similarity between two embeddings.
 */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error(`Dimension mismatch: ${a.length} vs ${b.length}`);
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i]! * b[i]!;
    normA += a[i]! ** 2;
    normB += b[i]! ** 2;
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dotProduct / denominator;
}
