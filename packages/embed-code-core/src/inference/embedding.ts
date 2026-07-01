/**
 * Token/Position/TokenType embedding lookup with int8 weights.
 *
 * BERT embedding = word_emb[token_ids] + pos_emb[positions] + type_emb[types]
 *
 * All embedding tables stored as int8 with per-channel scales.
 * Lookup dequantizes on-the-fly.
 */

import type { Int8Array } from 'node:buffer';

/**
 * Embedding lookup from int8 weight table and accumulate into hidden states.
 *
 *   hidden:       Float32Array [B × L × D]  (accumulation target)
 *   inputIds:     Int32Array   [B × L]       (token IDs to look up)
 *   weight:       Int8Array    [V × D]       (embedding table, row-major)
 *   scale:        Float32Array [V]           (per-token dequant scale — one per vocabulary entry)
 *   scaleFactor:  number                     (optional global multiplier, default 1.0)
 *   batch:        number                     (batch size)
 *   seqLen:       number                     (sequence length)
 *   dim:          number                     (hidden dimension)
 */
export function embeddingLookup(
  hidden: Float32Array,
  inputIds: Int32Array,
  weight: Int8Array,
  scale: Float32Array,
  scaleFactor: number,
  batch: number,
  seqLen: number,
  dim: number,
): void {
  for (let b = 0; b < batch; b++) {
    for (let s = 0; s < seqLen; s++) {
      const tokenId = inputIds[b * seqLen + s]!;
      const weightOffset = tokenId * dim;
      const hiddenOffset = (b * seqLen + s) * dim;

      for (let d = 0; d < dim; d++) {
        hidden[hiddenOffset + d] += weight[weightOffset + d]! * scale[tokenId]! * scaleFactor;
      }
    }
  }
}

/**
 * Add position embeddings to hidden states.
 *
 *   hidden:       Float32Array [B × L × D]  (accumulation target)
 *   weight:       Int8Array    [maxPos × D] (position embedding table)
 *   scale:        Float32Array [D]          (per-column dequant scale)
 *   scaleFactor:  number                    (optional global multiplier)
 *   batch:        number                    (batch size)
 *   seqLen:       number                    (sequence length)
 *   dim:          number                    (hidden dimension)
 */
export function positionEmbedding(
  hidden: Float32Array,
  weight: Int8Array,
  scale: Float32Array,
  scaleFactor: number,
  batch: number,
  seqLen: number,
  dim: number,
): void {
  for (let b = 0; b < batch; b++) {
    for (let s = 0; s < seqLen; s++) {
      const weightOffset = s * dim;
      const hiddenOffset = (b * seqLen + s) * dim;

      for (let d = 0; d < dim; d++) {
        hidden[hiddenOffset + d] += weight[weightOffset + d]! * scale[s]! * scaleFactor;
      }
    }
  }
}

/**
 * Add segment/token type embeddings to hidden states.
 *
 * For standard BERT, segment 0 = first sentence, segment 1 = second sentence.
 * For code embedding, all inputs typically use segment 0.
 *
 *   hidden:       Float32Array [B × L × D]  (accumulation target)
 *   typeIds:      Int32Array   [B × L]       (0 or 1)
 *   weight:       Int8Array    [2 × D]       (token type embedding table)
 *   scale:        Float32Array [2]           (per-type dequant scale)
 *   batch:        number
 *   seqLen:       number
 *   dim:          number
 */
export function tokenTypeEmbedding(
  hidden: Float32Array,
  typeIds: Int32Array,
  weight: Int8Array,
  scale: Float32Array,
  batch: number,
  seqLen: number,
  dim: number,
): void {
  for (let b = 0; b < batch; b++) {
    for (let s = 0; s < seqLen; s++) {
      const typeId = typeIds[b * seqLen + s]!;
      if (typeId === 0) continue; // skip segment 0 (optimization: hidden is already zeroed)

      const weightOffset = typeId * dim;
      const hiddenOffset = (b * seqLen + s) * dim;

      for (let d = 0; d < dim; d++) {
        hidden[hiddenOffset + d] += weight[weightOffset + d]! * scale[typeId]!;
      }
    }
  }
}
