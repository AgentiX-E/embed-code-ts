/**
 * Multi-head self-attention for BERT-base.
 *
 *   Q, K, V = Linear(x, W_qkv)  →  reshape into [B, heads, L, head_dim]
 *   scores = Q @ K^T / sqrt(head_dim)
 *   scores += attention_mask (for padding)
 *   attn_weights = softmax(scores)
 *   context = attn_weights @ V
 *   output = Linear(merge_heads(context), W_o)
 *
 * Uses int8 quantized weights with online dequantization.
 */

import { int8Linear } from './dequantize';
import { attentionSoftmax } from './activations';

/**
 * Run multi-head self-attention on input hidden states.
 *
 *   hidden:    Float32Array [B × L × D]  (input, modified in-place)
 *   qWeight:   Int8Array    [D × D]      (query weight [D × D] transposed)
 *   qBias:     Int8Array    [D]          (query bias)
 *   qScale:    Float32Array [D]
 *   qBScale:   Float32Array [D]
 *   kWeight:   Int8Array    [D × D]      (key weight)
 *   kBias:     Int8Array    [D]
 *   kScale:    Float32Array [D]
 *   kBScale:   Float32Array [D]
 *   vWeight:   Int8Array    [D × D]      (value weight)
 *   vBias:     Int8Array    [D]
 *   vScale:    Float32Array [D]
 *   vBScale:   Float32Array [D]
 *   oWeight:   Int8Array    [D × D]      (output projection weight)
 *   oBias:     Int8Array    [D]
 *   oScale:    Float32Array [D]
 *   oBScale:   Float32Array [D]
 *   attnMask:  Int32Array   [B × L]       (1=keep, 0=mask out)
 *   batch:     number
 *   seqLen:    number
 *   hiddenDim: number                     (D = 768)
 *   numHeads:  number                     (12)
 *   headDim:   number                     (64 = D / numHeads)
 *
 * Uses pre-allocated buffers (qBuf, kBuf, vBuf, scoresBuf, ctxBuf, outBuf)
 * to avoid per-call GC pressure.
 */
export function multiHeadAttention(
  hidden: Float32Array,
  qWeight: Int8Array,
  qBias: Int8Array,
  qScale: Float32Array,
  qBScale: Float32Array,
  kWeight: Int8Array,
  kBias: Int8Array,
  kScale: Float32Array,
  kBScale: Float32Array,
  vWeight: Int8Array,
  vBias: Int8Array,
  vScale: Float32Array,
  vBScale: Float32Array,
  oWeight: Int8Array,
  oBias: Int8Array,
  oScale: Float32Array,
  oBScale: Float32Array,
  attnMask: Int32Array,
  batch: number,
  seqLen: number,
  hiddenDim: number,
  numHeads: number,
  headDim: number,
  qBuf: Float32Array,
  kBuf: Float32Array,
  vBuf: Float32Array,
  scoresBuf: Float32Array,
  ctxBuf: Float32Array,
  outBuf: Float32Array,
): void {
  const m = batch * seqLen;
  const D = hiddenDim;
  const H = numHeads;
  const d = headDim;

  // 1. Linear projections: Q, K, V = hidden @ W^T + bias
  int8Linear(hidden, qWeight, qScale, qBias, qBScale, qBuf, m, D, D);
  int8Linear(hidden, kWeight, kScale, kBias, kBScale, kBuf, m, D, D);
  int8Linear(hidden, vWeight, vScale, vBias, vBScale, vBuf, m, D, D);

  // 2. Reshape Q, K, V to [B × H × L × d] and compute attention
  //   scores[b, h, i, j] = Σ_d Q[b, i, h*d+d] * K[b, j, h*d+d] / sqrt(d)
  const scaleFactor = 1.0 / Math.sqrt(d);

  // Score layout: [B × H × L × L] row-major
  // scoresBuf is assumed to be sized [B * H * L * L]

  let scoresIdx = 0;
  for (let b = 0; b < batch; b++) {
    for (let h = 0; h < H; h++) {
      const headOff = h * d;

      for (let i = 0; i < seqLen; i++) {
        const qi = (b * seqLen + i) * D + headOff;
        for (let j = 0; j < seqLen; j++) {
          const kj = (b * seqLen + j) * D + headOff;

          // Dot product over head dim
          let dot = 0;
          for (let dd = 0; dd < d; dd++) {
            dot += qBuf[qi + dd]! * kBuf[kj + dd]!;
          }
          scoresBuf[scoresIdx] = dot * scaleFactor;
          scoresIdx++;
        }
      }
    }
  }

  // 3. Apply attention mask
  // For each (b, h, i, j): if mask[b, j] == 0, score = -1e9
  for (let b = 0; b < batch; b++) {
    const maskOff = b * seqLen;
    for (let h = 0; h < H; h++) {
      for (let i = 0; i < seqLen; i++) {
        const rowOff = ((b * H + h) * seqLen + i) * seqLen;
        for (let j = 0; j < seqLen; j++) {
          if (attnMask[maskOff + j] === 0) {
            scoresBuf[rowOff + j] = -1e9;
          }
        }
      }
    }
  }

  // 4. Softmax over key dimension
  attentionSoftmax(scoresBuf, batch, H, seqLen);

  // 5. Context: ctx = attn_weights @ V
  //   ctx[b, h, i, dd] = Σ_j scores[b, h, i, j] * V[b, j, h*d+dd]
  let ctxIdx = 0;
  for (let b = 0; b < batch; b++) {
    for (let h = 0; h < H; h++) {
      const headOff = h * d;

      for (let i = 0; i < seqLen; i++) {
        const scoreRowOff = ((b * H + h) * seqLen + i) * seqLen;

        for (let dd = 0; dd < d; dd++) {
          let acc = 0;
          for (let j = 0; j < seqLen; j++) {
            const vIdx = (b * seqLen + j) * D + headOff + dd;
            acc += scoresBuf[scoreRowOff + j]! * vBuf[vIdx]!;
          }
          ctxBuf[ctxIdx] = acc;
          ctxIdx++;
        }
      }
    }
  }

  // 6. Merge heads: reshape [B × H × L × d] → [B × L × D]
  for (let b = 0; b < batch; b++) {
    for (let h = 0; h < H; h++) {
      const headOff = h * d;
      for (let i = 0; i < seqLen; i++) {
        const srcOff = ((b * H + h) * seqLen + i) * d;
        const dstOff = (b * seqLen + i) * D + headOff;
        for (let dd = 0; dd < d; dd++) {
          outBuf[dstOff + dd] = ctxBuf[srcOff + dd]!;
        }
      }
    }
  }

  // 7. Output projection: hidden = outBuf @ W_o^T + b_o
  int8Linear(outBuf, oWeight, oScale, oBias, oBScale, hidden, m, D, D);
}
