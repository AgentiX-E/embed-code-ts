/**
 * Feed-Forward Network (FFN) for BERT-base.
 *
 *   FFN(x) = GELU(x @ W_up^T + b_up) @ W_down^T + b_down
 *
 * Standard BERT-base dimensions:
 *   hidden_size = 768, intermediate_size = 3072
 *
 * Uses int8 quantized weights with online dequantization.
 */

import { int8Linear } from './dequantize';
import { gelu } from './activations';

/**
 * BERT Feed-Forward Network: GELU activation on intermediate layer.
 *
 *   hidden:    Float32Array [B × L × D]    (input, modified in-place)
 *   upWeight:  Int8Array    [I × D]         (intermediate projection weight)
 *   upBias:    Int8Array    [I]             (intermediate bias)
 *   upScale:   Float32Array [I]
 *   upBScale:  Float32Array [I]
 *   downWeight:Int8Array    [D × I]         (output projection weight)
 *   downBias:  Int8Array    [D]             (output bias)
 *   downScale: Float32Array [D]
 *   downBScale:Float32Array [D]
 *   batch:     number
 *   seqLen:    number
 *   hiddenDim: number                       (D = 768)
 *   intermediateDim: number                 (I = 3072)
 *   interBuf:  Float32Array [B × L × I]     (pre-allocated scratch buffer)
 */
export function feedForward(
  hidden: Float32Array,
  upWeight: Int8Array,
  upBias: Int8Array,
  upScale: Float32Array,
  upBScale: Float32Array,
  downWeight: Int8Array,
  downBias: Int8Array,
  downScale: Float32Array,
  downBScale: Float32Array,
  batch: number,
  seqLen: number,
  hiddenDim: number,
  intermediateDim: number,
  interBuf: Float32Array,
): void {
  const m = batch * seqLen;

  // 1. Intermediate projection: inter = hidden @ W_up^T + b_up
  int8Linear(hidden, upWeight, upScale, upBias, upBScale, interBuf, m, hiddenDim, intermediateDim);

  // 2. GELU activation (in-place on interBuf)
  gelu(interBuf);

  // 3. Output projection: hidden = inter @ W_down^T + b_down
  int8Linear(
    interBuf,
    downWeight,
    downScale,
    downBias,
    downBScale,
    hidden,
    m,
    intermediateDim,
    hiddenDim,
  );
}
