/**
 * Int8 dequantization and online GEMM for integer-quantized weights.
 *
 * Stores model weights as Int8Array with per-channel float32 scales.
 * During inference, weights are dequantized on-the-fly in the GEMM loop,
 * avoiding allocation of full float32 weight matrices (saving ~400 MB RAM).
 *
 * Quantization scheme (per-channel, per-output-dimension):
 *   For weight W[m×n] (output dim = n):
 *     scale[j] = max(|W[:, j]|) / 127.0
 *     W_int8[i, j] = round(clip(W[i, j] / scale[j], -128, 127))
 *
 * Dequantization (in GEMM):
 *   C[i, j] = Σ_k A[i, k] * float32(W_int8[k, j]) * scale[j]
 *
 * This keeps W as Int8Array (~137 MB) instead of Float32Array (~550 MB).
 */

/**
 * Int8 GEMM with online dequantization: C = A @ dequantize(W_int8, scale).
 *
 *   A:    Float32Array [m × k]  (row-major, input activations)
 *   W:    Int8Array    [k × n]  (row-major, quantized weights)
 *   scale: Float32Array [n]     (per-column dequantization scale)
 *   C:    Float32Array [m × n]  (row-major, output — caller allocated)
 *
 * Loop ordering i→p→j to keep W row access contiguous.
 * The inner loop dequantizes one int8 element per multiply.
 */
export function int8Gemm(
  a: Float32Array,
  w: Int8Array,
  scale: Float32Array,
  c: Float32Array,
  m: number,
  k: number,
  n: number,
): void {
  // Caller is responsible for initializing C (typically zero-filled or pre-loaded with bias)
  // We do NOT zero C here to allow bias pre-loading

  for (let i = 0; i < m; i++) {
    const aRowOff = i * k;
    const cRowOff = i * n;
    for (let p = 0; p < k; p++) {
      const aVal = a[aRowOff + p]!;
      if (aVal === 0) continue; // skip zeros (sparsity from attention mask)
      const wRowOff = p * n;
      for (let j = 0; j < n; j++) {
        c[cRowOff + j] += aVal * w[wRowOff + j]! * scale[j]!;
      }
    }
  }
}

/**
 * Int8 GEMM with bias: C = A @ dequantize(W_int8, scale) + bias.
 */
export function int8GemmBiased(
  a: Float32Array,
  w: Int8Array,
  scale: Float32Array,
  bias: Int8Array,
  biasScale: Float32Array,
  c: Float32Array,
  m: number,
  k: number,
  n: number,
): void {
  // Initialise with dequantized bias
  for (let i = 0; i < m; i++) {
    const off = i * n;
    for (let j = 0; j < n; j++) {
      c[off + j] = bias[j]! * biasScale[j]!;
    }
  }

  int8Gemm(a, w, scale, c, m, k, n);
}

/**
 * Int8 linear layer: output = input @ W_int8^T * scale + bias.
 *
 * This is the standard PyTorch `nn.Linear` pattern where:
 *   W shape: [out_features × in_features]  (transposed in memory)
 *   bias shape: [out_features]
 *
 * We transpose the weight access pattern to avoid a physical transpose:
 *   C[m, n] = Σ_k A[m, k] * W[n, k] * scale[n]
 *
 * where W is stored as [out_features × in_features] = [n × k].
 *
 *   a:     Float32Array [m × k]  (input)
 *   w:     Int8Array    [n × k]  (weight, transposed: out_features × in_features)
 *   scale: Float32Array [n]      (per-output-channel scale)
 *   bias:  Int8Array    [n]      (quantized bias)
 *   bScale:Float32Array [n]      (bias per-channel scale)
 *   c:     Float32Array [m × n]  (output, caller-allocated)
 */
export function int8Linear(
  a: Float32Array,
  w: Int8Array,
  scale: Float32Array,
  bias: Int8Array,
  bScale: Float32Array,
  c: Float32Array,
  m: number,
  k: number,
  n: number,
): void {
  // Init with dequantized bias
  for (let i = 0; i < m; i++) {
    const off = i * n;
    for (let j = 0; j < n; j++) {
      c[off + j] = bias[j]! * bScale[j]!;
    }
  }

  // GEMM with weight stored as [n × k] (transposed layout)
  for (let i = 0; i < m; i++) {
    const aRowOff = i * k;
    const cRowOff = i * n;
    for (let j = 0; j < n; j++) {
      const s = scale[j]!;
      const wRowOff = j * k;
      let acc = c[cRowOff + j]!; // start from bias
      for (let p = 0; p < k; p++) {
        acc += a[aRowOff + p]! * w[wRowOff + p]! * s;
      }
      c[cRowOff + j] = acc;
    }
  }
}

/**
 * Dequantize a single int8 tensor to Float32Array.
 *
 *   w:     Int8Array [len]  (quantized values)
 *   scale: Float32Array [n]  (per-channel scales)
 *   n:     number of channels
 *   out:   Float32Array [len] (caller-allocated output)
 *
 * Each element w[i] belongs to channel (i % n) and is scaled by scale[i % n].
 */
export function dequantize(w: Int8Array, scale: Float32Array, out: Float32Array, n: number): void {
  const len = w.length;
  for (let i = 0; i < len; i++) {
    out[i] = w[i]! * scale[i % n]!;
  }
}

/**
 * Per-channel quantization: float32 → int8.
 *
 *   f:  Float32Array [len]  (float32 input)
 *   w:  Int8Array    [len]  (int8 output, caller-allocated)
 *   n:  number       (number of channels)
 *
 * Returns the per-channel scales as Float32Array [n].
 */
export function quantizePerChannel(f: Float32Array, w: Int8Array, n: number): Float32Array {
  const len = f.length;
  const channelsPerEl = len / n;
  const scales = new Float32Array(n);

  // First pass: compute max absolute value per channel
  for (let c = 0; c < n; c++) {
    let maxAbs = 0;
    for (let e = 0; e < channelsPerEl; e++) {
      const idx = e * n + c;
      const absVal = Math.abs(f[idx]!);
      if (absVal > maxAbs) maxAbs = absVal;
    }
    scales[c] = maxAbs / 127.0;
    if (scales[c] === 0) scales[c] = 1.0; // avoid divide-by-zero
  }

  // Second pass: quantize
  for (let c = 0; c < n; c++) {
    const invScale = 1.0 / scales[c]!;
    for (let e = 0; e < channelsPerEl; e++) {
      const idx = e * n + c;
      const q = Math.round(f[idx]! * invScale);
      w[idx] = Math.max(-128, Math.min(127, q));
    }
  }

  return scales;
}
