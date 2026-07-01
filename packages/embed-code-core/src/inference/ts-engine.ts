/**
 * Pure-TypeScript BERT-base Inference Engine (pre-norm architecture).
 *
 * EVERY function is inlined — zero module imports for computation.
 * This eliminates tsup/tsx module resolution issues that can cause NaN.
 */
import type { IInferenceEngine, Tensor } from '../types';
import { InferenceError } from '../errors';
import { WeightBuffer, type ArchParams } from './weights';

// ── ALL FUNCTIONS INLINED ──

function ln(x: Float32Array, w: Int8Array, ws: Float32Array, b: Int8Array, bs: Float32Array,
  batch: number, seqLen: number, dim: number,
): void {
  const eps = 1e-12;
  for (let r = 0; r < batch * seqLen; r++) {
    const off = r * dim;
    let mean = 0;
    for (let d = 0; d < dim; d++) mean += x[off + d]!;
    mean /= dim;
    let variance = 0;
    for (let d = 0; d < dim; d++) { const diff = x[off + d]! - mean; variance += diff * diff; }
    variance /= dim;
    const invStd = 1 / Math.sqrt(variance + eps);
    for (let d = 0; d < dim; d++) {
      x[off + d] = ((x[off + d]! - mean) * invStd) * (w[d]! * ws[0]!) + b[d]! * bs[0]!;
    }
  }
}

function il(a: Float32Array, w: Int8Array, scale: Float32Array,
  bias: Int8Array, bScale: Float32Array,
  c: Float32Array, m: number, k: number, n: number,
): void {
  for (let i = 0; i < m; i++) {
    const off = i * n;
    for (let j = 0; j < n; j++) c[off + j] = bias[j]! * bScale[j]!;
  }
  for (let i = 0; i < m; i++) {
    const aRowOff = i * k, cRowOff = i * n;
    for (let j = 0; j < n; j++) {
      const s = scale[j]!, wRowOff = j * k;
      let acc = c[cRowOff + j]!;
      for (let p = 0; p < k; p++) acc += a[aRowOff + p]! * w[wRowOff + p]! * s;
      c[cRowOff + j] = acc;
    }
  }
}

function gelu(x: Float32Array): void {
  const SQRT_2_OVER_PI = 0.7978845608028654, COEFF = 0.044715;
  for (let i = 0; i < x.length; i++) {
    const v = x[i]!;
    x[i] = 0.5 * v * (1 + Math.tanh(SQRT_2_OVER_PI * (v + COEFF * v * v * v)));
  }
}

function embed(hidden: Float32Array, ids: Int32Array,
  weight: Int8Array, scale: Float32Array, factor: number,
  batch: number, seqLen: number, dim: number,
): void {
  for (let b = 0; b < batch; b++)
    for (let s = 0; s < seqLen; s++) {
      const tid = ids[b * seqLen + s]!, wo = tid * dim, ho = (b * seqLen + s) * dim;
      for (let d = 0; d < dim; d++)
        hidden[ho + d]! += weight[wo + d]! * scale[tid]! * factor;
    }
}

function posEmbed(hidden: Float32Array, weight: Int8Array, scale: Float32Array, factor: number,
  batch: number, seqLen: number, dim: number,
): void {
  for (let b = 0; b < batch; b++)
    for (let s = 0; s < seqLen; s++) {
      const wo = s * dim, ho = (b * seqLen + s) * dim;
      for (let d = 0; d < dim; d++)
        hidden[ho + d]! += weight[wo + d]! * scale[s]! * factor;
    }
}

function add(src: Float32Array, dst: Float32Array, len: number): void {
  for (let i = 0; i < len; i++) dst[i] += src[i]!;
}

// ── Engine ──

export class EmbedCodeTSEngine implements IInferenceEngine {
  private _loaded = false;
  private _weights: WeightBuffer | null = null;
  private _arch: ArchParams | null = null;

  isLoaded(): boolean { return this._loaded; }

  async load(source: any): Promise<void> {
    if (source instanceof WeightBuffer) this._weights = source;
    else if (typeof source === 'string') this._weights = await WeightBuffer.fromFile(source);
    else if (source instanceof ArrayBuffer) this._weights = new WeightBuffer(source);
    else throw new InferenceError('Expected WeightBuffer, file path, or ArrayBuffer.');
    this._arch = this._weights.archParams;
    this._loaded = true;
  }

  async run(feeds: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    if (!this._weights || !this._arch) throw new InferenceError('Engine not loaded.');
    const w = this._weights, arch = this._arch;
    const ids = (feeds['input_ids'] ?? Object.values(feeds)[0]).data as Int32Array;
    const mask = (feeds['attention_mask'] ?? feeds['attention_mask']).data as Int32Array;
    const dims = (feeds['input_ids'] ?? Object.values(feeds)[0]).dims;
    const B = dims[0]!, L = dims[1]!, D = arch.hiddenSize, H = arch.numHeads, d = arch.headDim,
      I = arch.intermediateSize, m = B * L;

    // Allocate buffers
    const hidden = new Float32Array(m * D);
    const saved = new Float32Array(m * D);
    const zb = new Int8Array(Math.max(D, I)), zs = new Float32Array(Math.max(D, I)).fill(1);
    const qb = new Float32Array(m * D), kb = new Float32Array(m * D), vb = new Float32Array(m * D);
    const sc = new Float32Array(B * H * L * L), ctx = new Float32Array(B * H * L * d);
    const mg = new Float32Array(m * D), ib = new Float32Array(m * I);

    // 1. Embedding
    embed(hidden, ids,
      w.getInt8('embeddings.word_embeddings.weight'),
      w.getFloat32('embeddings.word_embeddings.weight'), 1, B, L, D);
    if (w.has('embeddings.position_embeddings.weight'))
      posEmbed(hidden,
        w.getInt8('embeddings.position_embeddings.weight'),
        w.getFloat32('embeddings.position_embeddings.weight'), 1, B, L, D);
    if (w.has('embeddings.token_type_embeddings.weight')) {
      const tids = new Int32Array(m).fill(0);
      embed(hidden, tids,
        w.getInt8('embeddings.token_type_embeddings.weight'),
        w.getFloat32('embeddings.token_type_embeddings.weight'), 1, B, L, D);
    }
    ln(hidden,
      w.getInt8('embeddings.LayerNorm.weight'), w.getFloat32('embeddings.LayerNorm.weight'),
      w.getInt8('embeddings.LayerNorm.bias'), w.getFloat32('embeddings.LayerNorm.bias'),
      B, L, D);

    // 2. Transformer layers
    for (let l = 0; l < arch.numLayers; l++) {
      const p = `encoder.layer.${l}`;

      // Pre-norm + Attention
      saved.set(hidden);
      ln(hidden,
        w.getInt8(`${p}.attention.output.LayerNorm.weight`), w.getFloat32(`${p}.attention.output.LayerNorm.weight`),
        w.getInt8(`${p}.attention.output.LayerNorm.bias`), w.getFloat32(`${p}.attention.output.LayerNorm.bias`),
        B, L, D);
      il(hidden, w.getInt8(`${p}.attention.self.query.weight`), w.getFloat32(`${p}.attention.self.query.weight`), zb, zs, qb, m, D, D);
      il(hidden, w.getInt8(`${p}.attention.self.key.weight`), w.getFloat32(`${p}.attention.self.key.weight`), zb, zs, kb, m, D, D);
      il(hidden, w.getInt8(`${p}.attention.self.value.weight`), w.getFloat32(`${p}.attention.self.value.weight`), zb, zs, vb, m, D, D);

      // Attention scores
      const sf = 1 / Math.sqrt(d);
      let si = 0;
      for (let b = 0; b < B; b++)
        for (let h = 0; h < H; h++) {
          const ho2 = h * d;
          for (let i = 0; i < L; i++) {
            const qi = (b * L + i) * D + ho2;
            for (let j = 0; j < L; j++) {
              const kj = (b * L + j) * D + ho2;
              let dt = 0;
              for (let dd = 0; dd < d; dd++) dt += qb[qi + dd]! * kb[kj + dd]!;
              sc[si++] = dt * sf;
            }
          }
        }
      // Apply mask
      for (let b = 0; b < B; b++) {
        const mo = b * L;
        for (let h = 0; h < H; h++)
          for (let i = 0; i < L; i++) {
            const ro = ((b * H + h) * L + i) * L;
            for (let j = 0; j < L; j++)
              if (mask[mo + j] === 0) sc[ro + j] = -1e9;
          }
      }
      // Softmax
      for (let b = 0; b < B; b++)
        for (let h = 0; h < H; h++)
          for (let i = 0; i < L; i++) {
            const st = ((b * H + h) * L + i) * L;
            let mv = -Infinity;
            for (let j = 0; j < L; j++) if (sc[st + j]! > mv) mv = sc[st + j]!;
            let sm = 0;
            for (let j = 0; j < L; j++) { const e = Math.exp(sc[st + j]! - mv); sc[st + j] = e; sm += e; }
            if (sm > 0) for (let j = 0; j < L; j++) sc[st + j] = sc[st + j]! / sm;
          }
      // Context
      let ci = 0;
      for (let b = 0; b < B; b++)
        for (let h = 0; h < H; h++) {
          const ho2 = h * d;
          for (let i = 0; i < L; i++) {
            const sr = ((b * H + h) * L + i) * L;
            for (let dd = 0; dd < d; dd++) {
              let ac = 0;
              for (let j = 0; j < L; j++) ac += sc[sr + j]! * vb[(b * L + j) * D + ho2 + dd]!;
              ctx[ci++] = ac;
            }
          }
        }
      // Merge heads
      mg.fill(0);
      for (let b = 0; b < B; b++)
        for (let h = 0; h < H; h++) {
          const ho2 = h * d;
          for (let i = 0; i < L; i++) {
            const so = ((b * H + h) * L + i) * d, do2 = (b * L + i) * D + ho2;
            for (let dd = 0; dd < d; dd++) mg[do2 + dd] = ctx[so + dd]!;
          }
        }
      // Output projection + residual
      il(mg, w.getInt8(`${p}.attention.output.dense.weight`), w.getFloat32(`${p}.attention.output.dense.weight`), zb, zs, hidden, m, D, D);
      add(saved, hidden, m * D);

      // Pre-norm + FFN
      saved.set(hidden);
      ln(hidden,
        w.getInt8(`${p}.output.LayerNorm.weight`), w.getFloat32(`${p}.output.LayerNorm.weight`),
        w.getInt8(`${p}.output.LayerNorm.bias`), w.getFloat32(`${p}.output.LayerNorm.bias`),
        B, L, D);
      il(hidden, w.getInt8(`${p}.intermediate.dense.weight`), w.getFloat32(`${p}.intermediate.dense.weight`), zb, zs, ib, m, D, I);
      gelu(ib);
      il(ib, w.getInt8(`${p}.output.dense.weight`), w.getFloat32(`${p}.output.dense.weight`), zb, zs, hidden, m, I, D);
      add(saved, hidden, m * D);
    }

    return { last_hidden_state: { data: hidden, dims: [B, L, D], type: 'float32' } };
  }

  async dispose(): Promise<void> {
    this._weights = null; this._arch = null; this._loaded = false;
  }
}
