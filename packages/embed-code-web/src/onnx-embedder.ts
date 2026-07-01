/**
 * WebEmbedder — browser ONNX Runtime code embedding.
 *
 * Implements IEmbedder using onnxruntime-web (WASM default, WebGPU auto-upgrade).
 */
import type { IEmbedder, BatchOptions, ModelInfo } from '@agentix-e/embed-code-core';
import { WordPieceTokenizer, meanPool, l2Normalize } from '@agentix-e/embed-code-core';

export { type IEmbedder, type ModelInfo, type BatchOptions };

// Dynamic imports: onnxruntime-web is loaded lazily in browser context
let ortWeb: any = null;
async function getOrt(): Promise<any> {
  if (ortWeb) return ortWeb;
  ortWeb = await import('onnxruntime-web');
  return ortWeb;
}

export class WebEmbedder implements IEmbedder {
  private session: any = null;
  readonly dimensions = 768;
  readonly maxSequenceLength = 512;
  readonly modelInfo: ModelInfo;
  private tokenizer: WordPieceTokenizer;
  private modelBuffer: ArrayBuffer;

  private constructor(tokenizer: WordPieceTokenizer, modelBuffer: ArrayBuffer) {
    this.tokenizer = tokenizer;
    this.modelBuffer = modelBuffer;
    this.modelInfo = {
      name: 'nomic-embed-code',
      version: 'v1.5',
      dimensions: 768,
      maxSequenceLength: 512,
      vocabSize: tokenizer.vocabSize,
      quantization: 'int8',
    };
  }

  static async create(modelUrl: string, tokenizerJson: Record<string, any>): Promise<WebEmbedder> {
    const tokenizer = WordPieceTokenizer.fromJSON(tokenizerJson, 512);
    const resp = await fetch(modelUrl);
    const buffer = await resp.arrayBuffer();
    const embedder = new WebEmbedder(tokenizer, buffer);

    const ort = await getOrt();
    // WebGPU auto-upgrade
    try {
      embedder.session = await ort.InferenceSession.create(buffer, {
        executionProviders: ['webgpu'],
        graphOptimizationLevel: 'all',
      });
    } catch {
      embedder.session = await ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    }
    return embedder;
  }

  async embed(text: string): Promise<Float32Array> {
    if (!this.session) throw new Error('Session not initialized');
    const tokens = this.tokenizer.tokenize(text);

    const ort = await getOrt();
    const toBInt64 = (arr: Int32Array) => BigInt64Array.from(Array.from(arr).map((v) => BigInt(v)));

    const feeds = {
      input_ids: new ort.Tensor('int64', toBInt64(tokens.inputIds), [1, 512]),
      attention_mask: new ort.Tensor('int64', toBInt64(tokens.attentionMask), [1, 512]),
      token_type_ids: new ort.Tensor('int64', toBInt64(tokens.tokenTypeIds), [1, 512]),
    };

    const outputs = await this.session.run(feeds);
    const hidden = outputs.last_hidden_state.data as Float32Array;

    const pooled = meanPool(hidden, tokens.attentionMask, 1, 512, 768);
    l2Normalize(pooled, 1, 768);
    return pooled;
  }

  async embedBatch(texts: string[], options?: BatchOptions): Promise<Float32Array[]> {
    const results: Float32Array[] = [];
    let completed = 0;
    for (const text of texts) {
      const r = await this.embed(text);
      results.push(r);
      completed++;
      options?.onProgress?.(completed, texts.length);
    }
    return results;
  }

  async dispose(): Promise<void> {
    this.session?.release();
    this.session = null;
  }
}
