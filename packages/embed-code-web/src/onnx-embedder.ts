/**
 * WebEmbedder — browser ONNX Runtime code embedding.
 *
 * Implements IEmbedder using onnxruntime-web (WASM default, WebGPU auto-upgrade).
 */
import type { IEmbedder, BatchOptions, ModelInfo } from '@agentix-e/embed-code-core';
import {
  WordPieceTokenizer,
  meanPool,
  l2Normalize,
  int32ToBigInt64,
  processBatch,
} from '@agentix-e/embed-code-core';

export { type IEmbedder, type ModelInfo, type BatchOptions };

// Type-safe wrapper interface for onnxruntime-web
// Manual types avoid the complex onnxruntime-web type hierarchy
interface OrtInstance {
  InferenceSession: {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    create(path: ArrayBuffer | string, options?: Record<string, any>): Promise<OrtSession>;
  };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  Tensor: new (
    type: string,
    data: any,
    dims: number[],
  ) => { type: string; dims: readonly number[]; data: unknown };
}

interface OrtSession {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  run(feeds: Record<string, any>): Promise<Record<string, any>>;
  release(): void;
}

// Dynamic imports: onnxruntime-web is loaded lazily in browser context
let ortWeb: OrtInstance | null = null;
async function getOrt(): Promise<OrtInstance> {
  if (ortWeb) return ortWeb;
  ortWeb = (await import('onnxruntime-web')) as unknown as OrtInstance;
  return ortWeb;
}

export class WebEmbedder implements IEmbedder {
  private session: OrtSession | null = null;
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
    let buffer: ArrayBuffer;
    if (modelUrl.startsWith('file://')) {
      // Node.js file:// protocol — dynamic import of fs (not available in browser)
      // @ts-expect-error 2307 — node:fs module only exists in Node.js runtime
      const { readFileSync } = await import('node:fs');
      const raw = readFileSync(modelUrl.replace('file://', ''));
      buffer = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength);
    } else {
      const resp = await fetch(modelUrl);
      buffer = await resp.arrayBuffer();
    }
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

    const feeds = {
      input_ids: new ort.Tensor('int64', int32ToBigInt64(tokens.inputIds), [1, 512]),
      attention_mask: new ort.Tensor('int64', int32ToBigInt64(tokens.attentionMask), [1, 512]),
      token_type_ids: new ort.Tensor('int64', int32ToBigInt64(tokens.tokenTypeIds), [1, 512]),
    };

    const outputs = await this.session.run(feeds);
    const hidden = outputs.last_hidden_state.data as Float32Array;

    const pooled = meanPool(hidden, tokens.attentionMask, 1, 512, 768);
    l2Normalize(pooled, 1, 768);
    return pooled;
  }

  async embedBatch(texts: string[], options?: BatchOptions): Promise<Float32Array[]> {
    const results: Float32Array[] = new Array(texts.length);
    await processBatch(
      texts,
      async (text, index) => {
        results[index] = await this.embed(text);
      },
      options,
    );
    return results;
  }

  async dispose(): Promise<void> {
    this.session?.release();
    this.session = null;
  }
}
