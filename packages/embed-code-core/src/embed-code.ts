/**
 * EmbedCode — the main public API.
 *
 * Pure-TypeScript BERT-base code embedding engine.
 * Int8 weights are embedded directly in the npm package (incbin-style),
 * no ONNX Runtime or native bindings required.
 *
 * Usage:
 * ```typescript
 * import { EmbedCode } from '@agentix-e/embed-code-core';
 *
 * // Load from embedded weights buffer (recommended)
 * const embedder = await EmbedCode.fromPretrained({
 *   weightsBuffer: embeddedWeightsBuffer,
 *   tokenizerPath: path.join(__dirname, 'tokenizer.json'),
 * });
 *
 * // Or load from file path
 * const embedder = await EmbedCode.fromPretrained({
 *   modelPath: '/path/to/weights.int8.bin',
 *   tokenizerPath: '/path/to/tokenizer.json',
 * });
 *
 * const results = await embedder.embed([
 *   'search_query: Calculate factorial',
 *   'search_document: def fact(n): return 1 if n <= 1 else n * fact(n-1)',
 * ]);
 *
 * console.log(results.embeddings); // Float32Array [2, 768]
 * await embedder.dispose();
 * ```
 */

import * as path from 'node:path';
import { ModelNotFoundError } from './errors';
import { resolveModelConfig, EMBED_CODE_V1_CONFIG } from './model-descriptor';
import { EmbedCodeTSEngine } from './inference/ts-engine';
import { WeightBuffer } from './inference/weights';
import { Tokenizer } from './tokenizer';
import { poolEmbeddings, normalizeEmbeddings, cosineSimilarity } from './pooling';
import type {
  ModelConfig,
  ModelLoadOptions,
  EmbedOptions,
  EmbeddingResult,
  IInferenceEngine,
} from './types';

export class EmbedCode {
  private _engine: IInferenceEngine;
  private _config: Readonly<ModelConfig>;
  private _tokenizer: Tokenizer;
  private _loaded = false;

  private constructor(
    engine: IInferenceEngine,
    config: Readonly<ModelConfig>,
    tokenizer: Tokenizer,
  ) {
    this._engine = engine;
    this._config = config;
    this._tokenizer = tokenizer;
  }

  // ─── Factory ──────────────────────────────────────────────

  /**
   * Create an EmbedCode instance from pretrained weights.
   *
   * Supports:
   *   - Embedded weights buffer (ArrayBuffer/Uint8Array) — recommended incbin-style
   *   - File path to weights.int8.bin
   *
   * The model architecture is resolved from the co-located model-descriptor.json.
   * Falls back to EMBED_CODE_V1_CONFIG when no descriptor is present.
   */
  static async fromPretrained(options: ModelLoadOptions): Promise<EmbedCode> {
    if (!options.modelPath && !options.weightsBuffer) {
      throw new ModelNotFoundError(
        'modelPath or weightsBuffer is required.\n' +
          'Provide either:\n' +
          '  1. An embedded weights buffer: EmbedCode.fromPretrained({ weightsBuffer })\n' +
          '  2. A path to weights.int8.bin: EmbedCode.fromPretrained({ modelPath })\n' +
          '  3. Export weights locally: python scripts/export-weights.py --output weights.int8.bin',
      );
    }

    // Resolve architecture from model-descriptor.json
    const modelDir = options.modelPath
      ? path.dirname(options.modelPath)
      : path.dirname(options.tokenizerPath || __filename);
    const configSource = options.modelPath ?? modelDir;
    const { config } = resolveModelConfig(configSource, EMBED_CODE_V1_CONFIG);

    // Initialize tokenizer
    const tokenizer = new Tokenizer();
    const tokenizerPath =
      options.tokenizerPath || (options.modelPath ? path.join(modelDir, 'tokenizer.json') : '');

    if (tokenizerPath) {
      try {
        tokenizer.loadFromFile(tokenizerPath);
      } catch {
        console.warn(
          `[EmbedCode] Tokenizer not found at ${tokenizerPath}. ` +
            'Some tokenization features may be limited.',
        );
      }
    }

    // Initialize inference engine
    const engine = new EmbedCodeTSEngine();

    if (options.weightsBuffer) {
      let wb: WeightBuffer;
      if (options.weightsBuffer instanceof WeightBuffer) {
        wb = options.weightsBuffer;
      } else if (options.weightsBuffer instanceof ArrayBuffer) {
        wb = new WeightBuffer(options.weightsBuffer);
      } else {
        // Uint8Array or similar — copy to standalone ArrayBuffer
        const u8 = options.weightsBuffer as Uint8Array;
        const copy = new ArrayBuffer(u8.byteLength);
        new Uint8Array(copy).set(u8);
        wb = new WeightBuffer(copy);
      }
      await engine.load(wb);
    } else if (options.modelPath) {
      await engine.load(options.modelPath);
    }

    const instance = new EmbedCode(engine, config, tokenizer);
    instance._loaded = true;
    return instance;
  }

  // ─── Embed ────────────────────────────────────────────────

  /**
   * Generate embeddings for the given texts.
   *
   * For optimal results with nomic-embed-code, prefix texts with:
   *   - Queries: "search_query: {query}"
   *   - Code/Documents: "search_document: {code}"
   */
  async embed(texts: string[], options: EmbedOptions = {}): Promise<EmbeddingResult> {
    const maxTokens = options.maxTokens ?? this._config.maxTokens;
    const poolStrategy = options.poolingStrategy ?? this._config.poolingStrategy;
    const shouldNormalize = options.normalize ?? this._config.normalize;

    const startTime = performance.now();

    // 1. Tokenize
    const { inputIds, attentionMask } = this._tokenizer.encode(texts, maxTokens);
    const batchSize = texts.length;

    // Progress callback
    options.onProgress?.({ phase: 'tokenize', step: 1, total: 4 });

    // 2. Pure-TS inference
    options.signal?.throwIfAborted();

    const outputs = await this._engine.run({
      [this._config.inputIdsName]: {
        data: inputIds,
        dims: [batchSize, maxTokens],
        type: 'int64',
      },
      [this._config.attentionMaskName]: {
        data: attentionMask,
        dims: [batchSize, maxTokens],
        type: 'int64',
      },
    });

    options.onProgress?.({ phase: 'inference', step: 2, total: 4 });

    // 3. Pooling
    const hiddenStates =
      outputs[this._config.outputName] ||
      outputs['last_hidden_state'] ||
      outputs['token_embeddings'] ||
      Object.values(outputs)[0];

    if (!hiddenStates) {
      throw new Error(`Output "${this._config.outputName}" not found.`);
    }

    const hiddenData = hiddenStates.data as Float32Array;
    const hiddenDims = hiddenStates.dims;

    let embeddings: Float32Array;
    if (hiddenDims.length === 2) {
      embeddings = new Float32Array(hiddenData);
    } else {
      embeddings = poolEmbeddings(
        hiddenData,
        attentionMask,
        batchSize,
        maxTokens,
        hiddenDims[2] ?? this._config.embeddingDim,
        poolStrategy,
      );
    }

    options.onProgress?.({ phase: 'pool', step: 3, total: 4 });

    // 4. Normalize
    if (shouldNormalize) {
      normalizeEmbeddings(embeddings, batchSize, this._config.embeddingDim);
    }

    options.onProgress?.({ phase: 'normalize', step: 4, total: 4 });

    const elapsedMs = performance.now() - startTime;

    return {
      embeddings,
      shape: [batchSize, this._config.embeddingDim],
      elapsedMs,
    };
  }

  // ─── Similarity ───────────────────────────────────────────

  /** Compute cosine similarity between two embedding vectors. */
  similarity(a: Float32Array, b: Float32Array): number {
    return cosineSimilarity(a, b);
  }

  // ─── Accessors ────────────────────────────────────────────

  get config(): Readonly<ModelConfig> {
    return this._config;
  }

  get embeddingDim(): number {
    return this._config.embeddingDim;
  }

  get maxTokens(): number {
    return this._config.maxTokens;
  }

  get taskPrefixes() {
    return this._config.taskPrefixes;
  }

  get isLoaded(): boolean {
    return this._loaded;
  }

  // ─── Lifecycle ────────────────────────────────────────────

  async dispose(): Promise<void> {
    await this._engine.dispose();
    this._loaded = false;
  }
}
