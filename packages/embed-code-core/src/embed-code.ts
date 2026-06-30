/**
 * EmbedCode — the main public API.
 *
 * Usage:
 * ```typescript
 * import { EmbedCode, downloadModel } from '@agentix-e/embed-code-core';
 *
 * // Auto-download on first use (caches to ~/.cache/)
 * const modelPath = await downloadModel();
 *
 * const embedder = await EmbedCode.fromPretrained({ modelPath });
 *
 * const results = await embedder.embed([
 *   'search_query: Calculate factorial',
 *   'search_document: def fact(n): return 1 if n <= 1 else n * fact(n-1)',
 * ]);
 *
 * console.log(results.embeddings); // Float32Array [2, 3584]
 * await embedder.dispose();
 * ```
 */

import * as path from 'node:path';
import { ModelNotFoundError } from './errors';
import { resolveModelConfig, EMBED_CODE_V1_CONFIG } from './model-descriptor';
import { EmbedCodeInferenceEngine } from './inference/onnx-engine';
import { Tokenizer } from './tokenizer';
import { poolEmbeddings, normalizeEmbeddings, cosineSimilarity } from './pooling';
import type { ModelConfig, ModelLoadOptions, EmbedOptions, EmbeddingResult } from './types';

export class EmbedCode {
  private _engine: EmbedCodeInferenceEngine;
  private _config: Readonly<ModelConfig>;
  private _tokenizer: Tokenizer;
  private _loaded = false;

  private constructor(
    engine: EmbedCodeInferenceEngine,
    config: Readonly<ModelConfig>,
    tokenizer: Tokenizer,
  ) {
    this._engine = engine;
    this._config = config;
    this._tokenizer = tokenizer;
  }

  // ─── Factory ──────────────────────────────────────────────

  /**
   * Create an EmbedCode instance from a pretrained ONNX checkpoint.
   *
   * The model architecture is resolved from the model-descriptor.json
   * co-located with the ONNX file. Falls back to EMBED_CODE_V1_CONFIG
   * when no descriptor is present.
   *
   * @param options.modelPath  Path to the ONNX model file. Required.
   * @param options.tokenizerPath  Path to tokenizer.json (default: alongside model)
   * @param options.executionProvider  'cpu' (default), 'cuda', 'dml'
   */
  static async fromPretrained(options: ModelLoadOptions): Promise<EmbedCode> {
    if (!options.modelPath) {
      throw new ModelNotFoundError(
        'modelPath is required. Provide the path to an ONNX model file.\n' +
          'To obtain a model:\n' +
          '  1. Run: import { downloadModel } from "@agentix-e/embed-code-core"; await downloadModel();\n' +
          '  2. Or download a pre-converted ONNX model from GitHub Releases\n' +
          '  3. Or export locally: python scripts/export-onnx.py --output model.onnx',
      );
    }

    // Resolve architecture from model-descriptor.json
    const { config, descriptor } = resolveModelConfig(options.modelPath, EMBED_CODE_V1_CONFIG);

    if (descriptor) {
      console.log(
        `[EmbedCode] Loaded descriptor: ${descriptor.model.name} ${descriptor.model.version}` +
          ` (${descriptor.model.base_architecture}), ` +
          `${descriptor.onnx.size_bytes > 0 ? (descriptor.onnx.size_bytes / 1024 ** 2).toFixed(0) + ' MB' : ''}`,
      );
    }

    // Initialize tokenizer
    const tokenizer = new Tokenizer();
    const modelDir = path.dirname(options.modelPath);
    const tokenizerPath = options.tokenizerPath || path.join(modelDir, 'tokenizer.json');

    try {
      tokenizer.loadFromFile(tokenizerPath);
    } catch {
      console.warn(
        `[EmbedCode] Tokenizer not found at ${tokenizerPath}. ` +
          'Some tokenization features may be limited.',
      );
    }

    // Initialize inference engine
    const engine = new EmbedCodeInferenceEngine({
      executionProvider: options.executionProvider,
      intraOpNumThreads: options.intraOpNumThreads,
    });

    await engine.load(options.modelPath, { skipWarmup: options.skipWarmup });

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
   *
   * @param texts Text strings to embed
   * @param options Optional config overrides
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

    // 2. ONNX inference
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
      throw new Error(`ONNX output "${this._config.outputName}" not found.`);
    }

    const hiddenData = hiddenStates.data as Float32Array;
    const hiddenDims = hiddenStates.dims;

    let embeddings: Float32Array;
    if (hiddenDims.length === 2) {
      // Already pooled output
      embeddings = new Float32Array(hiddenData);
    } else {
      embeddings = poolEmbeddings(
        hiddenData,
        attentionMask,
        batchSize,
        maxTokens,
        hiddenDims[2],
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
