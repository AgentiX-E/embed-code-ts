/**
 * EmbedCode — the core embedding engine for nomic-embed-code.
 *
 * Handles the full pipeline:
 *  1. Tokenization (BPE)
 *  2. ONNX model inference
 *  3. Pooling (last-token for nomic-embed-code)
 *  4. Normalization (L2)
 *
 * The model provider (embedded/chunked/remote) is transparent to this class.
 * It receives an ArrayBuffer and feeds it to ONNX Runtime.
 */

import type {
  EmbedCodeOptions,
  EmbeddingResult,
  IModelProvider,
  IInferenceSession,
  ModelDescriptor,
} from './types';
import { Tokenizer } from './tokenizer';
import { poolEmbeddings, normalizeEmbeddings, cosineSimilarity } from './pooling';
import { resolveModel, listModels, getModels } from './registry';
import { createModelProvider } from './model-provider';

export class EmbedCode {
  private modelProvider: IModelProvider | null = null;
  private session: IInferenceSession | null = null;
  private tokenizer: Tokenizer;
  private descriptor!: ModelDescriptor;
  private options: EmbedCodeOptions;
  private ready = false;

  private constructor(options: EmbedCodeOptions = {}) {
    this.options = options;
    this.tokenizer = new Tokenizer();
  }

  /**
   * Create and initialize an EmbedCode instance.
   *
   * This is the primary entry point. It handles:
   * - Resolving the model descriptor from the registry
   * - Creating the appropriate model provider (embedded/chunked/remote)
   * - Loading and decoding weights
   * - Initializing the tokenizer
   * - Creating the ONNX inference session
   *
   * @example
   * ```ts
   * const embedder = await EmbedCode.create({
   *   model: 'nomic-embed-code-v1',
   *   cacheDir: './.cache/embed-code',
   *   onProgress: (p) => console.log(`${p.percent}%`),
   * });
   * ```
   */
  static async create(options: EmbedCodeOptions = {}): Promise<EmbedCode> {
    const instance = new EmbedCode(options);
    await instance.init();
    return instance;
  }

  /**
   * Generate embeddings for the given texts.
   *
   * For code search with nomic-embed-code, use task prefixes:
   * - Queries: `"search_query: {query}"`
   * - Code/Docs: `"search_document: {code}"`
   *
   * @param texts - One or more text strings to embed
   * @returns Embedding result with Float32Array and metadata
   *
   * @example
   * ```ts
   * const results = await embedder.embed([
   *   'search_query: Calculate factorial',
   *   'search_document: def fact(n): return 1 if n <= 1 else n * fact(n-1)',
   * ]);
   * // results.embeddings → Float32Array [2, 3584]
   * ```
   */
  async embed(texts: string | string[]): Promise<EmbeddingResult> {
    this.ensureReady();

    const inputs = Array.isArray(texts) ? texts : [texts];
    const startTime = performance.now();

    // Step 1: Tokenize
    const { inputIds, attentionMask } = this.tokenizer.encode(
      inputs,
      this.options.maxTokens ?? this.descriptor.runtime.maxTokens,
    );

    const batchSize = inputs.length;
    const seqLen = this.options.maxTokens ?? this.descriptor.runtime.maxTokens;

    // Step 2: Run ONNX inference
    const session = this.session!;
    const outputs = await session.run({
      input_ids: {
        data: inputIds,
        dims: [batchSize, seqLen],
        type: 'int64',
      },
      attention_mask: {
        data: attentionMask,
        dims: [batchSize, seqLen],
        type: 'int64',
      },
    });

    // Step 3: Extract hidden states and pool
    // The output name varies by model — try common names
    const hiddenStates =
      outputs['last_hidden_state'] ||
      outputs['token_embeddings'] ||
      outputs['sentence_embedding'] ||
      Object.values(outputs)[0];

    if (!hiddenStates) {
      throw new Error(
        'ONNX model output not found. Expected "last_hidden_state" or "token_embeddings".',
      );
    }

    const hiddenData = hiddenStates.data as Float32Array;
    const hiddenDims = hiddenStates.dims;

    let embeddings: Float32Array;

    // If the model already outputs pooled embeddings (e.g., sentence_embedding)
    if (hiddenDims.length === 2) {
      // Already pooled: [batchSize, embeddingDim]
      embeddings = hiddenData;
    } else {
      // Raw hidden states: [batchSize, seqLen, hiddenDim]
      embeddings = poolEmbeddings(
        hiddenData,
        attentionMask,
        {
          batchSize,
          seqLen,
          hiddenDim: hiddenDims[2]!,
        },
        this.descriptor.runtime.poolingStrategy,
      );
    }

    // Step 4: Normalize if configured
    if (this.options.normalize ?? this.descriptor.runtime.normalize) {
      const embedDim =
        hiddenDims.length === 2
          ? hiddenDims[1]!
          : hiddenDims[2]!;
      embeddings = normalizeEmbeddings(embeddings, batchSize, embedDim);
    }

    const elapsedMs = performance.now() - startTime;

    return {
      embeddings,
      shape: [batchSize, this.descriptor.runtime.embeddingDim],
      elapsedMs,
    };
  }

  /**
   * Compute cosine similarity between two embedding vectors.
   */
  similarity(a: Float32Array, b: Float32Array): number {
    return cosineSimilarity(a, b);
  }

  /**
   * Get the model descriptor (config, dimensions, etc.).
   */
  getDescriptor(): ModelDescriptor {
    return { ...this.descriptor };
  }

  /**
   * Get the embedding dimension.
   */
  get embeddingDim(): number {
    return this.descriptor.runtime.embeddingDim;
  }

  /**
   * Release all resources (session, buffers).
   */
  dispose(): void {
    this.session?.release();
    this.session = null;
    this.modelProvider?.dispose();
    this.modelProvider = null;
    this.ready = false;
  }

  // ─── Static Helpers ──────────────────────────────────────

  /** List all available models in the registry */
  static listModels = listModels;

  /** Get details for all registered models */
  static getModels = getModels;

  /** Resolve a specific model */
  static resolveModel = resolveModel;

  // ─── Private Methods ─────────────────────────────────────

  private async init(): Promise<void> {
    // Resolve model descriptor
    this.descriptor = resolveModel(this.options.model);

    // Create model provider (handles embedded/chunked/remote loading)
    this.modelProvider = await createModelProvider(this.descriptor, this.options);

    // Initialize tokenizer
    const tokenizerBuffer = await this.modelProvider.getTokenizerBuffer();
    await this.tokenizer.initialize(tokenizerBuffer);

    // Load model weights and create ONNX session
    const modelBuffer = await this.modelProvider.getModelBuffer();

    // Determine platform and create session
    this.session = await this.createSession(modelBuffer);

    this.ready = true;
  }

  private async createSession(
    modelBuffer: ArrayBuffer,
  ): Promise<IInferenceSession> {
    // Try Node.js platform first, fall back to web
    try {
      const { createInferenceSession: createNode } = await import(
        './platform/node'
      );
      return await createNode(modelBuffer);
    } catch (nodeErr) {
      try {
        const { createInferenceSession: createWeb } = await import(
          './platform/web'
        );
        return await createWeb(modelBuffer);
      } catch (webErr) {
        throw new Error(
          'No ONNX Runtime platform available. Install either:\n' +
            '  onnxruntime-node  (for Node.js)\n' +
            '  onnxruntime-web   (for browser)\n\n' +
            `Node.js error: ${(nodeErr as Error).message}\n` +
            `Browser error: ${(webErr as Error).message}`,
        );
      }
    }
  }

  private ensureReady(): void {
    if (!this.ready) {
      throw new Error(
        'EmbedCode is not initialized. Call EmbedCode.create() first.',
      );
    }
  }
}
