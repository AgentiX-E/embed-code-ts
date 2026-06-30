/**
 * @agentix-e/embed-code-ts
 * Core type definitions for the nomic-embed-code TypeScript embedding system.
 */

// ─── Model Types ───────────────────────────────────────────

/** Supported pooling strategies */
export type PoolingStrategy = 'mean' | 'cls' | 'last_token';

/** Model precision / quantization level */
export type ModelPrecision =
  | 'fp32'
  | 'fp16'
  | 'int8'
  | 'int4'
  | 'bnb4'
  | 'q4'
  | 'q4f16';

/** Model provider type — determines how weights are loaded at runtime */
export type ProviderType = 'embedded' | 'chunked' | 'remote';

/** Model descriptor in the registry */
export interface ModelDescriptor {
  /** Unique model identifier, e.g. 'nomic-embed-code-v1' */
  id: string;
  /** Human-readable label */
  label: string;
  /** Provider strategy for this model */
  provider: ProviderType;
  /** Precision */
  precision: ModelPrecision;
  /** Target model on HuggingFace */
  target: {
    repository: string;
    onnxFile: string;
    tokenizerFile?: string;
  };
  /** Runtime characteristics */
  runtime: {
    /** Output embedding dimension */
    embeddingDim: number;
    /** Maximum input tokens */
    maxTokens: number;
    /** Pooling strategy */
    poolingStrategy: PoolingStrategy;
    /** Normalize embeddings (L2) */
    normalize: boolean;
  };
  /** Remote download config (only for provider: 'remote') */
  remote?: {
    /** Base URL for downloading model chunks */
    baseUrl: string;
    /** Number of chunks the model is split into */
    chunkCount: number;
    /** Expected SHA256 of the assembled model */
    sha256: string;
    /** Total size in bytes */
    totalSize: number;
  };
  /** Incremental version number for cache busting */
  version: number;
}

// ─── Inference Types ───────────────────────────────────────

/** Options for creating an EmbedCode instance */
export interface EmbedCodeOptions {
  /** Model ID from the registry */
  model?: string;
  /** Override provider type */
  provider?: ProviderType;
  /** Custom ONNX model buffer (bypasses registry) */
  customModelBuffer?: ArrayBuffer;
  /** Custom tokenizer JSON buffer */
  customTokenizerBuffer?: ArrayBuffer;
  /** Cache directory for downloaded models */
  cacheDir?: string;
  /** Progress callback for downloads */
  onProgress?: (progress: DownloadProgress) => void;
  /** Maximum tokens per input (truncation) */
  maxTokens?: number;
  /** Normalize embeddings */
  normalize?: boolean;
}

/** Download progress event */
export interface DownloadProgress {
  /** Percentage complete (0-100) */
  percent: number;
  /** Bytes downloaded so far */
  downloaded: number;
  /** Total bytes to download */
  total: number;
  /** Current chunk being downloaded */
  chunk?: number;
  /** Total chunks */
  totalChunks?: number;
}

/** Embedding result */
export interface EmbeddingResult {
  /** Float32Array of embeddings [batchSize, embeddingDim] */
  embeddings: Float32Array;
  /** Shape of the output tensor */
  shape: [number, number];
  /** Time taken in milliseconds */
  elapsedMs: number;
}

// ─── Tokenizer Types ───────────────────────────────────────

/** Tokenizer configuration */
export interface TokenizerConfig {
  type: 'bpe' | 'wordpiece' | 'unigram';
  vocabSize: number;
  maxLength: number;
  padToken: string;
  padTokenId: number;
  bosToken: string;
  bosTokenId: number;
  eosToken: string;
  eosTokenId: number;
  unkToken: string;
  unkTokenId: number;
}

/** Tokenization result */
export interface TokenizationResult {
  inputIds: Int32Array;
  attentionMask: Int32Array;
  tokenTypeIds?: Int32Array;
}

// ─── Model Provider Types ──────────────────────────────────

/** Abstract interface for model weight providers */
export interface IModelProvider {
  readonly type: ProviderType;
  /** Get the ONNX model as an ArrayBuffer (or Uint8Array) */
  getModelBuffer(): Promise<ArrayBuffer>;
  /** Get the tokenizer JSON as an ArrayBuffer */
  getTokenizerBuffer(): Promise<ArrayBuffer>;
  /** Get model configuration */
  getConfig(): ModelDescriptor;
  /** Verify integrity via SHA256 (if available) */
  verify(): Promise<boolean>;
  /** Clean up resources */
  dispose(): void;
}

// ─── Platform Types ────────────────────────────────────────

/** Platform-specific ONNX inference session */
export interface IInferenceSession {
  /** Run inference with named inputs */
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  /** Release session resources */
  release(): void;
}

/** ONNX Runtime tensor */
export interface OrtTensor {
  data: Float32Array | Int32Array | BigInt64Array;
  dims: number[];
  type: string;
}

// ─── Cache Types ───────────────────────────────────────────

/** Cache entry for downloaded models */
export interface CacheEntry {
  modelId: string;
  version: number;
  sha256: string;
  filePath: string;
  downloadedAt: number;
  size: number;
}

// ─── Error Types ───────────────────────────────────────────

export class EmbedCodeError extends Error {
  constructor(
    message: string,
    public readonly code: ErrorCode,
    override readonly cause?: Error,
  ) {
    super(message, { cause });
    this.name = 'EmbedCodeError';
  }
}

export type ErrorCode =
  | 'MODEL_NOT_FOUND'
  | 'DOWNLOAD_FAILED'
  | 'INTEGRITY_CHECK_FAILED'
  | 'INFERENCE_FAILED'
  | 'TOKENIZATION_FAILED'
  | 'UNSUPPORTED_PLATFORM'
  | 'INVALID_CONFIG';
