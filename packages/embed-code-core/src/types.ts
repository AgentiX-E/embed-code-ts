/**
 * Type definitions for embed-code-ts.
 *
 * Mirroring the model-descriptor.json schema as TypeScript types.
 */

// ─── Model Descriptor (mirrors models/model-descriptor.json) ──

export interface ModelDescriptor {
  schema: number;
  model: {
    name: string;
    version: string;
    base_architecture: string;
    hf_repository: string;
    hf_revision: string;
    exported_at: string;
    precision: string;
  };
  onnx: {
    input_ids_name: string;
    attention_mask_name: string;
    output_name: string;
    input_shape?: number[];
    output_shape?: number[];
    opset: number;
    sha256: string;
    size_bytes: number;
  };
  architecture: ModelArchitecture;
  tokenizer: TokenizerDescriptor;
  pooling: PoolingDescriptor;
  task_prefixes: TaskPrefixDescriptor;
}

export interface ModelArchitecture {
  embedding_dim: number;
  num_layers: number;
  num_heads: number;
  num_kv_heads: number;
  head_dim: number;
  hidden_size: number;
  intermediate_size: number;
  vocab_size: number;
  max_position_embeddings: number;
  rope_theta: number;
  sliding_window: number | null;
  attention_dropout: number;
  use_sliding_window: boolean;
}

export interface TokenizerDescriptor {
  type: 'bpe';
  vocab_size: number;
  max_length: number;
  pad_token: string | null;
  pad_token_id: number;
  bos_token: string | null;
  bos_token_id: number | null;
  eos_token: string | null;
  eos_token_id: number | null;
  unk_token: string | null;
  unk_token_id: number | null;
}

export interface PoolingDescriptor {
  strategy: 'mean' | 'cls' | 'last_token';
  normalize: boolean;
}

export interface TaskPrefixDescriptor {
  query: string;
  document: string;
}

// ─── Model Load Options ─────────────────────────────────────

export interface ModelLoadOptions {
  /** Path to the ONNX model file. Required. */
  modelPath: string;
  /** ONNX Runtime execution provider: 'cpu' (default), 'cuda', 'dml' */
  executionProvider?: string;
  /** Number of intra-op threads for CPU provider */
  intraOpNumThreads?: number;
  /** Skip the warmup inference run */
  skipWarmup?: boolean;
  /** Path to tokenizer.json (default: resolve from model directory) */
  tokenizerPath?: string;
}

// ─── Embed Options ──────────────────────────────────────────

export interface EmbedOptions {
  /** Maximum tokens per input (truncation) */
  maxTokens?: number;
  /** Override pooling strategy */
  poolingStrategy?: 'mean' | 'cls' | 'last_token';
  /** Override normalization */
  normalize?: boolean;
  /** Abort signal for cancellation */
  signal?: AbortSignal;
  /** Progress callback */
  onProgress?: (progress: EmbedProgress) => void;
}

export interface EmbedProgress {
  phase: 'tokenize' | 'inference' | 'pool' | 'normalize';
  step: number;
  total: number;
}

// ─── Embedding Result ───────────────────────────────────────

export interface EmbeddingResult {
  /** Float32Array of embeddings [batchSize, embeddingDim] */
  embeddings: Float32Array;
  /** Shape of the output tensor */
  shape: [number, number];
  /** Time taken in milliseconds */
  elapsedMs: number;
}

// ─── Download Types ─────────────────────────────────────────

export interface DownloadOptions {
  /** Target file path (default: ~/.cache/agentix-embed-code-ts/nomic-embed-code-v1-int8.onnx) */
  dest?: string;
  /** Force re-download even if file exists */
  force?: boolean;
  /** Progress callback: (receivedMB, totalMB, speedMBs) */
  onProgress?: (received: number, total: number, speed: number) => void;
  /** Alternative download URL (for mirrors) */
  url?: string;
  /** Custom logger (defaults to console.error) */
  logger?: (msg: string) => void;
  /** Proxy configuration */
  proxy?: ProxyConfig;
  /** Precision variant: 'int8' (default for embed-code) */
  precision?: string;
}

export interface ProxyConfig {
  url: string;
  username?: string;
  password?: string;
}

// ─── Inference Engine Interface ─────────────────────────────

export interface IInferenceEngine {
  isLoaded(): boolean;
  load(modelPath: string, options?: { skipWarmup?: boolean }): Promise<void>;
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>;
  dispose(): Promise<void>;
}

export interface OrtTensor {
  data: Float32Array | Int32Array | BigInt64Array;
  dims: number[];
  type: string;
}

// ─── Tokenizer Types ────────────────────────────────────────

export interface TokenizationResult {
  inputIds: Int32Array;
  attentionMask: Int32Array;
}

// ─── Model Config (frozen singleton derived from descriptor) ─

export interface ModelConfig {
  readonly embeddingDim: number;
  readonly maxTokens: number;
  readonly poolingStrategy: 'mean' | 'cls' | 'last_token';
  readonly normalize: boolean;
  readonly inputIdsName: string;
  readonly attentionMaskName: string;
  readonly outputName: string;
  readonly taskPrefixes: TaskPrefixDescriptor;
}
