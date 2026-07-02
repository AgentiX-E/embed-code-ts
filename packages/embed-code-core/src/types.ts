/**
 * Shared types for @agentix-e/embed-code-core.
 */

// ─── Re-exports from interface modules ──────────────────────────

export type { IEmbedder, BatchOptions, ModelInfo } from './embedder-interface';
export type { IOrtBackend, IOrtSession, IOrtTensor } from './ort-backend-interface';

// ─── Tokenizer types ────────────────────────────────────────────

export interface TokenizedInput {
  inputIds: Int32Array;
  attentionMask: Int32Array;
  tokenTypeIds: Int32Array;
}

// ─── Embedding result ───────────────────────────────────────────

export interface EmbeddingResult {
  embeddings: Float32Array;
  shape: [number, number];
  elapsedMs: number;
}

// ─── Progress ────────────────────────────────────────────────────

export interface EmbedProgress {
  phase: 'tokenize' | 'inference' | 'pool' | 'normalize';
  step: number;
  total: number;
}

// ─── Model descriptor types (for model-descriptor.json) ──────────

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
  weights?: {
    input_ids_name: string;
    attention_mask_name: string;
    output_name: string;
    sha256: string;
    size_bytes: number;
  };
  architecture: {
    embedding_dim: number;
    num_layers: number;
    num_heads: number;
    head_dim: number;
    hidden_size: number;
    intermediate_size: number;
    vocab_size: number;
    max_position_embeddings: number;
  };
  tokenizer: {
    type: string;
    vocab_size: number;
    max_length: number;
    pad_token_id: number;
    bos_token_id: number | null;
    eos_token_id: number | null;
  };
  pooling: {
    strategy: 'mean' | 'cls' | 'last_token';
    normalize: boolean;
  };
  task_prefixes: {
    query: string;
    document: string;
  };
}

export interface ModelConfig {
  readonly embeddingDim: number;
  readonly maxTokens: number;
  readonly poolingStrategy: 'mean' | 'cls' | 'last_token';
  readonly normalize: boolean;
  readonly inputIdsName: string;
  readonly attentionMaskName: string;
  readonly outputName: string;
  readonly taskPrefixes: { query: string; document: string };
}

// ─── Download / proxy ────────────────────────────────────────────

export interface DownloadOptions {
  dest?: string;
  force?: boolean;
  onProgress?: (received: number, total: number, speed: number) => void;
  url?: string;
  logger?: (msg: string) => void;
  proxy?: ProxyConfig;
  precision?: string;
}

export interface ProxyConfig {
  url: string;
  username?: string;
  password?: string;
}
