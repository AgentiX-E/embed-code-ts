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
