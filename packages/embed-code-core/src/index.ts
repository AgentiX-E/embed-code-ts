/**
 * @agentix-e/embed-code-core
 *
 * Zero-dependency core: tokenizer, pooling, normalization, interfaces.
 * No ONNX Runtime dependency — platform backends provided by
 * @agentix-e/embed-code-node and @agentix-e/embed-code-web.
 */
export type { IEmbedder, BatchOptions, ModelInfo } from './embedder-interface';
export type { IOrtBackend, IOrtSession, IOrtTensor } from './ort-backend-interface';
export type {
  TokenizedInput,
  EmbeddingResult,
  EmbedProgress,
  DownloadOptions,
  ProxyConfig,
} from './types';

export { WordPieceTokenizer } from './tokenizer/wordpiece-tokenizer';
export { loadVocab } from './tokenizer/vocab-loader';
export { preTokenize, isPunctuation } from './tokenizer/pre-tokenizer';
export { meanPool, clsPool, lastTokenPool } from './pooler';
export { l2Normalize, cosineSimilarity } from './normalizer';
export { processBatch } from './batch-processor';
export { int32ToBigInt64 } from './int64-utils';

export {
  EmbedCodeError,
  ModelNotFoundError,
  DownloadError,
  ChecksumMismatchError,
  ProxyAuthError,
  InferenceError,
  TokenizationError,
  ModelNotCompiledError,
} from './errors';
