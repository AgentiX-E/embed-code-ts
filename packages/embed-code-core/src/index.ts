/**
 * @agentix-e/embed-code-core
 *
 * Pure-TypeScript nomic-embed-code inference engine.
 * Int8 weights embedded directly in the npm package (incbin-style).
 * No ONNX Runtime, no native bindings — zero-dependency inference.
 *
 * @example
 * ```typescript
 * import { EmbedCode } from '@agentix-e/embed-code-core';
 *
 * // Load from embedded weights (recommended incbin-style)
 * const embedder = await EmbedCode.fromPretrained({ weightsBuffer, tokenizerPath });
 *
 * const results = await embedder.embed([
 *   'search_query: Calculate factorial',
 *   'search_document: def fact(n): return 1 if n <= 1 else n * fact(n-1)',
 * ]);
 * await embedder.dispose();
 * ```
 *
 * @packageDocumentation
 */

// Main API
export { EmbedCode } from './embed-code';

// Model downloader (for environments without embedded weights)
export {
  downloadModel,
  defaultModelPath,
  getCachedModelPath,
  isModelCached,
} from './model-downloader';

// Model descriptor
export {
  resolveModelConfig,
  readModelDescriptor,
  EMBED_CODE_V1_CONFIG,
  EMBED_TEXT_V15_CONFIG,
} from './model-descriptor';

// Inference engine (pure TypeScript, zero dependencies)
export { EmbedCodeTSEngine } from './inference/ts-engine';
export { WeightBuffer } from './inference/weights';

// Tokenizer (BPE, for advanced use)
export { Tokenizer } from './tokenizer';

// Pooling utilities
export { poolEmbeddings, normalizeEmbeddings, cosineSimilarity } from './pooling';

// Types
export type {
  ModelDescriptor,
  ModelArchitecture,
  TokenizerDescriptor,
  PoolingDescriptor,
  TaskPrefixDescriptor,
  ModelConfig,
  ModelLoadOptions,
  EmbedOptions,
  EmbeddingResult,
  EmbedProgress,
  DownloadOptions,
  ProxyConfig,
  IInferenceEngine,
  Tensor,
  TokenizationResult,
} from './types';

// Errors
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
