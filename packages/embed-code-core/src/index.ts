/**
 * @agentix-e/embed-code-core
 *
 * TypeScript/Node.js implementation of nomic-embed-code —
 * a state-of-the-art code embedding model for semantic code search.
 *
 * The npm package is code-only (~50 KB).  The int8 ONNX model is distributed
 * as a GitHub Release asset and downloaded on first use.
 *
 * @example
 * ```typescript
 * import { EmbedCode, downloadModel } from '@agentix-e/embed-code-core';
 *
 * const modelPath = await downloadModel();
 * const embedder = await EmbedCode.fromPretrained({ modelPath });
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

// Model downloader
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

// Inference engine (for advanced use)
export { EmbedCodeInferenceEngine } from './inference/onnx-engine';

// Tokenizer (for advanced use)
export { Tokenizer } from './tokenizer';

// Pooling utilities
export {
  poolEmbeddings,
  normalizeEmbeddings,
  cosineSimilarity,
} from './pooling';

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
  OrtTensor,
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
