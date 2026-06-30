/**
 * @agentix-e/embed-code-ts
 *
 * nomic-embed-code int8 ONNX weights embedded in a TypeScript npm package.
 *
 * Incbin-style: binary model weights are compiled directly into the TypeScript
 * bundle as base64 constants, enabling zero-network, offline-first embedding
 * for code search and semantic analysis.
 *
 * @example
 * ```ts
 * import { EmbedCode } from '@agentix-e/embed-code-ts';
 *
 * const embedder = await EmbedCode.create({
 *   model: 'nomic-embed-code-v1',
 *   cacheDir: './.cache/embed-code',
 * });
 *
 * const results = await embedder.embed([
 *   'search_query: Calculate the n-th factorial',
 *   'search_document: def fact(n): return 1 if n <= 1 else n * fact(n-1)',
 * ]);
 *
 * console.log(results.embeddings);
 * // Float32Array [2, 3584]
 * ```
 *
 * @packageDocumentation
 */

// Core engine
export { EmbedCode } from './embed-code';

// Types
export type {
  EmbedCodeOptions,
  EmbeddingResult,
  DownloadProgress,
  ModelDescriptor,
  PoolingStrategy,
  ModelPrecision,
  ProviderType,
  TokenizerConfig,
  TokenizationResult,
  EmbedCodeError,
  ErrorCode,
} from './types';

// Model registry (for introspection)
export { MODEL_REGISTRY, resolveModel, listModels, getModels } from './registry';

// Providers (advanced use)
export { EmbeddedModelProvider } from './providers/embedded-provider';
export { ChunkedModelProvider } from './providers/chunked-provider';
export { RemoteModelProvider } from './providers/remote-provider';

// Utilities
export {
  base64ToArrayBuffer,
  arrayBufferToBase64,
  sha256,
} from './providers/embedded-provider';
export { poolEmbeddings, normalizeEmbeddings, cosineSimilarity } from './pooling';
export { Tokenizer } from './tokenizer';

// Platform adapters
export { createInferenceSession as createNodeSession } from './platform/node';
export { createInferenceSession as createWebSession } from './platform/web';
