/**
 * Model Registry — the single source of truth for all supported embedding models.
 *
 * Adding a new model is as simple as adding an entry to this registry.
 * The build system reads this to know what to download, convert, and embed.
 */

import type { ModelDescriptor } from './types';

export const MODEL_REGISTRY: Record<string, ModelDescriptor> = {
  // ─── Primary Model: nomic-embed-code (7B) ───────────────
  'nomic-embed-code-v1': {
    id: 'nomic-embed-code-v1',
    label: 'Nomic Embed Code v1 (7B)',
    provider: 'remote', // 7B params → Layer 3: fingerprint + download
    precision: 'int8',
    target: {
      repository: 'nomic-ai/nomic-embed-code',
      onnxFile: 'onnx/model_int8.onnx',
      tokenizerFile: 'tokenizer.json',
    },
    runtime: {
      embeddingDim: 3584,
      maxTokens: 32768,
      poolingStrategy: 'last_token',
      normalize: true,
    },
    remote: {
      baseUrl:
        'https://huggingface.co/nomic-ai/nomic-embed-code/resolve/main',
      chunkCount: 16, // Split 7GB into 16 chunks ~450MB each
      sha256: '', // Populated at build time
      totalSize: 0, // Populated at build time
    },
    version: 1,
  },

  // ─── Reference Model: nomic-embed-text v1.5 (137M) ─────
  // 137MB int8 ONNX — small enough for full embedding (Layer 1)
  'nomic-embed-text-v1.5': {
    id: 'nomic-embed-text-v1.5',
    label: 'Nomic Embed Text v1.5 (137M)',
    provider: 'embedded',
    precision: 'int8',
    target: {
      repository: 'nomic-ai/nomic-embed-text-v1.5',
      onnxFile: 'onnx/model_int8.onnx',
      tokenizerFile: 'tokenizer.json',
    },
    runtime: {
      embeddingDim: 768,
      maxTokens: 8192,
      poolingStrategy: 'mean',
      normalize: true,
    },
    version: 1,
  },

  // ─── Future: nomic-embed-code v2 (distilled, small) ────
  // Placeholder for when a distilled/small version is released
  // 'nomic-embed-code-v2': {
  //   id: 'nomic-embed-code-v2',
  //   label: 'Nomic Embed Code v2 (Distilled)',
  //   provider: 'embedded',  // ← small enough for Layer 1!
  //   precision: 'int8',
  //   target: { ... },
  //   runtime: { ... },
  //   version: 1,
  // },
};

/**
 * Resolve a model descriptor, with fallback logic.
 */
export function resolveModel(modelId?: string): ModelDescriptor {
  if (modelId && MODEL_REGISTRY[modelId]) {
    return MODEL_REGISTRY[modelId]!;
  }

  // Default to nomic-embed-code-v1
  const defaultModel = MODEL_REGISTRY['nomic-embed-code-v1'];
  if (!defaultModel) {
    throw new Error(
      'No default model found in registry. The registry may be corrupted.',
    );
  }
  return defaultModel;
}

/**
 * List all available model IDs.
 */
export function listModels(): string[] {
  return Object.keys(MODEL_REGISTRY);
}

/**
 * Get all registered models.
 */
export function getModels(): ModelDescriptor[] {
  return Object.values(MODEL_REGISTRY);
}
