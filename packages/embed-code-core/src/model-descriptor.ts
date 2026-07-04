/**
 * Model configuration — frozen singleton derived from model-descriptor.json.
 *
 * The descriptor is the single source of truth for all model constants.
 * This module resolves it and provides the validated, frozen ModelConfig.
 */

import type { ModelDescriptor, ModelConfig } from './types';
import * as fs from 'node:fs';
import * as path from 'node:path';

/**
 * Default config for nomic-embed-code-v1 (BERT-base, 137M parameters).
 * Used as fallback when no model-descriptor.json is found.
 */
export const EMBED_CODE_V1_CONFIG: Readonly<ModelConfig> = Object.freeze({
  embeddingDim: 768,
  maxTokens: 8192,
  poolingStrategy: 'mean' as const,
  normalize: true,
  inputIdsName: 'input_ids',
  attentionMaskName: 'attention_mask',
  outputName: 'last_hidden_state',
  taskPrefixes: {
    query: 'search_query: ',
    document: 'search_document: ',
  },
});

/**
 * Default config for nomic-embed-text-v1.5 (137M BERT-based).
 */
export const EMBED_TEXT_V15_CONFIG: Readonly<ModelConfig> = Object.freeze({
  embeddingDim: 768,
  maxTokens: 8192,
  poolingStrategy: 'mean' as const,
  normalize: true,
  inputIdsName: 'input_ids',
  attentionMaskName: 'attention_mask',
  outputName: 'last_hidden_state',
  taskPrefixes: {
    query: 'search_query: ',
    document: 'search_document: ',
  },
});

/** Fallback config to use when descriptor is absent */
const FALLBACK_CONFIG = EMBED_CODE_V1_CONFIG;

/**
 * Resolve model config from the model-descriptor.json co-located with the weights file.
 *
 * This is the single source of truth pattern — the descriptor that ships alongside
 * the weights defines the architecture, input/output names, and runtime config.
 * Falls back to EMBED_CODE_V1_CONFIG when no descriptor is present.
 *
 * @param modelPath Path to the weights file (weights.int8.bin)
 * @param fallbackConfig Config to use when no descriptor is found
 * @returns { config, descriptor } — validated ModelConfig and the raw descriptor (or null)
 */
export function resolveModelConfig(
  modelPath: string,
  fallbackConfig: Readonly<ModelConfig> = FALLBACK_CONFIG,
): { config: Readonly<ModelConfig>; descriptor: ModelDescriptor | null } {
  const modelDir = path.dirname(modelPath);
  const descriptorPath = path.join(modelDir, 'model-descriptor.json');

  let descriptor: ModelDescriptor;
  try {
    const raw = fs.readFileSync(descriptorPath, 'utf-8');
    descriptor = JSON.parse(raw);
  } catch {
    // Descriptor not found or unparseable — use fallback
    console.warn(
      '[embed-code-core] Model descriptor not found or invalid, using fallback configuration.',
    );
    return { config: fallbackConfig, descriptor: null };
  }

  // Validate critical descriptor values
  const arch = descriptor.architecture;
  if (typeof arch.embedding_dim !== 'number' || arch.embedding_dim <= 0) {
    throw new Error(
      `[embed-code-core] Invalid descriptor: embedding_dim=${arch.embedding_dim}, must be > 0`,
    );
  }
  if (typeof arch.max_position_embeddings !== 'number' || arch.max_position_embeddings <= 0) {
    throw new Error(
      `[embed-code-core] Invalid descriptor: max_position_embeddings=${arch.max_position_embeddings}, must be > 0`,
    );
  }
  const validStrategies = ['mean', 'cls', 'last_token'];
  const strategy = descriptor.pooling?.strategy;
  if (strategy !== undefined && !validStrategies.includes(strategy)) {
    throw new Error(
      `[embed-code-core] Invalid descriptor: pooling.strategy="${strategy}", must be one of: ${validStrategies.join(', ')}`,
    );
  }

  // Build ModelConfig from descriptor — try `weights`, fall back to `onnx` for compat
  const d = descriptor;
  const w = d.weights ?? d.onnx;
  const config: ModelConfig = {
    embeddingDim: d.architecture.embedding_dim,
    maxTokens: d.architecture.max_position_embeddings,
    poolingStrategy: d.pooling.strategy,
    normalize: d.pooling.normalize,
    inputIdsName: w?.input_ids_name ?? 'input_ids',
    attentionMaskName: w?.attention_mask_name ?? 'attention_mask',
    outputName: w?.output_name ?? 'last_hidden_state',
    taskPrefixes: d.task_prefixes,
  };

  const frozenConfig: ModelConfig = {
    ...config,
    taskPrefixes: Object.freeze({ ...config.taskPrefixes }),
  };
  return { config: Object.freeze(frozenConfig), descriptor };
}

/**
 * Read model descriptor from a directory.
 * Returns null if not found.
 */
export function readModelDescriptor(dirPath: string): ModelDescriptor | null {
  try {
    const descriptorPath = path.join(dirPath, 'model-descriptor.json');
    const raw = fs.readFileSync(descriptorPath, 'utf-8');
    return JSON.parse(raw) as ModelDescriptor;
  } catch {
    return null;
  }
}
