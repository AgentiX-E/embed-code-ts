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
 * Default config for nomic-embed-code-v1 (Qwen2.5-7B based).
 * Used as fallback when no model-descriptor.json is found.
 */
export const EMBED_CODE_V1_CONFIG: Readonly<ModelConfig> = Object.freeze({
  embeddingDim: 3584,
  maxTokens: 32768,
  poolingStrategy: 'last_token' as const,
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
 * Default config for nomic-embed-text-v1.5 (137M).
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
 * Resolve model config from the model-descriptor.json co-located with the ONNX file.
 *
 * This is the single source of truth pattern — the descriptor that ships alongside
 * the ONNX model defines the architecture, input/output names, and runtime config.
 * Falls back to EMBED_CODE_V1_CONFIG when no descriptor is present.
 *
 * @param modelPath Path to the ONNX model file
 * @param fallbackConfig Config to use when no descriptor is found
 * @returns { config, descriptor } — validated ModelConfig and the raw descriptor (or null)
 */
export function resolveModelConfig(
  modelPath: string,
  fallbackConfig: Readonly<ModelConfig> = FALLBACK_CONFIG,
): { config: Readonly<ModelConfig>; descriptor: ModelDescriptor | null } {
  const modelDir = path.dirname(modelPath);
  const descriptorPath = path.join(modelDir, 'model-descriptor.json');

  try {
    const raw = fs.readFileSync(descriptorPath, 'utf-8');
    const descriptor: ModelDescriptor = JSON.parse(raw);

    // Build ModelConfig from descriptor
    const config: ModelConfig = {
      embeddingDim: descriptor.architecture.embedding_dim,
      maxTokens: descriptor.architecture.max_position_embeddings,
      poolingStrategy: descriptor.pooling.strategy,
      normalize: descriptor.pooling.normalize,
      inputIdsName: descriptor.onnx.input_ids_name,
      attentionMaskName: descriptor.onnx.attention_mask_name,
      outputName: descriptor.onnx.output_name,
      taskPrefixes: descriptor.task_prefixes,
    };

    return { config: Object.freeze(config), descriptor };
  } catch {
    // Descriptor not found or invalid — use fallback
    return { config: fallbackConfig, descriptor: null };
  }
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
