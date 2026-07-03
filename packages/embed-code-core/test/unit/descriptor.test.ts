/**
 * Unit tests for model-descriptor module.
 *
 * Covers:
 *   - Fallback to default config when no descriptor present
 *   - Config resolution from valid model-descriptor.json
 *   - Invalid JSON descriptor fallback
 *   - readModelDescriptor return values
 *   - EMBED_CODE_V1_CONFIG and EMBED_TEXT_V15_CONFIG constants
 */
import { describe, it, expect } from 'vitest';
import {
  resolveModelConfig,
  readModelDescriptor,
  EMBED_CODE_V1_CONFIG,
  EMBED_TEXT_V15_CONFIG,
} from '../../src/model-descriptor';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

describe('EMBED_CODE_V1_CONFIG', () => {
  it('has correct nomic-embed-code BERT-base dimensions', () => {
    expect(EMBED_CODE_V1_CONFIG.embeddingDim).toBe(768);
    expect(EMBED_CODE_V1_CONFIG.maxTokens).toBe(8192);
    expect(EMBED_CODE_V1_CONFIG.poolingStrategy).toBe('mean');
    expect(EMBED_CODE_V1_CONFIG.normalize).toBe(true);
  });

  it('has correct input/output names', () => {
    expect(EMBED_CODE_V1_CONFIG.inputIdsName).toBe('input_ids');
    expect(EMBED_CODE_V1_CONFIG.attentionMaskName).toBe('attention_mask');
    expect(EMBED_CODE_V1_CONFIG.outputName).toBe('last_hidden_state');
  });

  it('has task prefixes', () => {
    expect(EMBED_CODE_V1_CONFIG.taskPrefixes.query).toBe('search_query: ');
    expect(EMBED_CODE_V1_CONFIG.taskPrefixes.document).toBe('search_document: ');
  });

  it('is frozen (immutable)', () => {
    expect(Object.isFrozen(EMBED_CODE_V1_CONFIG)).toBe(true);
  });
});

describe('EMBED_TEXT_V15_CONFIG', () => {
  it('has correct nomic-embed-text-v1.5 dimensions', () => {
    expect(EMBED_TEXT_V15_CONFIG.embeddingDim).toBe(768);
    expect(EMBED_TEXT_V15_CONFIG.maxTokens).toBe(8192);
    expect(EMBED_TEXT_V15_CONFIG.poolingStrategy).toBe('mean');
    expect(EMBED_TEXT_V15_CONFIG.normalize).toBe(true);
  });

  it('is frozen (immutable)', () => {
    expect(Object.isFrozen(EMBED_TEXT_V15_CONFIG)).toBe(true);
  });
});

describe('resolveModelConfig', () => {
  it('falls back to default config when no descriptor found', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-test-'));
    const modelPath = path.join(tmpDir, 'nonexistent.weights.bin');

    const { config, descriptor } = resolveModelConfig(modelPath);
    expect(descriptor).toBeNull();
    expect(config.embeddingDim).toBe(768);
    expect(config.poolingStrategy).toBe('mean');
    expect(config.normalize).toBe(true);

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('resolves config from valid model-descriptor.json', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-test-'));
    const modelPath = path.join(tmpDir, 'model.weights.bin');
    fs.writeFileSync(modelPath, 'dummy content');

    const descriptor = {
      schema: 1,
      model: {
        name: 'nomic-embed-code',
        version: 'v1',
        base_architecture: 'BERT-base',
        hf_repository: 'nomic-ai/nomic-embed-code',
        hf_revision: 'abc123',
        exported_at: '2026-01-01T00:00:00Z',
        precision: 'int8',
      },
      onnx: {
        input_ids_name: 'input_ids',
        attention_mask_name: 'attention_mask',
        output_name: 'last_hidden_state',
        input_shape: [1, 512],
        output_shape: [1, 512, 768],
        opset: 18,
        sha256: 'abc123def456',
        size_bytes: 137000000,
      },
      architecture: {
        embedding_dim: 768,
        num_layers: 12,
        num_heads: 12,
        num_kv_heads: 0,
        head_dim: 64,
        hidden_size: 768,
        intermediate_size: 3072,
        vocab_size: 40856,
        max_position_embeddings: 8192,
        rope_theta: 0,
        sliding_window: null,
        attention_dropout: 0.0,
        use_sliding_window: false,
      },
      tokenizer: {
        type: 'bpe',
        vocab_size: 40856,
        max_length: 8192,
        pad_token: '<|endoftext|>',
        pad_token_id: 0,
        bos_token: null,
        bos_token_id: null,
        eos_token: '<|endoftext|>',
        eos_token_id: 0,
        unk_token: null,
        unk_token_id: null,
      },
      pooling: { strategy: 'mean', normalize: true },
      task_prefixes: { query: 'search_query: ', document: 'search_document: ' },
    };

    fs.writeFileSync(path.join(tmpDir, 'model-descriptor.json'), JSON.stringify(descriptor));

    const { config, descriptor: result } = resolveModelConfig(modelPath);
    expect(result).not.toBeNull();
    expect(config.embeddingDim).toBe(768);
    expect(config.poolingStrategy).toBe('mean');
    expect(config.maxTokens).toBe(8192);
    expect(config.inputIdsName).toBe('input_ids');
    expect(config.attentionMaskName).toBe('attention_mask');
    expect(config.outputName).toBe('last_hidden_state');
    expect(config.taskPrefixes.query).toBe('search_query: ');
    expect(config.taskPrefixes.document).toBe('search_document: ');

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('falls back on invalid JSON descriptor', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-test-'));
    const modelPath = path.join(tmpDir, 'model.weights.bin');
    fs.writeFileSync(modelPath, 'dummy content');
    fs.writeFileSync(path.join(tmpDir, 'model-descriptor.json'), '{invalid json');

    const { config, descriptor } = resolveModelConfig(modelPath);
    expect(descriptor).toBeNull();
    expect(config.embeddingDim).toBe(768); // fallback

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('resolves config with weights field instead of onnx', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-test-'));
    const modelPath = path.join(tmpDir, 'model.weights.bin');
    fs.writeFileSync(modelPath, 'dummy content');

    const descriptor = {
      schema: 1,
      model: {
        name: 'nomic-embed-code',
        version: 'v1',
        base_architecture: 'BERT-base',
        hf_repository: 'nomic-ai/nomic-embed-code',
        hf_revision: 'abc123',
        exported_at: '2026-01-01T00:00:00Z',
        precision: 'int8',
      },
      weights: {
        input_ids_name: 'custom_input_ids',
        attention_mask_name: 'custom_attention_mask',
        output_name: 'custom_output',
      },
      architecture: {
        embedding_dim: 768,
        num_layers: 12,
        num_heads: 12,
        num_kv_heads: 0,
        head_dim: 64,
        hidden_size: 768,
        intermediate_size: 3072,
        vocab_size: 40856,
        max_position_embeddings: 8192,
        rope_theta: 0,
        sliding_window: null,
        attention_dropout: 0.0,
        use_sliding_window: false,
      },
      tokenizer: {
        type: 'bpe',
        vocab_size: 40856,
        max_length: 8192,
        pad_token: '<|endoftext|>',
        pad_token_id: 0,
        bos_token: null,
        bos_token_id: null,
        eos_token: '<|endoftext|>',
        eos_token_id: 0,
        unk_token: null,
        unk_token_id: null,
      },
      pooling: { strategy: 'mean', normalize: true },
      task_prefixes: { query: 'search_query: ', document: 'search_document: ' },
    };

    fs.writeFileSync(path.join(tmpDir, 'model-descriptor.json'), JSON.stringify(descriptor));

    const { config } = resolveModelConfig(modelPath);
    expect(config.inputIdsName).toBe('custom_input_ids');
    expect(config.attentionMaskName).toBe('custom_attention_mask');
    expect(config.outputName).toBe('custom_output');

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('falls back when descriptor has no weights or onnx', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-test-'));
    const modelPath = path.join(tmpDir, 'model.weights.bin');
    fs.writeFileSync(modelPath, 'dummy content');

    const descriptor = {
      schema: 1,
      model: { name: 'test' },
      architecture: { embedding_dim: 768, max_position_embeddings: 8192 },
      pooling: { strategy: 'mean', normalize: true },
      task_prefixes: { query: 'q', document: 'd' },
    };

    fs.writeFileSync(path.join(tmpDir, 'model-descriptor.json'), JSON.stringify(descriptor));

    const { config } = resolveModelConfig(modelPath);
    expect(config.inputIdsName).toBe('input_ids'); // fallback
    expect(config.attentionMaskName).toBe('attention_mask');
    expect(config.outputName).toBe('last_hidden_state');

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });
  it('uses explicit fallback config when provided', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-test-'));
    const modelPath = path.join(tmpDir, 'model.weights.bin');

    const { config, descriptor } = resolveModelConfig(modelPath, EMBED_TEXT_V15_CONFIG);
    expect(descriptor).toBeNull();
    expect(config.embeddingDim).toBe(768); // text-v1.5 fallback
    expect(config.poolingStrategy).toBe('mean');

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });
});

describe('readModelDescriptor', () => {
  it('returns null for non-existent directory', () => {
    const result = readModelDescriptor('/nonexistent/path/12345');
    expect(result).toBeNull();
  });

  it('returns descriptor for valid directory', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-test-'));
    const descriptor = { schema: 1, model: { name: 'test' } };
    fs.writeFileSync(path.join(tmpDir, 'model-descriptor.json'), JSON.stringify(descriptor));

    const result = readModelDescriptor(tmpDir);
    expect(result).not.toBeNull();
    expect(result!.model.name).toBe('test');

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('returns null for directory without descriptor', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-test-'));
    const result = readModelDescriptor(tmpDir);
    expect(result).toBeNull();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });
});
