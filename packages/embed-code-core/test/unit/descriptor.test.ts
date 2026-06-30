/**
 * Unit tests for model-descriptor module.
 */
import { describe, it, expect } from 'vitest';
import { resolveModelConfig } from '../../src/model-descriptor';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

describe('resolveModelConfig', () => {
  it('falls back to default config when no descriptor found', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-test-'));
    const modelPath = path.join(tmpDir, 'nonexistent.onnx');

    const { config, descriptor } = await resolveModelConfig(modelPath);
    expect(descriptor).toBeNull();
    expect(config.embeddingDim).toBe(3584);
    expect(config.poolingStrategy).toBe('last_token');
    expect(config.normalize).toBe(true);

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('resolves config from model-descriptor.json', async () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-test-'));
    const modelPath = path.join(tmpDir, 'model.onnx');
    fs.writeFileSync(modelPath, ''); // empty dummy

    const descriptor = {
      schema: 1,
      model: {
        name: 'test',
        version: 'v1',
        base_architecture: 'test-arch',
        hf_repository: 'test/model',
        exported_at: '',
        precision: 'int8',
      },
      onnx: {
        input_ids_name: 'input_ids',
        attention_mask_name: 'attention_mask',
        output_name: 'last_hidden_state',
        input_shape: [1, 512],
        output_shape: [1, 512, 768],
        opset: 20,
        sha256: 'abc',
        size_bytes: 1000,
      },
      architecture: {
        embedding_dim: 768,
        num_layers: 12,
        num_heads: 12,
        num_kv_heads: 12,
        head_dim: 64,
        hidden_size: 768,
        intermediate_size: 3072,
        vocab_size: 50000,
        max_position_embeddings: 8192,
        rope_theta: 10000,
        sliding_window: null,
        attention_dropout: 0,
        use_sliding_window: false,
      },
      tokenizer: {
        type: 'bpe',
        vocab_size: 50000,
        max_length: 8192,
        pad_token: '<pad>',
        pad_token_id: 0,
        bos_token: null,
        bos_token_id: null,
        eos_token: null,
        eos_token_id: null,
        unk_token: null,
        unk_token_id: null,
      },
      pooling: { strategy: 'mean', normalize: true },
      task_prefixes: { query: 'query: ', document: 'doc: ' },
    };

    fs.writeFileSync(path.join(tmpDir, 'model-descriptor.json'), JSON.stringify(descriptor));

    const { config, descriptor: result } = await resolveModelConfig(modelPath);
    expect(result).not.toBeNull();
    expect(config.embeddingDim).toBe(768);
    expect(config.poolingStrategy).toBe('mean');
    expect(config.maxTokens).toBe(8192);
    expect(config.inputIdsName).toBe('input_ids');

    fs.rmSync(tmpDir, { recursive: true, force: true });
  });
});
