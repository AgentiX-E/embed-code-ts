/**
 * Tests for the EmbedCode engine
 */

import { describe, it, expect, beforeAll } from 'vitest';

// Note: Full integration tests require onnxruntime-node and model weights.
// These unit tests focus on the core logic that doesn't require ONNX Runtime.

describe('EmbedCode', () => {
  it('should list available models', async () => {
    const { listModels } = await import('../src/registry');
    const models = listModels();
    expect(models).toContain('nomic-embed-code-v1');
    expect(models).toContain('nomic-embed-text-v1.5');
  });

  it('should resolve default model', async () => {
    const { resolveModel } = await import('../src/registry');
    const model = resolveModel();
    expect(model.id).toBe('nomic-embed-code-v1');
    expect(model.runtime.embeddingDim).toBe(3584);
    expect(model.runtime.poolingStrategy).toBe('last_token');
  });

  it('should resolve nomic-embed-text-v1.5 model', async () => {
    const { resolveModel } = await import('../src/registry');
    const model = resolveModel('nomic-embed-text-v1.5');
    expect(model.runtime.embeddingDim).toBe(768);
    expect(model.runtime.poolingStrategy).toBe('mean');
  });
});

describe('Tokenizer', () => {
  it('should initialize with basic vocabulary', async () => {
    const { Tokenizer } = await import('../src/tokenizer');

    const tokenizer = new Tokenizer();

    // Create a minimal tokenizer.json for testing
    const minimalVocab = {
      vocab: {
        '<|endoftext|>': 0,
        'hello': 1,
        'world': 2,
        'search': 3,
        'query': 4,
        'document': 5,
        'def': 6,
      },
      model: {
        type: 'bpe',
        vocab: {
          '<|endoftext|>': 0,
          'hello': 1,
          'world': 2,
          'search': 3,
          'query': 4,
          'document': 5,
          'def': 6,
        },
        merges: [],
      },
    };

    const buffer = new TextEncoder().encode(JSON.stringify(minimalVocab)).buffer;
    await tokenizer.initialize(buffer);

    expect(tokenizer.vocabSize).toBeGreaterThan(0);
  });
});

describe('Pooling', () => {
  it('should perform last-token pooling', async () => {
    const { poolEmbeddings } = await import('../src/pooling');

    // Create a simple tensor [1, 3, 2] (batch=1, seq=3, hidden=2)
    const hiddenStates = new Float32Array([
      0.1, 0.2, // token 0
      0.3, 0.4, // token 1
      0.5, 0.6, // token 2 (last non-padding)
    ]);
    const attentionMask = new Int32Array([1, 1, 1]);

    const result = poolEmbeddings(
      hiddenStates,
      attentionMask,
      { batchSize: 1, seqLen: 3, hiddenDim: 2 },
      'last_token',
    );

    // Should pick the last token's hidden states
    expect(result[0]).toBeCloseTo(0.5);
    expect(result[1]).toBeCloseTo(0.6);
  });

  it('should handle padding in last-token pooling', async () => {
    const { poolEmbeddings } = await import('../src/pooling');

    const hiddenStates = new Float32Array([
      0.1, 0.2, // token 0
      0.3, 0.4, // token 1 (last non-padding)
      0.5, 0.6, // padding
    ]);
    const attentionMask = new Int32Array([1, 1, 0]); // Last is padding

    const result = poolEmbeddings(
      hiddenStates,
      attentionMask,
      { batchSize: 1, seqLen: 3, hiddenDim: 2 },
      'last_token',
    );

    // Should pick token 1 (index 1 is last non-padding)
    expect(result[0]).toBeCloseTo(0.3);
    expect(result[1]).toBeCloseTo(0.4);
  });

  it('should perform mean pooling', async () => {
    const { poolEmbeddings } = await import('../src/pooling');

    const hiddenStates = new Float32Array([
      1.0, 2.0, // token 0
      3.0, 4.0, // token 1
      5.0, 6.0, // padding
    ]);
    const attentionMask = new Int32Array([1, 1, 0]);

    const result = poolEmbeddings(
      hiddenStates,
      attentionMask,
      { batchSize: 1, seqLen: 3, hiddenDim: 2 },
      'mean',
    );

    // Mean of token 0 and token 1: (1+3)/2=2, (2+4)/2=3
    expect(result[0]).toBeCloseTo(2.0);
    expect(result[1]).toBeCloseTo(3.0);
  });
});

describe('Embedding utilities', () => {
  it('should normalize embeddings', async () => {
    const { normalizeEmbeddings } = await import('../src/pooling');

    const embeddings = new Float32Array([3.0, 4.0]); // norm = 5
    normalizeEmbeddings(embeddings, 1, 2);

    expect(embeddings[0]).toBeCloseTo(0.6);
    expect(embeddings[1]).toBeCloseTo(0.8);
  });

  it('should compute cosine similarity', async () => {
    const { cosineSimilarity } = await import('../src/pooling');

    const a = new Float32Array([1.0, 0.0]);
    const b = new Float32Array([0.0, 1.0]);

    const sim = cosineSimilarity(a, b);
    expect(sim).toBeCloseTo(0.0); // Orthogonal

    const c = new Float32Array([1.0, 0.0]);
    const d = new Float32Array([1.0, 0.0]);
    expect(cosineSimilarity(c, d)).toBeCloseTo(1.0); // Parallel
  });
});

describe('base64ToArrayBuffer', () => {
  it('should decode base64 to ArrayBuffer', async () => {
    const { base64ToArrayBuffer, arrayBufferToBase64 } = await import(
      '../src/providers/embedded-provider'
    );

    const original = 'Hello, World!';
    const base64 = arrayBufferToBase64(
      new TextEncoder().encode(original).buffer,
    );
    const decoded = base64ToArrayBuffer(base64);
    const result = new TextDecoder().decode(decoded);

    expect(result).toBe(original);
  });
});
