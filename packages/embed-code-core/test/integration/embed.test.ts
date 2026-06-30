/**
 * Integration tests for EmbedCode (requires ONNX model).
 *
 * These tests are skipped when EMBED_CODE_MODEL_PATH is not set
 * or the model file is not found.
 */
import { describe, it, expect } from 'vitest';
import * as fs from 'node:fs';

const MODEL_PATH = process.env.EMBED_CODE_MODEL_PATH || '';

const hasModel = MODEL_PATH && fs.existsSync(MODEL_PATH);

describe.runIf(hasModel)('EmbedCode Integration', () => {
  it('loads model and generates embeddings', async () => {
    const { EmbedCode } = await import('../../src/embed-code');

    const embedder = await EmbedCode.fromPretrained({ modelPath: MODEL_PATH });
    expect(embedder.isLoaded).toBe(true);
    expect(embedder.embeddingDim).toBeGreaterThan(0);

    const result = await embedder.embed(
      [
        embedder.taskPrefixes.query + 'Calculate factorial',
        embedder.taskPrefixes.document + 'def fact(n): return 1 if n <= 1 else n * fact(n-1)',
      ],
      { maxTokens: 64 },
    );

    expect(result.embeddings).toBeInstanceOf(Float32Array);
    expect(result.shape[0]).toBe(2);
    expect(result.shape[1]).toBe(embedder.embeddingDim);
    expect(result.elapsedMs).toBeGreaterThan(0);

    await embedder.dispose();
    expect(embedder.isLoaded).toBe(false);
  }, 60000);

  it('computes similarity between query and code', async () => {
    const { EmbedCode } = await import('../../src/embed-code');
    const embedder = await EmbedCode.fromPretrained({ modelPath: MODEL_PATH });

    const result = await embedder.embed(
      [
        embedder.taskPrefixes.query + 'Sort an array',
        embedder.taskPrefixes.document + 'def sort(arr): return sorted(arr)',
      ],
      { maxTokens: 64 },
    );

    const dim = embedder.embeddingDim;
    const sim = embedder.similarity(result.embeddings.slice(0, dim), result.embeddings.slice(dim));

    expect(sim).toBeGreaterThan(-1.1);
    expect(sim).toBeLessThan(1.1);

    await embedder.dispose();
  }, 60000);
});
