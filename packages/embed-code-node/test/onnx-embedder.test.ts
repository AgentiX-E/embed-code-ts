import { describe, it, expect } from 'vitest';
import { NodeEmbedder } from '../src/onnx-embedder';
import * as fs from 'node:fs';
import * as path from 'node:path';

const MODEL_PATH = process.env.EMBED_CODE_MODEL_PATH || 'models/nomic-embed-code-v1.5.int8.onnx';
const hasModel = fs.existsSync(MODEL_PATH);

describe('NodeEmbedder', () => {
  it('create fails when tokenizer missing', async () => {
    await expect(NodeEmbedder.create({ modelPath: '/nonexistent/model.onnx' })).rejects.toThrow(
      'Tokenizer not found',
    );
  });

  it('create fails when model missing', async () => {
    const dir = path.join(process.cwd(), 'models');
    await expect(
      NodeEmbedder.create({ modelPath: path.join(dir, 'nonexistent.onnx') }),
    ).rejects.toThrow();
  });

  it('embed with real model returns 768-dim normalized vector', async () => {
    if (!hasModel) return;
    const embedder = await NodeEmbedder.create({ modelPath: MODEL_PATH });
    const result = await embedder.embed('search_query: hello world');
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(768);
    // L2-normalized: norm should be ~1.0
    let norm = 0;
    for (let i = 0; i < result.length; i++) norm += result[i] ** 2;
    expect(Math.sqrt(norm)).toBeCloseTo(1, 1);
    await embedder.dispose();
  }, 120000);

  it('embedBatch with real model returns array of embeddings', async () => {
    if (!hasModel) return;
    const embedder = await NodeEmbedder.create({ modelPath: MODEL_PATH });
    const results = await embedder.embedBatch(['query one', 'query two']);
    expect(results.length).toBe(2);
    expect(results[0].length).toBe(768);
    expect(results[1].length).toBe(768);
    await embedder.dispose();
  }, 120000);

  it('embedBatch with onProgress callback', async () => {
    if (!hasModel) return;
    const embedder = await NodeEmbedder.create({ modelPath: MODEL_PATH });
    const progress: number[] = [];
    const results = await embedder.embedBatch(['a', 'b', 'c'], {
      concurrency: 2,
      onProgress: (done, _total) => progress.push(done),
    });
    expect(results.length).toBe(3);
    expect(progress.length).toBeGreaterThanOrEqual(3);
    await embedder.dispose();
  }, 120000);

  it('embed throws when session disposed', async () => {
    if (!hasModel) return;
    const embedder = await NodeEmbedder.create({ modelPath: MODEL_PATH });
    await embedder.dispose();
    await expect(embedder.embed('test')).rejects.toThrow('Session not initialized');
  });

  it('modelInfo reflects correct dimensions', async () => {
    if (!hasModel) return;
    const embedder = await NodeEmbedder.create({ modelPath: MODEL_PATH });
    expect(embedder.dimensions).toBe(768);
    expect(embedder.maxSequenceLength).toBe(512);
    expect(embedder.modelInfo.name).toBe('nomic-embed-code');
    expect(embedder.modelInfo.quantization).toBe('int8');
    await embedder.dispose();
  });
});
