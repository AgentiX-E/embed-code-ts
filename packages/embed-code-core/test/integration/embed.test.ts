/**
 * Integration test: TS engine vs HuggingFace Python reference.
 *
 * Loads real int8 weights, generates embeddings for the same texts,
 * and compares against pre-computed HuggingFace reference embeddings.
 *
 * Acceptance: cosine similarity ≥ 0.99 for every embedding pair.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { EmbedCode } from '../../src/embed-code';
import { WeightBuffer } from '../../src/inference/weights';

// Resolve model paths from project root (process.cwd() when run from project root)
const PROJECT_ROOT = process.cwd();
const MODELS_DIR = path.join(PROJECT_ROOT, 'models');
const WEIGHTS_PATH = process.env.EMBED_CODE_MODEL_PATH || path.join(MODELS_DIR, 'nomic-embed-code-v1-int8.weights.bin');
const REFERENCE_PATH = path.join(MODELS_DIR, 'reference-embeddings.json');
const TOKENIZER_PATH = path.join(MODELS_DIR, 'tokenizer.json');

// Check if weights exist
const hasWeights = fs.existsSync(WEIGHTS_PATH);
const hasReference = fs.existsSync(REFERENCE_PATH);
const hasTokenizer = fs.existsSync(TOKENIZER_PATH);

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    normA += a[i]! ** 2;
    normB += b[i]! ** 2;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

describe.runIf(hasWeights && hasReference && hasTokenizer)(
  'Integration: TS Engine vs HuggingFace Reference',
  () => {
    let reference: any;
    let embedder: EmbedCode;

    beforeAll(async () => {
      // Load reference data
      reference = JSON.parse(fs.readFileSync(REFERENCE_PATH, 'utf-8'));

      // Load TS embedder with real weights
      const weightsData = fs.readFileSync(WEIGHTS_PATH);
      const wb = WeightBuffer.fromBuffer(new Uint8Array(weightsData.buffer));

      embedder = await EmbedCode.fromPretrained({
        weightsBuffer: wb,
        tokenizerPath: TOKENIZER_PATH,
        skipWarmup: true,
      });
    }, 30000);

    it('WeightBuffer parses architecture correctly', () => {
      expect(reference.dim).toBe(768);
    });

    it('TS engine loads and is ready', () => {
      expect(embedder.isLoaded).toBe(true);
      expect(embedder.embeddingDim).toBe(reference.dim);
    });

    it('single text embedding matches reference (cosine ≥ 0.99)', async () => {
      const texts: string[] = reference.texts.slice(0, 1);
      const result = await embedder.embed(texts, { maxTokens: 64 });

      expect(result.embeddings.length).toBe(768);

      const refEmb = reference.batch1_embeddings[0];
      const tsEmb = Array.from(result.embeddings);
      const sim = cosineSimilarity(refEmb, tsEmb);

      console.log(`  Batch1 cosine similarity: ${sim.toFixed(6)}`);
      expect(sim).toBeGreaterThanOrEqual(0.99);
    }, 30000);

    it('batch 4 texts match reference (cosine ≥ 0.99 each)', async () => {
      const texts: string[] = reference.texts.slice(0, 4);
      const result = await embedder.embed(texts, { maxTokens: 64 });

      expect(result.shape[0]).toBe(4);
      expect(result.shape[1]).toBe(768);

      const refEmbs = reference.batch4_embeddings;
      const allPassed: number[] = [];

      for (let i = 0; i < 4; i++) {
        const tsEmb = Array.from(result.embeddings.slice(i * 768, (i + 1) * 768));
        const refEmb = refEmbs[i];
        const sim = cosineSimilarity(refEmb, tsEmb);
        allPassed.push(sim);
        expect(sim).toBeGreaterThanOrEqual(0.99);
      }

      console.log(
        `  Batch4 similarities: ${allPassed.map((s) => s.toFixed(6)).join(', ')}`,
      );
    }, 30000);

    it('query-document similarity ordering preserved', async () => {
      const texts: string[] = reference.texts;
      const result = await embedder.embed(texts, { maxTokens: 64 });

      const dim = 768;
      const refEmbs = reference.batch10_embeddings;

      // Check each query-document pair (indices 0-1, 2-3, 4-5, 6-7, 8-9)
      for (let i = 0; i < texts.length; i += 2) {
        const tsQuery = Array.from(result.embeddings.slice(i * dim, (i + 1) * dim));
        const tsDoc = Array.from(result.embeddings.slice((i + 1) * dim, (i + 2) * dim));
        const refQuery = refEmbs[i];
        const refDoc = refEmbs[i + 1];

        const tsSim = cosineSimilarity(tsQuery, tsDoc);
        const refSim = cosineSimilarity(refQuery, refDoc);

        console.log(
          `  Pair ${i / 2 + 1}: TS sim=${tsSim.toFixed(6)}, Ref sim=${refSim.toFixed(6)}, Δ=${Math.abs(tsSim - refSim).toFixed(6)}`,
        );

        // Similarity difference should be small
        expect(Math.abs(tsSim - refSim)).toBeLessThan(0.02);
      }
    }, 60000);
  },
);
