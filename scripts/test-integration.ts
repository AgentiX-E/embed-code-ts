#!/usr/bin/env npx tsx
/**
 * Integration test runner for the pure-TS embedding engine.
 * Run: npx tsx scripts/test-integration.ts
 */
import * as fs from 'fs';
import * as path from 'path';
import { EmbedCode } from '../packages/embed-code-core/src/embed-code';
import { WeightBuffer } from '../packages/embed-code-core/src/inference/weights';

const MODELS_DIR = path.resolve(__dirname, '..', 'models');
const WEIGHTS = path.join(MODELS_DIR, 'nomic-embed-code-v1-int8.weights.bin');
const TOKENIZER = path.join(MODELS_DIR, 'tokenizer.json');
const REFERENCE = path.join(MODELS_DIR, 'reference-pre-norm.json');

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

async function main() {
  console.log('═'.repeat(60));
  console.log('Integration Test: TS Engine vs HuggingFace Reference');
  console.log('═'.repeat(60));

  // Load weights
  if (!fs.existsSync(WEIGHTS)) {
    console.error('❌ Weights file not found:', WEIGHTS);
    process.exit(1);
  }
  const rawData = fs.readFileSync(WEIGHTS);
  const wb = WeightBuffer.fromBuffer(new Uint8Array(rawData.buffer));
  console.log(
    `\n📦 Weights: ${(rawData.length / 1024 ** 2).toFixed(1)} MB, ${wb.names.length} tensors, arch: ${JSON.stringify(wb.archParams)}`,
  );

  // Load reference
  let reference: any = null;
  if (fs.existsSync(REFERENCE)) {
    reference = JSON.parse(fs.readFileSync(REFERENCE, 'utf-8'));
    console.log(`📐 Reference: ${reference.dim}-dim, ${reference.texts.length} texts`);
  }

  // Load engine
  console.log('\n🔧 Loading engine...');
  const t0 = Date.now();
  const embedder = await EmbedCode.fromPretrained({
    weightsBuffer: wb,
    tokenizerPath: TOKENIZER,
    skipWarmup: true,
  });
  console.log(
    `   Loaded in ${Date.now() - t0}ms, dim=${embedder.embeddingDim}, maxTokens=${embedder.maxTokens}`,
  );

  // Test 1: Single text embedding
  console.log('\n📏 Test 1: Single text embedding...');
  const t1 = Date.now();
  const r1 = await embedder.embed(['search_query: How to sort an array?'], { maxTokens: 32 });
  console.log(
    `   Shape: ${r1.shape}, time: ${Date.now() - t1}ms, first 5: [${Array.from(
      r1.embeddings.slice(0, 5),
    )
      .map((v) => v.toFixed(6))
      .join(', ')}]`,
  );
  let nans = 0;
  for (let i = 0; i < r1.embeddings.length; i++) if (isNaN(r1.embeddings[i])) nans++;
  console.log(`   NaN count: ${nans}/${r1.embeddings.length} ${nans === 0 ? '✅' : '❌'}`);

  // Test 2: Batch 4 texts
  const texts = [
    'search_query: How to sort an array using quicksort?',
    'search_document: def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x > arr[0]])',
    'search_query: Recursive factorial implementation',
    'search_document: def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)',
  ];
  console.log('\n📏 Test 2: Batch 4 texts...');
  const t2 = Date.now();
  const r2 = await embedder.embed(texts, { maxTokens: 64 });
  console.log(`   Shape: ${r2.shape}, time: ${Date.now() - t2}ms`);
  nans = 0;
  for (let i = 0; i < Math.min(100, r2.embeddings.length); i++) if (isNaN(r2.embeddings[i])) nans++;
  console.log(`   NaN count (first 100): ${nans} ${nans === 0 ? '✅' : '❌'}`);

  // Test 3: Query-document similarity
  console.log('\n📏 Test 3: Query-document similarity...');
  const dim = embedder.embeddingDim;
  const tsSim01 = cosineSimilarity(
    Array.from(r2.embeddings.slice(0, dim)),
    Array.from(r2.embeddings.slice(dim, dim * 2)),
  );
  const tsSim23 = cosineSimilarity(
    Array.from(r2.embeddings.slice(dim * 2, dim * 3)),
    Array.from(r2.embeddings.slice(dim * 3, dim * 4)),
  );
  console.log(`   sim(query0,doc1) = ${tsSim01.toFixed(6)}`);
  console.log(`   sim(query2,doc3) = ${tsSim23.toFixed(6)}`);

  // Test 4: Reference comparison (if available)
  if (reference) {
    console.log('\n📏 Test 4: Reference comparison...');
    const refTexts = reference.texts;
    const t3 = Date.now();
    const r3 = await embedder.embed(refTexts, { maxTokens: 64 });
    console.log(`   Shape: ${r3.shape}, time: ${Date.now() - t3}ms`);

    const refEmbs = reference.embeddings;
    const sims: number[] = [];
    for (let i = 0; i < refTexts.length; i++) {
      const tsEmb = Array.from(r3.embeddings.slice(i * dim, (i + 1) * dim));
      const refEmb = refEmbs[i];
      const sim = cosineSimilarity(tsEmb, refEmb);
      sims.push(sim);
    }
    const avgSim = sims.reduce((a, b) => a + b, 0) / sims.length;
    const minSim = Math.min(...sims);
    console.log(`   Avg cosine sim: ${avgSim.toFixed(6)}, min: ${minSim.toFixed(6)}`);
    console.log(`   ${minSim >= 0.95 ? '✅ Accuracy verified (≥0.95)' : '❌ Below threshold'}`);

    // Query-doc pair comparison
    console.log('\n   Pair similarities (TS vs Ref):');
    for (let i = 0; i < refTexts.length; i += 2) {
      if (i + 1 >= refTexts.length) break;
      const tsQE = Array.from(r3.embeddings.slice(i * dim, (i + 1) * dim));
      const tsDE = Array.from(r3.embeddings.slice((i + 1) * dim, (i + 2) * dim));
      const refQE = refEmbs[i];
      const refDE = refEmbs[i + 1];
      const tsS = cosineSimilarity(tsQE, tsDE);
      const refS = cosineSimilarity(refQE, refDE);
      console.log(
        `   Pair ${i / 2 + 1}: TS=${tsS.toFixed(6)}, Ref=${refS.toFixed(6)}, Δ=${Math.abs(tsS - refS).toFixed(6)}`,
      );
    }
  }

  await embedder.dispose();
  console.log('\n' + '═'.repeat(60));
  console.log('Integration test complete.');
  console.log('═'.repeat(60));
}

main().catch((e) => {
  console.error('❌', e.message);
  process.exit(1);
});
