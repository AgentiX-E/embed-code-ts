/**
 * Advanced Usage Example
 *
 * Demonstrates advanced features:
 * - Custom model buffer
 * - Custom tokenizer
 * - Batch processing
 * - Similarity search
 *
 * Run: tsx examples/advanced.ts
 */

import { EmbedCode, Tokenizer } from '../src/index';

async function main() {
  console.log('@agentix-e/embed-code-ts — Advanced Example\n');

  // ─── 1. Using a custom model (not from registry) ─────────

  // You can provide your own ONNX model buffer
  // const customModel = fs.readFileSync('my-model.onnx');
  // const embedder = await EmbedCode.create({
  //   customModelBuffer: customModel.buffer,
  // });

  // ─── 2. Batch processing with progress ───────────────────

  const embedder = await EmbedCode.create({
    model: 'nomic-embed-text-v1.5',
    cacheDir: './.cache/embed-code',
  });

  // Large batch: process in chunks
  const codebase = [
    'def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)',
    'def binary_search(arr, target): ...',
    'class Node: def __init__(self, val): self.val = val',
    'def quicksort(arr): ...',
    'async function fetchData(url) { return await fetch(url); }',
    'const sum = (a, b) => a + b;',
    'public class Main { public static void main(String[] args) {} }',
    'func greet(name string) string { return "Hello, " + name }',
  ];

  const BATCH_SIZE = 4;
  const allEmbeddings: Float32Array[] = [];

  console.log(`Processing ${codebase.length} code snippets in batches of ${BATCH_SIZE}...\n`);

  for (let i = 0; i < codebase.length; i += BATCH_SIZE) {
    const batch = codebase.slice(i, i + BATCH_SIZE);
    const batchWithPrefix = batch.map((code) => `search_document: ${code}`);

    const result = await embedder.embed(batchWithPrefix);

    // Extract individual embeddings
    const dim = embedder.embeddingDim;
    for (let b = 0; b < batch.length; b++) {
      const start = b * dim;
      allEmbeddings.push(result.embeddings.slice(start, start + dim));
    }

    console.log(`  Batch ${Math.floor(i / BATCH_SIZE) + 1}: ${result.elapsedMs.toFixed(1)}ms`);
  }

  console.log(`\nTotal embeddings: ${allEmbeddings.length}`);
  console.log(`Embedding dimension: ${embedder.embeddingDim}\n`);

  // ─── 3. Similarity search ────────────────────────────────

  const query = 'search_query: Find element in sorted array';
  const queryResult = await embedder.embed(query);
  const queryEmb = queryResult.embeddings;

  // Compute similarities
  const similarities = allEmbeddings.map((emb, idx) => ({
    index: idx,
    code: codebase[idx]!.substring(0, 60),
    similarity: embedder.similarity(queryEmb, emb),
  }));

  // Sort by similarity (descending)
  similarities.sort((a, b) => b.similarity - a.similarity);

  console.log('Search results for: "Find element in sorted array"');
  console.log('─'.repeat(60));
  for (const { code, similarity } of similarities.slice(0, 3)) {
    console.log(`  [${similarity.toFixed(4)}] ${code}...`);
  }

  // ─── 4. Model introspection ──────────────────────────────

  console.log('\nRegistered models:');
  for (const model of EmbedCode.getModels()) {
    console.log(`  ${model.id} — ${model.label}`);
    console.log(`    Provider: ${model.provider}, Dim: ${model.runtime.embeddingDim}`);
  }

  embedder.dispose();
}

main().catch(console.error);
