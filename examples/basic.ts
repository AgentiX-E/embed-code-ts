/**
 * Basic Usage Example
 *
 * Demonstrates how to use @agentix-e/embed-code-ts for code embedding.
 *
 * Run: tsx examples/basic.ts
 */

import { EmbedCode } from '../src/index';

async function main() {
  console.log('@agentix-e/embed-code-ts — Basic Example\n');

  // Create an embedder instance
  // For the fully-embedded model (nomic-embed-text-v1.5, 137MB int8)
  const embedder = await EmbedCode.create({
    model: 'nomic-embed-text-v1.5', // Use the smaller model for quick testing
    // model: 'nomic-embed-code-v1', // 7B model (requires download)
    cacheDir: './.cache/embed-code',
    onProgress: (progress) => {
      if (progress.percent % 20 === 0) {
        console.log(`  Download: ${progress.percent}%`);
      }
    },
  });

  const descriptor = embedder.getDescriptor();
  console.log(`Model: ${descriptor.label}`);
  console.log(`Embedding dim: ${descriptor.runtime.embeddingDim}`);
  console.log(`Pooling: ${descriptor.runtime.poolingStrategy}\n`);

  // Generate embeddings
  // For nomic-embed-code, use task prefixes:
  //   Queries: "search_query: {text}"
  //   Code/Docs: "search_document: {code}"
  const texts = [
    'search_query: Calculate the n-th factorial',
    'search_document: def fact(n): return 1 if n <= 1 else n * fact(n-1)',
    'search_query: Sort a list of numbers',
    'search_document: def sort_list(arr): return sorted(arr)',
  ];

  console.log('Generating embeddings...');
  const result = await embedder.embed(texts);

  console.log(`Done! Shape: [${result.shape.join(', ')}]`);
  console.log(`Time: ${result.elapsedMs.toFixed(1)}ms\n`);

  // Compute similarities
  const queryEmb = result.embeddings.slice(0, 3584); // query "factorial"
  const codeEmb = result.embeddings.slice(3584, 7168); // code "fact"

  const sim = embedder.similarity(
    new Float32Array(queryEmb),
    new Float32Array(codeEmb),
  );
  console.log(`Query-Code similarity: ${sim.toFixed(4)}`);

  // Clean up
  embedder.dispose();
}

main().catch(console.error);
