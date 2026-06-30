#!/usr/bin/env node
/**
 * Benchmark script for embed-code-ts.
 *
 * Runs embedding benchmarks with the real ONNX model. Measures:
 *   - Latency (ms) per batch
 *   - Throughput (tokens/second)
 *   - Memory usage
 *
 * Outputs benchmark-report.json for CI/Pages publication.
 *
 * Usage:
 *   node scripts/benchmark-ci.js
 *
 * Environment:
 *   EMBED_CODE_MODEL_PATH  — Path to ONNX model (required)
 *   BENCH_ITERATIONS       — Number of iterations per config (default: 5)
 */

const { performance } = require('perf_hooks');
const fs = require('fs');
const path = require('path');

const MODEL_PATH = process.env.EMBED_CODE_MODEL_PATH || '';
const ITERATIONS = parseInt(process.env.BENCH_ITERATIONS || '5', 10);

async function main() {
  if (!MODEL_PATH || !fs.existsSync(MODEL_PATH)) {
    console.error('EMBED_CODE_MODEL_PATH is not set or model not found.');
    console.error('Skipping benchmark — no model available.');
    // Write a minimal report so CI doesn't fail
    const report = {
      model: 'unknown',
      timestamp: new Date().toISOString(),
      configs: [],
      note: 'Model not available — benchmark skipped.',
    };
    fs.writeFileSync('benchmark-report.json', JSON.stringify(report, null, 2));
    return;
  }

  const core = require(path.resolve(__dirname, '..', 'packages/embed-code-core/dist/index.cjs'));

  console.log(`Benchmark: embed-code-ts`);
  console.log(`Model: ${MODEL_PATH}`);
  console.log(`Iterations: ${ITERATIONS}`);
  console.log('');

  const embedder = await core.EmbedCode.fromPretrained({ modelPath: MODEL_PATH, skipWarmup: false });
  const dim = embedder.embeddingDim;

  // Benchmark configs: different batch sizes and token lengths
  const configs = [
    { name: 'single-short', batchSize: 1, text: 'def hello(): return "world"' },
    { name: 'single-medium', batchSize: 1, text: 'def factorial(n):\n  if n <= 1:\n    return 1\n  return n * factorial(n - 1)' },
    { name: 'batch-4', batchSize: 4, text: 'def add(a, b): return a + b' },
    { name: 'batch-8', batchSize: 8, text: 'const x = 42' },
  ];

  const results = [];

  for (const cfg of configs) {
    const texts = Array(cfg.batchSize).fill(cfg.text).map((t) => embedder.taskPrefixes.document + t);
    const latencies = [];

    // Warmup (excluded from measurement)
    await embedder.embed(texts);

    for (let i = 0; i < ITERATIONS; i++) {
      const start = performance.now();
      const result = await embedder.embed(texts);
      const elapsed = performance.now() - start;
      latencies.push(elapsed);
    }

    const avgMs = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const minMs = Math.min(...latencies);
    const maxMs = Math.max(...latencies);

    // Estimate total tokens
    const tokenCount = cfg.text.length * cfg.batchSize;
    const throughputTokensPerSec = tokenCount / (avgMs / 1000);

    results.push({
      config: cfg.name,
      batchSize: cfg.batchSize,
      iterations: ITERATIONS,
      avgLatencyMs: Math.round(avgMs * 100) / 100,
      minLatencyMs: Math.round(minMs * 100) / 100,
      maxLatencyMs: Math.round(maxMs * 100) / 100,
      throughputTokensPerSec: Math.round(throughputTokensPerSec),
    });

    console.log(`  ${cfg.name.padEnd(16)} avg=${avgMs.toFixed(1)}ms  min=${minMs.toFixed(1)}ms  tokens/s=${throughputTokensPerSec.toFixed(0)}`);
  }

  await embedder.dispose();

  // Memory info
  const memUsage = process.memoryUsage();

  const report = {
    model: path.basename(MODEL_PATH),
    embeddingDim: dim,
    timestamp: new Date().toISOString(),
    system: {
      nodeVersion: process.version,
      platform: process.platform,
      arch: process.arch,
    },
    memory: {
      heapUsedMB: Math.round(memUsage.heapUsed / 1024 / 1024),
      heapTotalMB: Math.round(memUsage.heapTotal / 1024 / 1024),
      rssMB: Math.round(memUsage.rss / 1024 / 1024),
    },
    configs: results,
  };

  fs.writeFileSync('benchmark-report.json', JSON.stringify(report, null, 2));
  console.log('\nReport: benchmark-report.json');
}

main().catch((err) => {
  console.error('Benchmark failed:', err.message);
  const report = {
    error: err.message,
    timestamp: new Date().toISOString(),
    configs: [],
  };
  fs.writeFileSync('benchmark-report.json', JSON.stringify(report, null, 2));
  process.exitCode = 0; // Don't fail the CI
});
