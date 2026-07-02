#!/usr/bin/env node
/**
 * embed-code-ts Benchmark (ONNX Runtime, Node.js).
 *
 * Uses the new ONNX-powered NodeEmbedder for real inference benchmarks.
 *
 * Usage: EMBED_CODE_MODEL_PATH=models/model.onnx node scripts/benchmark-ci.js
 * Env: BENCH_ITERATIONS=10 (default: 5)
 */
const fs = require('fs');
const path = require('path');

const MODEL_PATH = process.env.EMBED_CODE_MODEL_PATH || '';
const ITERATIONS = parseInt(process.env.BENCH_ITERATIONS || '5', 10);
const JSON_OUT = process.argv.includes('--json')
  ? process.argv[process.argv.indexOf('--json') + 1]
  : 'benchmark-report.json';
const MD_OUT = process.argv.includes('--md')
  ? process.argv[process.argv.indexOf('--md') + 1]
  : null;
const HTML_OUT = process.argv.includes('--html')
  ? process.argv[process.argv.indexOf('--html') + 1]
  : null;
const VERBOSE = process.argv.includes('--verbose');

async function main() {
  // Fast path: no model
  if (!MODEL_PATH || !fs.existsSync(MODEL_PATH)) {
    console.error('Model not found. Export with: python3 scripts/export-model.py');
    process.exit(0);
  }

  console.log(`Benchmark: embed-code-ts (ONNX Runtime)`);
  console.log(`Model: ${path.basename(MODEL_PATH)}`);
  console.log(`Iterations: ${ITERATIONS}`);
  if (VERBOSE) console.log();

  // Load wordpiece tokenizer for direct tokenization
  const { WordPieceTokenizer } = require(
    path.resolve(__dirname, '..', 'packages/embed-code-core/dist/index.cjs'),
  );
  const tok = WordPieceTokenizer.fromFile(
    path.join(path.dirname(MODEL_PATH), 'tokenizer.json'),
    512,
  );

  // Load ONNX Runtime
  const ort = require('onnxruntime-node');
  const session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
  });

  const { meanPool, l2Normalize, cosineSimilarity } = require(
    path.resolve(__dirname, '..', 'packages/embed-code-core/dist/index.cjs'),
  );
  const DIM = 768;

  // Helper: embed text → 768-dim Float32Array
  async function embedOne(text) {
    const { inputIds, attentionMask, tokenTypeIds } = tok.tokenize(text);
    const feeds = {
      input_ids: new ort.Tensor(
        'int64',
        BigInt64Array.from(Array.from(inputIds, (n) => BigInt(n))),
        [1, 512],
      ),
      attention_mask: new ort.Tensor(
        'int64',
        BigInt64Array.from(Array.from(attentionMask, (n) => BigInt(n))),
        [1, 512],
      ),
      token_type_ids: new ort.Tensor(
        'int64',
        BigInt64Array.from(Array.from(tokenTypeIds, (n) => BigInt(n))),
        [1, 512],
      ),
    };
    const outputs = await session.run(feeds);
    const hidden = outputs.last_hidden_state.data;
    const pooled = meanPool(hidden, attentionMask, 1, 512, DIM);
    l2Normalize(pooled, 1, DIM);
    return pooled;
  }

  // System info
  const os = require('os');
  const system = {
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch,
    cpuModel: os.cpus()[0]?.model || 'unknown',
    cpuCores: os.cpus().length,
    ramGB: (os.totalmem() / 1024 ** 3).toFixed(1),
  };
  console.log(`System: Node.js ${system.nodeVersion} | ${system.cpuModel}`);

  // Configs
  const configs = [
    { name: 'single-short', text: 'search_document: def hello(): return "world"' },
    {
      name: 'single-medium',
      text: 'search_document: def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)',
    },
    {
      name: 'single-long',
      text:
        'search_document: ' +
        'function fib(n) { return n <= 1 ? n : fib(n-1) + fib(n-2); } '.repeat(5),
    },
  ];

  const results = [];

  // Cold start
  if (VERBOSE) console.log('Cold start:');
  const cs0 = performance.now();
  await embedOne(configs[0].text);
  const coldMs = performance.now() - cs0;
  if (VERBOSE) console.log(`  cold_start_ms = ${coldMs.toFixed(1)}`);

  // Latency
  for (const cfg of configs) {
    const latencies = [];
    await embedOne(cfg.text); // warmup
    for (let i = 0; i < ITERATIONS; i++) {
      const start = performance.now();
      await embedOne(cfg.text);
      latencies.push(performance.now() - start);
    }
    latencies.sort((a, b) => a - b);
    const avg = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    results.push({
      config: cfg.name,
      avgLatencyMs: Math.round(avg * 100) / 100,
      minLatencyMs: Math.round(latencies[0] * 100) / 100,
      maxLatencyMs: Math.round(latencies[latencies.length - 1] * 100) / 100,
      p50LatencyMs: Math.round(latencies[Math.floor(latencies.length / 2)] * 100) / 100,
      coldStartMs: cfg.name === 'single-short' ? Math.round(coldMs * 100) / 100 : undefined,
    });
    console.log(`  ${cfg.name.padEnd(16)} avg=${avg.toFixed(1)}ms`);
  }

  // Accuracy: query-document similarity
  console.log('\nAccuracy:');
  let accuracy = null;
  try {
    const qe = await embedOne('search_query: Recursive factorial implementation');
    const de = await embedOne(
      'search_document: def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)',
    );
    const ue = await embedOne(
      'search_document: class BinaryTree { constructor(v) { this.v = v; } }',
    );
    const qdSim = cosineSimilarity(Array.from(qe), Array.from(de));
    const quSim = cosineSimilarity(Array.from(qe), Array.from(ue));
    accuracy = {
      queryDocSimilarity: Math.round(qdSim * 10000) / 10000,
      queryUnrelatedSimilarity: Math.round(quSim * 10000) / 10000,
      betterThanUnrelated: qdSim > quSim,
    };
    console.log(
      `  query-doc: ${qdSim.toFixed(4)}  query-unrelated: ${quSim.toFixed(4)}  ${qdSim > quSim ? 'PASS' : 'FAIL'}`,
    );
  } catch (e) {
    console.log(`  Accuracy failed: ${e.message}`);
  }

  // Stability
  console.log('\nStability:');
  let stability = null;
  try {
    const snaps = [];
    for (let i = 0; i < 100; i++) {
      await embedOne('search_document: def stable(): pass');
      if (i % 25 === 0 || i === 99) {
        const mu = process.memoryUsage();
        snaps.push({ iteration: i, heapMB: Math.round((mu.heapUsed / 1024 / 1024) * 100) / 100 });
      }
    }
    const delta = snaps[snaps.length - 1].heapMB - snaps[0].heapMB;
    const deltaPct = snaps[0].heapMB > 0 ? (delta / snaps[0].heapMB) * 100 : 0;
    stability = {
      firstMB: snaps[0].heapMB,
      lastMB: snaps[snaps.length - 1].heapMB,
      deltaMB: Math.round(delta * 100) / 100,
      deltaPct: Math.round(deltaPct * 100) / 100,
      stable: Math.abs(deltaPct) <= 5,
    };
    console.log(
      `  ${stability.stable ? 'PASS' : 'WARN'}: ${stability.firstMB} → ${stability.lastMB} MB (Δ${stability.deltaPct}%)`,
    );
  } catch (e) {
    console.log(`  Stability failed: ${e.message}`);
  }

  session.release();

  const mem = process.memoryUsage();
  const report = {
    model: path.basename(MODEL_PATH),
    torchModule: 'onnxruntime',
    dim: DIM,
    timestamp: new Date().toISOString(),
    system,
    memory: {
      heapUsedMB: Math.round(mem.heapUsed / 1024 / 1024),
      rssMB: Math.round(mem.rss / 1024 / 1024),
    },
    latency: results,
    accuracy,
    stability,
  };

  fs.writeFileSync(JSON_OUT, JSON.stringify(report, null, 2));
  console.log(`\nJSON: ${JSON_OUT}`);
}

main().catch((err) => {
  console.error('Benchmark failed:', err.message);
  process.exitCode = 0;
});
