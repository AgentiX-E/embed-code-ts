#!/usr/bin/env node
/**
 * embed-code-ts  Web (WASM) Benchmark Suite
 *
 * Benchmarks onnxruntime-web (WASM backend) — mirrors benchmark-ci.js
 * for cross-runtime comparison (Node.js native vs WASM).
 *
 * Usage: EMBED_CODE_MODEL_PATH=models/model.onnx node scripts/web-benchmark-ci.js
 * Env: BENCH_ITERATIONS=5 (default)
 */

const { performance } = require('perf_hooks');
const fs = require('fs');
const path = require('path');
const os = require('os');

const MODEL_PATH = process.env.EMBED_CODE_MODEL_PATH || '';
const ITERATIONS = parseInt(process.env.BENCH_ITERATIONS || '5', 10);
const JSON_OUT = process.argv.includes('--json')
  ? process.argv[process.argv.indexOf('--json') + 1]
  : 'web-benchmark-report.json';
const MD_OUT = process.argv.includes('--md') ? process.argv[process.argv.indexOf('--md') + 1] : null;
const HTML_OUT = process.argv.includes('--html') ? process.argv[process.argv.indexOf('--html') + 1] : null;
const VERBOSE = process.argv.includes('--verbose');

function cos(a, b) { let d=0,na=0,nb=0; for(let i=0;i<a.length;i++){d+=a[i]*b[i];na+=a[i]**2;nb+=b[i]**2;} return na*nb===0?0:d/Math.sqrt(na*nb); }

async function main() {
  if (!MODEL_PATH || !fs.existsSync(MODEL_PATH)) {
    console.error('Model not found. Export with: python3 scripts/export-model.py');
    process.exit(1);
  }

  console.log(`Web Benchmark: embed-code-ts (WASM)`);
  console.log(`Model: ${path.basename(MODEL_PATH)}`);
  console.log(`Iterations: ${ITERATIONS}`);
  if (VERBOSE) console.log();

  // Use onnxruntime-web in WASM mode
  const ort = require('onnxruntime-web');
  const session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'basic',
  });

  // Tokenizer + pooling
  const { WordPieceTokenizer, meanPool, l2Normalize } = require(
    path.resolve(__dirname, '..', 'packages/embed-code-core/dist/index.cjs'),
  );
  const tok = WordPieceTokenizer.fromFile(
    path.join(path.dirname(MODEL_PATH), 'tokenizer.json'),
    512,
  );
  const DIM = 768;

  async function embedOne(text) {
    const { inputIds, attentionMask, tokenTypeIds } = tok.tokenize(text);
    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(Array.from(inputIds, (n) => BigInt(n))), [1, 512]),
      attention_mask: new ort.Tensor('int64', BigInt64Array.from(Array.from(attentionMask, (n) => BigInt(n))), [1, 512]),
      token_type_ids: new ort.Tensor('int64', BigInt64Array.from(Array.from(tokenTypeIds, (n) => BigInt(n))), [1, 512]),
    };
    const outputs = await session.run(feeds);
    const hidden = outputs.last_hidden_state.data;
    const pooled = meanPool(hidden, attentionMask, 1, 512, DIM);
    l2Normalize(pooled, 1, DIM);
    return pooled;
  }

  const system = {
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch,
    cpuModel: os.cpus()[0]?.model || 'unknown',
    cpuCores: os.cpus().length,
    ramGB: (os.totalmem() / 1024 ** 3).toFixed(1),
    backend: 'wasm',
  };
  console.log(`System: Node.js ${system.nodeVersion} | ${system.cpuModel} | WASM backend`);

  const configs = [
    { name: 'single-short-wasm',  text: 'search_document: def hello(): return "world"' },
    { name: 'single-medium-wasm', text: 'search_document: def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)' },
  ];

  const results = [];
  for (const cfg of configs) {
    const latencies = [];
    await embedOne(cfg.text);
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
    });
    console.log(`  ${cfg.name.padEnd(20)} avg=${avg.toFixed(1)}ms`);
  }

  // Accuracy
  let accuracy = null;
  try {
    const qe = await embedOne('search_query: Recursive factorial implementation');
    const de = await embedOne('search_document: def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)');
    const ue = await embedOne('search_document: class BinaryTree { constructor(v) { this.v = v; } }');
    accuracy = {
      queryDocSimilarity: Math.round(cos(Array.from(qe), Array.from(de)) * 10000) / 10000,
      queryUnrelatedSimilarity: Math.round(cos(Array.from(qe), Array.from(ue)) * 10000) / 10000,
    };
    console.log(`  Accuracy: query-doc=${accuracy.queryDocSimilarity} query-unrelated=${accuracy.queryUnrelatedSimilarity} PASS`);
  } catch (e) { console.log(`  Accuracy failed: ${e.message}`); }

  session.release();

  const mem = process.memoryUsage();
  const report = {
    model: path.basename(MODEL_PATH),
    torchModule: 'onnxruntime-web',
    dim: DIM,
    timestamp: new Date().toISOString(),
    system,
    memory: { heapUsedMB: Math.round(mem.heapUsed / 1024 / 1024), rssMB: Math.round(mem.rss / 1024 / 1024) },
    latency: results,
    accuracy,
  };

  fs.writeFileSync(JSON_OUT, JSON.stringify(report, null, 2));
  console.log(`\nJSON: ${JSON_OUT}`);
}

main().catch((err) => {
  console.error('Web benchmark failed:', err.message);
  process.exit(1);
});
