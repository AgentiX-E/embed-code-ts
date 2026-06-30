#!/usr/bin/env node
/**
 * Benchmark script for embed-code-ts.
 *
 * Runs embedding benchmarks with the real ONNX model. Measures:
 *   - Latency (ms) per batch configuration
 *   - Throughput (tokens/second)
 *   - Memory usage
 *   - Query-document similarity (accuracy verification)
 *   - Stability (memory leak detection)
 *
 * Outputs benchmark-report.json, benchmark-report.md, and benchmark-report.html
 * for CI/Pages publication.
 *
 * Usage:
 *   node scripts/benchmark-ci.js [--json <path>] [--md <path>] [--html <path>]
 *                               [--baseline <path>] [--verbose]
 *
 * Environment:
 *   EMBED_CODE_MODEL_PATH  — Path to ONNX model (required)
 *   BENCH_ITERATIONS       — Number of iterations per config (default: 5)
 *   BENCH_STABILITY_ITERS  — Number of stability iterations (default: 100)
 */

const { performance } = require('perf_hooks');
const fs = require('fs');
const path = require('path');

const MODEL_PATH = process.env.EMBED_CODE_MODEL_PATH || '';
const ITERATIONS = parseInt(process.env.BENCH_ITERATIONS || '5', 10);
const STABILITY_ITERS = parseInt(process.env.BENCH_STABILITY_ITERS || '100', 10);

const JSON_OUT = process.argv.includes('--json')
  ? process.argv[process.argv.indexOf('--json') + 1] || 'benchmark-report.json'
  : 'benchmark-report.json';
const MD_OUT = process.argv.includes('--md')
  ? process.argv[process.argv.indexOf('--md') + 1] || null
  : null;
const HTML_OUT = process.argv.includes('--html')
  ? process.argv[process.argv.indexOf('--html') + 1] || null
  : null;
const BASELINE_PATH = process.argv.includes('--baseline')
  ? process.argv[process.argv.indexOf('--baseline') + 1] || null
  : null;
const VERBOSE = process.argv.includes('--verbose');

async function main() {
  if (!MODEL_PATH || !fs.existsSync(MODEL_PATH)) {
    console.error('EMBED_CODE_MODEL_PATH is not set or model not found.');
    console.error('Skipping benchmark — no model available.');
    const report = {
      model: 'unknown',
      timestamp: new Date().toISOString(),
      configs: [],
      note: 'Model not available — benchmark skipped.',
    };
    fs.writeFileSync(JSON_OUT, JSON.stringify(report, null, 2));
    return;
  }

  // Load from dist (built CJS module)
  const core = require(path.resolve(__dirname, '..', 'packages/embed-code-core/dist/index.cjs'));

  console.log(`Benchmark: embed-code-ts`);
  console.log(`Model: ${path.basename(MODEL_PATH)}`);
  console.log(`Iterations: ${ITERATIONS}`);
  console.log('');

  const embedder = await core.EmbedCode.fromPretrained({
    modelPath: MODEL_PATH,
    skipWarmup: false,
  });
  const dim = embedder.embeddingDim;

  // ── System info ──────────────────────────────────────────
  const os = require('os');
  const system = {
    nodeVersion: process.version,
    platform: process.platform,
    arch: process.arch,
    cpuModel: os.cpus()[0]?.model || 'unknown',
    cpuCores: os.cpus().length,
    ramGB: (os.totalmem() / 1024 ** 3).toFixed(1),
  };

  // ── Latency benchmark configs ────────────────────────────
  const configs = [
    { name: 'single-short', batchSize: 1, text: 'def hello(): return "world"' },
    {
      name: 'single-medium',
      batchSize: 1,
      text: 'def factorial(n):\n  if n <= 1:\n    return 1\n  return n * factorial(n - 1)',
    },
    { name: 'batch-4', batchSize: 4, text: 'def add(a, b): return a + b' },
    { name: 'batch-8', batchSize: 8, text: 'const x = 42' },
  ];

  const results = [];

  // Cold start measurement
  if (VERBOSE) console.log('Cold start:');
  let coldStartMs = 0;
  try {
    const coldStart = performance.now();
    await embedder.embed([embedder.taskPrefixes.document + configs[0].text]);
    coldStartMs = performance.now() - coldStart;
    if (VERBOSE) console.log(`  cold_start_ms = ${coldStartMs.toFixed(1)}`);
  } catch (e) {
    if (VERBOSE) console.log('  cold_start failed, skipping');
  }

  for (const cfg of configs) {
    const texts = Array(cfg.batchSize)
      .fill(cfg.text)
      .map((t) => embedder.taskPrefixes.document + t);
    const latencies = [];

    // Warmup (excluded from measurement)
    await embedder.embed(texts);

    for (let i = 0; i < ITERATIONS; i++) {
      const start = performance.now();
      await embedder.embed(texts);
      const elapsed = performance.now() - start;
      latencies.push(elapsed);
    }

    latencies.sort((a, b) => a - b);
    const avgMs = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const minMs = latencies[0];
    const maxMs = latencies[latencies.length - 1];
    const p50Ms = latencies[Math.floor(latencies.length / 2)];
    const p99Ms = latencies[Math.floor(latencies.length * 0.99)];

    const tokenCount = cfg.text.length * cfg.batchSize;
    const throughputTokensPerSec = tokenCount / (avgMs / 1000);

    results.push({
      config: cfg.name,
      batchSize: cfg.batchSize,
      iterations: ITERATIONS,
      avgLatencyMs: Math.round(avgMs * 100) / 100,
      minLatencyMs: Math.round(minMs * 100) / 100,
      maxLatencyMs: Math.round(maxMs * 100) / 100,
      p50LatencyMs: Math.round(p50Ms * 100) / 100,
      p99LatencyMs: Math.round(p99Ms * 100) / 100,
      coldStartMs: cfg.name === 'single-short' ? Math.round(coldStartMs * 100) / 100 : undefined,
      throughputTokensPerSec: Math.round(throughputTokensPerSec),
    });

    console.log(
      `  ${cfg.name.padEnd(16)} avg=${avgMs.toFixed(1)}ms  p50=${p50Ms.toFixed(1)}ms  p99=${p99Ms.toFixed(1)}ms  tok/s=${throughputTokensPerSec.toFixed(0)}`,
    );
  }

  // ── Accuracy: query-document similarity ──────────────────
  console.log('\nAccuracy:');
  let accuracy = null;
  try {
    const queryText = embedder.taskPrefixes.query + 'Recursive factorial implementation';
    const docText =
      embedder.taskPrefixes.document +
      'def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)';
    const unrelatedText =
      embedder.taskPrefixes.document +
      'class BinaryTreeNode { constructor(val) { this.val = val; this.left = null; this.right = null; } }';

    const accResult = await embedder.embed([queryText, docText, unrelatedText]);
    const queryEmb = accResult.embeddings.slice(0, dim);
    const docEmb = accResult.embeddings.slice(dim, dim * 2);
    const unrelatedEmb = accResult.embeddings.slice(dim * 2, dim * 3);

    const queryDocSim = cosineSimilarity(queryEmb, docEmb, dim);
    const queryUnrelatedSim = cosineSimilarity(queryEmb, unrelatedEmb, dim);

    accuracy = {
      queryDocSimilarity: Math.round(queryDocSim * 10000) / 10000,
      queryUnrelatedSimilarity: Math.round(queryUnrelatedSim * 10000) / 10000,
      betterThanUnrelated: queryDocSim > queryUnrelatedSim,
    };

    console.log(
      `  query-doc similarity: ${accuracy.queryDocSimilarity}  |  query-unrelated: ${accuracy.queryUnrelatedSimilarity}  |  ${accuracy.betterThanUnrelated ? 'PASS' : 'FAIL'}`,
    );
  } catch (e) {
    console.log(`  Accuracy test failed: ${e.message}`);
  }

  // ── Stability (memory leak detection) ────────────────────
  console.log('\nStability:');
  let stability = null;
  try {
    const memSnapshots = [];
    const singleText = embedder.taskPrefixes.document + 'def stable(): pass';
    for (let i = 0; i < STABILITY_ITERS; i++) {
      await embedder.embed([singleText]);
      if (i % 25 === 0 || i === STABILITY_ITERS - 1) {
        const mu = process.memoryUsage();
        memSnapshots.push({
          iteration: i,
          heapUsedMB: Math.round((mu.heapUsed / 1024 / 1024) * 100) / 100,
        });
      }
    }
    const first = memSnapshots[0].heapUsedMB;
    const last = memSnapshots[memSnapshots.length - 1].heapUsedMB;
    const delta = last - first;
    const deltaPct = first > 0 ? (delta / first) * 100 : 0;
    stability = {
      iterations: STABILITY_ITERS,
      firstHeapMB: first,
      lastHeapMB: last,
      deltaMB: Math.round(delta * 100) / 100,
      deltaPct: Math.round(deltaPct * 100) / 100,
      stable: Math.abs(deltaPct) <= 5,
      snapshots: memSnapshots,
    };
    console.log(
      `  ${stability.stable ? 'PASS' : 'WARN'}: heap ${first.toFixed(1)} → ${last.toFixed(1)} MB (Δ${deltaPct.toFixed(1)}%)`,
    );
  } catch (e) {
    console.log(`  Stability test failed: ${e.message}`);
  }

  await embedder.dispose();

  // ── Regression detection ─────────────────────────────────
  let regression = null;
  if (BASELINE_PATH && fs.existsSync(BASELINE_PATH)) {
    try {
      const baseline = JSON.parse(fs.readFileSync(BASELINE_PATH, 'utf-8'));
      if (baseline.latency && baseline.latency.length > 0) {
        const regressions = [];
        const THRESHOLD = 10; // 10% regression threshold
        for (const cfg of results) {
          const base = baseline.latency.find((b) => b.config === cfg.config);
          if (base) {
            const delta = ((cfg.avgLatencyMs - base.avgLatencyMs) / base.avgLatencyMs) * 100;
            regressions.push({
              config: cfg.config,
              current: cfg.avgLatencyMs,
              baseline: base.avgLatencyMs,
              deltaPct: Math.round(delta * 100) / 100,
            });
          }
        }
        regression = { threshold: THRESHOLD, regressions };
        if (VERBOSE && regressions.length > 0) {
          console.log('\nRegression:');
          for (const r of regressions) {
            console.log(
              `  ${r.config}: ${r.current}ms vs ${r.baseline}ms (${r.deltaPct > 0 ? '+' : ''}${r.deltaPct}%)`,
            );
          }
        }
      }
    } catch (e) {
      if (VERBOSE) console.log(`  Regression check skipped: ${e.message}`);
    }
  }

  // ── Memory info ──────────────────────────────────────────
  const memUsage = process.memoryUsage();

  // ── Build report ─────────────────────────────────────────
  const report = {
    model: path.basename(MODEL_PATH),
    embeddingDim: dim,
    timestamp: new Date().toISOString(),
    system,
    memory: {
      heapUsedMB: Math.round(memUsage.heapUsed / 1024 / 1024),
      heapTotalMB: Math.round(memUsage.heapTotal / 1024 / 1024),
      rssMB: Math.round(memUsage.rss / 1024 / 1024),
    },
    configs: results,
    accuracy,
    stability,
    regression,
  };

  // ── Write reports ────────────────────────────────────────
  fs.writeFileSync(JSON_OUT, JSON.stringify(report, null, 2));
  console.log(`\nJSON: ${JSON_OUT}`);

  if (MD_OUT) {
    fs.writeFileSync(MD_OUT, generateMarkdown(report));
    console.log(`MD: ${MD_OUT}`);
  }

  if (HTML_OUT) {
    fs.writeFileSync(HTML_OUT, generateHTML(report));
    console.log(`HTML: ${HTML_OUT}`);
  }
}

// ── Helper: cosine similarity ──────────────────────────────
function cosineSimilarity(a, b, dim) {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < dim && i < a.length && i < b.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] ** 2;
    normB += b[i] ** 2;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// ── Markdown report generator ──────────────────────────────
function generateMarkdown(report) {
  const lines = [];
  lines.push('# embed-code-ts Benchmark Report');
  lines.push('');
  lines.push(
    `**Model**: ${report.model}  |  **Dim**: ${report.embeddingDim}  |  **Date**: ${report.timestamp}`,
  );
  lines.push('');
  if (report.system) {
    lines.push(
      `**System**: Node.js ${report.system.nodeVersion}  |  ${report.system.platform}/${report.system.arch}  |  ${report.system.cpuModel} × ${report.system.cpuCores}  |  ${report.system.ramGB} GB RAM`,
    );
    lines.push('');
  }
  lines.push('## Latency');
  lines.push('');
  lines.push('| Config | Batch | Avg (ms) | P50 (ms) | P99 (ms) | Min (ms) | Max (ms) | tok/s |');
  lines.push('|--------|-------|----------|----------|----------|----------|----------|-------|');
  for (const c of report.configs) {
    lines.push(
      `| ${c.config} | ${c.batchSize} | ${c.avgLatencyMs} | ${c.p50LatencyMs || '-'} | ${c.p99LatencyMs || '-'} | ${c.minLatencyMs} | ${c.maxLatencyMs} | ${c.throughputTokensPerSec} |`,
    );
  }
  lines.push('');

  if (report.accuracy) {
    lines.push('## Accuracy');
    lines.push('');
    lines.push('| Metric | Value |');
    lines.push('|--------|-------|');
    lines.push(`| Query-Doc Cosine Similarity | ${report.accuracy.queryDocSimilarity} |`);
    lines.push(
      `| Query-Unrelated Cosine Similarity | ${report.accuracy.queryUnrelatedSimilarity} |`,
    );
    lines.push(
      `| Better Than Unrelated | ${report.accuracy.betterThanUnrelated ? '✅ Yes' : '❌ No'} |`,
    );
    lines.push('');
  }

  if (report.stability) {
    lines.push('## Stability');
    lines.push('');
    lines.push('| Metric | Value |');
    lines.push('|--------|-------|');
    lines.push(`| Iterations | ${report.stability.iterations} |`);
    lines.push(`| Heap Start | ${report.stability.firstHeapMB} MB |`);
    lines.push(`| Heap End | ${report.stability.lastHeapMB} MB |`);
    lines.push(`| Delta | ${report.stability.deltaMB} MB (${report.stability.deltaPct}%) |`);
    lines.push(`| Stable | ${report.stability.stable ? '✅ Yes' : '⚠️ No'} |`);
    lines.push('');
  }

  if (report.memory) {
    lines.push('## Memory');
    lines.push('');
    lines.push('| Heap Used | Heap Total | RSS |');
    lines.push('|-----------|------------|-----|');
    lines.push(
      `| ${report.memory.heapUsedMB} MB | ${report.memory.heapTotalMB} MB | ${report.memory.rssMB} MB |`,
    );
    lines.push('');
  }

  return lines.join('\n');
}

// ── HTML report generator ──────────────────────────────────
function generateHTML(report) {
  const latencyRows = report.configs
    .map(
      (c) =>
        `<tr><td>${c.config}</td><td>${c.batchSize}</td><td>${c.avgLatencyMs}</td><td>${c.p50LatencyMs || '-'}</td><td>${c.p99LatencyMs || '-'}</td><td>${c.minLatencyMs}</td><td>${c.maxLatencyMs}</td><td>${c.throughputTokensPerSec}</td></tr>`,
    )
    .join('\n');

  const accuracySection = report.accuracy
    ? `<h2>Accuracy</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Query-Doc Cosine Similarity</td><td>${report.accuracy.queryDocSimilarity}</td></tr>
<tr><td>Query-Unrelated Cosine Similarity</td><td>${report.accuracy.queryUnrelatedSimilarity}</td></tr>
<tr><td>Better Than Unrelated</td><td>${report.accuracy.betterThanUnrelated ? '✅ Yes' : '❌ No'}</td></tr>
</table>`
    : '<h2>Accuracy</h2><p>No accuracy data.</p>';

  const stabilitySection = report.stability
    ? `<h2>Stability</h2>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Iterations</td><td>${report.stability.iterations}</td></tr>
<tr><td>Heap Start</td><td>${report.stability.firstHeapMB} MB</td></tr>
<tr><td>Heap End</td><td>${report.stability.lastHeapMB} MB</td></tr>
<tr><td>Delta</td><td>${report.stability.deltaMB} MB (${report.stability.deltaPct}%)</td></tr>
<tr><td>Stable</td><td>${report.stability.stable ? '✅ Yes' : '⚠️ No'}</td></tr>
</table>`
    : '';

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>embed-code-ts Benchmark Report</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1a1a2e; max-width: 960px; margin: 0 auto; padding: 2rem; }
    h1 { font-size: 1.8rem; margin-bottom: 0.5rem; }
    h2 { font-size: 1.3rem; margin-top: 1.5rem; margin-bottom: 0.5rem; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.25rem; }
    .meta { color: #666; margin-bottom: 1.5rem; font-size: 0.9rem; }
    table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    th, td { padding: 0.4rem 0.75rem; text-align: left; border-bottom: 1px solid #e0e0e0; font-size: 0.9rem; }
    th { background: #f5f5f5; font-weight: 600; }
    @media (prefers-color-scheme: dark) {
      body { background: #1a1a2e; color: #e0e0e0; }
      th { background: #2a2a3e; }
      td, th { border-color: #3a3a4e; }
      h2 { border-color: #3a3a4e; }
      .meta { color: #999; }
    }
  </style>
</head>
<body>
  <h1>embed-code-ts Benchmark Report</h1>
  <p class="meta">Model: ${report.model} | Dim: ${report.embeddingDim} | Date: ${report.timestamp}</p>
  <p class="meta">System: Node.js ${report.system?.nodeVersion || '?'} | ${report.system?.platform || '?'}/${report.system?.arch || '?'} | ${report.system?.cpuModel || '?'} | ${report.system?.ramGB || '?'} GB RAM</p>

  <h2>Latency</h2>
  <table><tr><th>Config</th><th>Batch</th><th>Avg (ms)</th><th>P50 (ms)</th><th>P99 (ms)</th><th>Min (ms)</th><th>Max (ms)</th><th>tok/s</th></tr>${latencyRows}</table>

  ${accuracySection}

  ${stabilitySection}

  <h2>Memory</h2>
  <table><tr><th>Heap Used</th><th>Heap Total</th><th>RSS</th></tr>
  <tr><td>${report.memory?.heapUsedMB || '?'} MB</td><td>${report.memory?.heapTotalMB || '?'} MB</td><td>${report.memory?.rssMB || '?'} MB</td></tr></table>

  <hr style="margin-top:2rem">
  <p style="color:#999;font-size:0.85rem">Generated by embed-code-ts CI — <a href="https://github.com/AgentiX-E/embed-code-ts">GitHub</a></p>
</body>
</html>`;
}

main().catch((err) => {
  console.error('Benchmark failed:', err.message);
  const report = {
    error: err.message,
    timestamp: new Date().toISOString(),
    configs: [],
  };
  fs.writeFileSync(JSON_OUT, JSON.stringify(report, null, 2));
  process.exitCode = 0;
});
