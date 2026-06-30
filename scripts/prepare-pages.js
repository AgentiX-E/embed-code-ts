#!/usr/bin/env node
/**
 * Prepare GitHub Pages deployment.
 *
 * Generates/updates docs/index.html with links to API docs, coverage, and benchmarks.
 * Also generates docs/coverage/index.html as a coverage dashboard.
 * Called by the CI deploy-pages job.
 */

const fs = require('fs');
const path = require('path');

const DOCS_DIR = path.resolve(__dirname, '..', 'docs');
const PKG_JSON = require(path.resolve(__dirname, '..', 'package.json'));

// Read benchmark data if available
let benchmarkData = null;
try {
  benchmarkData = JSON.parse(
    fs.readFileSync(path.join(DOCS_DIR, 'benchmark', 'benchmark-report.json'), 'utf-8'),
  );
} catch {
  // No benchmark data
}

// Try HTML benchmark report
let hasBenchmarkHTML = false;
try {
  hasBenchmarkHTML = fs.existsSync(path.join(DOCS_DIR, 'benchmark', 'benchmark-report.html'));
} catch {
  // No HTML benchmark
}

// Read coverage data
let coveragePct = null;
let coverageDetails = null;
try {
  const cov = JSON.parse(
    fs.readFileSync(path.join(DOCS_DIR, 'coverage', 'coverage-summary.json'), 'utf-8'),
  );
  if (cov?.total?.lines) {
    coveragePct = {
      lines: Math.round(cov.total.lines.pct),
      branches: Math.round(cov.total.branches.pct),
      functions: Math.round(cov.total.functions.pct),
      statements: Math.round(cov.total.statements.pct),
    };
    coverageDetails = {
      lines: cov.total.lines,
      branches: cov.total.branches,
      functions: cov.total.functions,
      statements: cov.total.statements,
    };
  }
} catch {
  // No coverage data
}

// Read integration coverage data
let integrationCoveragePct = null;
try {
  const cov = JSON.parse(
    fs.readFileSync(path.join(DOCS_DIR, 'coverage-integration', 'coverage-summary.json'), 'utf-8'),
  );
  if (cov?.total?.lines) {
    integrationCoveragePct = {
      lines: Math.round(cov.total.lines.pct),
      branches: Math.round(cov.total.branches.pct),
      functions: Math.round(cov.total.functions.pct),
      statements: Math.round(cov.total.statements.pct),
    };
  }
} catch {
  // No integration coverage
}

// ── Coverage badge helper ──────────────────────────────────
function coverageBadge(pct) {
  if (pct === null || pct === undefined) return '<span class="badge badge-unknown">N/A</span>';
  if (pct >= 95) return `<span class="badge badge-green">${pct}%</span>`;
  if (pct >= 80) return `<span class="badge badge-yellow">${pct}%</span>`;
  return `<span class="badge badge-red">${pct}%</span>`;
}

// Read package names and versions
const packages = [];
for (const pkgDir of ['packages/embed-code-core', 'packages/embed-code-cli']) {
  try {
    const pkgPath = path.resolve(__dirname, '..', pkgDir, 'package.json');
    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf-8'));
    packages.push({ name: pkg.name, version: pkg.version, dir: pkgDir.split('/').pop() });
  } catch {
    // skip
  }
}

// ── Generate coverage dashboard ────────────────────────────
const coverageDashboard = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>embed-code-ts — Coverage Dashboard</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1a1a2e; max-width: 960px; margin: 0 auto; padding: 2rem; }
    h1 { font-size: 2rem; margin-bottom: 0.5rem; }
    h2 { font-size: 1.3rem; margin-top: 1.5rem; margin-bottom: 0.5rem; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.25rem; }
    .subtitle { color: #666; margin-bottom: 1.5rem; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
    .badge-green { background: #d4edda; color: #155724; }
    .badge-yellow { background: #fff3cd; color: #856404; }
    .badge-red { background: #f8d7da; color: #721c24; }
    .badge-unknown { background: #e2e3e5; color: #383d41; }
    table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #e0e0e0; }
    th { background: #f5f5f5; font-weight: 600; }
    .card { border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; margin: 0.5rem; text-align: center; flex: 1; min-width: 120px; }
    .card .metric { font-size: 2rem; font-weight: 700; }
    .card .label { font-size: 0.85rem; color: #666; }
    .card-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem -0.5rem; }
    .green { color: #155724; }
    .yellow { color: #856404; }
    .red { color: #721c24; }
    a { color: #2563eb; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .nav { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
    .nav a { padding: 0.5rem 1rem; background: #f0f0f0; border-radius: 6px; font-weight: 500; }
    .nav a:hover { background: #e0e0e0; text-decoration: none; }
    .progress { height: 8px; background: #e0e0e0; border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }
    .progress-fill { height: 100%; border-radius: 4px; }
    .progress-fill.green { background: #28a745; }
    .progress-fill.yellow { background: #ffc107; }
    .progress-fill.red { background: #dc3545; }
  </style>
</head>
<body>
  <h1>Coverage Dashboard</h1>
  <p class="subtitle">embed-code-ts — Code coverage report</p>

  <div class="nav">
    <a href="../index.html">Home</a>
    <a href="../api/index.html">API Documentation</a>
    <a href="../benchmark/benchmark-report.html">Benchmark Report</a>
    ${fs.existsSync(path.join(DOCS_DIR, 'coverage', 'lcov-report', 'index.html')) ? '<a href="./lcov-report/index.html">Detailed LCOV Report</a>' : ''}
  </div>

  <h2>Unit Test Coverage</h2>
  <div class="card-row">
    <div class="card">
      <div class="label">Lines</div>
      <div class="metric ${coveragePct && coveragePct.lines >= 95 ? 'green' : coveragePct && coveragePct.lines >= 80 ? 'yellow' : 'red'}">${coveragePct ? coveragePct.lines : 'N/A'}%</div>
      <div class="progress"><div class="progress-fill ${coveragePct && coveragePct.lines >= 95 ? 'green' : coveragePct && coveragePct.lines >= 80 ? 'yellow' : 'red'}" style="width:${coveragePct ? coveragePct.lines : 0}%"></div></div>
      ${coverageDetails ? `<div class="label" style="margin-top:0.25rem">${coverageDetails.lines.covered}/${coverageDetails.lines.total}</div>` : ''}
    </div>
    <div class="card">
      <div class="label">Branches</div>
      <div class="metric ${coveragePct && coveragePct.branches >= 95 ? 'green' : coveragePct && coveragePct.branches >= 80 ? 'yellow' : 'red'}">${coveragePct ? coveragePct.branches : 'N/A'}%</div>
      <div class="progress"><div class="progress-fill ${coveragePct && coveragePct.branches >= 95 ? 'green' : coveragePct && coveragePct.branches >= 80 ? 'yellow' : 'red'}" style="width:${coveragePct ? coveragePct.branches : 0}%"></div></div>
      ${coverageDetails ? `<div class="label" style="margin-top:0.25rem">${coverageDetails.branches.covered}/${coverageDetails.branches.total}</div>` : ''}
    </div>
    <div class="card">
      <div class="label">Functions</div>
      <div class="metric ${coveragePct && coveragePct.functions >= 95 ? 'green' : coveragePct && coveragePct.functions >= 80 ? 'yellow' : 'red'}">${coveragePct ? coveragePct.functions : 'N/A'}%</div>
      <div class="progress"><div class="progress-fill ${coveragePct && coveragePct.functions >= 95 ? 'green' : coveragePct && coveragePct.functions >= 80 ? 'yellow' : 'red'}" style="width:${coveragePct ? coveragePct.functions : 0}%"></div></div>
      ${coverageDetails ? `<div class="label" style="margin-top:0.25rem">${coverageDetails.functions.covered}/${coverageDetails.functions.total}</div>` : ''}
    </div>
    <div class="card">
      <div class="label">Statements</div>
      <div class="metric ${coveragePct && coveragePct.statements >= 95 ? 'green' : coveragePct && coveragePct.statements >= 80 ? 'yellow' : 'red'}">${coveragePct ? coveragePct.statements : 'N/A'}%</div>
      <div class="progress"><div class="progress-fill ${coveragePct && coveragePct.statements >= 95 ? 'green' : coveragePct && coveragePct.statements >= 80 ? 'yellow' : 'red'}" style="width:${coveragePct ? coveragePct.statements : 0}%"></div></div>
      ${coverageDetails ? `<div class="label" style="margin-top:0.25rem">${coverageDetails.statements.covered}/${coverageDetails.statements.total}</div>` : ''}
    </div>
  </div>

  ${
    integrationCoveragePct
      ? `
  <h2>Integration Test Coverage</h2>
  <div class="card-row">
    <div class="card">
      <div class="label">Lines</div>
      <div class="metric ${integrationCoveragePct.lines >= 95 ? 'green' : integrationCoveragePct.lines >= 80 ? 'yellow' : 'red'}">${integrationCoveragePct.lines}%</div>
      <div class="progress"><div class="progress-fill ${integrationCoveragePct.lines >= 95 ? 'green' : integrationCoveragePct.lines >= 80 ? 'yellow' : 'red'}" style="width:${integrationCoveragePct.lines}%"></div></div>
    </div>
    <div class="card">
      <div class="label">Branches</div>
      <div class="metric ${integrationCoveragePct.branches >= 95 ? 'green' : integrationCoveragePct.branches >= 80 ? 'yellow' : 'red'}">${integrationCoveragePct.branches}%</div>
      <div class="progress"><div class="progress-fill ${integrationCoveragePct.branches >= 95 ? 'green' : integrationCoveragePct.branches >= 80 ? 'yellow' : 'red'}" style="width:${integrationCoveragePct.branches}%"></div></div>
    </div>
    <div class="card">
      <div class="label">Functions</div>
      <div class="metric ${integrationCoveragePct.functions >= 95 ? 'green' : integrationCoveragePct.functions >= 80 ? 'yellow' : 'red'}">${integrationCoveragePct.functions}%</div>
      <div class="progress"><div class="progress-fill ${integrationCoveragePct.functions >= 95 ? 'green' : integrationCoveragePct.functions >= 80 ? 'yellow' : 'red'}" style="width:${integrationCoveragePct.functions}%"></div></div>
    </div>
    <div class="card">
      <div class="label">Statements</div>
      <div class="metric ${integrationCoveragePct.statements >= 95 ? 'green' : integrationCoveragePct.statements >= 80 ? 'yellow' : 'red'}">${integrationCoveragePct.statements}%</div>
      <div class="progress"><div class="progress-fill ${integrationCoveragePct.statements >= 95 ? 'green' : integrationCoveragePct.statements >= 80 ? 'yellow' : 'red'}" style="width:${integrationCoveragePct.statements}%"></div></div>
    </div>
  </div>
  `
      : ''
  }

  <hr style="margin-top: 2rem;">
  <p style="color: #999; font-size: 0.85rem;">Generated by embed-code-ts CI — <a href="https://github.com/AgentiX-E/embed-code-ts">GitHub</a></p>
</body>
</html>
`;

// ── Build benchmark table rows ─────────────────────────────
function benchmarkTable() {
  if (!benchmarkData?.configs?.length) {
    return '<tr><td colspan="6">No benchmark data available</td></tr>';
  }
  return benchmarkData.configs
    .map(
      (c) =>
        `<tr>
      <td>${c.config}</td>
      <td>${c.batchSize}</td>
      <td>${c.avgLatencyMs} ms</td>
      <td>${c.p50LatencyMs || '-'} ms</td>
      <td>${c.p99LatencyMs || '-'} ms</td>
      <td>${c.throughputTokensPerSec} tok/s</td>
    </tr>`,
    )
    .join('\n');
}

function accuracySection() {
  if (!benchmarkData?.accuracy) return '';
  const a = benchmarkData.accuracy;
  return `
  <h2>Accuracy</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Query-Doc Cosine Similarity</td><td>${a.queryDocSimilarity}</td></tr>
    <tr><td>Query-Unrelated Cosine Similarity</td><td>${a.queryUnrelatedSimilarity}</td></tr>
    <tr><td>Better Than Unrelated</td><td>${a.betterThanUnrelated ? '✅ Yes' : '❌ No'}</td></tr>
  </table>`;
}

// ── Root landing page ──────────────────────────────────────
const html = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>embed-code-ts — nomic-embed-code int8 for Node.js</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1a1a2e; max-width: 960px; margin: 0 auto; padding: 2rem; }
    h1 { font-size: 2rem; margin-bottom: 0.5rem; }
    h2 { font-size: 1.4rem; margin-top: 2rem; margin-bottom: 0.5rem; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.25rem; }
    .subtitle { color: #666; margin-bottom: 1.5rem; }
    table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    th, td { padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #e0e0e0; }
    th { background: #f5f5f5; font-weight: 600; }
    .badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
    .badge-green { background: #d4edda; color: #155724; }
    .badge-yellow { background: #fff3cd; color: #856404; }
    a { color: #2563eb; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .nav { display: flex; gap: 1.5rem; margin: 1.5rem 0; flex-wrap: wrap; }
    .nav a { padding: 0.5rem 1rem; background: #f0f0f0; border-radius: 6px; font-weight: 500; }
    .nav a:hover { background: #e0e0e0; text-decoration: none; }
    .coverage-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
    .coverage-item { padding: 0.5rem 1rem; border: 1px solid #e0e0e0; border-radius: 6px; text-align: center; min-width: 80px; }
    .coverage-item .val { font-size: 1.5rem; font-weight: 700; }
    .coverage-item .lbl { font-size: 0.75rem; color: #666; }
  </style>
</head>
<body>
  <h1>embed-code-ts</h1>
  <p class="subtitle">nomic-embed-code int8 ONNX for Node.js — v${PKG_JSON.version}</p>

  <div class="nav">
    <a href="./api/index.html">API Documentation</a>
    <a href="./coverage/index.html">Coverage Report</a>
    ${hasBenchmarkHTML ? '<a href="./benchmark/benchmark-report.html">Benchmark Report</a>' : '<a href="./benchmark/benchmark-report.json">Benchmark Data (JSON)</a>'}
    <a href="https://github.com/AgentiX-E/embed-code-ts">GitHub Repository</a>
  </div>

  <h2>Packages</h2>
  <table>
    <tr><th>Package</th><th>Version</th><th>Description</th><th>API Docs</th></tr>
    ${packages
      .map(
        (p) => `
    <tr>
      <td><code>${p.name}</code></td>
      <td>${p.version}</td>
      <td>${p.dir === 'embed-code-core' ? 'Core inference engine for code embeddings' : 'Command-line interface'}</td>
      <td><a href="./api/modules/${p.dir === 'embed-code-core' ? '' : 'embed_code_cli_src_cli.html'}">TypeDoc</a></td>
    </tr>`,
      )
      .join('\n')}
  </table>

  <h2>Coverage</h2>
  ${
    coveragePct
      ? `
  <div class="coverage-row">
    <div class="coverage-item"><div class="lbl">Lines</div><div class="val">${coverageBadge(coveragePct.lines)}</div></div>
    <div class="coverage-item"><div class="lbl">Branches</div><div class="val">${coverageBadge(coveragePct.branches)}</div></div>
    <div class="coverage-item"><div class="lbl">Functions</div><div class="val">${coverageBadge(coveragePct.functions)}</div></div>
    <div class="coverage-item"><div class="lbl">Statements</div><div class="val">${coverageBadge(coveragePct.statements)}</div></div>
  </div>
  <p><a href="./coverage/index.html">Full Coverage Dashboard →</a></p>
  `
      : '<p>No coverage data available.</p>'
  }

  <h2>Benchmarks</h2>
  ${benchmarkData ? `<p>Model: <strong>${benchmarkData.model}</strong> | Dim: ${benchmarkData.embeddingDim} | Node.js ${benchmarkData.system?.nodeVersion || '?'} | ${benchmarkData.system?.platform || '?'}/${benchmarkData.system?.arch || '?'} | ${benchmarkData.system?.cpuModel || '?'}</p>` : '<p>No benchmark data available.</p>'}
  <table>
    <tr><th>Config</th><th>Batch</th><th>Avg Latency</th><th>P50</th><th>P99</th><th>Throughput</th></tr>
    ${benchmarkTable()}
  </table>
  ${accuracySection()}
  ${
    benchmarkData?.memory
      ? `
  <h2>Memory</h2>
  <table>
    <tr><th>Heap Used</th><th>Heap Total</th><th>RSS</th></tr>
    <tr>
      <td>${benchmarkData.memory.heapUsedMB} MB</td>
      <td>${benchmarkData.memory.heapTotalMB} MB</td>
      <td>${benchmarkData.memory.rssMB} MB</td>
    </tr>
  </table>`
      : ''
  }
  ${
    benchmarkData?.stability
      ? `
  <h2>Stability</h2>
  <table>
    <tr><th>Iterations</th><th>Heap Δ</th><th>Stable</th></tr>
    <tr>
      <td>${benchmarkData.stability.iterations}</td>
      <td>${benchmarkData.stability.deltaMB} MB (${benchmarkData.stability.deltaPct}%)</td>
      <td>${benchmarkData.stability.stable ? '✅ Yes' : '⚠️ No'}</td>
    </tr>
  </table>`
      : ''
  }

  <hr style="margin-top: 2rem;">
  <p style="color: #999; font-size: 0.85rem;">Generated by embed-code-ts CI — <a href="https://github.com/AgentiX-E/embed-code-ts">GitHub</a></p>
</body>
</html>
`;

// Write landing page
fs.writeFileSync(path.join(DOCS_DIR, 'index.html'), html);
console.log('Pages prepared: docs/index.html');

// Write coverage dashboard
const coverageDir = path.join(DOCS_DIR, 'coverage');
if (!fs.existsSync(coverageDir)) {
  fs.mkdirSync(coverageDir, { recursive: true });
}
fs.writeFileSync(path.join(coverageDir, 'index.html'), coverageDashboard);
console.log('Pages prepared: docs/coverage/index.html');
