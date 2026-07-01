/**
 * prepare-pages.js — Generate GitHub Pages static HTML files
 *
 * Creates:
 *   1. docs/index.html              — Root landing page with navigation cards
 *   2. docs/coverage/index.html     — Coverage dashboard
 *
 * Used by the CI deploy-pages job to avoid shell escaping issues.
 */
const fs = require('fs');
const path = require('path');

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) fs.mkdirSync(dirPath, { recursive: true });
}

function writeCoverageIndex() {
  ensureDir('docs/coverage');
  let html;
  try {
    const summary = JSON.parse(
      fs.readFileSync(path.join('docs', 'coverage', 'coverage-summary.json'), 'utf-8'),
    ).total;
    const pct = (k) => (summary[k]?.pct ?? 0).toFixed(1);
    const hasLcov = fs.existsSync(path.join('docs', 'coverage', 'lcov-report', 'index.html'));

    html = [
      '<!DOCTYPE html>',
      '<html lang="en"><head><meta charset="UTF-8"><title>Coverage · embed-code-ts</title>',
      '<style>',
      'body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.6;color:#1a1a2e}',
      'h1{border-bottom:3px solid #2563eb;padding-bottom:.5rem}',
      '.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.5rem 0}',
      '.card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:1.25rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.05)}',
      '.card .pct{font-size:2.5rem;font-weight:700;color:#16a34a}.card .label{color:#6b7280;font-size:.875rem;margin-top:.25rem}',
      '.btn{display:inline-block;padding:.75rem 2rem;background:#2563eb;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;margin-top:1rem}',
      '</style></head><body>',
      '<h1>📈 Coverage Report</h1>',
      '<p><a href="https://github.com/AgentiX-E/embed-code-ts">embed-code-ts</a></p>',
      '<div class="grid">',
      `  <div class="card"><div class="pct">${pct('lines')}%</div><div class="label">Lines</div></div>`,
      `  <div class="card"><div class="pct">${pct('branches')}%</div><div class="label">Branches</div></div>`,
      `  <div class="card"><div class="pct">${pct('functions')}%</div><div class="label">Functions</div></div>`,
      `  <div class="card"><div class="pct">${pct('statements')}%</div><div class="label">Statements</div></div>`,
      '</div>',
      '<p>Thresholds: ≥95% lines · functions · statements, ≥90% branches</p>',
      hasLcov ? '<a class="btn" href="lcov-report/index.html">📊 View Detailed Report</a>' : '',
      '</body></html>',
    ].join('\n');
  } catch (e) {
    html =
      '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Coverage</title></head><body><h1>📈 Coverage</h1><p>Report pending — check back after CI completes.</p></body></html>';
  }
  fs.writeFileSync('docs/coverage/index.html', html);
  console.log('[prepare-pages] Coverage index generated');
}

function writeRootLandingPage() {
  ensureDir('docs');
  const html = [
    '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>embed-code-ts · Docs</title>',
    '<style>',
    'body{font-family:system-ui,sans-serif;max-width:800px;margin:3rem auto;padding:0 1.5rem;line-height:1.7;color:#1a1a2e;background:#fafbfc}',
    'h1{font-size:2rem;border-bottom:3px solid #2563eb;padding-bottom:.5rem}',
    '.card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:1.5rem;margin:1.5rem 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}',
    '.card h2{margin-top:0;font-size:1.3rem}.card a{color:#2563eb;text-decoration:none;font-weight:500}',
    '.card p{color:#6b7280;margin:.5rem 0 0}',
    '</style></head><body>',
    '<h1>🚀 embed-code-ts</h1><p>Pure-TypeScript code embeddings — ONNX Runtime, int8 quantized.</p>',
    '<div class="card"><h2>📚 <a href="api/index.html">API Documentation</a></h2><p>TypeDoc reference for all packages</p></div>',
    '<div class="card"><h2>📈 <a href="coverage/">Test Coverage</a></h2><p>Code coverage reports</p></div>',
    '<div class="card"><h2>📖 <a href="https://github.com/AgentiX-E/embed-code-ts">Source Code</a></h2><p>GitHub repository</p></div>',
    '</body></html>',
  ].join('\n');
  fs.writeFileSync('docs/index.html', html);
  console.log('[prepare-pages] Root landing page generated');
}

writeCoverageIndex();
writeRootLandingPage();
console.log('[prepare-pages] All pages generated.');
