/**
 * prepare-pages.js — Generate GitHub Pages static HTML files with SEO optimization
 *
 * Creates:
 *   1. docs/index.html              — Root landing page with Schema.org + Open Graph
 *   2. docs/coverage/index.html     — Coverage dashboard
 *   3. docs/robots.txt             — Search engine crawling rules
 *   4. docs/sitemap.xml            — XML sitemap for search engines
 *   5. docs/llms.txt / docs/llms-full.txt — AI crawler discovery files (copied from root)
 *
 * Used by the CI deploy-pages job.
 */
const fs = require('fs');
const path = require('path');

const BASE_URL = 'https://agentix-e.github.io/embed-code-ts';
const REPO_URL = 'https://github.com/AgentiX-E/embed-code-ts';

function ensureDir(dirPath) {
  if (!fs.existsSync(dirPath)) fs.mkdirSync(dirPath, { recursive: true });
}

function writeRobotsTxt() {
  const content = ['User-agent: *', 'Allow: /', '', `Sitemap: ${BASE_URL}/sitemap.xml`, ''].join(
    '\n',
  );
  fs.writeFileSync('docs/robots.txt', content);
  console.log('[prepare-pages] robots.txt generated');
}

function writeSitemap() {
  const urls = [
    { loc: `${BASE_URL}/`, changefreq: 'weekly', priority: '1.0' },
    { loc: `${BASE_URL}/api/`, changefreq: 'weekly', priority: '0.9' },
    { loc: `${BASE_URL}/benchmark/`, changefreq: 'weekly', priority: '0.8' },
    { loc: `${BASE_URL}/coverage/`, changefreq: 'weekly', priority: '0.7' },
  ];

  const xml = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ...urls.map(
      (u) =>
        `  <url><loc>${u.loc}</loc><changefreq>${u.changefreq}</changefreq><priority>${u.priority}</priority></url>`,
    ),
    '</urlset>',
    '',
  ].join('\n');
  fs.writeFileSync('docs/sitemap.xml', xml);
  console.log('[prepare-pages] sitemap.xml generated');
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
      '<html lang="en"><head><meta charset="UTF-8">',
      '<meta name="viewport" content="width=device-width,initial-scale=1">',
      '<meta name="description" content="embed-code-ts test coverage report — line, branch, function, and statement coverage for the ONNX-powered TypeScript code embedding library.">',
      '<meta property="og:title" content="Coverage Report · embed-code-ts">',
      '<meta property="og:description" content="Test coverage dashboard for embed-code-ts — ONNX-powered code embeddings for Node.js and browser.">',
      '<meta property="og:url" content="https://agentix-e.github.io/embed-code-ts/coverage/">',
      '<meta property="og:type" content="website">',
      '<meta name="google-site-verification" content="kD2WXDZJEAkp_4Sjd55rGyHypMWzs_7oddb6PbHy_JE"/>',
      '<meta name="twitter:card" content="summary">',
      '<title>Coverage · embed-code-ts</title>',
      '<style>',
      'body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.6;color:#1a1a2e}',
      'h1{border-bottom:3px solid #2563eb;padding-bottom:.5rem}',
      '.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;margin:1.5rem 0}',
      '@media(max-width:640px){.grid{grid-template-columns:repeat(2,1fr)}}',
      '.card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:1.25rem;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.05)}',
      '.card .pct{font-size:2.5rem;font-weight:700;color:#16a34a}.card .label{color:#6b7280;font-size:.875rem;margin-top:.25rem}',
      '.btn{display:inline-block;padding:.75rem 2rem;background:#2563eb;color:#fff;border-radius:6px;text-decoration:none;font-weight:600;margin-top:1rem}',
      '</style>',
      '<script type="application/ld+json">',
      '{"@context":"https://schema.org","@type":"WebPage","name":"Coverage Report · embed-code-ts","description":"Test coverage dashboard for embed-code-ts","url":"https://agentix-e.github.io/embed-code-ts/coverage/","isPartOf":{"@type":"WebSite","name":"embed-code-ts","url":"https://agentix-e.github.io/embed-code-ts/"}}</script>',
      '</head><body>',
      '<h1>📈 Coverage Report</h1>',
      '<p><a href="https://github.com/AgentiX-E/embed-code-ts">embed-code-ts</a> — TypeScript code embedding library</p>',
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
    html = [
      '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">',
      '<meta name="viewport" content="width=device-width,initial-scale=1">',
      '<meta name="description" content="embed-code-ts test coverage report">',
      '<title>Coverage · embed-code-ts</title></head><body>',
      '<h1>📈 Coverage</h1><p>Report pending — check back after CI completes.</p>',
      '</body></html>',
    ].join('\n');
  }
  fs.writeFileSync('docs/coverage/index.html', html);
  console.log('[prepare-pages] Coverage index generated');
}

function copyLlmsFiles() {
  const files = ['llms.txt', 'llms-full.txt'];
  for (const file of files) {
    if (fs.existsSync(file)) {
      fs.copyFileSync(file, path.join('docs', file));
      console.log(`[prepare-pages] ${file} copied to docs/`);
    } else {
      console.warn(`[prepare-pages] ${file} not found at repo root, skipping`);
    }
  }
}

function writeRootLandingPage() {
  ensureDir('docs');
  const hasBenchmark =
    fs.existsSync(path.join('docs', 'benchmark', 'benchmark-report.html')) ||
    fs.existsSync(path.join('docs', 'benchmark', 'index.html'));
  const hasCoverage = fs.existsSync(path.join('docs', 'coverage', 'index.html'));
  const hasApi = fs.existsSync(path.join('docs', 'api', 'index.html'));

  const cards = [];
  if (hasApi)
    cards.push(
      '<div class="card"><h2>📚 <a href="api/index.html">API Documentation</a></h2><p>Full TypeDoc reference for all packages: core, node, web, cli</p></div>',
    );
  if (hasBenchmark) {
    const benchLink = fs.existsSync(path.join('docs', 'benchmark', 'index.html'))
      ? 'benchmark/index.html'
      : 'benchmark/benchmark-report.html';
    cards.push(
      `<div class="card"><h2>📊 <a href="${benchLink}">Benchmark Report</a></h2><p>Inference latency, throughput &amp; accuracy benchmarks</p></div>`,
    );
  }
  if (hasCoverage)
    cards.push(
      '<div class="card"><h2>📈 <a href="coverage/">Test Coverage</a></h2><p>Line, branch, function &amp; statement coverage reports</p></div>',
    );
  cards.push(
    '<div class="card"><h2>💻 <a href="https://github.com/AgentiX-E/embed-code-ts">Source Code</a></h2><p>GitHub repository — contribute, report issues, star the project</p></div>',
  );
  cards.push(
    '<div class="card"><h2>📦 <a href="https://www.npmjs.com/search?q=%40agentix-e%2Fembed-code">npm Packages</a></h2><p>@agentix-e/embed-code-core · node · web · cli</p></div>',
  );

  const html = [
    '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">',
    '<meta name="viewport" content="width=device-width,initial-scale=1">',
    '<meta name="description" content="embed-code-ts — ONNX-powered code embeddings for Node.js and browser. Pure TypeScript, int8 quantized BERT model, zero network dependency. Semantic code search and RAG for TypeScript/JavaScript.">',
    '<meta name="keywords" content="code embedding, semantic code search, ONNX, TypeScript, Node.js, browser, int8, BERT, RAG, machine learning, nomic, vector search">',
    '<meta name="author" content="AgentiX-E">',
    '<meta name="robots" content="index, follow">',
    '<link rel="canonical" href="https://agentix-e.github.io/embed-code-ts/">',
    // Open Graph
    '<meta property="og:title" content="embed-code-ts · ONNX Code Embeddings for Node.js & Browser">',
    '<meta property="og:description" content="TypeScript-first code embeddings — int8 quantized BERT model, zero network dependency. Semantic code search, RAG, and AI tools for JavaScript/TypeScript.">',
    '<meta property="og:url" content="https://agentix-e.github.io/embed-code-ts/">',
    '<meta property="og:type" content="website">',
    '<meta property="og:site_name" content="embed-code-ts">',
    '<meta property="og:locale" content="en_US">',
    // Twitter Card
    '<meta name="twitter:card" content="summary_large_image">',
    '<meta name="twitter:title" content="embed-code-ts · ONNX Code Embeddings">',
    '<meta name="twitter:description" content="TypeScript-first code embeddings — int8 quantized, zero network dependency. Works in Node.js and browser.">',
    // Schema.org structured data for LLMs
    '<script type="application/ld+json">',
    JSON.stringify({
      '@context': 'https://schema.org',
      '@type': 'SoftwareApplication',
      name: 'embed-code-ts',
      applicationCategory: 'DeveloperApplication',
      operatingSystem: 'Node.js, Browser',
      description:
        'ONNX-powered code embeddings for Node.js and browser — int8 quantized BERT model, zero network dependency. Pure TypeScript library for semantic code search and RAG.',
      url: 'https://agentix-e.github.io/embed-code-ts/',
      author: { '@type': 'Organization', name: 'AgentiX-E', url: 'https://github.com/AgentiX-E' },
      license: 'https://www.apache.org/licenses/LICENSE-2.0',
      codeRepository: 'https://github.com/AgentiX-E/embed-code-ts',
      programmingLanguage: 'TypeScript',
      offers: { '@type': 'Offer', price: '0', priceCurrency: 'USD' },
      softwareVersion: '0.1.1',
    }),
    '</script>',
    '<title>embed-code-ts · ONNX Code Embeddings for TypeScript</title>',
    '<style>',
    'body{font-family:system-ui,sans-serif;max-width:800px;margin:3rem auto;padding:0 1.5rem;line-height:1.7;color:#1a1a2e;background:#fafbfc}',
    'h1{font-size:2rem;border-bottom:3px solid #2563eb;padding-bottom:.5rem}',
    '.tagline{color:#6b7280;font-size:1.1rem;margin-bottom:2rem}',
    '.card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:1.5rem;margin:1.5rem 0;box-shadow:0 1px 3px rgba(0,0,0,.05);transition:box-shadow .2s}',
    '.card:hover{box-shadow:0 4px 12px rgba(0,0,0,.1)}',
    '.card h2{margin-top:0;font-size:1.3rem}.card a{color:#2563eb;text-decoration:none;font-weight:500}',
    '.card p{color:#6b7280;margin:.5rem 0 0}',
    '.footer{margin-top:3rem;padding-top:1.5rem;border-top:1px solid #e5e7eb;color:#9ca3af;font-size:.875rem}',
    '.footer a{color:#6b7280}',
    '</style></head><body>',
    '<h1>🚀 embed-code-ts</h1>',
    '<p class="tagline">ONNX-powered code embeddings for Node.js and browser — int8 quantized, zero network dependency. Pure TypeScript, production-ready.</p>',
    ...cards,
    '<div class="footer">',
    '<p>Built by <a href="https://github.com/AgentiX-E">AgentiX-E</a> · <a href="https://github.com/AgentiX-E/embed-code-ts">GitHub</a> · <a href="https://www.npmjs.com/search?q=%40agentix-e%2Fembed-code">npm</a> · Apache 2.0</p>',
    '</div>',
    '</body></html>',
  ].join('\n');
  fs.writeFileSync('docs/index.html', html);
  console.log('[prepare-pages] Root landing page generated (SEO-optimized)');
}

writeRobotsTxt();
writeSitemap();
copyLlmsFiles();
writeCoverageIndex();
writeRootLandingPage();
console.log('[prepare-pages] All pages generated with SEO metadata.');
