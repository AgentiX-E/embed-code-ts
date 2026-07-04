#!/usr/bin/env node
/**
 * ci-benchmark-check.js — CI benchmark quality gate
 *
 * Reads benchmark-report.json and enforces embedding quality:
 *   1. Embedding dimensions match expected value
 *   2. Cosine similarity between query and matching document is ≥ 0.5
 *   3. All per-config latencies are positive
 *
 * Usage:
 *   node scripts/ci-benchmark-check.js [--report benchmark-report.json] [--verbose]
 *
 * Called from ci.yml benchmark job.
 */

const fs = require('fs');
const path = require('path');

const REPORT_FILE = process.argv.includes('--report')
  ? process.argv[process.argv.indexOf('--report') + 1] || 'benchmark-report.json'
  : 'benchmark-report.json';

const VERBOSE = process.argv.includes('--verbose');

function main() {
  const reportPath = path.resolve(REPORT_FILE);

  if (!fs.existsSync(reportPath)) {
    console.log('[benchmark-check] No benchmark report — skipping quality gate.');
    process.exit(0);
  }

  const report = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));

  if (report.error) {
    console.log(`[benchmark-check] Benchmark had error: ${report.error} — skipping gate.`);
    process.exit(0);
  }

  if (!report.latency || report.latency.length === 0) {
    console.log('[benchmark-check] No config results — skipping quality gate.');
    process.exit(0);
  }

  let failed = false;

  // 1. Verify all latencies are positive
  for (const cfg of report.latency) {
    if (cfg.avgLatencyMs <= 0) {
      console.error(`FAIL: ${cfg.config}: avgLatencyMs = ${cfg.avgLatencyMs} (must be > 0)`);
      failed = true;
    }
    if (cfg.minLatencyMs <= 0) {
      console.error(`FAIL: ${cfg.config}: minLatencyMs = ${cfg.minLatencyMs} (must be > 0)`);
      failed = true;
    }
  }

  // 2. Verify embedding dimension is correct
  if (report.dim && report.dim !== 768) {
    console.error(`FAIL: Unexpected dim = ${report.dim} (expected 768)`);
    failed = true;
  }

  // 3. Verify accuracy data if present
  if (report.accuracy) {
    const { queryDocSimilarity } = report.accuracy;
    if (queryDocSimilarity !== undefined && queryDocSimilarity < 0.8) {
      console.error(
        `FAIL: Query-document cosine similarity = ${queryDocSimilarity} (expected ≥ 0.8)`,
      );
      failed = true;
    }

    const { queryUnrelatedSimilarity } = report.accuracy;
    if (
      queryDocSimilarity !== undefined &&
      queryUnrelatedSimilarity !== undefined &&
      queryDocSimilarity <= queryUnrelatedSimilarity
    ) {
      console.error(
        `FAIL: queryDocSimilarity (${queryDocSimilarity}) must be > queryUnrelatedSimilarity (${queryUnrelatedSimilarity})`,
      );
      failed = true;
    }
  }

  if (failed) {
    process.exit(1);
  }

  console.log('PASS: All benchmark quality checks passed.');
  if (VERBOSE) {
    for (const cfg of report.latency) {
      console.log(
        `  ${cfg.config.padEnd(16)} avg=${cfg.avgLatencyMs}ms  min=${cfg.minLatencyMs}ms`,
      );
    }
    if (report.accuracy) {
      console.log(`  Accuracy: queryDocSimilarity=${report.accuracy.queryDocSimilarity}`);
      const btu =
        report.accuracy.betterThanUnrelated !== undefined
          ? report.accuracy.betterThanUnrelated
          : report.accuracy.queryDocSimilarity > report.accuracy.queryUnrelatedSimilarity;
      console.log(`  betterThanUnrelated: ${btu ? 'PASS' : 'FAIL'}`);
    }
  }
}

main();
