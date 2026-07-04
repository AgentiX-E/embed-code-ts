#!/usr/bin/env node
/**
 * ci-coverage-check.js — CI coverage threshold verification
 *
 * Reads coverage/coverage-summary.json and enforces ≥95% on lines/functions/statements
 * and ≥90% on branches.  Exits with code 1 if any metric falls below the threshold.
 *
 * Usage:
 *   node scripts/ci-coverage-check.js [--tier unit|integration] [--verbose]
 */
const fs = require('fs');
const path = require('path');

const UNIT_THRESHOLDS = { lines: 95, branches: 85, functions: 95, statements: 95 };
const INTEGRATION_THRESHOLDS = { lines: 85, branches: 75, functions: 85, statements: 85 };
const TIER = process.argv.includes('--tier')
  ? process.argv[process.argv.indexOf('--tier') + 1] || 'unit'
  : 'unit';
const THRESHOLDS =
  TIER === 'integration' || TIER === 'release' ? INTEGRATION_THRESHOLDS : UNIT_THRESHOLDS;
const VERBOSE = process.argv.includes('--verbose');

function main() {
  const summaryPath = path.resolve('coverage', 'coverage-summary.json');

  if (!fs.existsSync(summaryPath)) {
    console.error(`[${TIER}] FAIL: coverage-summary.json not found.`);
    process.exit(1);
  }

  let summary;
  try {
    summary = JSON.parse(fs.readFileSync(summaryPath, 'utf-8'));
  } catch {
    console.error(`[${TIER}] FAIL: Cannot parse coverage JSON.`);
    process.exit(1);
  }

  if (!summary || !summary.total) {
    console.error(`[${TIER}] FAIL: No total coverage data.`);
    process.exit(1);
  }

  const s = summary.total;
  const allZero = ['lines', 'branches', 'functions', 'statements'].every(
    (k) => !s[k] || s[k].pct === 0,
  );
  if (allZero) {
    console.error(`[${TIER}] FAIL: All metrics are 0% — tests likely failed silently.`);
    process.exit(1);
  }

  let failed = false;
  for (const [metric, threshold] of Object.entries(THRESHOLDS)) {
    const pct = s[metric]?.pct ?? 0;
    const status = pct >= threshold ? '\u2705' : '\u274C';
    console.log(`${status} ${TIER} ${metric}: ${pct.toFixed(1)}% (threshold: ${threshold}%)`);
    if (pct < threshold) failed = true;
  }

  if (failed) {
    console.error(`\nFAIL: ${TIER} coverage thresholds not met.`);
    process.exit(1);
  }

  if (VERBOSE) console.log(`\n\u2705 All ${TIER} coverage thresholds met.`);
}

main();
