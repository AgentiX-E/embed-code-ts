#!/usr/bin/env node
/**
 * embed-code-ts Pipeline Tool
 *
 * Usage:
 *   node scripts/pipeline.js              # Full pipeline
 *   node scripts/pipeline.js --export      # Export weights only
 *   node scripts/pipeline.js --test        # Run tests only
 *   node scripts/pipeline.js --quick       # Skip export (quick mode)
 */

const { execSync, spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const MODEL_PATH =
  process.env.EMBED_CODE_MODEL_PATH || path.join(ROOT, 'models', 'nomic-embed-code-v1.5.int8.onnx');
const HF_MODEL = process.env.EMBED_CODE_HF_MODEL || 'nomic-ai/nomic-embed-code';

const COLORS = {
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  cyan: '\x1b[36m',
  reset: '\x1b[0m',
};
const info = (msg) => console.log(`${COLORS.green}[✓]${COLORS.reset} ${msg}`);
const warn = (msg) => console.log(`${COLORS.yellow}[!]${COLORS.reset} ${msg}`);
const fail = (msg) => {
  console.log(`${COLORS.red}[✗]${COLORS.reset} ${msg}`);
  process.exit(1);
};
const title = (msg) => console.log(`\n${COLORS.cyan}━━━ ${msg} ━━━${COLORS.reset}`);

function run(cmd, opts = {}) {
  const label = opts.label || cmd.split(' ').slice(0, 3).join(' ') + ' …';
  process.stdout.write(`  ${label}`);
  try {
    const result = execSync(cmd, {
      cwd: ROOT,
      stdio: opts.stdio || ['pipe', 'pipe', 'pipe'],
      timeout: opts.timeout || 600_000,
      env: { ...process.env, NODE_OPTIONS: '' },
    });
    console.log(` ${COLORS.green}✓${COLORS.reset}`);
    if (!opts.silent && result.length > 0 && opts.stdio !== 'inherit') {
      const out = result.toString().trim();
      if (out)
        out
          .split('\n')
          .slice(-8)
          .forEach((l) => console.log(`  ${l}`));
    }
    return true;
  } catch (e) {
    console.log(` ${COLORS.red}FAILED${COLORS.reset}`);
    if (e.stderr) console.error(e.stderr.toString().slice(-500));
    if (opts.fatal !== false) process.exit(1);
    return false;
  }
}

// ─── Pipeline Steps ────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const doExport = args.includes('--export') || args.length === 0;
  const doTest = args.includes('--test') || args.length === 0;
  const quick = args.includes('--quick');

  title('embed-code-ts Pipeline');

  // Step 1: Export Weights
  if (doExport && !quick) {
    title('Step 1: Export Weights');
    const pyCheck = run('python3 --version', { fatal: false, silent: true });
    if (!pyCheck) {
      warn('Python3 not found. Skipping weights export.');
      info('The weights file must already exist at: ' + MODEL_PATH);
    } else {
      const cmd = `python3 scripts/export-weights.py --output ${MODEL_PATH} --model ${HF_MODEL}`;
      run(cmd, { label: 'python3 export-weights.py', timeout: 1800_000 });
    }
  }

  // Step 2: Build TypeScript
  title('Step 2: Build TypeScript');
  run('npm run build', { label: 'tsc' });

  // Step 3: Run tests (if model exists)
  if (doTest) {
    title('Step 3: Run Tests');
    if (fs.existsSync(MODEL_PATH)) {
      info(
        `Using model: ${MODEL_PATH} (${(fs.statSync(MODEL_PATH).size / 1024 ** 2).toFixed(0)} MB)`,
      );
    } else {
      warn('Model not found. Some integration tests may be skipped.');
    }
    run('npx vitest run', { label: 'vitest' });
  }

  // Step 4: Summary
  title('Pipeline Complete');
  if (fs.existsSync(MODEL_PATH)) {
    const sha = execSync(`sha256sum ${MODEL_PATH} | cut -d' ' -f1`, { cwd: ROOT })
      .toString()
      .trim();
    info(`Model: ${MODEL_PATH} (SHA256: ${sha.slice(0, 16)}...)`);
  }
  info('Run "npm run release" to create a GitHub Release with the model.');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
