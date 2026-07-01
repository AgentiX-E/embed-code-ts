/**
 * Vitest global setup — checks for embed-code weights availability
 * before running integration tests.
 *
 * If the weights file is not found, sets VITEST_SKIP_ONNX_TESTS=true so
 * individual test files can conditionally skip model-dependent tests.
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export function setup(): void {
  const envPath = process.env.EMBED_CODE_MODEL_PATH;
  if (envPath && fs.existsSync(envPath)) {
    return;
  }

  const searchPaths = [
    path.resolve(__dirname, 'models'),
    path.resolve(__dirname),
    path.join(os.homedir(), '.cache', 'agentix-embed-code-ts'),
  ];

  const filenames = [
    'nomic-embed-code-v1-int8.weights.bin',
    'nomic-embed-text-v1.5-int8.weights.bin',
    'weights.int8.bin',
  ];

  for (const dir of searchPaths) {
    for (const name of filenames) {
      if (fs.existsSync(path.join(dir, name))) {
        return;
      }
    }
  }

  process.env.VITEST_SKIP_ONNX_TESTS = 'true';
  console.warn(
    '\n\u26a0\ufe0f  Embed-code weights file not found. Model-dependent tests will be skipped.\n' +
      '    Only pure-logic unit tests will run without the weights.\n' +
      '    CI integration tests include the weights file and enforce \u226595% coverage thresholds.\n',
  );
}
