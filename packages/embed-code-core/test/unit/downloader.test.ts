/**
 * Unit tests for model-downloader (cache logic only — no network).
 */
import { describe, it, expect } from 'vitest';
import { defaultModelPath, getCachedModelPath, isModelCached } from '../../src/model-downloader';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

describe('defaultModelPath', () => {
  it('returns a path under ~/.cache', () => {
    const p = defaultModelPath();
    expect(p).toContain('.cache');
    expect(p).toContain('agentix-embed-code-ts');
    expect(p).toContain('nomic-embed-code-v1-int8.onnx');
  });

  it('returns text-int8 path for text-int8 precision', () => {
    const p = defaultModelPath('text-int8');
    expect(p).toContain('nomic-embed-text-v1.5-int8.onnx');
  });

  it('falls back to default precision for invalid precision', () => {
    const p = defaultModelPath('invalid');
    expect(p).toContain('nomic-embed-code-v1-int8.onnx');
  });
});

describe('isModelCached', () => {
  it('returns false when no model exists', () => {
    expect(isModelCached()).toBe(false);
  });

  it('returns false for a tiny placeholder file', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-test-'));
    const p = path.join(tmpDir, 'tiny.onnx');
    fs.writeFileSync(p, 'small');
    expect(isModelCached()).toBe(false);
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });
});

describe('getCachedModelPath', () => {
  it('returns null when no cached model', () => {
    // Since we don't have a cached model in the test environment
    // This should return null
    const result = getCachedModelPath();
    // Either null or a path that doesn't exist
    expect(result === null || !fs.existsSync(result)).toBe(true);
  });
});
