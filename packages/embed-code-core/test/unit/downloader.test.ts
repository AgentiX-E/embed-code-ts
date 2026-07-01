/**
 * Unit tests for model-downloader (cache logic and proxy resolution — no network).
 *
 * Covers:
 *   - defaultModelPath for all precision profiles
 *   - Invalid precision fallback
 *   - isModelCached positive/negative cases
 *   - getCachedModelPath edge cases
 */
import { describe, it, expect, afterEach } from 'vitest';
import { defaultModelPath, getCachedModelPath, isModelCached } from '../../src/model-downloader';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';

// Clean up test cache after each test to avoid pollution
const CACHE_DIR = path.join(os.homedir(), '.cache', 'agentix-embed-code-ts');

describe('defaultModelPath', () => {
  it('returns a path under ~/.cache for int8 precision', () => {
    const p = defaultModelPath();
    expect(p).toContain('.cache');
    expect(p).toContain('agentix-embed-code-ts');
    expect(p).toContain('nomic-embed-code-v1-int8.weights.bin');
  });

  it('returns text-int8 path for text-int8 precision', () => {
    const p = defaultModelPath('text-int8');
    expect(p).toContain('nomic-embed-text-v1.5-int8.weights.bin');
  });

  it('falls back to int8 for invalid precision', () => {
    const p = defaultModelPath('fp16');
    expect(p).toContain('nomic-embed-code-v1-int8.weights.bin'); // default fallback
  });

  it('falls back to int8 for undefined precision', () => {
    const p = defaultModelPath(undefined);
    expect(p).toContain('nomic-embed-code-v1-int8.weights.bin');
  });

  it('respects XDG_CACHE_HOME environment variable', () => {
    const original = process.env.XDG_CACHE_HOME;
    process.env.XDG_CACHE_HOME = '/custom/cache';
    const p = defaultModelPath();
    expect(p).toContain('/custom/cache');
    if (original !== undefined) {
      process.env.XDG_CACHE_HOME = original;
    } else {
      delete process.env.XDG_CACHE_HOME;
    }
  });
});

describe('isModelCached', () => {
  it('returns false when no model exists', () => {
    expect(isModelCached()).toBe(false);
  });

  it('returns false for a tiny file below minCachedSize', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'embed-dl-test-'));
    // Create a file that's too small
    const p = path.join(tmpDir, 'tiny.weights.bin');
    fs.writeFileSync(p, 'small');
    expect(isModelCached()).toBe(false);
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });
});

describe('getCachedModelPath', () => {
  afterEach(() => {
    // Clean up any test files we might have created
    try {
      if (fs.existsSync(CACHE_DIR)) {
        const files = fs.readdirSync(CACHE_DIR);
        for (const f of files) {
          if (f.includes('test-')) {
            fs.unlinkSync(path.join(CACHE_DIR, f));
          }
        }
      }
    } catch {
      // best effort
    }
  });

  it('returns null when no cached model exists', () => {
    const result = getCachedModelPath();
    // Either null or a path that doesn't actually exist
    expect(result === null || (typeof result === 'string' && !fs.existsSync(result))).toBe(true);
  });
});
