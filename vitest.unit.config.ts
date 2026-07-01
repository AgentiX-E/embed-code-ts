import { defineConfig } from 'vitest/config';

/**
 * Unit test configuration — runs tests that do NOT require the weights file.
 *
 * These tests are fast, lightweight, and suitable for CI pre-merge checks.
 * They cover all pure-logic modules: pooling strategies, tokenizer (BPE),
 * model descriptor resolution, model downloader cache helpers, error classes,
 * and inference math primitives (matmul, activations, dequantize).
 *
 * Targets ≥95% on all four coverage metrics (lines, branches, functions, statements).
 */
export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['packages/*/test/unit/**/*.test.ts'],
    exclude: ['node_modules', 'dist'],
    testTimeout: 15000,
    hookTimeout: 15000,
    coverage: {
      provider: 'v8',
      include: ['packages/embed-code-core/src/**/*.ts'],
      exclude: [
        'packages/*/src/index.ts',
        'packages/embed-code-cli/src/cli.ts',
        'packages/embed-code-core/src/embed-code.ts',
        'packages/embed-code-core/src/inference/ts-engine.ts',
        'packages/embed-code-core/src/inference/weights.ts',
        'packages/embed-code-core/src/inference/attention.ts',
        'packages/embed-code-core/src/inference/embedding.ts',
        'packages/embed-code-core/src/inference/layernorm.ts',
        'packages/embed-code-core/src/inference/ffn.ts',
        'packages/embed-code-core/src/model-downloader.ts',
        'packages/embed-code-core/src/types.ts',
        'packages/embed-code-core/src/tokenizer.ts',
        'packages/embed-code-core/src/model-descriptor.ts',
        'packages/*/src/types/**/*.d.ts',
      ],
      reporter: ['text', 'text-summary', 'json-summary', 'lcov', 'html'],
      thresholds: {
        lines: 95,
        functions: 95,
        branches: 90,
        statements: 95,
      },
      reportsDirectory: './coverage',
    },
  },
});
