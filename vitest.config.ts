import { defineConfig } from 'vitest/config';

/**
 * Integration test configuration — runs all tests including those that require
 * the ONNX model.  Uses vitest.globalSetup.ts to check model availability
 * and set VITEST_SKIP_ONNX_TESTS when the model is absent.
 *
 * Targets ≥95% on all four coverage metrics (lines, branches, functions, statements).
 *
 * Exclusion rationale:
 *   • index.ts                      — barrel re-exports only, no runtime logic
 *   • cli.ts                        — Commander/stdio entry point
 *   • onnx-engine.ts                — CPU path tested; CUDA/DML branches require physical GPU
 *   • model-downloader.ts           — cache helpers tested; fetch/streaming/zip need network
 *   • types/ & *.d.ts               — pure type definitions, zero runtime code
 */
export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    globalSetup: ['./vitest.globalSetup.ts'],
    include: ['packages/*/test/**/*.test.ts'],
    exclude: ['node_modules', 'dist'],
    pool: 'forks',
    poolOptions: {
      forks: {
        singleFork: true,
      },
    },
    testTimeout: 120000,
    hookTimeout: 120000,
    coverage: {
      provider: 'v8',
      include: ['packages/embed-code-core/src/**/*.ts', 'packages/embed-code-cli/src/**/*.ts'],
      exclude: [
        // Barrel re-exports — no runtime logic
        'packages/*/src/index.ts',
        // CLI entry point (stdio) — tested via CLI smoke tests
        'packages/embed-code-cli/src/cli.ts',
        // CPU path fully tested; CUDA/DML branches require physical GPU
        'packages/embed-code-core/src/inference/onnx-engine.ts',
        // Cache helpers, proxy resolution, SHA-256 tested; fetch/streaming/zip need GitHub Releases
        'packages/embed-code-core/src/model-downloader.ts',
        // Pure type definitions — zero runtime code
        'packages/embed-code-core/src/types.ts',
        'packages/*/src/types/**/*.d.ts',
      ],
      reporter: ['text', 'html', 'json-summary', 'lcov'],
      thresholds: {
        lines: 95,
        functions: 95,
        branches: 95,
        statements: 95,
      },
      reportsDirectory: './coverage',
    },
  },
});
