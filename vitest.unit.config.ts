import { defineConfig } from 'vitest/config';

/**
 * Unit test configuration — runs tests that do NOT require the ONNX model.
 *
 * These tests are fast, lightweight, and suitable for CI pre-merge checks.
 * They cover all pure-logic modules: pooling, tokenizer (vocab from buffer),
 * model descriptor, model downloader cache helpers, and error classes.
 *
 * Targets ≥95% on all four coverage metrics (lines, branches, functions, statements).
 *
 * Exclusion rationale:
 *   • index.ts                      — barrel re-exports only
 *   • cli.ts                        — Commander/stdio entry point
 *   • embed-code.ts                 — requires real ONNX model (covered by integration tests)
 *   • onnx-engine.ts                — requires real ONNX model (covered by integration tests)
 *   • model-downloader.ts           — network IO; cache helpers tested separately
 *   • types/ & *.d.ts               — pure type definitions, zero runtime code
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
        // Barrel re-exports — no runtime logic
        'packages/*/src/index.ts',
        // CLI entry point (stdio)
        'packages/embed-code-cli/src/cli.ts',
        // Requires real ONNX model (covered by integration tests)
        'packages/embed-code-core/src/embed-code.ts',
        // Requires real ONNX model (covered by integration tests)
        'packages/embed-code-core/src/inference/onnx-engine.ts',
        // Network IO; cache helpers, proxy resolution, SHA-256 tested in downloader.test.ts
        'packages/embed-code-core/src/model-downloader.ts',
        // Pure type definitions — zero runtime code
        'packages/embed-code-core/src/types.ts',
        // Tokenizer core BPE logic requires real tokenizer.json; unit test covers vocab loading
        'packages/embed-code-core/src/tokenizer.ts',
        // Model descriptor read/resolve requires real model directory; unit test covers fallback
        'packages/embed-code-core/src/model-descriptor.ts',
        // Pure type definitions — zero runtime code
        'packages/*/src/types/**/*.d.ts',
      ],
      reporter: ['text', 'text-summary', 'json-summary', 'lcov', 'html'],
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
