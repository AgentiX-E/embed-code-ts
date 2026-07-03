import { defineConfig } from 'vitest/config';

/**
 * Unit test configuration — all pure-logic tests (no ONNX model needed).
 * Core: tokenizer, pooler, normalizer, vocab-loader, errors, pre-tokenizer.
 *
 * Targets ≥95% on all four coverage metrics.
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
        'packages/embed-code-core/src/types.ts',
        'packages/embed-code-core/src/embedder-interface.ts',
        'packages/embed-code-core/src/ort-backend-interface.ts',
        'packages/embed-code-core/src/model-descriptor.ts',
        'packages/embed-code-core/src/model-downloader.ts',
        'packages/*/src/types/**/*.d.ts',
      ],
      reporter: ['text', 'text-summary', 'json-summary', 'lcov', 'html'],
      reportsDirectory: './coverage',
    },
  },
});
