import { defineConfig } from 'vitest/config';

/**
 * Integration test configuration — all tests including those requiring ONNX model.
 * Targets ≥95% on all four coverage metrics.
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
      forks: { singleFork: true },
    },
    testTimeout: 120000,
    hookTimeout: 120000,
    coverage: {
      provider: 'v8',
      include: [
        'packages/embed-code-core/src/**/*.ts',
        'packages/embed-code-node/src/**/*.ts',
        'packages/embed-code-cli/src/**/*.ts',
      ],
      exclude: [
        'packages/*/src/index.ts',
        'packages/embed-code-cli/src/cli.ts',
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
