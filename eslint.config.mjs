import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  eslint.configs.recommended,
  // Strict type-checked rules for pure-logic source files
  // Excludes files that interface with `any`-typed ONNX runtime bindings,
  // dynamic imports, or JSON parsing
  ...tseslint.configs.recommendedTypeChecked.map((c) => ({
    ...c,
    files: ['packages/embed-code-core/src/errors.ts', 'packages/embed-code-core/src/pooling.ts'],
    languageOptions: {
      ...c.languageOptions,
      parserOptions: {
        ...c.languageOptions?.parserOptions,
        project: ['./tsconfig.eslint.json'],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      ...c.rules,
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-non-null-assertion': 'off',
      '@typescript-eslint/consistent-type-imports': ['error', { fixStyle: 'inline-type-imports' }],
      'no-console': ['warn', { allow: ['error', 'warn'] }],
      'no-debugger': 'error',
      'prefer-const': 'error',
      'no-var': 'error',
    },
  })),
  // Relaxed (non-type-checked) rules for ONNX engine and descriptor
  // (use `any` for onnxruntime-node, undici, JSON parsing)
  ...tseslint.configs.recommended.map((c) => ({
    ...c,
    files: [
      'packages/embed-code-core/src/inference/onnx-engine.ts',
      'packages/embed-code-core/src/model-descriptor.ts',
      'packages/embed-code-core/src/tokenizer.ts',
    ],
    rules: {
      ...c.rules,
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-non-null-assertion': 'off',
    },
  })),
  // Relaxed type-checked rules for CLI (dynamic imports, Commander)
  ...tseslint.configs.recommended.map((c) => ({
    ...c,
    files: ['packages/embed-code-cli/src/**/*.ts'],
    rules: {
      ...c.rules,
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-non-null-assertion': 'off',
    },
  })),
  // Relaxed rules for test and config files
  ...tseslint.configs.recommended.map((c) => ({
    ...c,
    files: ['packages/*/test/**/*.ts', '*.config.ts', 'vitest*.ts'],
    rules: {
      ...c.rules,
      '@typescript-eslint/no-explicit-any': 'off',
      'no-console': 'off',
    },
  })),
  {
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    },
  },
  {
    ignores: [
      '**/dist/**',
      '**/node_modules/**',
      '**/coverage/**',
      '**/docs/api/**',
      'scripts/**',
      'eslint.config.mjs',
      'package-lock.json',
      'models/**',
    ],
  },
);
