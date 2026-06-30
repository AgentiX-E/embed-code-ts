import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  eslint.configs.recommended,
  // Strict type-checked rules for source files
  ...tseslint.configs.recommendedTypeChecked.map((c) => ({
    ...c,
    files: ['packages/*/src/**/*.ts'],
    rules: {
      ...c.rules,
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-non-null-assertion': 'error',
      '@typescript-eslint/consistent-type-imports': ['error', { fixStyle: 'inline-type-imports' }],
      'no-console': ['warn', { allow: ['error', 'warn'] }],
      'no-debugger': 'error',
      'prefer-const': 'error',
      'no-var': 'error',
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
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.eslint.json'],
        tsconfigRootDir: import.meta.dirname,
      },
    },
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
