import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
  eslint.configs.recommended,
  // Type-checked rules for source files only
  ...tseslint.configs.recommendedTypeChecked.map((c) => ({
    ...c,
    files: ['packages/*/src/**/*.ts'],
  })),
  // Relaxed rules for test and config files
  ...tseslint.configs.recommended.map((c) => ({
    ...c,
    files: ['packages/*/test/**/*.ts', '*.config.ts'],
  })),
  {
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.eslint.json'],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unsafe-assignment': 'off',
      '@typescript-eslint/no-unsafe-member-access': 'off',
      '@typescript-eslint/no-unsafe-call': 'off',
      '@typescript-eslint/no-unsafe-return': 'off',
      '@typescript-eslint/no-unsafe-argument': 'off',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
      '@typescript-eslint/restrict-template-expressions': 'off',
      '@typescript-eslint/no-redundant-type-constituents': 'off',
      'no-console': 'off',
    },
  },
  {
    ignores: [
      '**/dist/**',
      '**/node_modules/**',
      '**/coverage/**',
      'scripts/**',
      'eslint.config.mjs',
      'package-lock.json',
      'models/**',
    ],
  },
);
