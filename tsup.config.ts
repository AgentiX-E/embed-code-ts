import { defineConfig } from 'tsup';

export default defineConfig([
  {
    entry: {
      index: 'src/index.ts',
      'platform/node': 'src/platform/node.ts',
      'platform/web': 'src/platform/web.ts',
    },
    format: ['esm', 'cjs'],
    dts: true,
    sourcemap: true,
    clean: true,
    splitting: false,
    treeshake: true,
    minify: false,
    external: ['onnxruntime-node', 'onnxruntime-web'],
  },
]);
