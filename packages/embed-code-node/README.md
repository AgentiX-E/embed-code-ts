# @agentix-e/embed-code-node

> Node.js ONNX Runtime adapter for nomic-embed-code — AVX2 native acceleration.

[![npm](https://img.shields.io/npm/v/@agentix-e/embed-code-node?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-node)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-node_src_onnx-embedder.html)

## Overview

`@agentix-e/embed-code-node` provides the **Node.js** ONNX Runtime inference engine for nomic-embed-code. It leverages `onnxruntime-node` for native AVX2/AVX-512 acceleration with zero-copy tensor operations.

## Installation

```bash
npm install @agentix-e/embed-code-node @agentix-e/embed-code-core
```

Requires Node.js ≥ 22.

## Quick Start

```typescript
import { NodeEmbedder } from '@agentix-e/embed-code-node';

const embedder = await NodeEmbedder.create({
  modelPath: './models/nomic-embed-code-v1.5.int8.onnx',
});

const embedding = await embedder.embed(
  'function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }',
);
// → Float32Array(768), L2 normalized

await embedder.dispose();
```

### Batch Embedding

```typescript
const embeddings = await embedder.embedBatch(
  ['search_query: sorting algorithm', 'search_document: def quicksort(arr): ...'],
  { concurrency: 4, onProgress: (done, total) => console.log(`${done}/${total}`) },
);
```

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/embed-code-ts/api/modules/embed-code-node_src_onnx-embedder.html](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-node_src_onnx-embedder.html)

Key exports:

- `NodeEmbedder` — ONNX Runtime Node.js embedder implementing `IEmbedder`
- `NodeEmbedderOptions` — Configuration for model path and tokenizer

## Performance

| Backend            | Hardware | Latency |
| ------------------ | -------- | ------- |
| `onnxruntime-node` | AVX-512  | ~5ms    |
| `onnxruntime-node` | AVX2     | ~12ms   |

## License

Apache 2.0
