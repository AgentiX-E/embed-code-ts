# @agentix-e/embed-code-node

> **Node.js ONNX Runtime code embedding engine** — AVX2/AVX-512 native acceleration for nomic-embed-code. Production-grade, offline-first.

[![npm](https://img.shields.io/npm/v/@agentix-e/embed-code-node?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-node)
[![npm downloads](https://img.shields.io/npm/dm/@agentix-e/embed-code-node?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-node)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-node_src_onnx-embedder.html)
[![Benchmark](https://img.shields.io/badge/benchmark-report-blue)](https://agentix-e.github.io/embed-code-ts/benchmark/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/AgentiX-E/embed-code-ts/blob/main/LICENSE)

## Overview

`@agentix-e/embed-code-node` provides the **Node.js ONNX Runtime inference backend** for nomic-embed-code. Built on `onnxruntime-node`, it delivers native AVX2/AVX-512 SIMD acceleration with zero-copy tensor operations for maximum throughput.

Perfect for: **Node.js servers**, **CLI tools**, **CI/CD pipelines**, **serverless functions**, and **offline/air-gapped environments**.

## Installation

```bash
npm install @agentix-e/embed-code-node @agentix-e/embed-code-core
```

Requires Node.js ≥ 22 on x86_64 with AVX2 support.

## Quick Start

```typescript
import { NodeEmbedder } from '@agentix-e/embed-code-node';

const embedder = await NodeEmbedder.create({
  modelPath: './models/nomic-embed-code-v1.5.int8.onnx',
});

// Single embedding
const embedding = await embedder.embed(
  'function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }',
);
// → Float32Array(768), L2 normalized

// Batch embedding with progress tracking
const embeddings = await embedder.embedBatch(
  [
    'search_query: sorting algorithm',
    'search_document: def quicksort(arr): ...',
    'search_query: binary tree traversal',
  ],
  {
    concurrency: 4,
    onProgress: (done, total) => console.log(`${done}/${total}`),
  },
);

await embedder.dispose();
```

## Features

- **AVX2/AVX-512 acceleration**: Native SIMD via onnxruntime-node
- **Zero-copy tensors**: Direct memory access for minimal overhead
- **Batch processing**: Concurrent embedding with configurable parallelism
- **Progress callbacks**: Real-time feedback for long-running batches
- **Offline-first**: Model weights bundled, no network needed after install
- **TypeScript-first**: Full type declarations included

## Performance

| Backend            | Hardware              | Latency (per token) | Throughput |
| ------------------ | --------------------- | ------------------- | ---------- |
| `onnxruntime-node` | AVX-512 (Xeon/EPYC)   | ~5ms                | ~200 req/s |
| `onnxruntime-node` | AVX2 (Core i7+/Ryzen) | ~12ms               | ~80 req/s  |

Benchmarks updated on every CI run: [Live Benchmark Report](https://agentix-e.github.io/embed-code-ts/benchmark/)

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/embed-code-ts/api/modules/embed-code-node_src_onnx-embedder.html](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-node_src_onnx-embedder.html)

Key exports:

- `NodeEmbedder` — ONNX Runtime Node.js embedder implementing `IEmbedder`
- `NodeEmbedderOptions` — Model path, tokenizer, and runtime configuration

## License

Apache 2.0
