# @agentix-e/embed-code-core

> Core inference engine for nomic-embed-code — int8 code embeddings for Node.js pure-TypeScript incbin inference.

[![npm](https://img.shields.io/npm/v/@agentix-e/embed-code-core?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-core)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-core.html)

## Overview

`@agentix-e/embed-code-core` is the heart of embed-code-ts — a production-grade Node.js/TypeScript implementation of Nomic's nomic-embed-code model. It provides state-of-the-art code-aware text embeddings with int8 quantization, no GPU required.

### Architecture

```
Input Text → [Tokenizer] → [inference engine] → [Pooling] → [Normalize] → Embedding
               (BPE)        (int8 weights)       (last/mean/    (L2 norm)
                                                cls)
```

## Installation

```bash

```

Requires Node.js ≥ 22.

## Quick Start

```typescript
import { EmbedCode, downloadModel } from '@agentix-e/embed-code-core';

// Auto-download model (~7 GB, first time only, cached thereafter)
const modelPath = await downloadModel({
  onProgress: (received, total, speed) =>
    console.log(`${received.toFixed(0)}/${total.toFixed(0)} MB @ ${speed.toFixed(1)} MB/s`),
});

const embedder = await EmbedCode.fromPretrained({ modelPath });

const results = await embedder.embed([
  'search_query: Calculate the n-th factorial',
  'search_document: def fact(n): return 1 if n <= 1 else n * fact(n-1)',
]);

console.log(results.embeddings); // Float32Array [2, 3584]
console.log(results.elapsedMs); // Inference time

await embedder.dispose();
```

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/embed-code-ts/api/modules/embed-code-core.html](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-core.html)

Key exports:

- `EmbedCode` — Main model class (`fromPretrained`, `embed`, `similarity`, `dispose`)
- `downloadModel` / `defaultModelPath` / `getCachedModelPath` / `isModelCached` — Model download & cache management
- `EmbedCodeInferenceEngine` — inference engine inference engine (advanced use)
- `Tokenizer` — BPE tokenizer (advanced use)
- `poolEmbeddings` / `normalizeEmbeddings` / `cosineSimilarity` — Pooling utilities
- `EmbedOptions` / `EmbeddingResult` / `EmbedProgress` / `DownloadOptions` — Type definitions
- Error hierarchy: `EmbedCodeError`, `ModelNotFoundError`, `DownloadError`, etc.
- Task prefix support via `embedder.taskPrefixes` (`{ query, document }`)

## Model Download

```typescript
import { downloadModel } from '@agentix-e/embed-code-core';

// Default: ~/.cache/embed-code-ts/nomic-embed-text-v1.5-int8.weights.bin
const path = await downloadModel();

// With proxy (corporate network)
const path = await downloadModel({
  proxy: { url: 'http://proxy.company.com:8080', username: 'user', password: 'pass' },
  onProgress: (received, total, speed) => console.log(`${received}/${total} MB @ ${speed} MB/s`),
});
```

Proxy can also be configured via environment variables:

- `EMBED_CODE_PROXY_URL` / `EMBED_CODE_PROXY_USERNAME` / `EMBED_CODE_PROXY_PASSWORD`
- `EMBED_CODE_PROXY_PASSWORD_FILE` — read password from a file (Docker/K8s secrets)
- Standard `HTTPS_PROXY` / `HTTP_PROXY`

## License

Apache 2.0
