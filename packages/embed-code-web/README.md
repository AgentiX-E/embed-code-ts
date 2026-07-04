# @agentix-e/embed-code-web

> Browser ONNX Runtime adapter for nomic-embed-code — WASM + WebGPU.

[![npm](https://img.shields.io/npm/v/@agentix-e/embed-code-web?color=orange)](https://www.npmjs.com/package/@agentix-e/embed-code-web)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-web_src_onnx-embedder.html)
[![Benchmark](https://img.shields.io/badge/benchmark-report-blue)](https://agentix-e.github.io/embed-code-ts/benchmark/)

## Overview

`@agentix-e/embed-code-web` provides the **browser** ONNX Runtime inference engine for nomic-embed-code. It uses `onnxruntime-web` with **WebGPU auto-upgrade** — falls back to WASM when WebGPU is unavailable.

## Installation

```bash
npm install @agentix-e/embed-code-web @agentix-e/embed-code-core
```

## Quick Start

```typescript
import { WebEmbedder } from '@agentix-e/embed-code-web';

// Load tokenizer from JSON (fetch from your server or bundle inline)
const tokenizerJson = await fetch('/models/tokenizer.json').then((r) => r.json());

const embedder = await WebEmbedder.create('/models/nomic-embed-code-v1.5.int8.onnx', tokenizerJson);

const embedding = await embedder.embed('const memo = new Map();');
// → Float32Array(768), L2 normalized

await embedder.dispose();
```

### WebGPU Auto-Upgrade

```typescript
// WebEmbedder.create() tries WebGPU first, then falls back to WASM:
//   WebGPU available → GPU accelerated (~20ms latency)
//   WebGPU unavailable → WASM backend (~50ms latency)
```

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/embed-code-ts/api/modules/embed-code-web_src_onnx-embedder.html](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-web_src_onnx-embedder.html)

Key exports:

- `WebEmbedder` — ONNX Runtime Web embedder implementing `IEmbedder` with WebGPU auto-upgrade

## Performance

| Backend           | Hardware | Latency |
| ----------------- | -------- | ------- |
| `onnxruntime-web` | WebGPU   | ~20ms   |
| `onnxruntime-web` | WASM     | ~50ms   |

## License

Apache 2.0
