# @agentix-e/embed-code-web

> **Browser ONNX Runtime code embedding engine** — WebGPU acceleration with automatic WASM fallback. Run nomic-embed-code entirely in the browser.

[![npm](https://img.shields.io/npm/v/@agentix-e/embed-code-web?color=orange)](https://www.npmjs.com/package/@agentix-e/embed-code-web)
[![npm downloads](https://img.shields.io/npm/dm/@agentix-e/embed-code-web?color=orange)](https://www.npmjs.com/package/@agentix-e/embed-code-web)
[![API Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-web_src_onnx-embedder.html)
[![Benchmark](https://img.shields.io/badge/benchmark-report-blue)](https://agentix-e.github.io/embed-code-ts/benchmark/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/AgentiX-E/embed-code-ts/blob/main/LICENSE)

## Overview

`@agentix-e/embed-code-web` brings **production-grade code embeddings to the browser**. Powered by `onnxruntime-web`, it uses WebGPU when available for GPU-accelerated inference (~20ms) and automatically falls back to WASM (~50ms) when WebGPU is not supported.

Perfect for: **in-browser code search**, **client-side RAG**, **privacy-sensitive applications**, **edge computing**, and **offline-capable web apps**.

## Installation

```bash
npm install @agentix-e/embed-code-web @agentix-e/embed-code-core
```

Works in all modern browsers: Chrome, Firefox, Safari, Edge.

## Quick Start

```typescript
import { WebEmbedder } from '@agentix-e/embed-code-web';

// Load tokenizer (fetch from your server or bundle inline)
const tokenizerJson = await fetch('/models/tokenizer.json').then(r => r.json());

// Create embedder — auto-selects WebGPU or WASM
const embedder = await WebEmbedder.create(
  '/models/nomic-embed-code-v1.5.int8.onnx',
  tokenizerJson,
);

// Generate embedding
const embedding = await embedder.embed('const memo = new Map();');
// → Float32Array(768), L2 normalized

await embedder.dispose();
```

## WebGPU Auto-Upgrade

`WebEmbedder.create()` automatically selects the best available backend:

```
WebGPU available? → GPU-accelerated (~20ms latency)
       ↓ NO
    WASM backend  (~50ms latency)
```

No configuration needed — the library handles backend selection transparently.

## Features

- **WebGPU acceleration**: GPU inference when available (Chrome 113+, Edge 113+)
- **WASM fallback**: Works in all modern browsers
- **Zero server dependency**: Model runs entirely in the browser
- **Privacy-first**: Code never leaves the user's device
- **TypeScript-first**: Full type declarations included
- **Tree-shakeable**: Bundle only what you use

## Performance

| Backend | Hardware | Latency (per token) | Throughput |
| --- | --- | --- | --- |
| `onnxruntime-web` | WebGPU | ~20ms | ~50 req/s |
| `onnxruntime-web` | WASM | ~50ms | ~20 req/s |

Benchmarks updated on every CI run: [Live Benchmark Report](https://agentix-e.github.io/embed-code-ts/benchmark/)

## Browser Support

| Browser | WebGPU | WASM | Status |
| --- | --- | --- | --- |
| Chrome 113+ | ✅ | ✅ | Fully supported |
| Edge 113+ | ✅ | ✅ | Fully supported |
| Firefox 120+ | ⚠️ (nightly) | ✅ | WASM only |
| Safari 17+ | ⚠️ (experimental) | ✅ | WASM only |

## API Documentation

📚 **Full API reference**: [agentix-e.github.io/embed-code-ts/api/modules/embed-code-web_src_onnx-embedder.html](https://agentix-e.github.io/embed-code-ts/api/modules/embed-code-web_src_onnx-embedder.html)

Key exports:

- `WebEmbedder` — ONNX Runtime Web embedder with WebGPU auto-upgrade

## License

Apache 2.0
