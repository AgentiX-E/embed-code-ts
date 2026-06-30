# @agentix-e/embed-code-ts

> nomic-embed-code int8 ONNX — code embeddings shipped with npm. incbin-style.

**State-of-the-art code embeddings. Ship AI directly in your npm dependencies.**

## Philosophy

Inspired by the C/C++ `incbin` technique and the proven [`@agentix-e/timesfm-ts`](https://github.com/AgentiX-E/timesfm-ts) architecture:

- **npm package is code-only** (~50 KB) — the model weights are distributed as a GitHub Release asset
- **model-descriptor.json** is the "incbin fingerprint" — embedded in the package, it tells the runtime exactly what model to get and how to verify it
- **On first use**, `downloadModel()` fetches the int8 ONNX from GitHub Releases, verifies SHA-256, and caches it to `~/.cache/`
- **Subsequent runs** are pure filesystem reads — zero network, zero latency
- **No postinstall scripts** — explicit, auditable download flow

This is the same pattern used by:
- [`@agentix-e/timesfm-ts`](https://github.com/AgentiX-E/timesfm-ts) — TimesFM time-series forecasting
- **Puppeteer** — Chromium download
- **sharp** — libvips download

## Quick Start

```bash
npm install @agentix-e/embed-code-core onnxruntime-node
```

```typescript
import { EmbedCode, downloadModel } from '@agentix-e/embed-code-core';

// Download model on first use (caches to ~/.cache/)
const modelPath = await downloadModel({
  onProgress: (received, total, speed) => {
    console.log(`${received.toFixed(0)} / ${total.toFixed(0)} MB @ ${speed.toFixed(1)} MB/s`);
  },
});

// Create embedder from pretrained ONNX
const embedder = await EmbedCode.fromPretrained({ modelPath });

// Generate code embeddings
const results = await embedder.embed([
  'search_query: Calculate the n-th factorial',
  'search_document: def fact(n): return 1 if n <= 1 else n * fact(n-1)',
]);

console.log(results.embeddings); // Float32Array [2, 3584]
console.log(results.elapsedMs);  // Inference time

// Compute similarity
const sim = embedder.similarity(
  results.embeddings.slice(0, 3584),
  results.embeddings.slice(3584),
);

await embedder.dispose();
```

## Task Prefixes

nomic-embed-code requires task prefixes:

| Role | Prefix |
|------|--------|
| Query | `search_query: {query}` |
| Code / Document | `search_document: {code}` |

Access via `embedder.taskPrefixes`:
```typescript
const { query, document } = embedder.taskPrefixes;
const results = await embedder.embed([
  query + 'Calculate factorial',
  document + 'def fact(n): return 1 if n <= 1 else n * fact(n-1)',
]);
```

## Architecture

```
┌──────────────────────────────────────────┐
│         @agentix-e/embed-code-core         │
│  ~50 KB code-only npm package             │
│                                            │
│  ┌────────────┐  ┌──────────────────────┐ │
│  │ EmbedCode   │  │ Model Downloader     │ │
│  │ (public API)│  │ downloadModel()      │ │
│  └─────┬──────┘  │ → GitHub Releases    │ │
│        │         │ → SHA-256 verify      │ │
│        │         │ → ~/.cache/           │ │
│        │         └──────────────────────┘ │
│        │                                   │
│  ┌─────▼──────────────────────────────┐   │
│  │ ONNX Inference Engine               │   │
│  │ onnxruntime-node → CPU/CUDA/DML     │   │
│  └────────────────────────────────────┘   │
│                                            │
│  ┌──────────┐ ┌──────────┐ ┌───────────┐ │
│  │ Tokenizer │ │ Pooling  │ │ Descriptor│ │
│  │ (BPE)     │ │ (last/   │ │ Resolver  │ │
│  │           │ │  mean/cls)│ │           │ │
│  └──────────┘ └──────────┘ └───────────┘ │
└──────────────────────────────────────────┘
         │
         │ downloads from
         ▼
┌──────────────────────────────────────────┐
│  GitHub Release: embed-code-latest-int8   │
│  nomic-embed-code-v1-int8.zip             │
│  ├── nomic-embed-code-v1-int8.onnx        │
│  └── model-descriptor.json                │
└──────────────────────────────────────────┘
```

## API Reference

### `downloadModel(options?)`

Downloads the int8 ONNX model from GitHub Releases.

```typescript
const modelPath = await downloadModel({
  precision: 'int8',        // 'int8' (7B) | 'text-int8' (137M)
  force: false,             // Force re-download
  onProgress: (receivedMB, totalMB, speedMBs) => {},
  url: 'https://...',       // Custom URL or mirror
  proxy: { url: 'http://proxy:8080' },
});
// → ~/.cache/agentix-embed-code-ts/nomic-embed-code-v1-int8.onnx
```

### `EmbedCode.fromPretrained(options)`

Creates an embedder from a pretrained ONNX checkpoint.

```typescript
const embedder = await EmbedCode.fromPretrained({
  modelPath: '/path/to/model.onnx',  // Required
  executionProvider: 'cpu',          // 'cpu' | 'cuda' | 'dml'
  intraOpNumThreads: 4,             // CPU thread count
  skipWarmup: false,
  tokenizerPath: '/path/to/tokenizer.json',
});
```

### `embedder.embed(texts, options?)`

Generates embeddings for input texts.

```typescript
const result = await embedder.embed(texts, {
  maxTokens: 512,
  poolingStrategy: 'last_token',
  normalize: true,
  signal: abortController.signal,
  onProgress: (progress) => {},
});
// → { embeddings: Float32Array, shape: [N, dim], elapsedMs: number }
```

### `embedder.similarity(a, b)`

Computes cosine similarity between two embedding vectors.

### `embedder.dispose()`

Releases ONNX session and resources.

### `defaultModelPath()`, `getCachedModelPath()`, `isModelCached()`

Cache management utilities.

## Exporting Models

To export nomic-embed-code from PyTorch to ONNX int8:

```bash
# Install Python deps
pip install optimum onnx onnxruntime torch transformers

# Export with int8 quantization
python3 scripts/export-onnx.py \
  --output models/nomic-embed-code-v1-int8.onnx \
  --model nomic-ai/nomic-embed-code \
  --precision int8

# Or use the full pipeline
npm run pipeline -- --export
```

## GitHub Release

To create a GitHub Release with the exported model:

```bash
npm run release
```

This runs the pipeline, exports the model, and creates a release tagged `embed-code-latest-int8` that `downloadModel()` can fetch.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `EMBED_CODE_PROXY_URL` | Proxy for model download |
| `EMBED_CODE_PROXY_USERNAME` | Proxy username |
| `EMBED_CODE_PROXY_PASSWORD` | Proxy password |
| `EMBED_CODE_MODEL_PATH` | Override model path for pipeline |
| `EMBED_CODE_HF_MODEL` | Override HuggingFace model ID |
| `XDG_CACHE_HOME` | Override cache directory root |

## License

Apache-2.0 — see [LICENSE](./LICENSE)

nomic-embed-code is also Apache-2.0 licensed by Nomic AI.
