# @agentix-e/embed-code-ts

> nomic-embed-code int8 ONNX weights embedded in a TypeScript npm package — incbin-style, offline-first.

**State-of-the-art code embeddings. Ship AI directly in your npm dependencies.**

## Philosophy

Inspired by the C/C++ `incbin` technique — where binary data is compiled directly into the executable at build time — `@agentix-e/embed-code-ts` embeds the int8-quantized ONNX model weights as base64-encoded TypeScript constants. No network requests at runtime. No external file dependencies. Just `import` and inference.

## Architecture

The package implements a **three-layer embedding strategy** to gracefully handle models of any size:

| Layer | Name | Size Range | Mechanism | Used For |
|-------|------|-----------|-----------|----------|
| **1** | Fully Embedded | < 200MB | Base64 TS constants | Tokenizer, config, small models |
| **2** | Chunked Embedded | 200MB–2GB | Lazy-loaded 50MB chunks | Medium distilled models |
| **3** | Fingerprint + Download | > 2GB | SHA256 manifest, CDN fetch | nomic-embed-code (7B) |

This architecture is **future-proof**: when Nomic releases a distilled/smaller nomic-embed-code, upgrading from Layer 3 to Layer 1 is a one-line registry change.

## Quick Start

```bash
npm install @agentix-e/embed-code-ts onnxruntime-node
```

```typescript
import { EmbedCode } from '@agentix-e/embed-code-ts';

// Create an embedder (auto-downloads model on first use with caching)
const embedder = await EmbedCode.create({
  model: 'nomic-embed-code-v1',
  cacheDir: './.cache/embed-code',
  onProgress: (p) => console.log(`Downloading: ${p.percent}%`),
});

// Generate code embeddings
const results = await embedder.embed([
  'search_query: Calculate the n-th factorial',
  'search_document: def fact(n): return 1 if n <= 1 else n * fact(n-1)',
]);

// results.embeddings → Float32Array [2, 3584]
// results.elapsedMs → inference time

// Compute similarity
const similarity = embedder.similarity(
  results.embeddings.slice(0, 3584),
  results.embeddings.slice(3584),
);

embedder.dispose();
```

## Task Prefixes

nomic-embed-code requires task prefixes for optimal performance:

- **Queries**: `search_query: {your query}`
- **Code/Documents**: `search_document: {your code}`

## Supported Models

| Model | Parameters | Embedding Dim | Provider | Status |
|-------|-----------|---------------|----------|--------|
| `nomic-embed-code-v1` | 7B | 3584 | Remote | ✅ |
| `nomic-embed-text-v1.5` | 137M | 768 | Embedded | ✅ |

## API

### `EmbedCode.create(options)`

Creates and initializes an embedding engine.

```typescript
interface EmbedCodeOptions {
  model?: string;                    // Model ID from registry
  provider?: 'embedded' | 'chunked' | 'remote';
  cacheDir?: string;                 // Cache directory for downloads
  onProgress?: (progress: DownloadProgress) => void;
  maxTokens?: number;               // Max tokens per input
  normalize?: boolean;              // L2 normalize embeddings
}
```

### `embedder.embed(texts)`

Generate embeddings for one or more text inputs.

### `embedder.similarity(a, b)`

Compute cosine similarity between two embedding vectors.

### Static Methods

- `EmbedCode.listModels()` — List available model IDs
- `EmbedCode.getModels()` — Get detailed model info
- `EmbedCode.resolveModel(id)` — Resolve a specific model

## Build System

The incbin build pipeline:

```bash
# Download model + generate weight files
npm run build:weights

# Full build (weights + TypeScript bundle)
npm run build:all

# Build specific model
npm run build:weights -- --model nomic-embed-text-v1.5

# Dry run (see what would be built)
npm run build:weights -- --dry-run
```

## Extending

Adding a new model is a matter of adding one entry to the registry in `src/registry.ts`:

```typescript
'my-new-model': {
  id: 'my-new-model',
  label: 'My New Model',
  provider: 'embedded',  // or 'chunked' or 'remote'
  precision: 'int8',
  target: {
    repository: 'org/model-name',
    onnxFile: 'onnx/model_int8.onnx',
  },
  runtime: {
    embeddingDim: 1024,
    maxTokens: 8192,
    poolingStrategy: 'mean',
    normalize: true,
  },
  version: 1,
},
```

Then run `npm run build:weights -- --model my-new-model`.

## License

Apache-2.0 — see [LICENSE](./LICENSE)

The nomic-embed-code and nomic-embed-text models are also Apache-2.0 licensed by Nomic AI.
