# embed-code-ts Usage Documentation

> Node.js/TypeScript nomic-embed-code — Code-Aware Text Embeddings

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Getting the Model](#2-getting-the-model)
3. [API Usage](#3-api-usage)
4. [CLI Tools](#4-cli-tools)
5. [Configuration Reference](#5-configuration-reference)
6. [Output Description](#6-output-description)
7. [Model Update](#7-model-update)
8. [Troubleshooting](#8-troubleshooting)
9. [Performance Guide](#9-performance-guide)

---

## 1. Quick Start

### Installation

```bash
git clone https://github.com/AgentiX-E/embed-code-ts.git
cd embed-code-ts
npm install
npm run build:all
```

### Minimal Example (requires ONNX model file, see Section 2)

```typescript
import { EmbedCode, downloadModel } from '@agentix-e/embed-code-core';

// 1. Download model (first time only, cached thereafter)
const modelPath = await downloadModel({
  onProgress: (received, total, speed) => {
    console.log(`${received.toFixed(0)} / ${total.toFixed(0)} MB @ ${speed.toFixed(1)} MB/s`);
  },
});

// 2. Create embedder
const embedder = await EmbedCode.fromPretrained({ modelPath });

// 3. Generate embeddings
const results = await embedder.embed([
  'search_query: How to sort an array in Python?',
  'search_document: def quicksort(arr): return arr if len(arr) <= 1 else quicksort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x > arr[0]])',
]);

// 4. Use results
console.log('Embeddings shape:', results.embeddings.length); // 7168 (= 2 × 3584)
console.log('Time:', results.elapsedMs, 'ms');

// 5. Compute similarity
const dim = 3584;
const sim = embedder.similarity(results.embeddings.slice(0, dim), results.embeddings.slice(dim));
console.log('Similarity:', sim);

// 6. Release resources
await embedder.dispose();
```

> **For full API, configuration, CLI usage** → see subsequent sections of this document.
> **For model export and automation pipeline** → see [MODEL-UPDATE.md](MODEL-UPDATE.md).

---

## 2. Getting the Model

### 2.1 Auto Download (Recommended)

```bash
npm install @agentix-e/embed-code-core onnxruntime-node
```

```typescript
import { downloadModel } from '@agentix-e/embed-code-core';

const modelPath = await downloadModel();
// → ~/.cache/embed-code-ts/nomic-embed-text-v1.5-int8.onnx
```

The model is downloaded once, SHA-256 verified, and cached for all subsequent uses.

### 2.2 One-Click Full Pipeline

```bash
# Node.js — One-click: Download → Export → Validate → Test → Benchmark
npm run pipeline

# Or pure Node.js
node scripts/pipeline.js
```

> See [MODEL-UPDATE.md](MODEL-UPDATE.md) for details.

### 2.3 Manual Export from HuggingFace

**Step 1: Install Python dependencies**

```bash
pip install optimum onnx onnxruntime torch
```

**Step 2: Run the export script**

```bash
# Basic usage — auto-download nomic-embed-code and export int8 ONNX
python3 scripts/export-onnx.py \
  --output models/nomic-embed-text-v1.5-int8.onnx \
  --model nomic-ai/nomic-embed-code \
  --precision int8

# Skip validation (faster)
python3 scripts/export-onnx.py \
  --output models/nomic-embed-text-v1.5-int8.onnx \
  --skip-validation
```

**Export flow**:

```
HuggingFace Hub              Local Disk
┌──────────────────┐        ┌─────────────────────────────────┐
│ nomic-ai/         │ pip    │ models/                          │
│ nomic-embed-code  │──────→│ nomic-embed-text-v1.5-int8.onnx │
│ (safetensors)     │ export │ (~140 MB)                         │
└──────────────────┘        └─────────────────────────────────┘
      ~550 MB                   ~140 MB (int8)
```

### 2.4 Model File Specifications

| Property | Value                                             |
| -------- | ------------------------------------------------- |
| Filename | `nomic-embed-text-v1.5-int8.onnx`                 |
| Size     | ~140 MB                                           |
| Format   | ONNX (opset 18, int8 quantized)                   |
| Input    | `input_ids: [batch, 512]` (int64)                 |
| Input    | `attention_mask: [batch, 512]` (int64)            |
| Output   | `last_hidden_state: [batch, 512, 3584]` (float32) |
| Backend  | ONNX Runtime: CPU / CUDA / DirectML               |

### 2.5 Hardware Requirements

| Component      | Minimum      | Recommended |
| -------------- | ------------ | ----------- |
| RAM            | 512 MB       | 2 GB+       |
| Disk           | 150 MB free  | SSD         |
| GPU (Optional) | 2 GB VRAM    | 4 GB+ VRAM  |
| CPU Mode       | ✅ Available | Fast enough |

---

## 3. API Usage

### 3.1 EmbedCode

Core class through which all operations are performed.

```typescript
import { EmbedCode } from '@agentix-e/embed-code-core';
```

#### `fromPretrained(options)`

Load a pretrained model.

```typescript
const embedder = await EmbedCode.fromPretrained({
  modelPath: './models/nomic-embed-text-v1.5-int8.onnx', // Required
  executionProvider: 'cpu', // Optional: 'cpu' | 'cuda' | 'dml'
  intraOpNumThreads: 4, // Optional: CPU thread count
  skipWarmup: false, // Optional: skip warmup inference
  tokenizerPath: '/path/to/tokenizer.json', // Optional: custom tokenizer
});
```

#### `embed(texts, options?)`

Generate embeddings for one or more texts.

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

#### `embedStream(texts, options?)`

Generate embeddings with streaming progress updates.

```typescript
const stream = embedder.embedStream(texts);
for await (const chunk of stream) {
  console.log('Progress:', chunk.progress);
}
```

#### `similarity(a, b)`

Compute cosine similarity between two embedding vectors.

```typescript
const sim = embedder.similarity(embeddingA, embeddingB);
// → number in range [-1, 1]
```

#### `dispose()`

Release ONNX Runtime resources and GPU memory.

```typescript
await embedder.dispose();
```

### 3.2 Full Example: Code Search

```typescript
import { EmbedCode, downloadModel } from '@agentix-e/embed-code-core';

async function codeSearch() {
  // Load model
  const modelPath = await downloadModel();
  const embedder = await EmbedCode.fromPretrained({ modelPath });

  const { query, document } = embedder.taskPrefixes;

  // Code snippets to search over
  const snippets = [
    'function fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }',
    'function quickSort(arr) { if (arr.length <= 1) return arr; const pivot = arr[0]; return [...quickSort(arr.filter(x => x < pivot)), pivot, ...quickSort(arr.filter(x => x >= pivot))]; }',
    'class BinarySearchTree { insert(value) { /* ... */ } find(value) { /* ... */ } }',
  ];

  // Generate embeddings
  const results = await embedder.embed([
    query + 'recursive fibonacci sequence',
    ...snippets.map((s) => document + s),
  ]);

  const dim = 3584;
  const queryEmb = results.embeddings.slice(0, dim);
  const snippetEmbs = snippets.map((_, i) =>
    results.embeddings.slice((i + 1) * dim, (i + 2) * dim),
  );

  // Compute similarities
  const similarities = snippetEmbs.map((emb) => embedder.similarity(queryEmb, emb));
  const bestIdx = similarities.indexOf(Math.max(...similarities));

  console.log('Best match:', snippets[bestIdx]);
  console.log('Similarity:', similarities[bestIdx].toFixed(4));

  await embedder.dispose();
}

codeSearch();
```

### 3.3 Using Task Prefixes

nomic-embed-code requires task prefixes for optimal embedding quality:

```typescript
const { query, document } = embedder.taskPrefixes;

// Search query (what you're looking for)
const queryEmbedding = await embedder.embed([query + 'authentication and user login service']);

// Code document (what you're searching over)
const codeEmbedding = await embedder.embed([document + 'export class AuthService { /* ... */ }']);
```

---

## 4. CLI Tools

### Basic Usage

```bash
# Download model
embed-code setup

# Generate embedding for text
embed-code embed "function hello() { console.log('Hello World'); }"

# Embed from a file
embed-code embed -f source.py

# With custom model
embed-code embed -m ./custom.onnx -f source.ts

# Show model info
embed-code info
embed-code info -m ./custom.onnx
```

### CLI Parameters

| Parameter              | Required | Description                                          |
| ---------------------- | -------- | ---------------------------------------------------- |
| `<text>`               | ❌       | Text to embed (alternative to -f)                    |
| `-f, --file <path>`    | ❌       | Input file path                                      |
| `-m, --model <path>`   | ❌       | ONNX model file path                                 |
| `-o, --output <path>`  | ❌       | Output file path                                     |
| `--max-tokens <n>`     | ❌       | Max input tokens (default: 512)                      |
| `--no-normalize`       | ❌       | Disable L2 normalization                             |
| `--pooling <strategy>` | ❌       | Pooling: last_token, mean, cls (default: last_token) |

---

## 5. Configuration Reference

### Embed Options Full Parameters

```typescript
interface EmbedOptions {
  maxTokens: number; // Max input token count (default: 512)
  poolingStrategy: 'last_token' | 'mean' | 'cls'; // Pooling (default: 'last_token')
  normalize: boolean; // L2 normalize output (default: true)
  signal?: AbortSignal; // Abort controller signal
  onProgress?: (progress: EmbedProgress) => void; // Progress callback
}

interface EmbedProgress {
  current: number; // Current text index
  total: number; // Total text count
  elapsedMs: number; // Elapsed time
}
```

### Recommended Configuration

```typescript
// Production
{
  maxTokens: 512,
  poolingStrategy: 'last_token',
  normalize: true,
}

// Maximum accuracy
{
  maxTokens: 512,
  poolingStrategy: 'mean',
  normalize: true,
}
```

### Input Length Recommendations

| Scenario              | maxTokens | Notes                        |
| --------------------- | --------- | ---------------------------- |
| Single function/class | 128-256   | Fast, covers most use cases  |
| Full source file      | 512       | nomic-embed-code max context |
| Multi-file contexts   | N/A       | Chunk and embed individually |

---

## 6. Output Description

### Single Text Embedding

```
embeddings: Float32Array(3584) — L2-normalized embedding vector
elapsedMs: number            — Inference time
```

### Batch Embedding

```
embeddings: Float32Array(N × 3584) — Flat array of N embeddings concatenated
elapsedMs: number                — Total inference time
```

### Embedding Properties

| Property   | Value        | Description                                  |
| ---------- | ------------ | -------------------------------------------- |
| Dimensions | 3584         | Fixed embedding size (nomic-embed-code-v1.5) |
| Normalized | true         | L2 norm ≈ 1.0 when normalize is enabled      |
| Type       | Float32Array | 32-bit floating point                        |
| Range      | [-1, 1]      | Values typically in [-0.1, 0.1]              |

---

## 7. Model Update

### Check Current nomic-embed-code Latest Version

nomic-embed-code model versions are published on HuggingFace:
https://huggingface.co/nomic-ai/nomic-embed-code

### Update to Latest Model

```bash
# Re-export latest version
python3 scripts/export-onnx.py \
  --model nomic-ai/nomic-embed-code \
  --output models/nomic-embed-text-v1.5-int8.onnx \
  --precision int8 \
  --skip-validation

# Verify
npm test
```

### Verify Model Compatibility

```bash
# Run all tests to confirm compatibility
npm test
```

If all tests pass, the new model is compatible with current code.

---

## 8. Troubleshooting

### Model Load Failure

```
Error: ONNX engine not loaded. Call load() first.
```

**Solution**:

1. Verify model file exists and is ≥ 100 MB
2. Run `python3 scripts/export-onnx.py --output <path>` to re-export
3. Check that `onnxruntime-node` is installed

### Model Download Fails Behind Proxy

```
Error: fetch failed
```

**Solution**:

```bash
# Option A: Standard proxy env vars
export HTTPS_PROXY=http://proxy.company.com:8080

# Option B: Embed-Code specific env vars
export EMBED_CODE_PROXY_URL=http://proxy.company.com:8080
export EMBED_CODE_PROXY_USERNAME=user
export EMBED_CODE_PROXY_PASSWORD=pass
```

### Out of Memory (OOM)

```
JavaScript heap out of memory
```

**Solution**:

```typescript
// Process large batches in chunks
const CHUNK = 50;
const allEmbeddings = [];
for (let i = 0; i < texts.length; i += CHUNK) {
  const chunk = texts.slice(i, i + CHUNK);
  const { embeddings } = await embedder.embed(chunk);
  allEmbeddings.push(embeddings);
}
```

### ONNX Runtime Not Found

```
Cannot find module 'onnxruntime-node'
```

**Solution**:

```bash
npm install onnxruntime-node
```

---

## 9. Performance Guide

### Expected Inference Speed

| Hardware       | 1 text (512 tokens) | 32 texts (batch) |
| -------------- | ------------------- | ---------------- |
| CPU (32 cores) | 2-5 ms              | 30-60 ms         |
| CPU (8 cores)  | 5-10 ms             | 80-150 ms        |
| GPU (8GB)      | 0.5-1 ms            | 10-20 ms         |
| GPU (24GB)     | 0.2-0.5 ms          | 5-10 ms          |

### Optimization Suggestions

1. **Batch inference**: Process multiple texts together — 5-10x throughput improvement
2. **Use appropriate maxTokens**: Set to the minimum needed for your use case
3. **GPU acceleration**: `executionProvider: 'cuda'` for 5-20x speedup
4. **Cache the embedder**: Reuse the same `EmbedCode` instance for multiple calls
5. **Pooling strategy**: `last_token` is fastest; `mean` and `cls` are slightly slower but may yield better quality

### Memory Optimization

```typescript
// Process large batches in chunks
const CHUNK = 50;
for (let i = 0; i < allTexts.length; i += CHUNK) {
  const chunk = allTexts.slice(i, i + CHUNK);
  const { embeddings } = await embedder.embed(chunk);
  // Save chunk results...
}
```

---

## Architecture Overview

```
┌─────────────────────────────────────────┐
│              embed-code-ts               │
├─────────────────────────────────────────┤
│                                          │
│  Input Text (string)                     │
│      │                                   │
│      ▼                                   │
│  ┌──────────┐    ┌───────────────┐      │
│  │Tokenizer │───→│EmbedCodeInfer │      │
│  │          │    │ (ONNX Runtime) │      │
│  │•BPE      │    │               │      │
│  │•Pad/Trunc│    │ C++ native inf│      │
│  │•Attn Mask│    │ CPU/CUDA/DML  │      │
│  └──────────┘    │ INT8 quantized│      │
│                  └───────────────┘      │
│                         │               │
│                         ▼               │
│                    ┌──────────┐         │
│                    │  Pooler  │         │
│                    │          │         │
│                    │•Last Token│        │
│                    │•Mean Pool│         │
│                    │•CLS Pool │         │
│                    └──────────┘         │
│                         │               │
│                         ▼               │
│                    ┌──────────┐         │
│                    │Normalizer│         │
│                    │          │         │
│                    │•L2 Norm  │         │
│                    └──────────┘         │
│                         │               │
│                         ▼               │
│  Output: Float32Array(3584)             │
│                                          │
└─────────────────────────────────────────┘
```

---

## References

- **Project Repository**: https://github.com/AgentiX-E/embed-code-ts
- **nomic-embed-code on HuggingFace**: https://huggingface.co/nomic-ai/nomic-embed-code
- **Nomic Embed Paper**: https://arxiv.org/abs/2402.01613
- **ONNX Runtime**: https://onnxruntime.ai/
