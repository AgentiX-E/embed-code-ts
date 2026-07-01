# embed-code-ts

> Node.js/TypeScript implementation of Nomic's nomic-embed-code — int8 code embeddings with incbin-style model delivery.

[![CI](https://github.com/AgentiX-E/embed-code-ts/actions/workflows/ci.yml/badge.svg)](https://github.com/AgentiX-E/embed-code-ts/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-TypeDoc-blue)](https://agentix-e.github.io/embed-code-ts/api/)
[![Benchmark Report](https://img.shields.io/badge/benchmark-latest-blue)](https://agentix-e.github.io/embed-code-ts/benchmark/)
[![Coverage](https://img.shields.io/badge/coverage-report-blue)](https://agentix-e.github.io/embed-code-ts/coverage/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![TypeScript](https://img.shields.io/badge/TypeScript-6.0-blue)](https://www.typescriptlang.org/)
[![Node.js](https://img.shields.io/badge/Node.js-%3E%3D22-green)](https://nodejs.org/)

## Overview

**embed-code-ts** brings Nomic's nomic-embed-code to the Node.js ecosystem. It provides state-of-the-art code-aware text embeddings — feed it source code or natural language queries and get normalized embedding vectors, no GPU required.

The model weights are distributed with an **incbin-inspired approach**: the npm package is code-only (~50 KB), while the model is downloaded on-demand from GitHub Releases, SHA-256 verified, and cached locally.

### Architecture

```
Input Text → [Tokenizer] → [inference engine] → [Pooling] → [Normalize] → Embedding
               (BPE)       (nomic-embed-code   (last_token/    (L2 norm)
                             int8 weights)          mean/cls)
```

### Key Features

- **Code-aware embeddings** — optimized for source code and natural language
- **Int8 quantized** — model runs fast on CPU with minimal memory
- **Incbin-style delivery** — npm package is code-only; model downloaded on-demand and cached
- **Task prefix support** — `search_query:`, `search_document:` for semantic search
- **Multiple pooling strategies** — last_token, mean, cls pooling
- **Production-grade** — built on inference engine's native C++ backend (CPU, CUDA, DirectML)
- **Verified accuracy** — cosine similarity preservation verified in benchmarks, see [latest benchmark](https://agentix-e.github.io/embed-code-ts/benchmark/)

## Packages

| Package                      | npm                                                                                                                                    | Description                                                    |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| `@agentix-e/embed-code-core` | [![npm](https://img.shields.io/npm/v/@agentix-e/embed-code-core?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-core) | Core inference engine + tokenizer + pooling + model downloader |
| `@agentix-e/embed-code-cli`  | [![npm](https://img.shields.io/npm/v/@agentix-e/embed-code-cli?color=blue)](https://www.npmjs.com/package/@agentix-e/embed-code-cli)   | CLI tool (includes `embed-code setup` auto model download)     |

> **Layered strategy**: npm packages contain only code (~50 KB), models (~137 MB int8 weights for 7B Qwen2.5-based nomic-embed-code) are downloaded on-demand via GitHub Releases.

## Quick Start

### Option 1 — npm install (recommended, code only)

```bash


# Auto download model ~7 GB (first time only)
node -e "const {downloadModel}=require('@agentix-e/embed-code-core');downloadModel()"
```

### Option 2 — Programmatic usage

```typescript
import { EmbedCode, downloadModel } from '@agentix-e/embed-code-core';

// Auto download model (first time only, cached thereafter)
const modelPath = await downloadModel({
  onProgress: (received, total, speed) => {
    console.log(`${received.toFixed(0)} / ${total.toFixed(0)} MB @ ${speed.toFixed(1)} MB/s`);
  },
});

// Create embedder from pretrained weights
const embedder = await EmbedCode.fromPretrained({ modelPath });

// Generate code embeddings
const results = await embedder.embed([
  'search_query: Calculate the n-th factorial',
  'search_document: def fact(n): return 1 if n <= 1 else n * fact(n-1)',
]);

console.log(results.embeddings); // Float32Array [2, 3584]
console.log(results.elapsedMs); // Inference time

// Compute similarity
const sim = embedder.similarity(results.embeddings.slice(0, 3584), results.embeddings.slice(3584));

await embedder.dispose();
```

### Option 3 — Build from source + HuggingFace export

```bash
git clone https://github.com/AgentiX-E/embed-code-ts.git
cd embed-code-ts && npm install && npm run build:all

# One-click pipeline
npm run pipeline
```

> Detailed docs: [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md) | [docs/MODEL-UPDATE.md](docs/MODEL-UPDATE.md)

## EmbedConfig Reference

| Parameter         | Type    | Default      | Description                             |
| ----------------- | ------- | ------------ | --------------------------------------- |
| `maxTokens`       | number  | 32768        | Maximum input token count               |
| `poolingStrategy` | string  | `last_token` | Pooling: `last_token`, `mean`, or `cls` |
| `normalize`       | boolean | true         | L2 normalize output embeddings          |

## Task Prefixes

nomic-embed-code requires task prefixes for optimal performance:

| Role            | Prefix                    |
| --------------- | ------------------------- |
| Query           | `search_query: {query}`   |
| Code / Document | `search_document: {code}` |

Access via `embedder.taskPrefixes`:

```typescript
const { query, document } = embedder.taskPrefixes;
const results = await embedder.embed([
  query + 'Calculate factorial',
  document + 'def fact(n): return 1 if n <= 1 else n * fact(n-1)',
]);
```

## Output Shape Reference

| Output               | Shape    | Description                         |
| -------------------- | -------- | ----------------------------------- |
| `results.embeddings` | `(N, D)` | Flat Float32Array, N texts × D dims |
| `results.elapsedMs`  | number   | Inference time in milliseconds      |

Where `D` = 3584 (nomic-embed-code-v1.5 hidden dimension).

## Project Structure

```
embed-code-ts/
├── packages/
│   ├── embed-code-core/         # Core inference engine
│   │   ├── src/
│   │   │   ├── index.ts         # Public API
│   │   │   ├── embed-code.ts    # EmbedCode class
│   │   │   ├── errors.ts        # Error hierarchy
│   │   │   ├── types.ts         # Type definitions
│   │   │   ├── tokenizer.ts     # BPE tokenizer
│   │   │   ├── pooling.ts       # Pooling strategies + normalize
│   │   │   ├── model-descriptor.ts # Model descriptor resolver
│   │   │   ├── model-downloader.ts # Model downloader + proxy
│   │   │   ├── inference/
│   │   │   │   └── ts-engine.ts  # inference engine inference engine
│   │   │   └── types/
│   │   │       ├── weights.ts       # inference engine type shims
│   │   │       └── undici.d.ts     # undici ProxyAgent type shims
│   │   └── test/
│   │       ├── unit/               # Fast unit tests (no model needed)
│   │       ├── integration/        # Real-model integration tests
│   │       └── test-fixtures.ts    # Deterministic test fixtures
│   └── embed-code-cli/         # CLI tool
│       ├── src/
│       │   └── cli.ts          # Commander-based CLI
│       └── README.md
├── scripts/
│   ├── pipeline.js              # Node.js fully automated pipeline
│   ├── benchmark-ci.js          # CI benchmark runner
│   ├── ci-benchmark-check.js    # Benchmark quality gate
│   ├── ci-coverage-check.js     # Coverage threshold verification
│   ├── prepare-pages.js         # GitHub Pages preparation
│   └── export-weights.py           # PyTorch → int8 weights exporter
├── benchmarks/
│   └── baseline.json            # Performance regression baseline
├── .github/
│   └── workflows/               # CI/CD automation
│       ├── ci.yml               # PR checks + integration tests + benchmark + deploy
│       ├── release.yml          # Tag-triggered npm publish (OIDC)
│       ├── model-release.yml    # Model channel: export → validate → GH Release
│       └── nightly.yml          # Daily HF revision check
├── models/                      # Model descriptor (model weights gitignored)
├── vitest.config.ts             # Integration test config (≥95% thresholds)
├── vitest.unit.config.ts        # Unit test config (≥95% thresholds)
└── vitest.globalSetup.ts        # Model availability check
```

## Development

```bash
# Install
npm install && npm run build:all

# One-click full pipeline (model export + tests + benchmarks)
npm run pipeline

# Run tests only
npm test
npm run test:watch

# Unit tests only (no model needed, fast)
npm run test:unit

# Unit tests with coverage (≥95% thresholds enforced)
npm run test:unit:coverage

# Full coverage report (unit + integration, ≥95% thresholds)
npm run test:coverage

# Lint + format
npm run lint
npm run format:check
```

## References

- **Model**: [nomic-ai/nomic-embed-code](https://huggingface.co/nomic-ai/nomic-embed-code) on HuggingFace
- **Paper**: [Nomic Embed: Training a Reproducible Long Context Text Embedder](https://arxiv.org/abs/2402.01613)
- **inference engine**: [github.com/AgentiX-E/embed-code-ts](https://github.com/AgentiX-E/embed-code-ts)

## Known Limitations

- **Max context**: 32768 tokens per input (nomic-embed-code native limit based on Qwen2.5-7B)
- **No fine-tuning API**: The model runs in zero-shot mode. Fine-tuning is not yet supported.
- **Model version**: Currently supports nomic-embed-code-v1. Future model versions will be supported via the model descriptor system.
- **Task prefixes required**: For optimal performance, inputs should use `search_query:` or `search_document:` prefixes.

## Documentation & Reports

| Resource          | Description                                                                      | URL                                                                                                      |
| ----------------- | -------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 📚 **API Docs**   | Full TypeDoc reference for all packages                                          | [agentix-e.github.io/embed-code-ts/api/](https://agentix-e.github.io/embed-code-ts/api/)                 |
| 📊 **Benchmark**  | Inference latency, throughput & embedding quality reports                        | [agentix-e.github.io/embed-code-ts/benchmark/](https://agentix-e.github.io/embed-code-ts/benchmark/)     |
| 📈 **Coverage**   | Line, branch, function & statement coverage (≥95% on all covered source modules) | [agentix-e.github.io/embed-code-ts/coverage/](https://agentix-e.github.io/embed-code-ts/coverage/)       |
| 📦 **npm (core)** | `@agentix-e/embed-code-core`                                                     | [npmjs.com/package/@agentix-e/embed-code-core](https://www.npmjs.com/package/@agentix-e/embed-code-core) |
| 📦 **npm (cli)**  | `@agentix-e/embed-code-cli`                                                      | [npmjs.com/package/@agentix-e/embed-code-cli](https://www.npmjs.com/package/@agentix-e/embed-code-cli)   |

## System Requirements

| Component             | Minimum                 | Recommended                        |
| --------------------- | ----------------------- | ---------------------------------- |
| **OS**                | Linux / macOS / Windows | Linux (production)                 |
| **Node.js**           | ≥ 22.x                  | ≥ 22.x                             |
| **RAM**               | 512 MB                  | 2 GB+                              |
| **Disk (code)**       | 10 MB                   | —                                  |
| **Disk (model)**      | 7 GB                    | SSD                                |
| **GPU** (optional)    | 2 GB VRAM               | 4 GB+ VRAM (CUDA)                  |
| **Python** (optional) | ≥ 3.10                  | Only needed for HuggingFace export |

### Pre-install dependencies

| Usage method                          | Requires pre-install                                   |
| ------------------------------------- | ------------------------------------------------------ |
| **npm install + auto model download** | Node.js ≥ 22 + ``                      |
| **Export model from HuggingFace**     | Python ≥ 3.10 + `pip install torch transformers` |
| **Build from source**                 | Node.js ≥ 22 + npm                                     |

> `` includes prebuilt C++ native modules, supports Linux x64 / arm64, macOS x64 / arm64 (Apple Silicon), Windows x64. **No additional system packages required**.

## CLI Quick Reference

```bash
# Download model
embed-code setup                              # Default: ~/.cache/embed-code-ts/
embed-code setup -o ./models/model.weights.bin       # Custom path
embed-code setup -f                           # Force re-download
embed-code setup --precision int8             # Download INT8 quantized model

# Model info
embed-code info                               # Show model metadata + system info
embed-code info -m ./custom.weights.bin              # Custom model path

# Download with proxy (corporate / restricted networks)
# Option A: Standard environment variables (auto-detected)
export HTTPS_PROXY=http://proxy.company.com:8080
embed-code setup

# Option B: Explicit proxy with authentication
embed-code setup --proxy-url http://proxy.company.com:8080
embed-code setup --proxy-url http://proxy.company.com:8080 --proxy-username user
embed-code setup --proxy-url http://proxy:8080 --proxy-username user --proxy-password pass
# Password is also available from environment variable (more secure):
EMBED_CODE_PROXY_PASSWORD=pass embed-code setup --proxy-url http://proxy:8080 --proxy-username user
# Or via file for Docker/Kubernetes secrets:
EMBED_CODE_PROXY_PASSWORD_FILE=/run/secrets/proxy-password embed-code setup --proxy-url http://proxy:8080 --proxy-username user

# Option C: EMBED_CODE-specific environment variables
EMBED_CODE_PROXY_URL=http://proxy:8080 EMBED_CODE_PROXY_USERNAME=user EMBED_CODE_PROXY_PASSWORD=pass embed-code setup

# Embed (model path priority)
embed-code embed "function fib(n) { return n <= 1 ? n : fib(n-1) + fib(n-2); }"
embed-code embed -f source.ts                            # From file
embed-code embed -m ./custom.weights.bin -f source.py           # Explicit model path
EMBED_CODE_MODEL_PATH=./prod.weights.bin embed-code embed "..." # Environment variable

# Model path resolution priority: ① --model ② $EMBED_CODE_MODEL_PATH ③ default cache ④ auto download
```

## License

This project is open source under [Apache 2.0](LICENSE).

### Relationship with Nomic nomic-embed-code

- nomic-embed-code ([nomic-ai/nomic-embed-code](https://huggingface.co/nomic-ai/nomic-embed-code)) is licensed under **Apache 2.0**
- The TypeScript/Node.js code in this project is an **original implementation**, also released under Apache 2.0
- nomic-embed-code pretrained model weights (downloaded from HuggingFace) follow Nomic's model license terms
- This project's `scripts/export-weights.py` is used to help users export models, does not directly distribute model weights
- Weight files in GitHub Releases are derivative works exported by users from HuggingFace

### License compatibility

| Component                | License            | Description        |
| ------------------------ | ------------------ | ------------------ |
| embed-code-ts code       | Apache 2.0         | Fully original     |
| nomic-embed-code weights | Apache 2.0 (Nomic) | HuggingFace hosted |
| inference engine             | MIT (Microsoft)    | npm dependency     |
